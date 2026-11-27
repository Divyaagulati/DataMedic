"""Universal checks: univariate and multivariate outliers, distribution shape."""

from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine.adapters.outliers import DETECTORS
from datamedic.engine.checks.framework import (
    CheckContext,
    Finding,
    detected,
    not_assessed,
    not_detected,
    pct,
)
from datamedic.engine.checks.spec import CheckSpec
from datamedic.engine.checks.stats import as_float, num, robust_z

MIN_DISTINCT_FOR_OUTLIERS = 11
MULTI_TOP_SHARE = 0.01
MULTI_IFOREST_THRESHOLD = 0.65
MULTI_IFOREST_ONLY_THRESHOLD = 0.7
MULTI_MIN_ROWS = 100


def _continuous(ctx: CheckContext) -> list[str]:
    return [
        c
        for c in ctx.numeric_features()
        if ctx.data[c].nunique(dropna=True) >= MIN_DISTINCT_FOR_OUTLIERS
    ]


def univariate_outliers(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    columns = _continuous(ctx)
    if not columns:
        return [not_assessed(spec, "no numeric feature has more than 10 distinct values.")]
    findings = []
    clean = []
    for column in columns:
        values = as_float(ctx.data[column])
        values = values[np.isfinite(values)]
        if values.size < 4:
            continue
        q1, q3 = np.quantile(values, [0.25, 0.75])
        iqr = q3 - q1
        methods: dict[str, int | None] = {}
        if iqr > 0:
            methods["iqr_1_5"] = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
            methods["iqr_3"] = int(((values < q1 - 3 * iqr) | (values > q3 + 3 * iqr)).sum())
        else:
            methods["iqr_1_5"] = methods["iqr_3"] = None
        sd = values.std(ddof=1)
        methods["z_gt_3"] = int((np.abs(values - values.mean()) / sd > 3).sum()) if sd > 0 else None
        rz = robust_z(values)
        methods["modified_z_gt_3_5"] = int((np.abs(rz) > 3.5).sum()) if rz is not None else None
        extreme = max(methods["iqr_3"] or 0, methods["modified_z_gt_3_5"] or 0)
        if extreme == 0:
            clean.append(column)
            continue
        share = pct(extreme, values.size)
        findings.append(
            detected(
                spec,
                ctx,
                "medium" if share > 1 else "low",
                f"'{column}' has {extreme} extreme values",
                "Some values lie far from the bulk of the column. They may be valid rare "
                "observations, unit or entry errors, or a different population; this check "
                "cannot tell which.",
                {
                    "counts_by_method": methods,
                    "values": int(values.size),
                    "q1": num(q1),
                    "q3": num(q3),
                    "median": num(np.median(values)),
                    "lower_fence_3iqr": num(q1 - 3 * iqr) if iqr > 0 else None,
                    "upper_fence_3iqr": num(q3 + 3 * iqr) if iqr > 0 else None,
                    "skewness": num(pd.Series(values).skew(), 3),
                },
                columns=[column],
                affected_rows=extreme,
                recommendation="Inspect the extreme values with someone who knows how the data "
                "were measured. Do not delete them automatically; if they are valid, consider "
                "robust methods or transformations instead.",
            )
        )
    if not findings:
        return [
            not_detected(
                spec,
                "No numeric feature has extreme values by IQR x 3 or the modified z-score.",
                {"columns_checked": clean},
            )
        ]
    return findings


def multivariate_outliers(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    columns = ctx.numeric_features()
    if len(columns) < 2:
        return [
            not_assessed(spec, f"at least 2 numeric features are needed (found {len(columns)}).")
        ]
    sample, info = ctx.sample(columns, ctx.budget.expensive_rows)
    matrix = np.column_stack([as_float(sample[c]) for c in columns])
    matrix[~np.isfinite(matrix)] = np.nan
    medians = np.nanmedian(matrix, axis=0)
    usable = ~np.isnan(medians)
    matrix = matrix[:, usable]
    used = [c for c, ok in zip(columns, usable, strict=True) if ok]
    if matrix.shape[1] < 2 or matrix.shape[0] < MULTI_MIN_ROWS:
        return [not_assessed(spec, "fewer than 100 rows or 2 usable numeric features.")]
    rows_idx = np.where(np.isnan(matrix))
    matrix[rows_idx] = np.take(medians[usable], rows_idx[1])
    scores = {det.name: det.score(matrix, ctx.seed) for det in DETECTORS}
    k = max(1, int(round(MULTI_TOP_SHARE * matrix.shape[0])))
    tops = [set(np.argsort(-s, kind="stable")[:k].tolist()) for s in scores.values()]
    consensus = set.intersection(*tops)
    iso = scores["isolation_forest"]
    strong = sorted(i for i in consensus if iso[i] >= MULTI_IFOREST_THRESHOLD)
    # ECOD and COPOD score each feature's tails separately, so rows that only break a
    # relationship between features are visible to Isolation Forest alone. They are reported
    # here for review but not counted as flagged (one detector is not a consensus).
    iforest_only = sorted(
        i for i in tops[0] if i not in consensus and iso[i] >= MULTI_IFOREST_ONLY_THRESHOLD
    )
    evidence: dict[str, Any] = {
        "detectors": [f"{d.name} ({d.version})" for d in DETECTORS],
        "rows_scored": int(matrix.shape[0]),
        "columns": used,
        "top_share_per_detector": MULTI_TOP_SHARE,
        "rows_in_all_three_tops": len(consensus),
        "isolation_forest_threshold": MULTI_IFOREST_THRESHOLD,
        "rows_flagged": len(strong),
        "max_isolation_forest_score": num(iso.max(), 3),
        "isolation_forest_only_rows": len(iforest_only),
        "isolation_forest_only_threshold": MULTI_IFOREST_ONLY_THRESHOLD,
    }
    if iforest_only:
        evidence["isolation_forest_only_row_ids_example"] = (
            ctx.frame.loc[sample.index[iforest_only], "_dm_row_id"].astype(int).tolist()[:10]
        )
    if not strong:
        return [
            not_detected(
                spec,
                "No row is unusual by all three detectors at the required strength.",
                evidence,
                columns=used,
                sample_info=info,
            )
        ]
    row_ids = ctx.frame.loc[sample.index[strong], "_dm_row_id"].astype(int).tolist()
    evidence["row_ids_example"] = row_ids[:10]
    affected = len(strong)
    if info is not None:
        affected = int(round(len(strong) * ctx.n_rows / matrix.shape[0]))
        evidence["estimated_rows_in_full_data"] = affected
    return [
        detected(
            spec,
            ctx,
            "low",
            f"{len(strong)} rows have unusual combinations of values",
            "These rows were ranked among the most unusual by Isolation Forest, ECOD and COPOD "
            "simultaneously: their values are jointly extreme across several features, even "
            "when no single value is extreme on its own. Unusual is not the same as wrong.",
            evidence,
            columns=used,
            affected_rows=affected,
            recommendation="Review the flagged rows (row ids in the evidence) for entry or "
            "processing errors. Keep them if they are genuine; consider robust models.",
            sample_info=info,
        )
    ]


def distribution_shape(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    # Binary indicators (<= 2 distinct values) are skipped: their "skewness" only restates the
    # class balance. Count-like columns (0, 1, 2, ...) are kept.
    columns = [
        c
        for c in ctx.numeric_features()
        if ctx.data[c].notna().sum() >= 8 and ctx.data[c].nunique(dropna=True) > 2
    ]
    if not columns:
        return [not_assessed(spec, "no numeric feature has 8 or more values.")]
    findings = []
    for column in columns:
        values = pd.Series(as_float(ctx.data[column])).replace([np.inf, -np.inf], np.nan).dropna()
        skewness = values.skew()
        kurt = values.kurt()
        zeros = float((values == 0).mean())
        reasons = []
        if pd.notna(skewness) and abs(skewness) >= 1:
            reasons.append(f"skewness {skewness:.2f}")
        if pd.notna(kurt) and kurt >= 7:
            reasons.append(f"excess kurtosis {kurt:.1f} (heavy tails)")
        if zeros >= 0.5:
            reasons.append(f"{zeros * 100:.0f}% zeros")
        if not reasons:
            continue
        severity = "low" if (pd.notna(skewness) and abs(skewness) >= 2) or zeros >= 0.5 else "info"
        findings.append(
            detected(
                spec,
                ctx,
                severity,
                f"'{column}' is strongly asymmetric or heavy-tailed",
                "The distribution is not symmetric: " + "; ".join(reasons) + ". This is common "
                "for counts, amounts and lab values and is not an error.",
                {
                    "skewness": num(skewness, 3),
                    "excess_kurtosis": num(kurt, 3),
                    "zero_share_pct": num(zeros * 100, 2),
                    "values": int(values.size),
                },
                columns=[column],
                recommendation="For models sensitive to scale (linear models, distance-based "
                "methods) consider a log or rank transformation fitted on training data. "
                "Tree models usually need none.",
            )
        )
    return findings or [
        not_detected(
            spec,
            "No numeric feature is strongly skewed, heavy-tailed or zero-inflated.",
            {"columns_checked": columns},
        )
    ]

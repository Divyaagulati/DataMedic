"""Universal checks: missingness, duplicates, variance, cardinality, suspicious values."""

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine.checks.framework import (
    CheckContext,
    Finding,
    detected,
    not_detected,
    pct,
)
from datamedic.engine.checks.spec import CheckSpec
from datamedic.engine.checks.stats import num

HIGH_CARD_MIN_UNIQUE = 50
HIGH_CARD_RATIO = 0.5
NEAR_CONSTANT_SHARE = 0.95
PLACEHOLDER_NUMBERS = (-9999.0, -999.0, -99.0, 999.0, 9999.0, 99999.0)
PLACEHOLDER_TEXT = frozenset(
    {
        "?",
        "-",
        "--",
        "---",
        ".",
        "unknown",
        "unk",
        "missing",
        "none",
        "not available",
        "not applicable",
        "n.a.",
        "n.a",
        "na.",
        "null.",
        "undefined",
        "tbd",
        "xxx",
        "999",
    }
)


def _missing_severity(share: float) -> str:
    if share >= 50:
        return "high"
    if share >= 20:
        return "medium"
    if share >= 5:
        return "low"
    return "info"


def missing_by_column(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    data = ctx.data
    counts = data.isna().sum()
    affected = counts[counts > 0]
    if affected.empty:
        return [
            not_detected(spec, "No column has missing values.", {"columns_checked": len(counts)})
        ]
    findings = []
    for column, n_missing in affected.sort_values(ascending=False).items():
        share = pct(int(n_missing), ctx.n_rows)
        empty = int(n_missing) == ctx.n_rows
        severity = "high" if empty else _missing_severity(share)
        findings.append(
            detected(
                spec,
                ctx,
                severity,
                f"'{column}' is entirely empty"
                if empty
                else f"'{column}' has {share:g}% missing values",
                f"{int(n_missing)} of {ctx.n_rows} rows have no value in this column.",
                {"missing": int(n_missing), "missing_pct": share, "rows": ctx.n_rows},
                columns=[str(column)],
                affected_rows=int(n_missing),
                recommendation=(
                    "An empty column carries no information; consider excluding it."
                    if empty
                    else "Find out why values are missing before imputing. Consider a "
                    "missing-value indicator, imputation fitted on training data only, or "
                    "excluding the column if it is mostly empty."
                ),
            )
        )
    return findings


def missing_patterns(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    indicators = ctx.data.isna()
    partly = [c for c in indicators.columns if 0 < indicators[c].sum() < ctx.n_rows]
    findings: list[Finding] = []
    pairs: list[dict[str, Any]] = []
    if len(partly) >= 2:
        corr = indicators[partly].astype(float).corr()  # phi coefficient for 0/1 indicators
        for i, a in enumerate(partly):
            for b in partly[i + 1 :]:
                phi = corr.at[a, b]
                if pd.notna(phi) and phi >= 0.5:
                    both = int((indicators[a] & indicators[b]).sum())
                    pairs.append(
                        {"a": str(a), "b": str(b), "phi": num(phi, 3), "both_missing": both}
                    )
    pairs.sort(key=lambda p: -float(p["phi"] or 0))
    if pairs:
        cols = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
        findings.append(
            detected(
                spec,
                ctx,
                "low",
                f"{len(pairs)} column pairs tend to be missing together",
                "Values are missing together more often than chance. This often means a shared "
                "cause, such as a skipped form section, a failed device or a merge.",
                {"pairs": pairs[:20], "pairs_total": len(pairs), "threshold_phi": 0.5},
                columns=cols,
                recommendation="Check whether these columns come from the same source or step; "
                "a group-level indicator may describe the pattern better than imputing each "
                "column.",
            )
        )
    n_cols = indicators.shape[1]
    if n_cols >= 4:
        row_share = indicators.sum(axis=1) / n_cols
        sparse_rows = int((row_share >= 0.5).sum())
        if sparse_rows:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium" if pct(sparse_rows, ctx.n_rows) >= 5 else "low",
                    f"{sparse_rows} rows are missing at least half of their values",
                    "These rows contain little information and can distort imputation.",
                    {"rows": sparse_rows, "threshold_share_of_columns": 0.5, "columns": n_cols},
                    affected_rows=sparse_rows,
                    recommendation="Review how these rows were created (partial records, "
                    "incomplete merges). Do not delete them automatically.",
                )
            )
    if not findings:
        return [
            not_detected(
                spec,
                "No columns are missing together and no row is mostly empty.",
                {"columns_with_partial_missingness": len(partly)},
            )
        ]
    return findings


def exact_duplicates(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    dup = ctx.data.duplicated(keep="first")
    n_dup = int(dup.sum())
    if n_dup == 0:
        return [not_detected(spec, "No two rows are identical.", {"rows": ctx.n_rows})]
    in_groups = int(ctx.data.duplicated(keep=False).sum())
    share = pct(n_dup, ctx.n_rows)
    return [
        detected(
            spec,
            ctx,
            "medium" if share >= 5 else "low",
            f"{n_dup} rows duplicate an earlier row",
            f"{n_dup} rows ({share:g}%) are exact copies of an earlier row; {in_groups} rows "
            "are involved in total. Duplicates inflate the weight of some records and can "
            "leak between training and test data.",
            {"duplicate_rows": n_dup, "rows_in_duplicate_groups": in_groups, "share_pct": share},
            affected_rows=n_dup,
            recommendation="Confirm whether these are true repeats (e.g. re-entered records) "
            "or legitimate repeated measurements. Keep duplicates in the same split when "
            "evaluating models.",
        )
    ]


def identifier_duplicates(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    findings = []
    for column in ctx.roles.all("id"):
        values = ctx.data[column]
        present = values.dropna()
        counts = present.value_counts()
        repeated = counts[counts > 1]
        missing = int(values.isna().sum())
        if repeated.empty and missing == 0:
            findings.append(
                not_detected(
                    spec,
                    f"Every value of '{column}' is unique and present.",
                    {"unique": int(counts.size)},
                    columns=[column],
                )
            )
            continue
        rows = int(repeated.sum())
        findings.append(
            detected(
                spec,
                ctx,
                "high" if rows else "medium",
                f"Identifier '{column}' is not unique"
                if rows
                else f"Identifier '{column}' has missing values",
                f"{repeated.size} identifier values occur more than once ({rows} rows); "
                f"{missing} rows have no identifier.",
                {
                    "repeated_values": int(repeated.size),
                    "rows_with_repeated_value": rows,
                    "missing": missing,
                    "max_repeats": int(repeated.max()) if rows else 1,
                    "examples": ctx.examples(column, repeated.index[:5]),
                },
                columns=[column],
                affected_rows=rows + missing,
                recommendation="If one record per identifier is expected, investigate the "
                "repeats before modelling. If repeats are expected (several rows per entity), "
                "declare the column as a group instead.",
            )
        )
    return findings


def constant_columns(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    findings = []
    for column in ctx.features():
        series = ctx.data[column]
        present = series.dropna()
        if present.empty:
            continue  # entirely empty: reported by the missingness check
        counts = present.value_counts()
        top_share = counts.iloc[0] / len(present)
        if counts.size == 1:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium",
                    f"'{column}' is constant",
                    "Every present value is the same, so the column cannot help distinguish rows.",
                    {
                        "distinct_values": 1,
                        "present": int(len(present)),
                        "missing": int(series.isna().sum()),
                    },
                    columns=[column],
                    affected_rows=None,
                    recommendation="Exclude the column from modelling unless its missing "
                    "pattern is meaningful.",
                )
            )
        elif top_share >= NEAR_CONSTANT_SHARE:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "low",
                    f"'{column}' is near-constant",
                    f"One value covers {top_share * 100:.1f}% of present values.",
                    {
                        "top_value_share_pct": num(top_share * 100, 2),
                        "distinct_values": int(counts.size),
                        "threshold_pct": 95,
                    },
                    columns=[column],
                    recommendation="Check whether the rare values matter (rare events can be "
                    "informative). Otherwise the column adds little.",
                )
            )
    return findings or [
        not_detected(
            spec,
            "No feature is constant or near-constant.",
            {"columns_checked": len(ctx.features())},
        )
    ]


def high_cardinality(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    findings = []
    for column in ctx.features("categorical", "text", "identifier"):
        present = ctx.data[column].dropna()
        if present.empty:
            continue
        n_unique = int(present.nunique())
        ratio = n_unique / len(present)
        if ctx.type_of(column) == "identifier" or (
            n_unique > HIGH_CARD_MIN_UNIQUE and ratio > HIGH_CARD_RATIO
        ):
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium" if ratio >= 0.95 else "low",
                    f"'{column}' has very many distinct values",
                    f"{n_unique} distinct values over {len(present)} present rows "
                    f"({ratio * 100:.0f}%). Such columns are usually identifiers or free text; "
                    "used as a category they let a model memorise rows.",
                    {
                        "distinct": n_unique,
                        "present": int(len(present)),
                        "distinct_pct": num(ratio * 100, 2),
                        "inferred_type": ctx.type_of(column),
                    },
                    columns=[column],
                    recommendation="Declare it as an identifier, exclude it, or derive a "
                    "coarser feature from it; avoid one-hot encoding it as is.",
                )
            )
    return findings or [
        not_detected(
            spec,
            "No categorical or text feature has very high cardinality.",
            {"columns_checked": len(ctx.features())},
        )
    ]


def suspicious_values(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    findings = []
    data = ctx.data
    for column in ctx.features("numeric"):
        values = data[column].dropna().astype("float64")
        if values.empty:
            continue
        n_inf = int(np.isinf(values).sum())
        if n_inf:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium",
                    f"'{column}' contains infinite values",
                    f"{n_inf} values are infinite, typically from a division by zero upstream.",
                    {"infinite": n_inf},
                    columns=[column],
                    affected_rows=n_inf,
                    recommendation="Trace how the column was computed; treat infinities as "
                    "missing only once their origin is understood.",
                )
            )
        finite = values[np.isfinite(values)]
        for code in PLACEHOLDER_NUMBERS:
            hits = int((finite == code).sum())
            if hits == 0 or hits / len(finite) < 0.01:
                continue
            rest = finite[finite != code]
            if rest.empty:
                continue
            lo, hi = rest.quantile(0.01), rest.quantile(0.99)
            span = max(hi - lo, 1e-9)
            if lo - 0.5 * span <= code <= hi + 0.5 * span:
                continue  # the code lies within the column's normal range: plausibly real
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium",
                    f"'{column}' may use {code:g} as a placeholder",
                    f"{hits} values equal {code:g}, far outside the range of the other values "
                    f"({num(lo, 4)} to {num(hi, 4)}). Codes like this often mean 'unknown'.",
                    {
                        "value": code,
                        "count": hits,
                        "share_pct": pct(hits, len(finite)),
                        "other_values_p1": num(lo),
                        "other_values_p99": num(hi),
                    },
                    columns=[column],
                    affected_rows=hits,
                    recommendation="Confirm the meaning with the data source. If it means "
                    "'unknown', convert it to missing in a reviewed preprocessing step.",
                )
            )
    for column in ctx.features("categorical", "text"):
        series = data[column].dropna().astype(str)
        if series.empty:
            continue
        normalised = series.str.strip().str.lower()
        mask = normalised.isin(PLACEHOLDER_TEXT)
        hits = int(mask.sum())
        if hits:
            tokens = sorted(set(normalised[mask]))
            findings.append(
                detected(
                    spec,
                    ctx,
                    "low",
                    f"'{column}' contains placeholder text",
                    f"{hits} values look like placeholders for an unknown value.",
                    {
                        "count": hits,
                        "share_pct": pct(hits, len(series)),
                        "placeholders": tokens[:10],
                    },
                    columns=[column],
                    affected_rows=hits,
                    recommendation="Decide whether these mean 'missing'; if so, convert them in "
                    "a reviewed step so they are not treated as a real category.",
                )
            )
    now = (
        pd.Timestamp(ctx.now).tz_localize(None)
        if pd.Timestamp(ctx.now).tzinfo
        else pd.Timestamp(ctx.now)
    )
    for column in ctx.features("datetime"):
        series = data[column].dropna()
        if series.empty:
            continue
        naive = (
            series.dt.tz_convert("UTC").dt.tz_localize(None) if series.dt.tz is not None else series
        )
        future = int((naive > now + timedelta(days=1)).sum())
        ancient = int((naive < pd.Timestamp("1900-01-01")).sum())
        if future or ancient:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium",
                    f"'{column}' has implausible dates",
                    f"{future} dates are in the future and {ancient} are before 1900.",
                    {
                        "future": future,
                        "before_1900": ancient,
                        "checked_against": now.date().isoformat(),
                    },
                    columns=[column],
                    affected_rows=future + ancient,
                    recommendation="Check for data-entry errors, wrong century, or default "
                    "dates. Future dates can be legitimate for scheduled events.",
                )
            )
    if not findings:
        return [
            not_detected(
                spec,
                "No placeholders, infinite values or implausible dates were found.",
                {"columns_checked": len(ctx.features())},
            )
        ]
    return findings

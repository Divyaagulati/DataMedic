"""Universal checks involving relationships: target, correlation, leakage."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from datamedic.engine.checks.framework import (
    CheckContext,
    Finding,
    detected,
    not_assessed,
    not_detected,
    pct,
)
from datamedic.engine.checks.spec import CheckSpec
from datamedic.engine.checks.stats import as_float, binary_auc, cramers_v, num

CORR_THRESHOLD = 0.9
VIF_THRESHOLD = 10.0
MAX_CORR_COLUMNS = 50
CRAMER_THRESHOLD = 0.8
PROXY_AUC = 0.98
PROXY_SPEARMAN = 0.98
PROXY_PURITY = 0.99


def _target(ctx: CheckContext) -> tuple[str, pd.Series]:
    name = ctx.target()
    assert name is not None
    return name, ctx.data[name]


def _encoded_binary(target: pd.Series) -> tuple[np.ndarray, list[str]] | None:
    """Binary target as booleans (positive = the second class in sorted order)."""
    labels = target.dropna().astype(str)
    classes = sorted(labels.unique())
    if len(classes) != 2:
        return None
    return (target.astype(str) == classes[1]).to_numpy(), classes


# ---- Missingness vs target ------------------------------------------------------------------


def missing_vs_target(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    name, target = _target(ctx)
    has_target = target.notna()
    features = [
        c for c in ctx.features() if 0 < ctx.data.loc[has_target, c].isna().sum() < has_target.sum()
    ]
    if not features:
        return [
            not_detected(
                spec,
                "No feature has missing values among rows with a target.",
                {"features_checked": len(ctx.features())},
            )
        ]
    findings = []
    for column in features:
        missing = ctx.data.loc[has_target, column].isna()
        tvals = target[has_target]
        evidence: dict[str, Any]
        if ctx.roles.task_type == "regression":
            y = as_float(tvals)
            a, b = y[missing.to_numpy()], y[~missing.to_numpy()]
            if a.size < 5 or b.size < 5:
                continue
            p = float(stats.mannwhitneyu(a, b).pvalue)
            auc = binary_auc(y, missing.to_numpy())
            effect = None if auc is None else 2 * auc - 1  # rank-biserial correlation
            flagged = effect is not None and abs(effect) >= 0.2 and p < 0.01
            evidence = {
                "rank_biserial": num(effect, 3),
                "p_value": num(p, 6),
                "rows_missing": int(missing.sum()),
            }
            strength = abs(effect or 0)
        else:
            v, p_val, _ = cramers_v(missing.astype(str), tvals)
            flagged = v is not None and p_val is not None and v >= 0.1 and p_val < 0.01
            evidence = {
                "cramers_v": num(v, 3),
                "p_value": num(p_val, 6),
                "rows_missing": int(missing.sum()),
                "target_rate_by_missing": _rates(missing, tvals),
            }
            strength = v or 0
        if flagged:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium" if strength >= 0.3 else "low",
                    f"Missing values in '{column}' are related to '{name}'",
                    "Rows where this column is missing have a different target distribution. "
                    "The fact that a value is missing carries information about the outcome.",
                    evidence,
                    columns=[column, name],
                    affected_rows=int(missing.sum()),
                    recommendation="Find out why the value is missing. If missingness is "
                    "caused by the outcome (e.g. recorded after it), it leaks; if it is a "
                    "legitimate signal, model it explicitly with a missing indicator.",
                )
            )
    return findings or [
        not_detected(
            spec, "Missingness is not associated with the target.", {"features_checked": features}
        )
    ]


def _rates(indicator: pd.Series, target: pd.Series) -> dict[str, dict[str, float]]:
    table = pd.crosstab(
        indicator.map({True: "missing", False: "present"}), target.astype(str), normalize="index"
    )
    return {
        str(k): {str(c): round(float(v) * 100, 2) for c, v in row.items()}
        for k, row in table.iloc[:, :10].iterrows()
    }


# ---- Correlation ----------------------------------------------------------------------------


def numeric_redundancy(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    columns = ctx.numeric_features()[:MAX_CORR_COLUMNS]
    if len(columns) < 2:
        return [
            not_assessed(spec, f"at least 2 numeric features are needed (found {len(columns)}).")
        ]
    sample, info = ctx.sample(columns, ctx.budget.moderate_rows)
    frame = pd.DataFrame({c: as_float(sample[c]) for c in columns}).replace(
        [np.inf, -np.inf], np.nan
    )
    pearson = frame.corr(method="pearson", min_periods=10)
    spearman = frame.corr(method="spearman", min_periods=10)
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            p, s = pearson.at[a, b], spearman.at[a, b]
            strongest = max(abs(p) if pd.notna(p) else 0, abs(s) if pd.notna(s) else 0)
            if strongest >= CORR_THRESHOLD:
                pairs.append(
                    {
                        "a": a,
                        "b": b,
                        "pearson": num(p, 4),
                        "spearman": num(s, 4),
                        "strength": strongest,
                    }
                )
    vifs: dict[str, float | None] = {}
    if len(columns) >= 2 and not pearson.isna().to_numpy().any():
        inverse = np.linalg.pinv(pearson.to_numpy())
        vifs = {c: num(inverse[i, i], 2) for i, c in enumerate(columns)}
    high_vif = {c: v for c, v in vifs.items() if v is None or v >= VIF_THRESHOLD}
    if not pairs and not high_vif:
        return [
            not_detected(
                spec,
                "No pair of numeric features is correlated at |r| >= 0.9 and no VIF reaches 10.",
                {
                    "columns": columns,
                    "max_vif": num(
                        max((v for v in vifs.values() if v is not None), default=None)
                        if vifs
                        else None,
                        2,
                    ),
                },
                sample_info=info,
            )
        ]
    pairs.sort(key=lambda p: -float(p["strength"]))
    strongest = pairs[0]["strength"] if pairs else 0
    for pair in pairs:
        pair.pop("strength")
    severe = strongest >= 0.99 or any(v is None or v >= 100 for v in high_vif.values())
    cols = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs} | set(high_vif))
    return [
        detected(
            spec,
            ctx,
            "medium" if severe else "low",
            f"{len(pairs)} strongly correlated pairs; {len(high_vif)} features with high VIF",
            "Some numeric features carry nearly the same information. This makes linear-model "
            "coefficients unstable and importance scores misleading; it does not say that one "
            "feature causes another.",
            {
                "pairs": pairs[:20],
                "pairs_total": len(pairs),
                "threshold_abs_r": CORR_THRESHOLD,
                "vif": dict(sorted(high_vif.items(), key=lambda kv: -(kv[1] or 1e9)))
                if high_vif
                else {},
                "vif_threshold": VIF_THRESHOLD,
            },
            columns=cols,
            recommendation="Keep one feature per group, combine them, or use regularised or "
            "tree models. Decide with domain knowledge which feature is the primary measure.",
            sample_info=info,
        )
    ]


def categorical_association(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    columns = [
        c
        for c in ctx.features("categorical", "boolean")
        if 2 <= ctx.data[c].nunique(dropna=True) <= 50
    ][:30]
    if len(columns) < 2:
        return [
            not_assessed(
                spec,
                f"at least 2 categorical features with 2-50 levels are "
                f"needed (found {len(columns)}).",
            )
        ]
    sample, info = ctx.sample(columns, ctx.budget.expensive_rows)
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(columns):
        for b in columns[i + 1 :]:
            v, p, n = cramers_v(sample[a], sample[b])
            if v is not None and v >= CRAMER_THRESHOLD:
                pairs.append({"a": a, "b": b, "cramers_v": num(v, 4), "p_value": num(p, 6), "n": n})
    if not pairs:
        return [
            not_detected(
                spec,
                "No pair of categorical features is associated at Cramer's V >= 0.8.",
                {"columns": columns},
                sample_info=info,
            )
        ]
    pairs.sort(key=lambda p: -float(p["cramers_v"] or 0))
    return [
        detected(
            spec,
            ctx,
            "medium" if float(pairs[0]["cramers_v"] or 0) >= 0.95 else "low",
            f"{len(pairs)} pairs of categorical features are strongly associated",
            "These categorical features largely determine one another (for example a code "
            "and its label). They add little information together.",
            {"pairs": pairs[:20], "pairs_total": len(pairs), "threshold": CRAMER_THRESHOLD},
            columns=sorted({p["a"] for p in pairs} | {p["b"] for p in pairs}),
            recommendation="Keep the most meaningful one of each pair, or confirm they encode "
            "different things.",
            sample_info=info,
        )
    ]


# ---- Target ---------------------------------------------------------------------------------


def class_imbalance(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    name, target = _target(ctx)
    counts = target.dropna().astype(str).value_counts()
    if counts.size < 2:
        return [not_assessed(spec, "the target has fewer than two classes (see target quality).")]
    total = int(counts.sum())
    minority_share = counts.iloc[-1] / total * 100
    evidence = {
        "classes": {str(k): int(v) for k, v in counts.head(50).items()},
        "minority_class": str(counts.index[-1]),
        "minority_share_pct": num(minority_share, 3),
        "imbalance_ratio": num(counts.iloc[0] / counts.iloc[-1], 2),
    }
    if minority_share >= 20:
        return [
            not_detected(
                spec,
                f"The smallest class holds {minority_share:.1f}% of rows.",
                evidence,
                columns=[name],
            )
        ]
    severity = "high" if minority_share < 1 else "medium" if minority_share < 10 else "low"
    return [
        detected(
            spec,
            ctx,
            severity,
            f"'{name}' is imbalanced",
            f"The smallest class ('{counts.index[-1]}') has {int(counts.iloc[-1])} rows "
            f"({minority_share:.1f}%). Accuracy will look good even for a model that never "
            "predicts it.",
            evidence,
            columns=[name],
            affected_rows=int(counts.iloc[-1]),
            recommendation="Use stratified splits and metrics that reflect the minority class "
            "(balanced accuracy, PR AUC). Consider class weights; resample only inside "
            "training folds.",
        )
    ]


def target_quality(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    name, target = _target(ctx)
    findings = []
    missing = int(target.isna().sum())
    if missing:
        share = pct(missing, ctx.n_rows)
        findings.append(
            detected(
                spec,
                ctx,
                "high" if share >= 20 else "medium",
                f"{missing} rows have no value for the target '{name}'",
                "Rows without a target cannot be used to train or evaluate a supervised model.",
                {"missing": missing, "missing_pct": share},
                columns=[name],
                affected_rows=missing,
                recommendation="Find out why outcomes are missing; excluding these rows can "
                "bias the sample if missingness depends on the outcome.",
            )
        )
    present = target.dropna()
    n_classes = int(present.astype(str).nunique())
    if ctx.roles.task_type == "regression":
        if as_float(present).std() == 0 or n_classes <= 1:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "high",
                    f"The target '{name}' is constant",
                    "A constant target cannot be predicted meaningfully.",
                    {"distinct_values": n_classes},
                    columns=[name],
                    recommendation="Check that the right column was chosen as the target.",
                )
            )
    else:
        counts = present.astype(str).value_counts()
        if n_classes < 2:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "high",
                    f"The target '{name}' has a single class",
                    "Only one class is present, so no classifier can be trained.",
                    {"classes": n_classes},
                    columns=[name],
                    recommendation="Check that the right column and rows were selected.",
                )
            )
        elif ctx.roles.task_type == "binary" and n_classes != 2:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium",
                    f"The target '{name}' has {n_classes} classes, not 2",
                    "The task type is binary but more than two classes are present.",
                    {"classes": n_classes},
                    columns=[name],
                    recommendation="Choose 'multiclass', or map the classes to two groups "
                    "in a reviewed preprocessing step.",
                )
            )
        rare = counts[counts < 10]
        if n_classes >= 2 and not rare.empty:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "medium",
                    f"{rare.size} target classes have fewer than 10 rows",
                    "Very small classes make cross-validation unreliable: some folds may "
                    "contain none of them.",
                    {"rare_classes": {str(k): int(v) for k, v in rare.head(20).items()}},
                    columns=[name],
                    affected_rows=int(rare.sum()),
                    recommendation="Collect more examples, merge related classes, or use fewer "
                    "folds; report per-class metrics.",
                )
            )
    return findings or [
        not_detected(
            spec,
            "The target is complete and usable.",
            {"classes": n_classes, "missing": 0},
            columns=[name],
        )
    ]


# ---- Leakage --------------------------------------------------------------------------------


def target_proxy(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    name, target = _target(ctx)
    task = ctx.roles.task_type
    findings = []
    checked = 0
    binary = _encoded_binary(target) if task == "binary" else None
    for column in ctx.features():
        if column == name:
            continue
        feature = ctx.data[column]
        both = feature.notna() & target.notna()
        if both.sum() < 20:
            continue
        kind = ctx.type_of(column)
        checked += 1
        evidence: dict[str, object] | None = None
        strength = 0.0
        if kind in ("numeric", "boolean") and binary is not None:
            labels, classes = binary
            auc = binary_auc(as_float(feature[both]), labels[both.to_numpy()])
            if auc is not None:
                strength = max(auc, 1 - auc)
                if strength >= PROXY_AUC:
                    evidence = {"roc_auc": num(auc, 4), "classes": classes, "rows": int(both.sum())}
        elif kind == "numeric" and task == "regression":
            rho = stats.spearmanr(as_float(feature[both]), as_float(target[both])).statistic
            if pd.notna(rho):
                strength = abs(float(rho))
                if strength >= PROXY_SPEARMAN:
                    evidence = {"spearman": num(rho, 4), "rows": int(both.sum())}
        elif kind in ("categorical", "boolean", "text") and task in ("binary", "multiclass"):
            frame = pd.DataFrame({"f": feature[both].astype(str), "t": target[both].astype(str)})
            n_categories = frame["f"].nunique()
            if n_categories <= len(frame) / 5:
                majority = frame.groupby("f")["t"].agg(lambda s: s.value_counts().iloc[0]).sum()
                strength = majority / len(frame)
                if strength >= PROXY_PURITY and frame["t"].nunique() > 1:
                    evidence = {
                        "purity_pct": num(strength * 100, 3),
                        "categories": int(n_categories),
                        "rows": int(len(frame)),
                    }
        if evidence is not None:
            perfect = strength >= 0.999
            findings.append(
                detected(
                    spec,
                    ctx,
                    "critical" if perfect else "high",
                    f"'{column}' almost determines the target '{name}'",
                    "This single feature separates the target "
                    + ("perfectly" if perfect else "nearly perfectly")
                    + ". Genuine predictors are rarely this strong; the feature may be "
                    "recorded after the outcome, derived from it, or a copy of it.",
                    evidence,
                    columns=[column, name],
                    recommendation="Confirm the feature is known at prediction time and does "
                    "not encode the outcome. If in doubt, exclude it and compare results.",
                )
            )
    if checked == 0:
        return [not_assessed(spec, "no feature has 20 or more rows paired with the target.")]
    return findings or [
        not_detected(
            spec,
            "No single feature nearly determines the target.",
            {"features_checked": checked},
            columns=[name],
        )
    ]


def identifier_leakage(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    name, target = _target(ctx)
    candidates = sorted(
        set(ctx.roles.all("id"))
        | {c["name"] for c in ctx.schema if c["inferred_type"] == "identifier"}
    )
    candidates = [c for c in candidates if c != name]
    if not candidates:
        return [not_assessed(spec, "no identifier column is declared or inferred.")]
    if ctx.roles.task_type == "multiclass":
        return [not_assessed(spec, "an ordering test needs a binary or numeric target.")]
    if ctx.roles.task_type == "binary":
        encoded = _encoded_binary(target)
        if encoded is None:
            return [not_assessed(spec, "the binary target does not have exactly two classes.")]
        y = pd.Series(encoded[0].astype(float), index=target.index).where(target.notna())
    else:
        y = pd.Series(as_float(target), index=target.index)
    findings = []
    for column in candidates:
        ident = ctx.data[column]
        order = ident.rank(method="average")  # sort order of the identifier values
        both = order.notna() & y.notna()
        if both.sum() < 30:
            continue
        result = stats.spearmanr(order[both], y[both])
        rho, p = float(result.statistic), float(result.pvalue)
        if np.isfinite(rho) and abs(rho) >= 0.3 and p < 0.001:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "high",
                    f"The order of '{column}' is related to '{name}'",
                    "Records with higher identifier values tend to have different outcomes. "
                    "The data may have been sorted or numbered by outcome, so a model can learn "
                    "the identifier instead of real signal.",
                    {"spearman": num(rho, 3), "p_value": num(p, 8), "rows": int(both.sum())},
                    columns=[column, name],
                    recommendation="Never use identifiers as features. Make sure evaluation "
                    "splits are random with respect to identifier order (or grouped by entity).",
                )
            )
    return findings or [
        not_detected(
            spec,
            "No identifier's order is associated with the target.",
            {"identifiers_checked": candidates},
        )
    ]

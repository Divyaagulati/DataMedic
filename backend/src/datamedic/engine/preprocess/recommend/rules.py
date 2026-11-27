"""Recommendation rules (standard library only).

Each rule reads measured facts (schema counts, profile statistics, detected findings,
declared roles) and either drafts recommendations with their evidence, records why an
option was considered and *not recommended*, or records that the question could not be
assessed (``not_assessed``) and why. Rules never look at column names, never recommend
deleting rows, and never recommend removing a column: selection steps are proposed in
``report`` mode, so the user decides.

Thresholds are module constants so they are documented and testable.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.recommend.facts import (
    ENTITY_ROLES,
    TIME_ROLES,
    Column,
    Facts,
    FindingRef,
)

RULES_VERSION = "1.0.0"

MIN_RELIABLE = 30  # observed values behind "moderate"/"strong" evidence
SMALL_DATA_ROWS = 100  # fewer rows: learned parameters are noisy (ranking penalty)
IMPUTE_MAX_MISSING_PCT = 60.0  # above: imputation would invent most values
INDICATOR_MIN_MISSING_PCT = 20.0  # at or above: missingness itself may carry information
CATEGORY_MODE_MAX_MISSING_PCT = 5.0  # below: most-frequent category; above: own category
SKEW = 1.0  # |skewness| at or above: asymmetric (same as diagnostics distribution.shape)
STRONG_SKEW = 2.0
KNN_MIN_ABS_CORR = 0.3  # neighbour-based imputation needs informative other columns
ITERATIVE_MIN_ABS_CORR = 0.5
HIGH_CARDINALITY = 20  # distinct categories above: one-hot becomes wide
ID_LIKE_UNIQUE_PCT = 90.0  # distinct share at or above: identifier-like
RARE_LEVEL_PCT = 1.0  # categories rarer than this share of present rows are "rare"
MIN_RARE_LEVELS = 2
SELECTION_MIN_FEATURES = 20
SELECTION_MAX_ROWS_PER_FEATURE = 20  # fewer rows per feature: selection may be considered
UNDERSAMPLE_MIN_MINORITY = 500  # smaller minority: undersampling discards too much
SMOTE_MIN_MINORITY = 6  # SMOTE's default k_neighbors = 5 needs 6 minority rows
RESOURCE_HEAVY_ROWS = 20_000  # above: KNN, iterative, mutual information and SMOTE are slow

NO_GUARANTEE = (
    "A recommendation is a hypothesis to test, not a guaranteed improvement: compare it with "
    "the baseline on identical folds in the Experiment Lab."
)


@dataclass
class Draft:
    rule_id: str
    op: str
    columns: tuple[str, ...]
    params: dict[str, Any]
    title: str
    reason: str
    evidence: list[dict[str, Any]]
    expected_benefit: str
    risks: list[str]
    information_loss: tuple[str, str]  # (none | low | moderate | high, detail)
    required_decisions: list[str]
    strength: tuple[str, str]  # (strong | moderate | weak, explanation)
    task_fit: tuple[str, str]  # (relevant | indirect, detail)
    limitations: list[str] = field(default_factory=list)
    conflict_group: str | None = None
    finding_ids: list[str] = field(default_factory=list)
    assumption_note: str | None = None  # set when a statistical assumption is doubtful
    heavy: bool = False  # resource-intensive at larger sizes


class Collector:
    def __init__(self) -> None:
        self.drafts: list[Draft] = []
        self.not_assessed: list[dict[str, Any]] = []
        self.not_recommended: list[dict[str, Any]] = []
        self.advisories: list[dict[str, Any]] = []

    def skip(self, rule_id: str, topic: str, reason: str, columns: Sequence[str] = ()) -> None:
        self.not_assessed.append(
            {"rule_id": rule_id, "topic": topic, "columns": list(columns), "reason": reason}
        )

    def reject(self, rule_id: str, op: str, reason: str, columns: Sequence[str] = ()) -> None:
        self.not_recommended.append(
            {"rule_id": rule_id, "op": op, "columns": list(columns), "reason": reason}
        )

    def advise(
        self,
        code: str,
        title: str,
        detail: str,
        columns: Sequence[str] = (),
        findings: Sequence[FindingRef] = (),
    ) -> None:
        self.advisories.append(
            {
                "code": code,
                "title": title,
                "detail": detail,
                "columns": list(columns),
                "finding_ids": sorted({f.id for f in findings}),
            }
        )


def _ev(source: str, metric: str, value: Any, **extra: Any) -> dict[str, Any]:
    return {"source": source, "metric": metric, "value": value, **extra}


def _finding_ev(f: FindingRef, metric: str, value: Any, column: str | None = None) -> dict:  # type: ignore[type-arg]
    return _ev("finding", metric, value, check_id=f.check_id, finding_id=f.id, column=column)


def _names(columns: Sequence[Column]) -> tuple[str, ...]:
    return tuple(c.name for c in columns)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _task_fit_inputs(facts: Facts) -> tuple[str, str]:
    return (
        "relevant",
        "The linear and k-means baselines need complete numeric inputs; gradient boosting "
        "handles gaps natively, so its results may change less.",
    )


# ---- Missing values ---------------------------------------------------------------------------


def missing_numeric(facts: Facts, out: Collector) -> None:
    rule = "missing.numeric"
    cols = [c for c in facts.features("numeric") if c.n_missing > 0]
    if not cols:
        return
    mostly_empty = [c for c in cols if c.missing_pct >= IMPUTE_MAX_MISSING_PCT]
    if mostly_empty:
        out.reject(
            rule,
            "impute.*",
            f"Missing in at least {IMPUTE_MAX_MISSING_PCT:g}% of rows: imputation would invent "
            "most values. Decide whether the column is usable (an indicator can still record "
            "when a value was measured).",
            _names(mostly_empty),
        )
        out.advise(
            "MOSTLY_EMPTY_COLUMNS",
            "Columns that are mostly empty",
            "These columns are missing in most rows. DataMedic does not remove them; decide "
            "whether to exclude them (role 'Exclude') or keep only a missing-value indicator.",
            _names(mostly_empty),
        )
    cols = [c for c in cols if c.missing_pct < IMPUTE_MAX_MISSING_PCT and c.n_present > 0]
    if not cols:
        return
    findings = [
        f for c in cols for f in facts.detected("universal.missingness.column_rate", c.name)
    ]
    evidence = [
        _ev("schema", "missing_pct", c.missing_pct, column=c.name, missing=c.n_missing)
        for c in cols
    ]
    skewed = [c for c in cols if abs(c.stat("skewness") or 0) >= SKEW]
    for c in skewed:
        evidence.append(_ev("profile", "skewness", c.stat("skewness"), column=c.name))
    worst = max(c.missing_pct for c in cols)
    observed = min(c.n_present for c in cols)
    strength = (
        ("strong", f"Missing counts are exact; up to {worst:g}% of rows are affected.")
        if worst >= 5 and observed >= MIN_RELIABLE
        else ("moderate", f"Exact counts, but at most {worst:g}% of rows are affected.")
        if observed >= MIN_RELIABLE
        else ("weak", f"Some columns have fewer than {MIN_RELIABLE} observed values.")
    )
    decisions = [
        "Confirm why values are missing: median imputation assumes it is unrelated to the "
        "unobserved value itself (this cannot be verified from the data).",
        "Decide whether to keep a missing-value indicator alongside the imputed value.",
    ]
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="impute.median",
            columns=_names(cols),
            params={},
            title=f"Fill missing values in {_plural(len(cols), 'numeric column')} with the "
            "training-fold median",
            reason=f"{_plural(len(cols), 'numeric feature')} have missing values (up to "
            f"{worst:g}% of rows). The median is used rather than the mean because it is not "
            "pulled by skewed distributions or extreme values"
            + (f" ({len(skewed)} of these columns are skewed)." if skewed else "."),
            evidence=evidence
            + [
                _finding_ev(f, "missing_pct", f.evidence.get("missing_pct"), f.columns[0])
                for f in findings
            ],
            expected_benefit="Keeps every row usable by models that cannot handle gaps, without "
            "discarding rows.",
            risks=[
                "Imputed values are estimates: variance shrinks and relationships with other "
                "columns weaken.",
                "If values are missing because they were extreme, the median is biased.",
            ],
            information_loss=(
                "low",
                "Observed values are unchanged; that a value was missing is lost unless an "
                "indicator is added.",
            ),
            required_decisions=decisions,
            strength=strength,
            task_fit=_task_fit_inputs(facts),
            limitations=["Mean imputation is not suggested: it is pulled by skew and outliers."],
            conflict_group="impute.numeric",
            finding_ids=[f.id for f in findings],
        )
    )
    _neighbour_imputation(facts, out, cols, findings)


def _neighbour_imputation(
    facts: Facts, out: Collector, cols: list[Column], findings: list[FindingRef]
) -> None:
    rule = "missing.numeric"
    corr = facts.inp.profile_correlations
    if corr is None or corr.get("status") != "ok":
        out.skip(
            rule,
            "neighbour-based imputation (KNN, iterative)",
            "No correlation profile is available, so there is no evidence that other columns "
            "can predict the missing values.",
            _names(cols),
        )
        return
    others = [c.name for c in facts.features("numeric", "boolean")]
    strengths = {c.name: facts.strongest_correlation(c.name, others) for c in cols}
    knn = [c for c in cols if (strengths[c.name] or 0) >= KNN_MIN_ABS_CORR]
    if not knn:
        out.reject(
            rule,
            "impute.knn",
            f"No incomplete column is correlated (|r| ≥ {KNN_MIN_ABS_CORR}) with another numeric "
            "feature, so neighbours would carry little information beyond the median.",
            _names(cols),
        )
        return
    evidence = [
        _ev("profile", "max_abs_correlation", strengths[c.name], column=c.name)
        for c in cols
        if strengths[c.name] is not None
    ]
    sampled = " (correlations computed on a seeded sample)" if facts.correlations_sampled else ""
    share = len(knn) / len(cols)
    strength = (
        (
            "moderate",
            f"{len(knn)} of {len(cols)} incomplete columns have a correlated numeric "
            f"neighbour (|r| ≥ {KNN_MIN_ABS_CORR}){sampled}.",
        )
        if share >= 0.5
        else (
            "weak",
            f"Only {len(knn)} of {len(cols)} incomplete columns have a correlated "
            f"numeric neighbour{sampled}.",
        )
    )
    if facts.n_rows > Limits.KNN_MAX_FIT_ROWS:
        out.reject(
            rule,
            "impute.knn",
            f"KNN imputation is capped at {Limits.KNN_MAX_FIT_ROWS:,} fitting rows.",
            _names(cols),
        )
    else:
        out.drafts.append(
            Draft(
                rule_id=rule,
                op="impute.knn",
                columns=_names(cols),
                params={},
                title=f"Alternative: fill missing values from the 5 most similar training rows "
                f"({_plural(len(cols), 'column')})",
                reason="Other numeric features are correlated with the incomplete columns, so "
                "similar rows may predict missing values better than a single median.",
                evidence=evidence,
                expected_benefit="Imputed values can follow relationships between columns "
                "instead of collapsing to one value.",
                risks=[
                    "Slower, and sensitive to the scale and choice of predictor columns.",
                    "Can overstate relationships: imputed values look more typical than real ones.",
                ],
                information_loss=("low", "Observed values are unchanged."),
                required_decisions=[
                    "Compare with median imputation in the Experiment Lab before choosing.",
                ],
                strength=strength,
                task_fit=_task_fit_inputs(facts),
                conflict_group="impute.numeric",
                finding_ids=[f.id for f in findings],
                heavy=True,
            )
        )
    iterative = [c for c in cols if (strengths[c.name] or 0) >= ITERATIVE_MIN_ABS_CORR]
    if iterative and facts.n_rows <= Limits.ITERATIVE_MAX_FIT_ROWS:
        out.drafts.append(
            Draft(
                rule_id=rule,
                op="impute.iterative",
                columns=_names(cols),
                params={},
                title=f"Alternative: model each incomplete column from the others "
                f"({_plural(len(cols), 'column')})",
                reason=f"{_plural(len(iterative), 'incomplete column')} are strongly correlated "
                f"(|r| ≥ {ITERATIVE_MIN_ABS_CORR}) with another numeric feature, so a "
                "regression-based imputation may recover more of their structure.",
                evidence=evidence,
                expected_benefit="Uses the multivariate structure of the data for each "
                "missing value.",
                risks=[
                    "Can overfit relationships and understate uncertainty.",
                    "scikit-learn marks IterativeImputer as experimental.",
                ],
                information_loss=("low", "Observed values are unchanged."),
                required_decisions=[
                    "Compare with median and KNN imputation in the Experiment Lab.",
                ],
                strength=(
                    "moderate" if facts.n_rows >= MIN_RELIABLE else "weak",
                    f"{len(iterative)} column(s) with |r| ≥ {ITERATIVE_MIN_ABS_CORR}{sampled}.",
                ),
                task_fit=_task_fit_inputs(facts),
                conflict_group="impute.numeric",
                finding_ids=[f.id for f in findings],
                heavy=True,
            )
        )


def missing_categorical(facts: Facts, out: Collector) -> None:
    rule = "missing.categorical"
    cols = [
        c
        for c in facts.features("categorical", "boolean")
        if 0 < c.missing_pct < IMPUTE_MAX_MISSING_PCT
    ]
    if not cols:
        return
    low = [
        c
        for c in cols
        if c.missing_pct < CATEGORY_MODE_MAX_MISSING_PCT or c.inferred_type == "boolean"
    ]
    high = [c for c in cols if c not in low]
    if low:
        out.drafts.append(
            Draft(
                rule_id=rule,
                op="impute.mode",
                columns=_names(low),
                params={},
                title=f"Fill a few missing categories with the most frequent training value "
                f"({_plural(len(low), 'column')})",
                reason="These categorical or true/false columns have a small share of missing "
                "values.",
                evidence=[_ev("schema", "missing_pct", c.missing_pct, column=c.name) for c in low],
                expected_benefit="Keeps rows usable with minimal distortion when few values are "
                "missing.",
                risks=["Inflates the most frequent category."],
                information_loss=("low", "That a value was missing is lost without an indicator."),
                required_decisions=["Confirm missing values do not have a meaning of their own."],
                strength=("moderate", "Exact missing counts from the schema."),
                task_fit=_task_fit_inputs(facts),
            )
        )
    if high:
        fill = "Missing"
        for c in high:
            values = {str(v.get("value")) for v in (c.frequencies or {}).get("values", [])}
            if fill in values:
                fill = "(missing)"
        out.drafts.append(
            Draft(
                rule_id=rule,
                op="impute.constant",
                columns=_names(high),
                params={"fill_value": fill},
                title=f"Treat missing as its own category '{fill}' "
                f"({_plural(len(high), 'column')})",
                reason=f"At least {CATEGORY_MODE_MAX_MISSING_PCT:g}% of values are missing; "
                "filling them with the most frequent category would distort it, while an "
                "explicit category keeps the information that the value was absent.",
                evidence=[_ev("schema", "missing_pct", c.missing_pct, column=c.name) for c in high],
                expected_benefit="Models can learn whether 'missing' itself is informative.",
                risks=["Adds a category; if missingness is random it adds noise."],
                information_loss=("none", "No observed value changes."),
                required_decisions=["Confirm the label does not clash with a real category."],
                strength=("moderate", "Exact missing counts from the schema."),
                task_fit=_task_fit_inputs(facts),
            )
        )


def missing_indicator(facts: Facts, out: Collector) -> None:
    rule = "missing.indicator"
    chosen: list[Column] = []
    evidence: list[dict[str, Any]] = []
    findings: list[FindingRef] = []
    for c in facts.features("numeric", "categorical", "boolean"):
        if not 0 < c.missing_pct < 100:
            continue
        assoc = facts.detected("universal.missingness.target_association", c.name)
        if assoc:
            chosen.append(c)
            findings.extend(assoc)
            for f in assoc:
                metric = "rank_biserial" if "rank_biserial" in f.evidence else "cramers_v"
                evidence.append(_finding_ev(f, metric, f.evidence.get(metric), c.name))
        elif c.missing_pct >= INDICATOR_MIN_MISSING_PCT:
            chosen.append(c)
            evidence.append(_ev("schema", "missing_pct", c.missing_pct, column=c.name))
    if not chosen:
        return
    associated = bool(findings)
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="missing.indicator",
            columns=_names(chosen),
            params={},
            title=f"Record where values were missing ({_plural(len(chosen), 'indicator column')})",
            reason=(
                "Whether a value is missing is associated with the target for some columns "
                "(diagnostics), so the fact of missingness may itself be predictive."
                if associated
                else f"At least {INDICATOR_MIN_MISSING_PCT:g}% of values are missing in these "
                "columns; imputation alone would hide which values were estimated."
            ),
            evidence=evidence,
            expected_benefit="Keeps the information that a value was missing after imputation.",
            risks=[
                "Adds one column per selected column.",
                "If the reasons for missingness differ when the model is used, the indicator can "
                "mislead (it may encode how data were collected rather than the outcome).",
            ],
            information_loss=("none", "Adds columns; nothing is changed or removed."),
            required_decisions=[
                "Decide whether the missingness process will be the same when the model is used.",
            ],
            strength=("strong", "Association with the target was detected by diagnostics.")
            if associated
            else ("moderate", "Exact missing counts; no tested association with the target."),
            task_fit=("relevant", "Directly adds information for the model.")
            if associated
            else ("indirect", "Helps when missingness is informative; untested here."),
            conflict_group="missing.indicator",
            finding_ids=sorted({f.id for f in findings}),
        )
    )


# ---- Distributions ----------------------------------------------------------------------------


def skewness(facts: Facts, out: Collector) -> None:
    rule = "distribution.skew"
    if not facts.has_profile:
        out.skip(rule, "distribution transforms", "No profile is available yet.")
        return
    groups: dict[str, list[Column]] = {
        "transform.log": [],
        "transform.log1p": [],
        "transform.yeo_johnson": [],
    }
    for c in facts.features("numeric"):
        skew = c.stat("skewness")
        count = c.stat("count") or 0
        if skew is None or abs(skew) < SKEW or count < Limits.TRANSFORM_MIN_VALUES:
            continue
        if (c.n_unique or 0) <= 2 or (c.stat("n_infinite") or 0) > 0:
            continue
        minimum, negatives = c.stat("min"), c.stat("n_negative") or 0
        if negatives > 0 or minimum is None:
            groups["transform.yeo_johnson"].append(c)
        elif (c.stat("n_zeros") or 0) > 0:
            groups["transform.log1p"].append(c)
        else:
            groups["transform.log"].append(c)
    labels = {
        "transform.log": "all observed values are positive",
        "transform.log1p": "values are non-negative and include zeros",
        "transform.yeo_johnson": "values include negatives",
    }
    for op, cols in groups.items():
        if not cols:
            continue
        group = f"transform.{op.split('.')[1]}"
        evidence = [
            _ev("profile", "skewness", c.stat("skewness"), column=c.name, observed=c.stat("count"))
            for c in cols
        ]
        findings = [f for c in cols for f in facts.detected("universal.distribution.shape", c.name)]
        strongest = max(abs(c.stat("skewness")) for c in cols)
        observed = min(int(c.stat("count") or 0) for c in cols)
        strength = (
            ("strong", f"|skewness| up to {strongest:.2f} over at least {observed} values.")
            if strongest >= STRONG_SKEW and observed >= MIN_RELIABLE
            else ("moderate", f"|skewness| up to {strongest:.2f}.")
            if observed >= MIN_RELIABLE
            else ("weak", f"Skewness estimated from as few as {observed} values.")
        )
        task_fit = (
            ("relevant", "k-means distances are dominated by long tails.")
            if facts.target is None
            else (
                "indirect",
                "Matters for linear and distance-based models; tree-based "
                "models are unaffected by monotonic transformations.",
            )
        )
        common = dict(
            evidence=evidence
            + [
                _finding_ev(f, "skewness", f.evidence.get("skewness"), f.columns[0])
                for f in findings
            ],
            strength=strength,
            task_fit=task_fit,
            conflict_group=group,
            finding_ids=[f.id for f in findings],
        )
        names = _names(cols)
        if op != "transform.yeo_johnson":
            out.drafts.append(
                Draft(
                    rule_id=rule,
                    op=op,
                    columns=names,
                    params={},
                    title=f"{'Log' if op == 'transform.log' else 'Log(1 + x)'} transform "
                    f"{_plural(len(cols), 'skewed column')}",
                    reason=f"These columns are right-skewed (|skewness| ≥ {SKEW:g}) and "
                    f"{labels[op]}, so a log scale is defined and often interpretable "
                    "(ratios, fold changes).",
                    expected_benefit="Compresses long tails so a few large values dominate "
                    "linear fits and distances less.",
                    risks=[
                        "Coefficients and thresholds are no longer in original units.",
                        "Does not help if the skew comes from a mixture of populations.",
                    ],
                    information_loss=("none", "Invertible: no information is lost."),
                    required_decisions=["Accept that results are on a log scale."],
                    **common,  # type: ignore[arg-type]
                )
            )
        out.drafts.append(
            Draft(
                rule_id=rule,
                op="transform.yeo_johnson",
                columns=names,
                params={},
                title=("Alternative: " if op != "transform.yeo_johnson" else "")
                + f"Yeo-Johnson power transform of {_plural(len(cols), 'skewed column')}",
                reason=f"These columns are skewed (|skewness| ≥ {SKEW:g}); Yeo-Johnson learns a "
                "power per column from the training fold and works with zero and negative "
                "values.",
                expected_benefit="Makes distributions more symmetric with a fitted, invertible "
                "transformation.",
                risks=[
                    "The learned power is less interpretable than a log.",
                    "Fitted per fold, so the scale differs slightly between folds.",
                ],
                information_loss=("none", "Invertible: no information is lost."),
                required_decisions=["Accept a fitted, less interpretable scale."],
                **common,  # type: ignore[arg-type]
            )
        )
        strong = [c for c in cols if abs(c.stat("skewness")) >= STRONG_SKEW]
        if strong:
            out.drafts.append(
                Draft(
                    rule_id=rule,
                    op="transform.quantile",
                    columns=names,
                    params={"output_distribution": "normal"},
                    title=f"Alternative: rank-based (quantile) transform of "
                    f"{_plural(len(cols), 'column')}",
                    reason=f"{_plural(len(strong), 'column')} are strongly skewed "
                    f"(|skewness| ≥ {STRONG_SKEW:g}); a quantile transform forces a normal shape "
                    "whatever the original distribution.",
                    expected_benefit="Removes the influence of tails entirely.",
                    risks=[
                        "Discards distances between values: only their order is kept.",
                        "Ties and values beyond the training range are compressed.",
                    ],
                    information_loss=(
                        "moderate",
                        "Lossy: original spacing between values is discarded (destructive).",
                    ),
                    required_decisions=[
                        "Decide whether the measurement scale matters for "
                        "interpretation (if it does, keep the original "
                        "distribution or use a log)."
                    ],
                    **common,  # type: ignore[arg-type]
                )
            )


def outliers(facts: Facts, out: Collector) -> None:
    rule = "outliers.extreme"
    check = "universal.outliers.univariate"
    if not facts.has_diagnostics:
        out.skip(rule, "outlier treatment", "Diagnostics have not run.")
        return
    found = [(c, facts.detected(check, c.name)) for c in facts.features("numeric")]
    found = [(c, fs) for c, fs in found if fs]
    if not found:
        return
    cols = [c for c, _ in found]
    findings = [f for _, fs in found for f in fs]
    evidence = []
    for c, fs in found:
        counts = fs[0].evidence.get("counts_by_method", {})
        evidence.append(_finding_ev(fs[0], "extreme_values_iqr_3", counts.get("iqr_3"), c.name))
        evidence.append(_finding_ev(fs[0], "values", fs[0].evidence.get("values"), c.name))
    skewed = [c for c in cols if abs(c.stat("skewness") or 0) >= STRONG_SKEW]
    assumption = (
        f"{_plural(len(skewed), 'column')} are strongly skewed: IQR fences assume a roughly "
        "symmetric distribution, so many valid tail values may be marked. Consider a "
        "transformation first."
        if skewed
        else None
    )
    severe = any(f.severity in ("medium", "high", "critical") for f in findings)
    strength = (
        (
            "strong",
            "Diagnostics found extreme values by two rules (IQR × 3, modified z-score) "
            "in more than 1% of rows of at least one column.",
        )
        if severe
        else (
            "moderate",
            "Diagnostics found a few extreme values (IQR × 3 or modified z-score > 3.5).",
        )
    )
    params = {"method": "iqr", "iqr_multiplier": 3.0}
    common = dict(
        evidence=evidence,
        strength=strength,
        conflict_group="outliers.treatment",
        finding_ids=[f.id for f in findings],
        assumption_note=assumption,
    )
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="outliers.flag",
            columns=_names(cols),
            params=dict(params),
            title=f"Flag extreme values in {_plural(len(cols), 'column')} for review",
            reason="Diagnostics found values far outside the bulk of these columns. Flagging "
            "marks them without deciding they are errors.",
            expected_benefit="Makes extreme values visible to the model and to reviewers while "
            "keeping every observed value.",
            risks=[
                "Adds one true/false column per selected column.",
                "Rare but valid cases are flagged too.",
            ],
            information_loss=("none", "Values are unchanged; flag columns are added."),
            required_decisions=[
                "Review the flagged values with someone who knows how the data were measured."
            ],
            task_fit=(
                "indirect",
                "Adds information; most useful when extreme values behave differently.",
            ),
            **common,  # type: ignore[arg-type]
        )
    )
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="outliers.cap",
            columns=_names(cols),
            params=dict(params),
            title=f"Alternative: cap extreme values at fences learned per training fold "
            f"({_plural(len(cols), 'column')})",
            reason="If the extreme values are entry or unit errors, capping limits their "
            "influence without removing rows.",
            expected_benefit="Reduces the leverage of a few values on linear fits and scales.",
            risks=[
                "Overwrites observed values (destructive): real extreme cases become "
                "indistinguishable from the fence.",
                "Can hide genuine effects in the tails.",
            ],
            information_loss=("high", "Observed magnitudes beyond the fences are lost."),
            required_decisions=[
                "Confirm the extreme values are errors rather than real cases "
                "(capping is destructive and needs explicit approval to commit)."
            ],
            task_fit=(
                "indirect",
                "Mainly affects linear models; trees are insensitive to monotone tails.",
            ),
            **common,  # type: ignore[arg-type]
        )
    )
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="scale.robust",
            columns=_names(cols),
            params={},
            title=f"Alternative: robust scaling (median and IQR) of {_plural(len(cols), 'column')}",
            reason="Robust scaling reduces the influence of extreme values on the scale "
            "without changing any value's rank or overwriting it.",
            expected_benefit="Scale is not dominated by a few extreme values; nothing is "
            "overwritten.",
            risks=[
                "Extreme values stay extreme after scaling: models sensitive to them are "
                "still affected."
            ],
            information_loss=("none", "Invertible linear rescaling."),
            required_decisions=["Compare with flagging and capping in the Experiment Lab."],
            task_fit=(
                "indirect",
                "Affects scale-sensitive models only; the linear baseline "
                "also standardises its inputs.",
            ),
            **common,  # type: ignore[arg-type]
        )
    )


# ---- Categories -------------------------------------------------------------------------------


def categories(facts: Facts, out: Collector) -> None:
    rule = "categories.cardinality"
    for c in facts.features("categorical"):
        n = c.n_unique or 0
        unique_pct = c.unique_pct or 0.0
        findings = facts.detected("universal.cardinality.high", c.name)
        if n > HIGH_CARDINALITY and unique_pct >= ID_LIKE_UNIQUE_PCT:
            out.reject(
                rule,
                "encode.*",
                f"{unique_pct:.0f}% of present values are distinct: the column behaves like an "
                "identifier, and encoding it lets a model memorise rows.",
                [c.name],
            )
            out.advise(
                "IDENTIFIER_LIKE_COLUMN",
                "A categorical column looks like an identifier",
                "Almost every value is distinct. Assign it the 'Row identifier' or 'Exclude' "
                "role, or derive a coarser feature from it.",
                [c.name],
                findings,
            )
            continue
        if n > HIGH_CARDINALITY:
            evidence = [_ev("schema", "distinct_categories", n, column=c.name)] + [
                _finding_ev(f, "distinct_pct", f.evidence.get("distinct_pct"), c.name)
                for f in findings
            ]
            common = dict(
                evidence=evidence,
                strength=(
                    "strong" if c.n_present >= MIN_RELIABLE else "weak",
                    f"Exact count of {n} distinct categories.",
                ),
                conflict_group=f"encode.cardinality:{c.name}",
                finding_ids=[f.id for f in findings],
            )
            out.drafts.append(
                Draft(
                    rule_id=rule,
                    op="encode.frequency",
                    columns=(c.name,),
                    params={},
                    title=f"Encode a high-cardinality column ({n} categories) by frequency",
                    reason=f"One-hot encoding {n} categories would add {n} sparse columns; "
                    "frequency encoding keeps one numeric column learned from the training fold.",
                    expected_benefit="Avoids a very wide, sparse input; suited to tree models.",
                    risks=[
                        "Categories with equal frequency become indistinguishable.",
                        "Frequency may be unrelated to the outcome.",
                    ],
                    information_loss=(
                        "moderate",
                        "Category identity is replaced by its frequency.",
                    ),
                    required_decisions=[
                        "Decide whether category identity matters (if so, "
                        "compare with grouped one-hot encoding)."
                    ],
                    task_fit=("relevant", "Determines how the model sees this column."),
                    **common,  # type: ignore[arg-type]
                )
            )
            out.drafts.append(
                Draft(
                    rule_id=rule,
                    op="encode.onehot",
                    columns=(c.name,),
                    params={"max_categories": HIGH_CARDINALITY},
                    title=f"Alternative: one-hot encode the {HIGH_CARDINALITY} most frequent "
                    f"categories and group the rest ({n} categories)",
                    reason="Keeps category identity for the frequent levels while bounding the "
                    "number of added columns.",
                    expected_benefit="Interpretable indicators for common categories.",
                    risks=[
                        f"Adds up to {HIGH_CARDINALITY + 1} columns (dimensionality).",
                        "Rare categories are merged into one group.",
                    ],
                    information_loss=(
                        "moderate",
                        f"{n - HIGH_CARDINALITY} rare categories are merged.",
                    ),
                    required_decisions=["Accept merging rare categories."],
                    task_fit=("relevant", "Determines how the model sees this column."),
                    **common,  # type: ignore[arg-type]
                )
            )
            continue
        freq = c.frequencies or {}
        if not freq.get("complete") or c.n_present < MIN_RELIABLE:
            continue
        levels = freq.get("values", [])
        rare = [v for v in levels if float(v.get("pct") or 0) < RARE_LEVEL_PCT]
        if len(rare) >= MIN_RARE_LEVELS and len(levels) - len(rare) >= 2:
            keep = len(levels) - len(rare)
            out.drafts.append(
                Draft(
                    rule_id="categories.rare",
                    op="encode.onehot",
                    columns=(c.name,),
                    params={"max_categories": keep},
                    title=f"One-hot encode with {len(rare)} rare categories grouped",
                    reason=f"{len(rare)} categories each cover less than {RARE_LEVEL_PCT:g}% of "
                    "rows; individual indicators for them are mostly zeros and unreliable "
                    "across folds.",
                    evidence=[
                        _ev("profile", "rare_categories", len(rare), column=c.name),
                        _ev("profile", "categories", len(levels), column=c.name),
                    ],
                    expected_benefit="Fewer, better-supported indicator columns.",
                    risks=["Rare categories lose their identity (merged into one group)."],
                    information_loss=("low", f"{len(rare)} rare categories are merged."),
                    required_decisions=[
                        "Check that no rare category is important on its own "
                        "(e.g. a rare diagnosis)."
                    ],
                    strength=("moderate", "Complete frequency table from the profile."),
                    task_fit=("relevant", "Determines how the model sees this column."),
                )
            )


# ---- Target -----------------------------------------------------------------------------------


def imbalance(facts: Facts, out: Collector) -> None:
    rule = "target.imbalance"
    topic = "class imbalance"
    if facts.target is None:
        out.skip(rule, topic, "No target role is declared; declare a target and task type.")
        return
    if not facts.classification:
        out.skip(
            rule,
            topic,
            "The target is a regression target; class imbalance does not apply.",
            [facts.target],
        )
        return
    check = "universal.target.imbalance"
    found = facts.detected(check)
    if not found:
        if not facts.check_ran(check):
            out.skip(
                rule,
                topic,
                "The class-imbalance diagnostic has not produced a result for "
                "the current target (re-run diagnostics).",
                [facts.target],
            )
        return
    f = found[0]
    share = float(f.evidence.get("minority_share_pct") or 0)
    ratio = f.evidence.get("imbalance_ratio")
    classes = f.evidence.get("classes") or {}
    minority = min((int(v) for v in classes.values()), default=0)
    evidence = [
        _finding_ev(f, "minority_share_pct", share, facts.target),
        _finding_ev(f, "imbalance_ratio", ratio, facts.target),
        _finding_ev(f, "minority_rows", minority, facts.target),
    ]
    strength = (
        ("strong", f"The smallest class holds {share:g}% of rows ({minority} rows).")
        if minority >= MIN_RELIABLE
        else (
            "moderate",
            f"The smallest class holds {share:g}% of rows but only {minority} "
            "rows, so any estimate for it is noisy.",
        )
    )
    common = dict(
        evidence=evidence,
        strength=strength,
        conflict_group="imbalance",
        finding_ids=[f.id],
        task_fit=("relevant", "Directly changes how the model weighs the minority class."),
    )
    entity = facts.role_columns(ENTITY_ROLES)
    time = facts.role_columns(TIME_ROLES)
    decisions = [
        "Choose the metric that matters (minority recall, precision, PR-AUC) before "
        "comparing; balancing trades precision for recall."
    ]
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="imbalance.class_weight",
            columns=(),
            params={},
            title="Weight classes inversely to their frequency during training",
            reason=f"The smallest class holds {share:g}% of rows. Class weights make the model "
            "pay more attention to it without creating or discarding any row.",
            expected_benefit="Often improves minority recall and balanced accuracy.",
            risks=[
                "Predicted probabilities are no longer calibrated to real frequencies.",
                "Usually lowers precision for the minority class.",
            ],
            information_loss=("none", "No row is created or removed."),
            required_decisions=decisions,
            **common,  # type: ignore[arg-type]
        )
    )
    if minority >= 2:
        out.drafts.append(
            Draft(
                rule_id=rule,
                op="imbalance.random_oversample",
                columns=(),
                params={},
                title="Alternative: duplicate minority-class training rows at random",
                reason="Random oversampling balances the training rows by repeating real "
                "minority examples (inside each training fold only).",
                expected_benefit="Balances training without inventing values.",
                risks=[
                    "Repeated rows encourage overfitting to the few minority examples.",
                    "Probabilities are no longer calibrated.",
                ],
                information_loss=("none", "Training rows are repeated; test rows untouched."),
                required_decisions=decisions,
                assumption_note="Rows are time-ordered: repeated past rows over-weight older "
                "periods."
                if time
                else None,
                **common,  # type: ignore[arg-type]
            )
        )
    categorical = [c.name for c in facts.features("categorical", "text", "datetime")]
    if entity:
        out.reject(
            rule,
            "imbalance.smote",
            "Rows belong to entities (patients, accounts, "
            "groups): SMOTE would interpolate between different entities, creating "
            "synthetic rows that are not independent observations.",
            entity,
        )
    elif categorical:
        out.reject(
            rule,
            "imbalance.smote",
            "SMOTE interpolates numeric features; this "
            "dataset has categorical, text or date features, where interpolated values "
            "are not real categories.",
            categorical,
        )
    elif minority < SMOTE_MIN_MINORITY:
        out.reject(
            rule,
            "imbalance.smote",
            f"The smallest class has fewer than "
            f"{SMOTE_MIN_MINORITY} rows, too few for its 5 nearest neighbours.",
        )
    else:
        out.drafts.append(
            Draft(
                rule_id=rule,
                op="imbalance.smote",
                columns=(),
                params={},
                title="Alternative: create synthetic minority rows by interpolation (SMOTE)",
                reason="All features are numeric, so SMOTE can create new minority examples "
                "between neighbouring real ones (inside each training fold only).",
                expected_benefit="Balances training with less duplication than random "
                "oversampling.",
                risks=[
                    "Synthetic rows are not real observations and can fall in implausible regions.",
                    "Needs complete numeric inputs: impute first.",
                ],
                information_loss=("none", "Training rows are added; test rows untouched."),
                required_decisions=decisions + ["Accept training on synthetic rows."],
                assumption_note="Rows are time-ordered: synthetic rows blend different periods."
                if time
                else None,
                heavy=True,
                **common,  # type: ignore[arg-type]
            )
        )
    if minority >= UNDERSAMPLE_MIN_MINORITY:
        out.drafts.append(
            Draft(
                rule_id=rule,
                op="imbalance.random_undersample",
                columns=(),
                params={},
                title="Alternative: discard majority-class training rows at random",
                reason=f"The smallest class has {minority} rows, enough to train on a balanced "
                "subset.",
                expected_benefit="Faster training on balanced data.",
                risks=["Discards real training rows (destructive for training)."],
                information_loss=("high", "Majority training rows are discarded in each fold."),
                required_decisions=decisions + ["Accept discarding training rows."],
                **common,  # type: ignore[arg-type]
            )
        )
    else:
        out.reject(
            rule,
            "imbalance.random_undersample",
            f"The smallest class has fewer than "
            f"{UNDERSAMPLE_MIN_MINORITY} rows; undersampling would discard most of the "
            "training data.",
        )


# ---- Redundancy and selection (report only) ---------------------------------------------------


def redundancy(facts: Facts, out: Collector) -> None:
    rule = "redundancy.correlation"
    found = facts.detected("universal.correlation.redundancy")
    if not found:
        return
    f = found[0]
    numeric = {c.name for c in facts.features("numeric", "boolean")}
    pairs = [
        p for p in f.evidence.get("pairs", []) if p.get("a") in numeric and p.get("b") in numeric
    ]
    cols = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    if len(cols) < 2:
        return
    strongest = max(abs(float(p.get("pearson") or 0)) for p in pairs)
    threshold = float(f.evidence.get("threshold_abs_r") or 0.9)
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="select.correlation",
            columns=tuple(cols),
            params={"threshold": max(0.5, min(threshold, 1.0))},
            title=f"Review {_plural(len(pairs), 'highly correlated pair')} (proposal only)",
            reason=f"Diagnostics found pairs with |r| up to {strongest:.2f}. This step reports "
            "which column of each pair could be redundant in each training fold; it removes "
            "nothing.",
            evidence=[_finding_ev(f, "pairs", len(pairs)), _finding_ev(f, "max_abs_r", strongest)],
            expected_benefit="Shows whether redundancy proposals are stable across folds before "
            "you decide anything.",
            risks=[
                "Correlated columns can each carry distinct meaning; correlation is not redundancy."
            ],
            information_loss=("none", "Report only: nothing is removed."),
            required_decisions=[
                "Decide per pair whether one column is truly redundant; remove "
                "it explicitly only if so."
            ],
            strength=("strong", f"Pairwise |r| ≥ {threshold:g} in diagnostics."),
            task_fit=("indirect", "Matters for linear-model stability, little for trees."),
            conflict_group="redundancy",
            finding_ids=[f.id],
        )
    )


def constant(facts: Facts, out: Collector) -> None:
    rule = "variance.constant"
    found = facts.detected("universal.variance.constant")
    numeric = {c.name for c in facts.features("numeric", "boolean")}
    cols = sorted(
        {
            c
            for f in found
            if f.evidence.get("distinct_values") == 1
            for c in f.columns
            if c in numeric
        }
    )
    if not cols:
        return
    ids = [f.id for f in found if set(f.columns) & set(cols)]
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="select.variance",
            columns=tuple(cols),
            params={"threshold": 0.0, "action": "report"},
            title=f"Report {_plural(len(cols), 'constant column')} in each training fold",
            reason="These columns have a single value, so they cannot help any model. The step "
            "reports them per fold; it removes nothing.",
            evidence=[_ev("finding", "distinct_values", 1, column=c) for c in cols],
            expected_benefit="Confirms per fold that the columns carry no information.",
            risks=["A rarely varying column can still matter in new data."],
            information_loss=("none", "Report only: nothing is removed."),
            required_decisions=["Decide whether to exclude the columns (role 'Exclude')."],
            strength=("strong", "Exactly one distinct value."),
            task_fit=("indirect", "No direct effect in report mode."),
            finding_ids=ids,
        )
    )


def selection(facts: Facts, out: Collector) -> None:
    rule = "selection.wide"
    topic = "supervised feature selection"
    if not facts.supervised:
        out.skip(rule, topic, "No target role with a task type is declared.")
        return
    numeric = [c for c in facts.features("numeric", "boolean")]
    n = len(numeric)
    per_feature = facts.n_rows / n if n else float("inf")
    if n < SELECTION_MIN_FEATURES or per_feature >= SELECTION_MAX_ROWS_PER_FEATURE:
        out.reject(
            rule,
            "select.*",
            f"{n} numeric features for {facts.n_rows} rows "
            f"({per_feature:.0f} rows per feature): selection is unlikely to help and "
            "rankings are unstable between folds.",
        )
        return
    test = "f_regression" if facts.task == "regression" else "anova_f"
    k = max(1, min(n // 2, 50))
    out.drafts.append(
        Draft(
            rule_id=rule,
            op="select.univariate",
            columns=_names(numeric),
            params={"test": test, "k": k, "action": "report"},
            title=f"Rank {n} features against the target in each fold (report only)",
            reason=f"There are only {per_feature:.1f} rows per numeric feature; ranking shows "
            "which features carry signal and, in the Experiment Lab, how stable that ranking is "
            "between folds.",
            evidence=[_ev("schema", "numeric_features", n), _ev("schema", "rows", facts.n_rows)],
            expected_benefit="Makes weak and unstable features visible before deciding anything.",
            risks=[
                "Rankings reflect association in this sample, not causation or validation.",
                "Unstable between folds when signals are weak.",
            ],
            information_loss=("none", "Report only: nothing is removed."),
            required_decisions=["Decide which columns must always be kept ('keep')."],
            strength=("moderate", "Based on the shape of the data (rows per feature)."),
            task_fit=("relevant", "Concerns which inputs the model uses."),
            conflict_group="selection",
            heavy=False,
        )
    )


# ---- Advisories (no operation) ----------------------------------------------------------------


def advisories(facts: Facts, out: Collector) -> None:
    for f in facts.detected("universal.leakage.target_proxy"):
        out.advise(
            "POSSIBLE_TARGET_PROXY",
            "A feature almost determines the target",
            "Confirm it is known at prediction time and does not encode the outcome; otherwise "
            "assign it the 'Exclude' role. No preprocessing can fix leakage.",
            [c for c in f.columns if c != facts.target],
            [f],
        )
    missing_target = [
        f for f in facts.detected("universal.target.quality") if "missing" in f.evidence
    ]
    if missing_target:
        out.advise(
            "MISSING_TARGET_ROWS",
            "Some rows have no target value",
            "Experiments evaluate only rows with a target; the excluded count is reported.",
            [facts.target] if facts.target else [],
            missing_target,
        )
    duplicates = facts.detected("universal.duplicates.exact_rows")
    if duplicates:
        out.advise(
            "DUPLICATE_ROWS",
            "Exact duplicate rows exist",
            "Experiment folds keep identical rows together so a duplicate is never in training "
            "and test at once. DataMedic does not delete them.",
            findings=duplicates,
        )


RULES: tuple[Callable[[Facts, Collector], None], ...] = (
    missing_numeric,
    missing_categorical,
    missing_indicator,
    skewness,
    outliers,
    categories,
    imbalance,
    redundancy,
    constant,
    selection,
    advisories,
)

"""Biomedical Intelligence Lite: a descriptive cohort summary of one version (worker only).

Everything here is descriptive and for research use only. It never claims clinical validity,
causality or biomarker discovery. Inputs are the canonical version frame and its declared roles;
every section that lacks the metadata it needs is reported ``not_assessed`` with a reason.

Privacy: groups (classes, batches, sites) with fewer than ``MIN_CELL`` units are suppressed
(counts shown as ``None`` with ``suppressed: true``; their statistics are not computed). Labels
of sensitive columns (and of a sensitive target) are replaced by "group 1", "class 1", ...
No row values are returned; only aggregates.

Statistics:
- Comparisons use one value per patient (the patient's mean) when a patient/entity role is
  declared, so repeated observations do not inflate the evidence; otherwise rows.
- Binary target: Hedges' g (bias-corrected standardised mean difference) and the Mann–Whitney
  rank-biserial correlation with its p-value. Multiclass: Kruskal–Wallis epsilon². Regression:
  Spearman correlation. p-values are adjusted across biomarkers with Benjamini–Hochberg (q).
- Batch/site: Cramér's V (categorical target) or epsilon² (numeric target) for the association
  with the outcome; Kruskal–Wallis epsilon² for each biomarker across batches.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from datamedic.engine import ROW_ID

SUMMARY_VERSION = "1.0.0"
MIN_CELL = 10
MAX_BIOMARKERS = 40
MAX_GROUPS = 12
MAX_LABEL = 48
SMALL_COHORT = 100
EPV_MIN = 10
DISCLAIMER = (
    "Research use only. These are descriptive statistics of this dataset. They are not "
    "clinical evidence, do not establish causality or biomarker validity, and must not be used "
    "for diagnosis or treatment decisions."
)
PATIENT_ROLES = ("bio.patient_id", "group")
BATCH_ROLES = ("bio.batch",)
CLINICAL_COVARIATES = ("bio.age", "bio.sex", "bio.bmi")
LIMITATIONS = [
    "Associations are not causal and can be produced by confounding, selection or batch effects.",
    "p-values and q-values describe this sample under the tests' assumptions; many comparisons "
    "are made, so isolated small p-values are expected by chance.",
    "Groups with fewer than 10 units are suppressed for privacy; their statistics are not shown.",
    "Only declared roles are used: undeclared patients, batches or sites cannot be assessed.",
    "Advanced genomics, single-cell, pathway and omics-specific analyses are not performed.",
]


def _na(reason: str) -> dict[str, Any]:
    return {"status": "not_assessed", "reason": reason}


def _r(x: Any, d: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, d) if np.isfinite(v) else None


def _sig(p: float | None, alpha: float = 0.01) -> bool:
    return p is not None and p < alpha


def _num(x: float | None) -> float:
    return 0.0 if x is None else float(x)


def _label(value: Any) -> str:
    text = str(value)
    return text if len(text) <= MAX_LABEL else text[: MAX_LABEL - 1] + "…"


def _first(
    roles: Mapping[str, Sequence[str]], keys: Sequence[str], frame: pd.DataFrame
) -> str | None:
    for k in keys:
        for c in roles.get(k, ()):
            if c in frame.columns:
                return c
    return None


def _bh(pvalues: list[float | None]) -> list[float | None]:
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    if not idx:
        return [None] * len(pvalues)
    ps = np.array([pvalues[i] for i in idx], dtype=float)
    order = np.argsort(ps)
    m = len(ps)
    q = np.empty(m)
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        prev = min(prev, ps[i] * m / (rank + 1))
        q[i] = prev
    out: list[float | None] = [None] * len(pvalues)
    for j, i in enumerate(idx):
        out[i] = _r(q[j], 6)
    return out


def _groups(
    s: pd.Series, sensitive: bool, prefix: str
) -> tuple[list[dict[str, Any]], dict[Any, str]]:
    """Group counts with suppression; ``names`` maps original values to shown labels."""
    counts = s.dropna().astype(object).value_counts()
    rows, names = [], {}
    for i, (value, n) in enumerate(counts.items()):
        shown = f"{prefix} {i + 1}" if sensitive else _label(value)
        names[value] = shown
        if i >= MAX_GROUPS:
            continue
        small = int(n) < MIN_CELL
        rows.append({"label": shown, "n": None if small else int(n), "suppressed": small})
    return rows, names


class _Ctx:
    def __init__(
        self,
        frame: pd.DataFrame,
        roles: Mapping[str, Sequence[str]],
        task_type: str | None,
        sensitive: set[str],
        numeric: list[str],
    ) -> None:
        self.frame = frame.drop(columns=[ROW_ID], errors="ignore")
        self.roles = roles
        self.task_type = task_type
        self.sensitive = sensitive
        self.patient = _first(roles, PATIENT_ROLES, self.frame)
        target = list(roles.get("target", ()))
        self.target = target[0] if len(target) == 1 and target[0] in self.frame.columns else None
        self.batches = [c for k in BATCH_ROLES for c in roles.get(k, ()) if c in self.frame.columns]
        protected = {
            c
            for k, cs in roles.items()
            if k
            in (
                "target",
                "id",
                "group",
                "time",
                "weight",
                "exclude",
                "bio.patient_id",
                "bio.sample_id",
                "bio.batch",
                "bio.biomarker_id",
            )
            for c in cs
        }
        self.biomarkers = [c for c in numeric if c in self.frame.columns and c not in protected][
            :MAX_BIOMARKERS
        ]
        self.classification = task_type in ("binary", "multiclass")


# ---- Sections ------------------------------------------------------------------------------------


def _cohort(c: _Ctx) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(c.frame), "columns": int(c.frame.shape[1])}
    if c.patient is None:
        out["patients"] = _na(
            "No patient or entity role (bio.patient_id or group) is declared, so unique patients "
            "and repeated observations cannot be counted."
        )
        return out
    ids = c.frame[c.patient]
    per = ids.dropna().astype(object).value_counts()
    out["patients"] = {
        "status": "assessed",
        "column": c.patient,
        "unique": int(len(per)),
        "rows_without_id": int(ids.isna().sum()),
        "repeated_patients": int((per > 1).sum()),
        "single_observation_patients": int((per == 1).sum()),
        "observations_per_patient": {
            "min": int(per.min()) if len(per) else None,
            "median": _r(per.median(), 2) if len(per) else None,
            "mean": _r(per.mean(), 2) if len(per) else None,
            "max": int(per.max()) if len(per) else None,
        },
    }
    if c.target is not None and c.classification:
        varying = c.frame[[c.patient, c.target]].dropna().groupby(c.patient)[c.target].nunique()
        out["patients"]["patients_with_changing_target"] = int((varying > 1).sum())
    return out


def _target(c: _Ctx) -> dict[str, Any]:
    if c.target is None:
        return _na("No single target column is declared.")
    s = c.frame[c.target]
    base: dict[str, Any] = {
        "status": "assessed",
        "column": c.target,
        "task_type": c.task_type,
        "missing": int(s.isna().sum()),
        "sensitive": c.target in c.sensitive,
    }
    if c.classification:
        classes, _ = _groups(s, c.target in c.sensitive, "class")
        present = s.dropna()
        counts = present.astype(object).value_counts()
        base["classes"] = classes
        if len(counts) >= 2:
            base["imbalance"] = {
                "minority_share": _r(counts.min() / counts.sum()),
                "majority_to_minority": _r(counts.max() / max(counts.min(), 1), 2),
                "minority_n": None if counts.min() < MIN_CELL else int(counts.min()),
            }
        return base
    values = pd.to_numeric(s, errors="coerce").dropna()
    if c.target in c.sensitive:
        base["distribution"] = {"withheld": "The target is sensitive."}
    elif len(values):
        base["distribution"] = {
            "n": int(len(values)),
            "mean": _r(values.mean()),
            "sd": _r(values.std(ddof=1)),
            "median": _r(values.median()),
            "q1": _r(values.quantile(0.25)),
            "q3": _r(values.quantile(0.75)),
        }
    return base


def _units(c: _Ctx, columns: list[str]) -> tuple[pd.DataFrame, str]:
    """One row per patient (mean of numeric columns, constant target) when a patient role
    exists; otherwise rows. Patients whose classification target changes are dropped."""
    keep = [x for x in columns if x in c.frame.columns]
    if c.patient is None:
        return c.frame[keep + ([c.target] if c.target else [])], "row"
    data = c.frame[[c.patient] + keep + ([c.target] if c.target else [])].dropna(subset=[c.patient])
    numeric = data[[c.patient] + keep].apply(
        lambda s: s if s.name == c.patient else pd.to_numeric(s, errors="coerce")
    )
    agg = numeric.groupby(c.patient).mean()
    if c.target:
        if c.classification:
            t = data.groupby(c.patient)[c.target].agg(
                lambda s: s.iloc[0] if s.nunique() <= 1 else None
            )
        else:
            t = pd.to_numeric(data[c.target], errors="coerce").groupby(data[c.patient]).mean()
        agg[c.target] = t
    return agg.reset_index(drop=True), "patient"


def _biomarkers(c: _Ctx) -> dict[str, Any]:
    if not c.biomarkers:
        return _na("No numeric feature column is available to describe.")
    rows: list[dict[str, Any]] = []
    for col in c.biomarkers:
        s = pd.to_numeric(c.frame[col], errors="coerce")
        present = s.dropna()
        item: dict[str, Any] = {
            "column": col,
            "n": int(len(present)),
            "missing": int(s.isna().sum()),
            "missing_pct": _r(100 * s.isna().mean(), 2),
            "sensitive": col in c.sensitive,
        }
        if len(present) >= MIN_CELL:
            q1, q3 = present.quantile(0.25), present.quantile(0.75)
            iqr = q3 - q1
            out = int(((present < q1 - 1.5 * iqr) | (present > q3 + 1.5 * iqr)).sum())
            far = int(((present < q1 - 3 * iqr) | (present > q3 + 3 * iqr)).sum())
            item.update(
                mean=_r(present.mean()),
                sd=_r(present.std(ddof=1)),
                median=_r(present.median()),
                q1=_r(q1),
                q3=_r(q3),
                outliers_iqr_1_5=out,
                outliers_iqr_3=far,
                outlier_pct=_r(100 * out / len(present), 2),
            )
            if c.patient is not None:
                within = s.groupby(c.frame[c.patient]).std(ddof=1)
                total = present.std(ddof=1)
                if total and np.isfinite(total) and within.notna().any():
                    item["within_patient_sd_ratio"] = _r(within.mean() / total)
        rows.append(item)
    return {"status": "assessed", "items": rows}


def _comparisons(c: _Ctx) -> dict[str, Any]:
    if c.target is None or c.task_type is None:
        return _na("A target with a task type is needed to compare biomarkers across outcomes.")
    if not c.biomarkers:
        return _na("No numeric feature column is available to compare.")
    units, unit = _units(c, c.biomarkers)
    t = units[c.target]
    out: list[dict[str, Any]] = []
    pvals: list[float | None] = []
    sensitive_target = c.target in c.sensitive
    if c.classification:
        present = t.dropna().astype(object)
        counts = present.value_counts()
        kept = sorted((v for v, n in counts.items() if n >= MIN_CELL), key=str)
        if len(kept) < 2:
            return _na(f"Fewer than two outcome groups have at least {MIN_CELL} {unit}s.")
        names = {
            v: (f"class {i + 1}" if sensitive_target else _label(v))
            for i, v in enumerate(counts.index)
        }
    for col in c.biomarkers:
        x = pd.to_numeric(units[col], errors="coerce")
        res: dict[str, Any] = {"column": col}
        p = None
        try:
            if c.task_type == "binary":
                a = x[t.astype(object) == kept[0]].dropna()
                b = x[t.astype(object) == kept[1]].dropna()
                if min(len(a), len(b)) < MIN_CELL:
                    res.update(_na(f"A group has fewer than {MIN_CELL} {unit}s with values."))
                else:
                    sp = np.sqrt(
                        ((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                        / (len(a) + len(b) - 2)
                    )
                    d = (b.mean() - a.mean()) / sp if sp > 0 else np.nan
                    g = d * (1 - 3 / (4 * (len(a) + len(b)) - 9))
                    u = sps.mannwhitneyu(b, a, alternative="two-sided")
                    p = float(u.pvalue)
                    res.update(
                        status="assessed",
                        groups=[
                            {
                                "label": names[kept[0]],
                                "n": int(len(a)),
                                "mean": _r(a.mean()),
                                "median": _r(a.median()),
                            },
                            {
                                "label": names[kept[1]],
                                "n": int(len(b)),
                                "mean": _r(b.mean()),
                                "median": _r(b.median()),
                            },
                        ],
                        effect={
                            "name": "hedges_g",
                            "value": _r(g),
                            "direction": f"{names[kept[1]]} minus {names[kept[0]]}",
                        },
                        rank_biserial=_r(2 * u.statistic / (len(a) * len(b)) - 1),
                        p_value=_r(p, 6),
                    )
            elif c.task_type == "multiclass":
                samples = [x[t.astype(object) == v].dropna() for v in kept]
                samples = [s for s in samples if len(s) >= MIN_CELL]
                if len(samples) < 2:
                    res.update(_na(f"Fewer than two groups have {MIN_CELL} {unit}s with values."))
                else:
                    h = sps.kruskal(*samples)
                    n, k = sum(len(s) for s in samples), len(samples)
                    p = float(h.pvalue)
                    res.update(
                        status="assessed",
                        effect={
                            "name": "epsilon_squared",
                            "value": _r((h.statistic - k + 1) / (n - k)),
                        },
                        p_value=_r(p, 6),
                    )
            else:
                pair = pd.concat([x, pd.to_numeric(t, errors="coerce")], axis=1).dropna()
                if len(pair) < MIN_CELL * 2:
                    res.update(_na(f"Fewer than {MIN_CELL * 2} {unit}s have both values."))
                else:
                    rho = sps.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                    p = float(rho.pvalue)
                    res.update(
                        status="assessed",
                        effect={"name": "spearman_rho", "value": _r(rho.statistic)},
                        n=int(len(pair)),
                        p_value=_r(p, 6),
                    )
        except (ValueError, ZeroDivisionError, FloatingPointError):
            res.update(_na("The comparison could not be computed for this column."))
        pvals.append(p if p is not None and np.isfinite(p) else None)
        out.append(res)
    for item, q in zip(out, _bh(pvals), strict=True):
        if item.get("status") == "assessed":
            item["q_value"] = q
    out.sort(key=lambda r: -abs((r.get("effect") or {}).get("value") or 0))
    return {
        "status": "assessed",
        "unit": unit,
        "note": "One value per patient (patient mean)"
        if unit == "patient"
        else "One value per row",
        "items": out,
    }


def _association(x: pd.Series, y: pd.Series, y_categorical: bool) -> dict[str, Any] | None:
    pair = pd.concat([x.astype(object), y], axis=1).dropna()
    if len(pair) < MIN_CELL * 2:
        return None
    if y_categorical:
        table = pd.crosstab(pair.iloc[:, 0], pair.iloc[:, 1].astype(object))
        if table.shape[0] < 2 or table.shape[1] < 2:
            return None
        chi2, p, _, _ = sps.chi2_contingency(table)
        v = np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1)))
        return {"name": "cramers_v", "value": _r(v), "p_value": _r(p, 6)}
    samples = [
        pd.to_numeric(g, errors="coerce").dropna()
        for _, g in pair.iloc[:, 1].groupby(pair.iloc[:, 0])
    ]
    samples = [s for s in samples if len(s) >= MIN_CELL]
    if len(samples) < 2:
        return None
    h = sps.kruskal(*samples)
    n, k = sum(len(s) for s in samples), len(samples)
    return {
        "name": "epsilon_squared",
        "value": _r((h.statistic - k + 1) / (n - k)),
        "p_value": _r(h.pvalue, 6),
    }


def _batches(c: _Ctx) -> dict[str, Any]:
    if not c.batches:
        return _na(
            "No batch or site role (bio.batch) is declared, so batch and site effects "
            "cannot be assessed."
        )
    items = []
    for col in c.batches[:3]:
        s = c.frame[col]
        groups, _ = _groups(s, col in c.sensitive, "group")
        item: dict[str, Any] = {"column": col, "groups": groups, "n_groups": int(s.nunique())}
        if c.target is not None:
            item["outcome_association"] = _association(s, c.frame[c.target], c.classification)
        shifts = []
        for b in c.biomarkers:
            a = _association(s, pd.to_numeric(c.frame[b], errors="coerce"), False)
            if a is not None:
                shifts.append({"column": b, **a})
        shifts.sort(key=lambda r: -(r["value"] or 0))
        item["biomarker_shifts"] = shifts[:15]
        items.append(item)
    return {"status": "assessed", "items": items}


def _covariates(c: _Ctx) -> list[dict[str, Any]]:
    out = []
    for key in CLINICAL_COVARIATES:
        for col in c.roles.get(key, ()):
            if col in c.frame.columns and c.target is not None and col != c.target:
                numeric = key != "bio.sex"
                if numeric:
                    a = (
                        _association(
                            c.frame[c.target], pd.to_numeric(c.frame[col], errors="coerce"), False
                        )
                        if c.classification
                        else None
                    )
                    if not c.classification:
                        pair = pd.concat(
                            [
                                pd.to_numeric(c.frame[col], errors="coerce"),
                                pd.to_numeric(c.frame[c.target], errors="coerce"),
                            ],
                            axis=1,
                        ).dropna()
                        if len(pair) >= MIN_CELL * 2:
                            rho = sps.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
                            a = {
                                "name": "spearman_rho",
                                "value": _r(rho.statistic),
                                "p_value": _r(rho.pvalue, 6),
                            }
                else:
                    a = _association(c.frame[col], c.frame[c.target], c.classification)
                if a is not None:
                    out.append({"column": col, "role": key, **a})
    return out


def _confounding(
    c: _Ctx, batch: dict[str, Any], covariates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for item in batch.get("items", []):
        assoc = item.get("outcome_association")
        if assoc and _num(assoc["value"]) >= 0.2 and _sig(assoc["p_value"]):
            warnings.append(
                {
                    "kind": "batch_outcome_association",
                    "severity": "high",
                    "column": item["column"],
                    "message": f"The outcome differs between levels of '{item['column']}' "
                    f"({assoc['name']} {assoc['value']}). Batch or site may be confounded with the "
                    "outcome; a model can learn the batch instead of the biology.",
                }
            )
        strong = [
            s
            for s in item.get("biomarker_shifts", [])
            if _num(s["value"]) >= 0.1 and _sig(s["p_value"])
        ]
        if strong:
            warnings.append(
                {
                    "kind": "batch_effect",
                    "severity": "medium",
                    "column": item["column"],
                    "columns": [s["column"] for s in strong][:10],
                    "message": f"{len(strong)} biomarker(s) shift strongly between levels of "
                    f"'{item['column']}' (epsilon² ≥ 0.1): possible batch effect. Consider "
                    "batch-aware "
                    "evaluation (group the split by batch) before trusting comparisons.",
                }
            )
    for cov in covariates:
        if (cov["value"] is not None and abs(cov["value"]) >= 0.2) and _sig(cov["p_value"]):
            warnings.append(
                {
                    "kind": "covariate_outcome_association",
                    "severity": "medium",
                    "column": cov["column"],
                    "message": f"'{cov['column']}' ({cov['role']}) is associated with the outcome "
                    f"({cov['name']} {cov['value']}). It may confound biomarker–outcome "
                    "associations; "
                    "compare within its strata or adjust before interpreting them.",
                }
            )
    return warnings


def _entities(c: _Ctx, key_like: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"possible_identifiers": key_like}
    if c.patient is None:
        out["patient_leakage"] = _na(
            "No patient role is declared: rows of one patient could fall in both training and "
            "test folds and this cannot be checked."
        )
    else:
        out["patient_leakage"] = {
            "status": "assessed",
            "column": c.patient,
            "message": "Declared patients are kept together by grouped splits in the Experiment "
            "Lab and Models. Check that training runs used a grouped split.",
        }
    return out


def _small_cohort(c: _Ctx, cohort: dict[str, Any], target: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    pat = cohort.get("patients", {})
    units = pat.get("unique") if pat.get("status") == "assessed" else len(c.frame)
    unit = "patients" if pat.get("status") == "assessed" else "rows"
    if units is not None and units < SMALL_COHORT:
        out.append(
            {
                "kind": "small_cohort",
                "message": f"Only {units} {unit}: estimates are imprecise and models are likely to "
                "overfit; treat all results as exploratory.",
            }
        )
    minority = (target.get("imbalance") or {}).get("minority_n")
    if minority is not None and c.biomarkers:
        epv = minority / max(len(c.biomarkers), 1)
        if epv < EPV_MIN:
            out.append(
                {
                    "kind": "events_per_variable",
                    "message": f"About {epv:.1f} minority-class rows per candidate predictor "
                    f"(fewer than {EPV_MIN}): models with many predictors are prone to "
                    "overfitting.",
                }
            )
    return out


def summarise(
    frame: pd.DataFrame,
    roles: Mapping[str, Sequence[str]],
    task_type: str | None,
    sensitive: set[str],
    numeric_columns: list[str],
    key_like: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    c = _Ctx(frame, roles, task_type, sensitive, numeric_columns)
    cohort = _cohort(c)
    target = _target(c)
    batch = _batches(c)
    covariates = _covariates(c)
    return {
        "version": SUMMARY_VERSION,
        "disclaimer": DISCLAIMER,
        "min_cell": MIN_CELL,
        "roles": {
            "patient": c.patient,
            "target": c.target,
            "task_type": c.task_type,
            "batches": c.batches,
            "biomarkers": c.biomarkers,
            "sensitive": sorted(sensitive),
        },
        "cohort": cohort,
        "target": target,
        "biomarkers": _biomarkers(c),
        "comparisons": _comparisons(c),
        "batches": batch,
        "covariates": covariates,
        "confounding": _confounding(c, batch, covariates),
        "entities": _entities(c, key_like or []),
        "small_cohort": _small_cohort(c, cohort, target),
        "limitations": LIMITATIONS,
    }

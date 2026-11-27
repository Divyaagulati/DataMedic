"""Reliability audit of a training run (worker only). See ``audit_spec`` for the catalogue.

Each check receives an ``AuditContext`` (the training run's configuration and results, its
out-of-fold predictions, the frozen split, the cohort of the evaluation source, roles,
lineage facts and the latest diagnostics findings) and returns one ``Finding`` (the existing
diagnostics schema). Checks are isolated: an exception becomes ``error`` carrying only the
exception type. The audit never modifies datasets, roles, pipelines or results.
"""

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn import metrics as skm
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from datamedic.engine import ROW_ID
from datamedic.engine.checks.framework import Finding
from datamedic.engine.ml.audit_spec import AUDIT_CHECKS, AuditSpec, evaluation_status
from datamedic.engine.ml.spec import CLASSIFICATION, METRICS, PRIMARY
from datamedic.engine.preprocess.split import Splits

ENTITY_ROLES = ("group", "bio.patient_id", "fin.account_id", "ind.asset_id", "biz.customer_id")
TIME_ROLES = ("time", "fin.timestamp", "ind.timestamp")
MAX_SINGLE_FEATURES = 200
MAX_ROWS = 50_000


@dataclass
class AuditContext:
    task: str
    target: str
    data: pd.DataFrame  # cohort of the evaluation source (raw values)
    features: list[str]  # raw feature columns (roles excluded)
    splits: Splits
    summary: Mapping[str, Any]
    models: Mapping[str, Mapping[str, Any]]
    evidence: Mapping[str, Any]
    mode: str  # unbiased_direct | unbiased_replay | exploratory
    eligibility: Mapping[str, Any]
    roles: Mapping[str, Sequence[str]]
    split_config: Mapping[str, Any]
    preprocess_steps: Sequence[Mapping[str, Any]]  # with "needs_target" and "op"
    lineage_steps: Sequence[Mapping[str, Any]]  # committed steps of the selected version
    sensitive: frozenset[str]
    datetime_columns: list[str]
    diagnostics: Mapping[str, Any]  # {"state": current|stale|missing, "findings": [...]}
    seed: int = 0
    max_rows: int = MAX_ROWS
    max_single_features: int = MAX_SINGLE_FEATURES
    results: dict[str, Finding] = field(default_factory=dict)

    @property
    def classification(self) -> bool:
        return self.task in CLASSIFICATION

    def first_role(self, keys: Sequence[str]) -> str | None:
        for k in keys:
            if self.roles.get(k):
                return sorted(self.roles[k])[0]
        return None


SPECS = {s.id: s for s in AUDIT_CHECKS}


def _finding(
    spec: AuditSpec,
    status: str,
    severity: str,
    explanation: str,
    evidence: dict[str, Any],
    *,
    title: str | None = None,
    columns: list[str] | None = None,
    recommendation: str | None = None,
    reason: str | None = None,
    affected_rows: int | None = None,
    n_rows: int | None = None,
    sample_info: dict[str, Any] | None = None,
) -> Finding:
    return Finding(
        check_id=spec.id,
        check_version=spec.version,
        domain=spec.domain,
        category=spec.category,
        status=status,
        severity=severity,
        title=title or spec.title,
        explanation=explanation,
        evidence={"basis": spec.basis, **evidence},
        columns=columns or [],
        affected_rows=affected_rows,
        affected_pct=round(100 * affected_rows / n_rows, 4)
        if affected_rows is not None and n_rows
        else None,
        recommendation=recommendation,
        limitations=list(spec.limitations),
        not_assessed_reason=reason,
        sampled=sample_info is not None,
        sample_info=sample_info,
    )


def _ok(
    spec: AuditSpec, explanation: str, evidence: dict[str, Any] | None = None, **kw: Any
) -> Finding:
    return _finding(spec, "not_detected", "info", explanation, evidence or {}, **kw)


def _na(spec: AuditSpec, reason: str) -> Finding:
    return _finding(spec, "not_assessed", "info", reason, {}, reason=reason)


def _hit(
    spec: AuditSpec,
    severity: str,
    title: str,
    explanation: str,
    evidence: dict[str, Any],
    recommendation: str,
    **kw: Any,
) -> Finding:
    if spec.basis == "statistical" and not title.lower().startswith(("possible", "suspected")):
        title = "Possible: " + title[0].lower() + title[1:]
    return _finding(
        spec,
        "detected",
        severity,
        explanation,
        evidence,
        title=title,
        recommendation=recommendation,
        **kw,
    )


def _sample(ctx: AuditContext) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if len(ctx.data) <= ctx.max_rows:
        return ctx.data, None
    sampled = ctx.data.sample(n=ctx.max_rows, random_state=ctx.seed)
    return sampled, {"rows": ctx.max_rows, "of": len(ctx.data), "seed": ctx.seed}


def _codes(ctx: AuditContext, frame: pd.DataFrame) -> np.ndarray:
    y = frame[ctx.target]
    if ctx.classification:
        return np.asarray(pd.factorize(y.astype(object), sort=True)[0])
    return np.asarray(y.to_numpy(dtype=float))


def _feature_matrix(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        return np.asarray(s.to_numpy(dtype=float, na_value=np.nan)).reshape(-1, 1)
    if pd.api.types.is_datetime64_any_dtype(s):
        codes, _ = pd.factorize(s, sort=True)  # time order as ranks; NaT -> -1
        out = np.asarray(codes, dtype=float)
        out[codes < 0] = np.nan
        return out.reshape(-1, 1)
    raw = np.asarray(pd.factorize(s.astype(object))[0])
    values = raw.astype(float)
    values[raw < 0] = np.nan
    return values.reshape(-1, 1)


# ---- A1 single feature --------------------------------------------------------------------------


def _single_feature_scores(ctx: AuditContext) -> tuple[dict[str, float], dict[str, Any]]:
    cache = getattr(ctx, "_single_cache", None)
    if cache is not None:
        return cache  # type: ignore[no-any-return]
    frame, sample = _sample(ctx)
    y = _codes(ctx, frame)
    ids = frame[ROW_ID].to_numpy()
    features = list(ctx.features)
    coverage = {"assessed": len(features), "total": len(features)}
    if len(features) > ctx.max_single_features:
        variance = {
            c: float(np.nanvar(_feature_matrix(frame[c]))) if len(frame) else 0.0 for c in features
        }
        features = sorted(features, key=lambda c: (-variance[c], c))[: ctx.max_single_features]
        coverage["assessed"] = len(features)
    scores: dict[str, float] = {}
    for c in features:
        x = _feature_matrix(frame[c])
        if np.all(np.isnan(x)):
            continue
        oof = (
            np.full(len(frame), np.nan)
            if not ctx.classification or len(np.unique(y)) == 2
            else None
        )
        oof_multi = np.full(len(frame), -1) if oof is None else None
        for f in ctx.splits.folds:
            tr, te = np.isin(ids, f.train_ids), np.isin(ids, f.test_ids)
            if not tr.any() or not te.any():
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if ctx.classification:
                    model = DecisionTreeClassifier(max_depth=3, random_state=ctx.seed)
                    model.fit(x[tr], y[tr])
                    if oof is not None:
                        proba = model.predict_proba(x[te])
                        classes = list(model.classes_)
                        oof[te] = proba[:, classes.index(1)] if 1 in classes else 0.0
                    else:
                        assert oof_multi is not None
                        oof_multi[te] = model.predict(x[te])
                else:
                    model = DecisionTreeRegressor(max_depth=3, random_state=ctx.seed)
                    model.fit(x[tr], y[tr])
                    assert oof is not None
                    oof[te] = model.predict(x[te])
        if oof is not None:
            mask = ~np.isnan(oof)
            if mask.sum() < 10:
                continue
            if ctx.classification:
                if len(np.unique(y[mask])) < 2:
                    continue
                auc = float(skm.roc_auc_score(y[mask], oof[mask]))
                scores[c] = max(auc, 1 - auc)
            else:
                if float(np.var(y[mask])) == 0:
                    continue
                scores[c] = float(skm.r2_score(y[mask], oof[mask]))
        else:
            assert oof_multi is not None
            mask = oof_multi >= 0
            if mask.sum() < 10:
                continue
            scores[c] = float(skm.balanced_accuracy_score(y[mask], oof_multi[mask]))
    result = (scores, {"coverage": coverage, "sample": sample})
    ctx._single_cache = result  # type: ignore[attr-defined]
    return result


def _best_model_primary(ctx: AuditContext) -> tuple[str | None, float | None]:
    primary = PRIMARY[ctx.task]
    best, value = None, None
    for m, r in ctx.models.items():
        if m == "dummy" or not r.get("comparable"):
            continue
        v = r["metrics"].get(primary, {}).get("mean")
        if v is not None and (value is None or v > value):
            best, value = m, v
    return best, value


def check_single_feature(ctx: AuditContext, spec: AuditSpec) -> Finding:
    if not ctx.features:
        return _na(spec, "No feature columns to assess.")
    scores, meta = _single_feature_scores(ctx)
    metric = (
        "r2"
        if ctx.task == "regression"
        else ("roc_auc" if ctx.task == "binary" else "balanced_accuracy")
    )
    threshold = 0.95 if ctx.task == "regression" else 0.98
    flagged = {c: v for c, v in scores.items() if v >= threshold}
    best, best_value = _best_model_primary(ctx)
    dominant = {}
    if best_value is not None and best_value >= 0.9 and metric == PRIMARY[ctx.task]:
        dominant = {c: v for c, v in scores.items() if v >= 0.95 * best_value and c not in flagged}
    evidence = {
        "metric": metric,
        "threshold": threshold,
        "top": [
            {"column": c, "score": round(v, 4)}
            for c, v in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
        ],
        "coverage": meta["coverage"],
        "best_model": best,
        "best_model_score": best_value,
    }
    if flagged:
        worst = max(flagged.values())
        cols = sorted(flagged, key=lambda c: -flagged[c])
        return _hit(
            spec,
            "critical" if worst >= 0.995 else "high",
            f"Possible target leakage: {len(cols)} feature(s) alone predict the target",
            f"Each of these features alone reaches {metric} >= {threshold} out of fold. This is "
            "a statistical suspicion, not a confirmed leak.",
            {**evidence, "flagged": [{"column": c, "score": round(flagged[c], 4)} for c in cols]},
            "Check that each flagged feature is known before the outcome at prediction time; "
            "if not, mark it 'exclude' in the roles and re-run.",
            columns=cols,
            sample_info=meta["sample"],
        )
    if dominant:
        cols = sorted(dominant, key=lambda c: -dominant[c])
        return _hit(
            spec,
            "medium",
            "Possible dominant feature: one feature nearly matches the full model",
            "A single feature reaches at least 95% of the best model's score; the model may "
            "rely on it almost entirely.",
            {**evidence, "dominant": [{"column": c, "score": round(dominant[c], 4)} for c in cols]},
            "Review whether the feature is legitimately available at prediction time.",
            columns=cols,
            sample_info=meta["sample"],
        )
    return _ok(
        spec,
        f"No single assessed feature reaches {metric} >= {threshold} out of fold.",
        evidence,
        sample_info=meta["sample"],
    )


# ---- A2 derived ---------------------------------------------------------------------------------


def check_derived(ctx: AuditContext, spec: AuditSpec) -> Finding:
    if not ctx.features:
        return _na(spec, "No feature columns to assess.")
    frame, sample = _sample(ctx)
    y = frame[ctx.target]
    n = len(frame)
    flagged: list[dict[str, Any]] = []
    for c in ctx.features:
        s = frame[c]
        present = s.notna() & y.notna()
        if present.sum() < 20:
            continue
        s, yy = s[present], y[present]
        numeric = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if ctx.classification:
            n_cat = s.nunique()
            if not numeric and 2 <= n_cat <= max(2, n // 10):
                counts = pd.crosstab(s.astype(object), yy.astype(object))
                purity = float(counts.max(axis=1).sum() / counts.to_numpy().sum())
                if purity >= 0.99 and yy.nunique() > 1:
                    flagged.append(
                        {"column": c, "relation": "category_purity", "value": round(purity, 4)}
                    )
                    continue
            if numeric and yy.nunique() > 1:
                within = s.groupby(yy.astype(object)).nunique()
                if (within == 1).all() and s.nunique() > 1:
                    flagged.append({"column": c, "relation": "constant_within_class", "value": 1.0})
        elif numeric:
            if s.nunique() > 2 and yy.nunique() > 2:
                rho = sps.spearmanr(s.to_numpy(dtype=float), yy.to_numpy(dtype=float)).statistic
                if rho is not None and np.isfinite(rho) and abs(float(rho)) >= 0.999:
                    flagged.append(
                        {"column": c, "relation": "spearman", "value": round(float(rho), 6)}
                    )
        else:
            n_cat = s.nunique()
            if 2 <= n_cat <= max(2, n // 10):
                means = yy.groupby(s.astype(object)).transform("mean")
                total = float(((yy - yy.mean()) ** 2).sum())
                if total > 0:
                    r2 = 1 - float(((yy - means) ** 2).sum()) / total
                    if r2 >= 0.999:
                        flagged.append(
                            {"column": c, "relation": "category_means", "value": round(r2, 6)}
                        )
    if flagged:
        return _hit(
            spec,
            "critical",
            f"Possible target-derived features: {len(flagged)} near-deterministic relation(s)",
            "These features relate to the target almost deterministically. That can be "
            "genuine, but it is typical of values computed from the outcome.",
            {"flagged": flagged},
            "Confirm how each feature is produced; exclude it if it is computed from, or "
            "recorded after, the outcome.",
            columns=[f["column"] for f in flagged],
            sample_info=sample,
        )
    return _ok(spec, "No near-deterministic feature-target relation found.", {}, sample_info=sample)


# ---- A3 identifier -----------------------------------------------------------------------------


def _id_like(s: pd.Series) -> bool:
    present = s.dropna()
    if len(present) < 20:
        return False
    ratio = present.nunique() / len(present)
    if ratio < 0.95:
        return False
    if pd.api.types.is_integer_dtype(present):
        return True
    if pd.api.types.is_float_dtype(present):
        return bool((present % 1 == 0).all())
    return not pd.api.types.is_bool_dtype(present) and not pd.api.types.is_datetime64_any_dtype(
        present
    )


def check_identifier(ctx: AuditContext, spec: AuditSpec) -> Finding:
    candidates = [c for c in ctx.features if _id_like(ctx.data[c])]
    if not candidates:
        return _ok(spec, "No identifier-like column is a model input.", {"id_like": []})
    scores, meta = _single_feature_scores(ctx)
    y = _codes(ctx, ctx.data)
    flagged = []
    for c in candidates:
        s = ctx.data[c]
        if pd.api.types.is_numeric_dtype(s):
            order = s.to_numpy(dtype=float, na_value=np.nan)
        else:
            order = s.astype(object).rank(method="average").to_numpy(dtype=float)
        mask = ~np.isnan(order)
        rho, p = (np.nan, np.nan)
        if mask.sum() > 20 and len(np.unique(y[mask])) > 1:
            res = sps.spearmanr(order[mask], y[mask])
            rho, p = float(res.statistic), float(res.pvalue)
        score = scores.get(c)
        strong_rank = np.isfinite(rho) and abs(rho) >= 0.3 and p < 0.001
        strong_score = score is not None and score >= (0.3 if ctx.task == "regression" else 0.7)
        if strong_rank or strong_score:
            flagged.append(
                {
                    "column": c,
                    "spearman": None if not np.isfinite(rho) else round(rho, 4),
                    "p_value": None if not np.isfinite(p) else float(f"{p:.3g}"),
                    "single_feature_score": None if score is None else round(score, 4),
                }
            )
    if flagged:
        return _hit(
            spec,
            "high",
            f"Possible identifier leakage: {len(flagged)} identifier-like input(s) carry "
            "target signal",
            "Identifier-like columns should not predict the outcome; an association suggests "
            "the data was sorted or numbered by outcome.",
            {"flagged": flagged, "id_like": candidates},
            "Give these columns the 'id' role (or 'exclude') so they are not model inputs.",
            columns=[f["column"] for f in flagged],
            sample_info=meta["sample"],
        )
    return _ok(
        spec,
        "Identifier-like inputs exist but show no target signal.",
        {"id_like": candidates},
    )


# ---- A4 duplicates ------------------------------------------------------------------------------


def _normalised(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = {}
    for c in columns:
        s = frame[c]
        if pd.api.types.is_float_dtype(s):
            v = s.to_numpy(dtype=float, na_value=np.nan)
            with np.errstate(divide="ignore", invalid="ignore"):
                mag = np.where(v == 0, 0, np.floor(np.log10(np.abs(v))))
            factor = 10 ** (5 - np.nan_to_num(mag))
            out[c] = np.round(v * factor) / factor
        elif pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            out[c] = s
        else:
            out[c] = s.astype(object).map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    return pd.DataFrame(out, index=frame.index)


def check_duplicates(ctx: AuditContext, spec: AuditSpec) -> Finding:
    if not ctx.features:
        return _na(spec, "No feature columns to compare.")
    data = ctx.data
    ids = data[ROW_ID].to_numpy()
    feats = list(ctx.features)
    hashes = {
        "exact": pd.util.hash_pandas_object(data[feats + [ctx.target]], index=False).to_numpy(),
        "features_only": pd.util.hash_pandas_object(data[feats], index=False).to_numpy(),
        "normalised": pd.util.hash_pandas_object(
            _normalised(data, feats + [ctx.target]), index=False
        ).to_numpy(),
    }
    counts = {k: 0 for k in hashes}
    per_fold = []
    for f in ctx.splits.folds:
        tr, te = np.isin(ids, f.train_ids), np.isin(ids, f.test_ids)
        row: dict[str, Any] = {"fold": f.index}
        for k, h in hashes.items():
            crossing = int(np.isin(h[te], h[tr]).sum())
            row[k] = crossing
            counts[k] += crossing
        per_fold.append(row)
    evidence = {"test_rows_with_training_duplicate": counts, "per_fold": per_fold}
    if counts["exact"] or counts["normalised"]:
        n = max(counts["exact"], counts["normalised"])
        return _hit(
            spec,
            "high",
            f"Duplicate rows on both sides of the split ({n} test rows)",
            "Some test rows have an identical (or, after normalisation, identical) row among "
            "the training rows, so the model is partly tested on rows it has seen.",
            evidence,
            "Use DataMedic's default split (duplicates are grouped automatically) or remove "
            "duplicates in a pipeline after review.",
            affected_rows=n,
            n_rows=len(data),
        )
    note = (
        f"{counts['features_only']} test rows share feature values (but not the label) with "
        "a training row."
        if counts["features_only"]
        else "No test row has an identical training row."
    )
    return _ok(spec, note, evidence)


# ---- A5/A6 groups -------------------------------------------------------------------------------


def check_groups_cross_fold(ctx: AuditContext, spec: AuditSpec) -> Finding:
    group = ctx.first_role(ENTITY_ROLES)
    if group is None or group not in ctx.data.columns:
        return _na(spec, "No group (entity) role is declared, so entity overlap cannot be checked.")
    ids = ctx.data[ROW_ID].to_numpy()
    values = ctx.data[group].astype(object).to_numpy()
    per_fold = []
    total = 0
    for f in ctx.splits.folds:
        tr = set(values[np.isin(ids, f.train_ids)].tolist())
        te = set(values[np.isin(ids, f.test_ids)].tolist())
        overlap = len((tr & te) - {None})
        per_fold.append({"fold": f.index, "entities_in_both": overlap})
        total += overlap
    evidence = {
        "group_column": group,
        "split_method": ctx.split_config.get("method"),
        "per_fold": per_fold,
    }
    if total:
        return _hit(
            spec,
            "critical",
            "The same entities appear in training and test rows",
            f"{total} entity occurrences span training and test rows across folds; scores are "
            "inflated because the model sees each entity before being tested on it.",
            evidence,
            "Use a grouped split (the default when an entity role is declared).",
            columns=[group],
        )
    return _ok(
        spec,
        "No entity appears in both the training and test rows of any fold.",
        evidence,
        columns=[group],
    )


KEY_MIN_ROWS = 50
KEY_MIN_DISTINCT = 20
KEY_MIN_DISTINCT_SHARE = 0.05


def _key_like_repeats(s: pd.Series) -> float | None:
    """Rows per value when a feature looks like a repeated entity key: integer-valued or text,
    many distinct values (>= 20 and >= 5% of rows), yet repeated (>= 1.2 rows per value).
    Evidence only: measurements with many repeated integer values can match too."""
    present = s.dropna()
    if len(present) < KEY_MIN_ROWS or pd.api.types.is_bool_dtype(present):
        return None
    if pd.api.types.is_datetime64_any_dtype(present):
        return None
    if pd.api.types.is_float_dtype(present) and not bool((present % 1 == 0).all()):
        return None
    n_unique = int(present.astype(object).nunique())
    if n_unique < KEY_MIN_DISTINCT or n_unique / len(present) < KEY_MIN_DISTINCT_SHARE:
        return None
    rows_per = len(present) / n_unique
    return round(rows_per, 3) if rows_per >= 1.2 else None


def check_groups_undeclared(ctx: AuditContext, spec: AuditSpec) -> Finding:
    group = ctx.first_role(ENTITY_ROLES)
    if group is not None:
        return _ok(spec, "An entity role is declared.", {"group_column": group})
    signals: list[dict[str, Any]] = []
    key_like: list[dict[str, Any]] = []
    for c in ctx.features:
        if c in ctx.data.columns:
            rows_per = _key_like_repeats(ctx.data[c])
            if rows_per is not None:
                key_like.append(
                    {"column": c, "kind": "key_like_feature_repeats", "rows_per_value": rows_per}
                )
    for c in ctx.roles.get("id", ()):
        if c in ctx.data.columns:
            s = ctx.data[c].dropna()
            if len(s) and len(s) / max(s.nunique(), 1) >= 1.2:
                signals.append(
                    {
                        "column": c,
                        "kind": "identifier_repeats",
                        "rows_per_value": round(len(s) / s.nunique(), 3),
                    }
                )
    if ctx.features:
        dup = ctx.data.duplicated(subset=list(ctx.features), keep=False)
        share = float(dup.mean()) if len(dup) else 0.0
        if share >= 0.2:
            signals.append(
                {"kind": "repeated_feature_signatures", "share_of_rows": round(share, 4)}
            )
    if signals:
        return _hit(
            spec,
            "high",
            "Possible repeated entities without a group role",
            "Rows may belong to repeated entities (e.g. patients or customers), but no entity "
            "role is declared, so folds cannot keep them together.",
            {"signals": signals + key_like},
            "If rows belong to the same entity, declare that column's group role and re-run.",
            columns=[s["column"] for s in signals + key_like if "column" in s],
        )
    if key_like:
        return _hit(
            spec,
            "medium",
            "Possible entity key used as a feature without a group role",
            "A model input has many distinct values that repeat across rows, as an entity "
            "identifier (e.g. a patient or customer code) would. Without a group role the "
            "folds may place rows of one entity in both training and test rows, and the model "
            "may learn the identifier. This is evidence, not a confirmed identifier: a "
            "measurement with many repeated values can look the same.",
            {"signals": key_like},
            "If the column identifies an entity, give it the group role (or 'id' / 'exclude') "
            "and train again.",
            columns=[k["column"] for k in key_like],
        )
    return _ok(
        spec,
        "No sign of undeclared repeated entities among the inspected signals (declared "
        "identifiers, key-like feature columns and repeated feature signatures).",
        {"signals": []},
    )


# ---- A7/A8 temporal ----------------------------------------------------------------------------


def check_temporal(ctx: AuditContext, spec: AuditSpec) -> Finding:
    time = ctx.first_role(TIME_ROLES)
    if time is None or time not in ctx.data.columns:
        reason = "No time role is declared, so temporal leakage cannot be assessed."
        if ctx.datetime_columns:
            reason += " Date or time columns exist; declare a time role if rows are time-ordered."
        return _na(spec, reason)
    method = ctx.split_config.get("method")
    if method != "time":
        return _hit(
            spec,
            "critical",
            "A declared time role was ignored by the split",
            "Rows are time-ordered but the folds are not, so models are tested on rows earlier "
            "than some training rows.",
            {"time_column": time, "split_method": method},
            "Use the time split (the default when a time role is declared).",
            columns=[time],
        )
    ids = ctx.data[ROW_ID].to_numpy()
    values = ctx.data[time]
    violations = []
    for f in ctx.splits.folds:
        tr = values[np.isin(ids, f.train_ids)]
        te = values[np.isin(ids, f.test_ids)]
        if len(tr) and len(te) and not (tr.max() < te.min()):
            violations.append(f.index)
    evidence = {"time_column": time, "split_method": method, "folds_out_of_order": violations}
    if violations:
        return _hit(
            spec,
            "critical",
            "Training rows after test rows in a time split",
            "Some fold trains on rows later than its test rows.",
            evidence,
            "Re-run with the time split; report this as a defect if it persists.",
            columns=[time],
        )
    return _ok(
        spec, "Every training row precedes every test row in every fold.", evidence, columns=[time]
    )


def check_post_outcome(ctx: AuditContext, spec: AuditSpec) -> Finding:
    return _na(
        spec,
        "Outcome-time and feature-time roles are not available with universal roles; "
        "post-outcome features cannot be assessed in this version.",
    )


# ---- A9-A12 pipeline ----------------------------------------------------------------------------


def check_fit_on_full_data(ctx: AuditContext, spec: AuditSpec) -> Finding:
    if ctx.mode == "exploratory":
        return _hit(
            spec,
            "critical",
            "Evaluated on a version fitted on all rows (exploratory run)",
            "Committed steps of this version learned from every row, including rows used here "
            "as test rows. The scores are likely optimistic and are not a leakage-controlled "
            "estimate.",
            {"mode": ctx.mode, "eligibility": dict(ctx.eligibility)},
            "Evaluate the version through replay from its eligible ancestor, or evaluate an "
            "earlier version that is not fitted on all rows.",
        )
    if ctx.mode == "unbiased_replay":
        return _ok(
            spec,
            "The committed steps were replayed inside every training fold from the eligible "
            "ancestor; the replay was proven faithful before training.",
            {"mode": ctx.mode, "replay": ctx.eligibility.get("replay")},
        )
    return _ok(spec, "The evaluated version was not fitted on all rows.", {"mode": ctx.mode})


def check_boundaries(ctx: AuditContext, spec: AuditSpec) -> Finding:
    rows = list(ctx.evidence.get("boundaries", []))
    if not rows:
        return _na(spec, "No fold completed preprocessing, so no boundary evidence exists.")
    bad = [
        r
        for r in rows
        if not r["fit_rows_equal_training_rows"]
        or not r["test_rows_complete"]
        or r["synthetic_rows_in_test"]
        or r["weights_used_in_scoring"]
    ]
    if bad:
        return _hit(
            spec,
            "critical",
            "A preprocessing boundary was violated",
            "In some fold a learned step was not fitted on exactly the training rows, or test "
            "rows were incomplete or included synthetic rows.",
            {"folds": bad},
            "Do not use these results; report the run as a defect.",
        )
    return _ok(
        spec,
        "In every fold, learned steps were fitted on the training rows only, every test row "
        "was scored, and no synthetic row reached the test rows.",
        {"folds": rows},
    )


def check_tuning(ctx: AuditContext, spec: AuditSpec) -> Finding:
    tuning = ctx.summary.get("tuning", {})
    records = [r for r in tuning.get("per_fold", []) if r.get("mode") == "nested"]
    if tuning.get("used") != "nested" or not records:
        return _ok(
            spec,
            "No model selection was performed (default settings), so no outer test row could "
            "influence a selection.",
            {"tuning": tuning.get("used"), "fallback_reason": tuning.get("fallback_reason")},
        )
    bad = [r["fold"] for r in records if not r.get("inner_test_ids_within_outer_training")]
    if bad:
        return _hit(
            spec,
            "critical",
            "Model selection used rows outside the outer training rows",
            "Inner folds of some outer folds were not contained in the outer training rows.",
            {"folds": bad},
            "Do not use these results; report the run as a defect.",
        )
    return _ok(
        spec,
        "Every inner fold lay inside its outer training rows; settings were selected before "
        "the outer test rows were scored.",
        {
            "folds": [r["fold"] for r in records],
            "selection_metric": [r.get("selection_metric") for r in records],
        },
    )


def check_target_encoding(ctx: AuditContext, spec: AuditSpec) -> Finding:
    in_fold = [s["op"] for s in ctx.preprocess_steps if s.get("needs_target")]
    committed = [s["op"] for s in ctx.lineage_steps if s.get("needs_target")]
    if ctx.mode == "exploratory" and committed:
        return _hit(
            spec,
            "critical",
            "Steps that read the target were fitted on all rows",
            "The evaluated version contains committed steps that learned from target values of "
            "every row, including test rows.",
            {"committed_target_steps": committed},
            "Evaluate through replay from the eligible ancestor.",
        )
    return _ok(
        spec,
        "Every step that reads the target was fitted inside training folds (training labels only)."
        if in_fold or committed
        else "No step reads the target.",
        {
            "in_fold_target_steps": in_fold,
            "replayed_target_steps": committed if ctx.mode != "exploratory" else [],
        },
    )


# ---- A13-A20 reliability ------------------------------------------------------------------------


def check_overfitting(ctx: AuditContext, spec: AuditSpec) -> Finding:
    primary = PRIMARY[ctx.task]
    rows: list[dict[str, Any]] = []
    for m, r in ctx.models.items():
        if m == "dummy" or not r.get("train_scores"):
            continue
        train = [t.get(primary) for t in r["train_scores"] if t and t.get(primary) is not None]
        test = r["metrics"].get(primary, {}).get("mean")
        if not train or test is None:
            continue
        gap = float(np.mean(train)) - float(test)
        limit = 0.25 if m == "random_forest" else 0.15
        rows.append(
            {
                "model": m,
                "train": round(float(np.mean(train)), 4),
                "test": round(float(test), 4),
                "gap": round(gap, 4),
                "threshold": limit,
            }
        )
    if not rows:
        return _na(spec, "No model has both training and test scores.")
    flagged = [r for r in rows if float(r["gap"]) > float(r["threshold"])]
    if flagged:
        return _hit(
            spec,
            "medium",
            "Possible overfitting: large gap between training and test scores",
            f"The {primary} on training rows exceeds the test score by more than the "
            "threshold for some models.",
            {"metric": primary, "models": rows},
            "Prefer more regularised settings or simpler models; check the sample size.",
        )
    return _ok(
        spec,
        "Training and test scores are within the thresholds.",
        {"metric": primary, "models": rows},
    )


def check_fold_variability(ctx: AuditContext, spec: AuditSpec) -> Finding:
    if len(ctx.splits.folds) < 3:
        return _na(spec, "Fewer than 3 folds.")
    primary = PRIMARY[ctx.task]
    bounded = METRICS[primary][2]
    rows = []
    for m, r in ctx.models.items():
        st = r["metrics"].get(primary, {})
        if st.get("sd") is None or st.get("mean") is None:
            continue
        cv = abs(st["sd"] / st["mean"]) if st["mean"] else None
        spread = st["max"] - st["min"]
        flagged = (bounded and st["sd"] > 0.05) or (cv is not None and cv > 0.2) or spread > 0.15
        rows.append(
            {
                "model": m,
                "sd": st["sd"],
                "mean": st["mean"],
                "range": round(spread, 4),
                "cv": None if cv is None else round(cv, 4),
                "flagged": bool(flagged),
            }
        )
    if not rows:
        return _na(spec, f"{primary} is unavailable in too many folds.")
    flagged = [r for r in rows if r["flagged"] and r["model"] != "dummy"]
    if flagged:
        return _hit(
            spec,
            "medium",
            "Possible instability: scores vary strongly between folds",
            "Fold scores differ widely, so the mean is an uncertain summary.",
            {"metric": primary, "models": rows},
            "Interpret differences between models cautiously; more data or fewer, larger folds "
            "can help.",
        )
    return _ok(
        spec,
        "Fold-to-fold variation is within the thresholds.",
        {"metric": primary, "models": rows},
    )


def check_class_imbalance(ctx: AuditContext, spec: AuditSpec) -> Finding:
    if not ctx.classification:
        return _na(spec, "Applies to classification targets only.")
    counts = ctx.data[ctx.target].astype(object).value_counts()
    share = float(counts.min() / counts.sum())
    evidence = {
        "minority_share": round(share, 4),
        "classes": int(len(counts)),
        "majority_rate": round(float(counts.max() / counts.sum()), 4),
    }
    if share < 0.10:
        return _hit(
            spec,
            "high" if share < 0.02 else "medium",
            f"Class imbalance: the rarest class is {share:.1%} of rows",
            "Accuracy can look high by predicting the majority class; balanced accuracy and "
            "PR-AUC are more informative.",
            evidence,
            "Compare models on balanced accuracy and PR-AUC; consider class weights inside folds.",
        )
    return _ok(spec, "Classes are not strongly imbalanced.", evidence)


def check_sample_size(ctx: AuditContext, spec: AuditSpec) -> Finding:
    ids = ctx.data[ROW_ID].to_numpy()
    issues = []
    smallest_test = min(len(f.test_ids) for f in ctx.splits.folds)
    if smallest_test < 30:
        issues.append({"issue": "small_test_fold", "rows": smallest_test, "severity": "high"})
    p = max(len(ctx.features), 1)
    if ctx.classification:
        y = _codes(ctx, ctx.data)
        per_fold_min = min(
            int(np.bincount(y[np.isin(ids, f.test_ids)], minlength=int(y.max()) + 1).min())
            for f in ctx.splits.folds
        )
        if per_fold_min < 5:
            issues.append(
                {"issue": "rare_class_in_test_fold", "rows": per_fold_min, "severity": "high"}
            )
        if ctx.task == "binary":
            epv = float(np.bincount(y).min()) / p
            if epv < 10:
                issues.append(
                    {"issue": "events_per_feature", "value": round(epv, 2), "severity": "medium"}
                )
    elif len(ctx.data) / p < 10:
        issues.append(
            {
                "issue": "rows_per_feature",
                "value": round(len(ctx.data) / p, 2),
                "severity": "medium",
            }
        )
    if issues:
        severity = "high" if any(i["severity"] == "high" for i in issues) else "medium"
        return _hit(
            spec,
            severity,
            "Possible insufficient sample size",
            "Some size rule of thumb is not met; estimates are noisy.",
            {"issues": issues, "features": p, "rows": len(ctx.data)},
            "Use fewer folds or features, or collect more data; interpret scores cautiously.",
        )
    return _ok(spec, "Sample-size rules of thumb are met.", {"features": p, "rows": len(ctx.data)})


LEAK_CHECKS = (
    "leakage.target.single_feature",
    "leakage.target.derived",
    "leakage.identifier",
    "leakage.duplicates.cross_fold",
    "leakage.groups.cross_fold",
    "leakage.groups.undeclared",
)


def check_suspicious(ctx: AuditContext, spec: AuditSpec) -> Finding:
    high = []
    for m, r in ctx.models.items():
        if m == "dummy":
            continue
        for metric in ("roc_auc", "balanced_accuracy", "r2"):
            v = r["metrics"].get(metric, {}).get("mean")
            if v is not None and v >= 0.99:
                high.append({"model": m, "metric": metric, "mean": round(v, 4)})
                break
    leaks = [k for k in LEAK_CHECKS if k in ctx.results and ctx.results[k].status == "detected"]
    if high:
        return _hit(
            spec,
            "high",
            "Suspiciously high performance",
            "A model scores 0.99 or more. That is rare for real problems and often comes from "
            "leakage; it is not proof of leakage."
            + (" Leakage checks also detected issues." if leaks else ""),
            {"models": high, "related_detections": leaks},
            "Review the related leakage findings and the most important features.",
        )
    return _ok(spec, "No model scores 0.99 or more.", {"related_detections": leaks})


def check_dummy(ctx: AuditContext, spec: AuditSpec) -> Finding:
    if "dummy" not in ctx.models:
        return _na(spec, "The dummy baseline was not trained.")
    primary = PRIMARY[ctx.task]
    rows = [
        c
        for c in ctx.summary.get("comparisons", [])
        if c["reference"] == "dummy" and c["metric"] == primary
    ]
    if not rows:
        return _na(spec, "No comparable model to compare with the dummy baseline.")
    better = [
        c
        for c in rows
        if c["verdict"]
        not in ("negligible", "within_fold_variation", "consistently_worse", "insufficient_folds")
    ]
    if not better:
        return _hit(
            spec,
            "medium",
            "Possible lack of signal: no model clearly beats the dummy baseline",
            f"Every model's {primary} difference from the dummy is negligible or within fold "
            "variation.",
            {"comparisons": rows},
            "Check the target definition and features; more informative features may be needed.",
        )
    return _ok(
        spec,
        "At least one model beats the dummy baseline beyond fold variation.",
        {"comparisons": rows},
    )


def check_calibration(ctx: AuditContext, spec: AuditSpec) -> Finding:
    if not ctx.classification:
        return _na(spec, "Applies to classification models with probabilities only.")
    rows = []
    for m, r in ctx.models.items():
        if m == "dummy":
            continue
        pooled = r.get("pooled", {})
        if not pooled.get("available") or pooled.get("n", 0) < 100:
            continue
        ece = pooled["metrics"].get("ece")
        if ece is None:
            continue
        rows.append({"model": m, "ece": round(ece, 4), "brier": pooled["metrics"].get("brier")})
    if not rows:
        return _na(spec, "No model has probabilities on at least 100 rows.")
    resampling = [s["op"] for s in ctx.preprocess_steps if str(s["op"]).startswith("imbalance.")]
    worst = max(r["ece"] for r in rows)
    evidence = {"models": rows, "resampling_or_weights": resampling}
    if worst > 0.1:
        return _hit(
            spec,
            "high" if worst > 0.2 else "medium",
            "Possible poor probability calibration",
            "Predicted probabilities differ from observed frequencies."
            + (" Resampling or class weights shift probabilities by design." if resampling else ""),
            evidence,
            "Use probabilities cautiously; recalibrate on training folds before relying on them.",
        )
    return _ok(spec, "Calibration error is at most 0.1 for every model.", evidence)


def check_metadata(ctx: AuditContext, spec: AuditSpec) -> Finding:
    items = []
    if ctx.first_role(ENTITY_ROLES) is None:
        undeclared = ctx.results.get("leakage.groups.undeclared")
        if undeclared is not None and undeclared.status == "detected":
            items.append(
                {
                    "item": "entity_role",
                    "message": "Repeated entities are suspected but no entity role is declared.",
                }
            )
    if ctx.first_role(TIME_ROLES) is None and ctx.datetime_columns:
        items.append(
            {
                "item": "time_role",
                "message": "Date or time columns exist but no time role is declared.",
            }
        )
    state = ctx.diagnostics.get("state")
    if state == "missing":
        items.append(
            {"item": "diagnostics", "message": "No diagnostics run exists for this version."}
        )
    elif state == "stale":
        items.append(
            {
                "item": "diagnostics",
                "message": "The latest diagnostics run is older than the current roles.",
            }
        )
    if items:
        return _finding(
            spec,
            "detected",
            "medium" if any(i["item"] in ("entity_role", "time_role") for i in items) else "info",
            "Some metadata is missing or stale, which limits what the audit can assess.",
            {"items": items},
            title="Missing metadata limits the assessment",
            recommendation="Review the roles (entity, time) and re-run diagnostics; then re-run "
            "the audit.",
        )
    return _ok(spec, "No missing metadata limits the assessment.", {"items": []})


IMPLS: dict[str, Callable[[AuditContext, AuditSpec], Finding]] = {
    "leakage.target.single_feature": check_single_feature,
    "leakage.target.derived": check_derived,
    "leakage.identifier": check_identifier,
    "leakage.duplicates.cross_fold": check_duplicates,
    "leakage.groups.cross_fold": check_groups_cross_fold,
    "leakage.groups.undeclared": check_groups_undeclared,
    "leakage.temporal.ordering": check_temporal,
    "leakage.temporal.post_outcome": check_post_outcome,
    "leakage.pipeline.fit_on_full_data": check_fit_on_full_data,
    "leakage.pipeline.boundaries": check_boundaries,
    "leakage.tuning.isolation": check_tuning,
    "leakage.pipeline.target_encoding": check_target_encoding,
    "reliability.overfitting": check_overfitting,
    "reliability.fold_variability": check_fold_variability,
    "reliability.class_imbalance": check_class_imbalance,
    "reliability.sample_size": check_sample_size,
    "reliability.suspicious_performance": check_suspicious,
    "reliability.dummy_competitive": check_dummy,
    "reliability.calibration": check_calibration,
    "reliability.metadata": check_metadata,
}


def run_audit(ctx: AuditContext) -> tuple[list[Finding], dict[str, Any]]:
    findings = []
    for spec in AUDIT_CHECKS:
        try:
            finding = IMPLS[spec.id](ctx, spec)
        except MemoryError:
            raise
        except Exception as exc:  # noqa: BLE001  (isolated; the type only, never values)
            finding = Finding(
                check_id=spec.id,
                check_version=spec.version,
                domain=spec.domain,
                category=spec.category,
                status="error",
                severity="info",
                title=spec.title,
                explanation="The check failed unexpectedly; details are withheld because they "
                "could contain data values.",
                evidence={"basis": spec.basis, "error_type": type(exc).__name__},
                limitations=list(spec.limitations),
                error_code="CHECK_ERROR",
            )
        ctx.results[spec.id] = finding
        findings.append(finding)
    status = evaluation_status(
        [
            {
                "check_id": f.check_id,
                "status": f.status,
                "severity": f.severity,
                "basis": f.evidence.get("basis"),
            }
            for f in findings
        ],
        exploratory=ctx.mode == "exploratory",
    )
    summary = {
        "total": len(findings),
        "by_status": {
            s: sum(1 for f in findings if f.status == s)
            for s in ("detected", "not_detected", "not_assessed", "error")
        },
        "detected_by_severity": {
            s: sum(1 for f in findings if f.status == "detected" and f.severity == s)
            for s in ("critical", "high", "medium", "low", "info")
        },
        "detected_by_basis": {
            b: sum(1 for f in findings if f.status == "detected" and f.evidence.get("basis") == b)
            for b in ("confirmed", "configuration", "statistical")
        },
        "check_errors": sum(1 for f in findings if f.status == "error"),
        "evaluation_status": status,
    }
    return findings, summary

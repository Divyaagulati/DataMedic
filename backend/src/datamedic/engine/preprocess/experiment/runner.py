"""Run candidate preprocessing pipelines on identical frozen folds (Phase 4 Part D).

For every candidate and every fold:

1. **Training rows** of the fold go through ``PreprocessExecutor.fit_resample``: every
   learned parameter is fitted on them only, and resampling or class weights (the last,
   training-only step) apply to them only.
2. **Test rows** go through ``transform``: nothing is learned from them, nothing is
   resampled, no row is removed. The runner verifies that every test row was scored.
3. Each baseline model is fitted on the training table (with its documented minimal
   in-fold preprocessing and any per-row training weights) and scored on the test rows.

All candidates use the same ``Splits`` object, so they are evaluated on exactly the same
training and test row ids. Failures are isolated per fold and per model: a failing fold
never affects another candidate. The input frame is never modified. Nothing here writes a
dataset version: the output is a JSON-serialisable report.
"""

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.catalog import SAMPLE_WEIGHT
from datamedic.engine.preprocess.executor import INTERNAL_PREFIX, PreprocessExecutor
from datamedic.engine.preprocess.experiment import metrics as m
from datamedic.engine.preprocess.experiment import quality, stability
from datamedic.engine.preprocess.experiment.models import (
    MODEL_NOTES,
    build_model,
    feature_kinds,
    model_matrix,
)
from datamedic.engine.preprocess.split import Splits
from datamedic.engine.preprocess.validate import ValidationResult

SUSPICIOUS = 0.99
PRACTICAL_DIFFERENCE = 0.01  # bounded metrics; unbounded: 1% of the baseline mean
FEW_PER_CLASS = 5
SELECTION_OPS = (
    "select.variance",
    "select.correlation",
    "select.mutual_information",
    "select.univariate",
)

LIMITATIONS = [
    "Scores come from cross-validation on this dataset only; they estimate, with uncertainty, "
    "how a pipeline might perform on similar data.",
    "Fold scores are not independent (training rows overlap), so no p-values are given; "
    "differences are described per fold.",
    "Models use library defaults with no hyper-parameter tuning; a pipeline that helps one "
    "model may not help another.",
    "Every learned preprocessing parameter is fitted on the training rows of each fold; "
    "validation rows are never resampled, removed or used for fitting.",
    "A higher mean score alone does not make a pipeline better: weigh fold-to-fold "
    "variation, practical size of the difference, data-quality changes and information loss.",
    "Experiments never change the dataset: no version is written.",
]
CLUSTERING_LIMITATIONS = [
    "Clustering metrics are internal (cohesion and separation of the found clusters); they "
    "are not accuracy against any ground truth and favour compact, spherical clusters.",
    "Different preprocessing changes the space in which clusters are measured, so scores "
    "between candidates are only loosely comparable.",
]


class Cancelled(Exception):
    """Raised by the runner when ``should_cancel`` returns True."""


@dataclass(frozen=True)
class Candidate:
    key: str
    label: str
    validation: ValidationResult


@dataclass(frozen=True)
class ExperimentSpec:
    task: str  # binary | multiclass | regression | clustering
    target: str | None
    models: tuple[str, ...]
    seed: int = 0
    n_clusters: int | None = None


def cohort(frame: pd.DataFrame, target: str | None) -> tuple[pd.DataFrame, int]:
    """Rows that can be evaluated: those with a target value (all rows for clustering)."""
    if target is None:
        return frame, 0
    keep = frame[target].notna()
    return frame[keep], int((~keep).sum())


def _error(exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, EngineError):
        return {"code": exc.code, "message": exc.message}
    return {
        "code": "FAILED",
        "message": "Failed unexpectedly; details are withheld because "
        "they could contain data values.",
        "error_type": type(exc).__name__,
    }


def _library_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return sorted({w.category.__name__ for w in caught})


def _raw_features(frame: pd.DataFrame, v: ValidationResult) -> list[str]:
    excluded = set(v.protected) | ({v.target} if v.target else set())
    return [
        c for c in frame.columns if c not in excluded and not str(c).startswith(INTERNAL_PREFIX)
    ]


class _Labels:
    """Class labels as integer codes (models and metrics never see raw label values)."""

    def __init__(self, values: pd.Series) -> None:
        unique = pd.unique(values.astype(object))
        try:
            self.labels = sorted(unique.tolist())
        except TypeError:
            self.labels = sorted(unique.tolist(), key=lambda v: (type(v).__name__, str(v)))
        self.index = {v: i for i, v in enumerate(self.labels)}

    def encode(self, values: pd.Series) -> np.ndarray:
        return np.array([self.index[v] for v in values.astype(object)], dtype=np.int64)


def run_experiment(
    frame: pd.DataFrame,
    splits: Splits,
    candidates: list[Candidate],
    spec: ExperimentSpec,
    *,
    on_progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    # One native thread: thread-pool start-up dominated small fits (gradient boosting was 4x
    # slower with the default pools on the reference machine), and a worker process should
    # not use every core of the host.
    with threadpool_limits(limits=1):
        return _run(frame, splits, candidates, spec, on_progress, should_cancel)


def _run(
    frame: pd.DataFrame,
    splits: Splits,
    candidates: list[Candidate],
    spec: ExperimentSpec,
    on_progress: Callable[[int, int, str], None] | None,
    should_cancel: Callable[[], bool] | None,
) -> dict[str, Any]:
    data, excluded = cohort(frame, spec.target)
    classification = spec.task in ("binary", "multiclass")
    labels = _Labels(data[spec.target]) if classification and spec.target else None
    total = len(candidates) * len(splits.folds)
    done = 0
    results = []
    ignored: set[str] = set()
    for candidate in candidates:
        folds = []
        for fold in splits.folds:
            if should_cancel is not None and should_cancel():
                raise Cancelled()
            folds.append(_run_fold(data, splits, fold, candidate, spec, labels, ignored))
            done += 1
            if on_progress is not None:
                on_progress(done, total, candidate.label)
        results.append(_summarise_candidate(candidate, folds, spec, labels))
    baseline = results[0]
    comparisons = [
        c
        for r in results[1:]
        if r["comparable"] and baseline["comparable"]
        for c in _compare(r, baseline, spec)
    ]
    sensitive_target = spec.target is not None and spec.target in candidates[0].validation.sensitive
    class_names = None
    if labels is not None:
        class_names = (
            [f"class {i + 1}" for i in range(len(labels.labels))]
            if sensitive_target
            else [str(v) for v in labels.labels]
        )
    return {
        "task": spec.task,
        "target": spec.target,
        "models": list(spec.models),
        "model_notes": {k: MODEL_NOTES[k] for k in spec.models},
        "metrics": m.TASK_METRICS["binary" if spec.task == "binary" else spec.task],
        "primary_metric": m.PRIMARY[spec.task],
        "metric_info": {
            k: {"label": v[0], "higher_is_better": v[1], "bounded": v[2]}
            for k, v in m.METRICS.items()
        },
        "classes": class_names,
        "positive_class": class_names[1] if class_names and spec.task == "binary" else None,
        "cohort": {"rows": len(data), "excluded_missing_target": excluded},
        "folds": [
            {"fold": f.index, "n_train": len(f.train_ids), "n_test": len(f.test_ids)}
            for f in splits.folds
        ],
        "candidates": results,
        "comparisons": comparisons,
        "warnings": _warnings(results, comparisons, spec, data, splits, labels, ignored),
        "limitations": LIMITATIONS + (CLUSTERING_LIMITATIONS if spec.task == "clustering" else []),
    }


def _run_fold(
    data: pd.DataFrame,
    splits: Splits,
    fold: Any,
    candidate: Candidate,
    spec: ExperimentSpec,
    labels: _Labels | None,
    ignored: set[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "fold": fold.index,
        "status": "ok",
        "error": None,
        "models": {},
        "selection": [],
        "warnings": [],
    }
    train, test = splits.frames(data, fold)
    executor = PreprocessExecutor(candidate.validation, seed=spec.seed)
    start = time.perf_counter()
    try:
        training = executor.fit_resample(train)
        held_out = executor.transform(test)
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001  (isolated per fold, reported without values)
        out.update(status="failed", error=_error(exc))
        return out
    out["preprocess_ms"] = round((time.perf_counter() - start) * 1000, 1)
    tested = held_out.frame[ROW_ID].to_numpy()
    if len(tested) != len(fold.test_ids) or not np.array_equal(np.sort(tested), fold.test_ids):
        out.update(
            status="failed",
            error={
                "code": "NOT_DIRECTLY_COMPARABLE",
                "message": "Not every test row of the fold was scored.",
            },
        )
        return out
    out["warnings"] = sorted({w["code"] for r in training.reports for w in r.warnings})
    out["selection"] = _selection(executor, train, spec.seed + fold.index)
    out["quality"] = quality.fold_quality(
        test,
        held_out.frame,
        _raw_features(test, candidate.validation),
        executor.feature_columns(held_out.frame),
        held_out.reports,
        (len(train), len(training.frame)),
    )
    features = executor.feature_columns(training.frame)
    kinds = feature_kinds(training.frame, features)
    ignored.update(kinds.ignored)
    x_train, x_test = model_matrix(training.frame, kinds), model_matrix(held_out.frame, kinds)
    weights = (
        training.frame[SAMPLE_WEIGHT].to_numpy(dtype=float)
        if SAMPLE_WEIGHT in training.frame.columns
        else None
    )
    for name in spec.models:
        out["models"][name] = _fit_score(
            name, spec, kinds, x_train, x_test, training.frame, held_out.frame, weights, labels
        )
    return out


def _selection(executor: PreprocessExecutor, train: pd.DataFrame, seed: int) -> list[dict]:  # type: ignore[type-arg]
    steps = executor.steps
    if not any(v.spec.op in SELECTION_OPS for v in steps):
        return []
    out = []
    current = train
    for v, t in zip(steps, executor.transformers(), strict=True):
        if v.spec.op in SELECTION_OPS:
            entry: dict[str, Any] = {
                "step": v.index,
                "op": v.spec.op,
                "action": v.step.params.get("action"),
                **stability.fold_selection(t.decisions()),
            }
            try:
                entry["probe"] = stability.probe_selection(t, current, seed)
            except MemoryError:
                raise
            except Exception:  # noqa: BLE001  (probes are diagnostic; never fail the fold)
                entry["probe"] = None
            out.append(entry)
        if v is not steps[-1]:
            current = t.transform(current)
    return out


def _fit_score(
    name: str,
    spec: ExperimentSpec,
    kinds: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    weights: np.ndarray | None,
    labels: _Labels | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ok", "metrics": None, "reasons": {}, "error": None}
    if name != "dummy" and not kinds.numeric and not kinds.categorical:
        result.update(
            status="failed",
            error={
                "code": "NO_FEATURES",
                "message": "No usable feature columns after preprocessing.",
            },
        )
        return result
    model = build_model(name, spec.task, kinds, spec.seed, spec.n_clusters)
    fit_kw = {"model__sample_weight": weights} if weights is not None else {}
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            start = time.perf_counter()
            if spec.task == "clustering":
                model.fit(x_train, **fit_kw)
            else:
                assert spec.target is not None
                y_train = (
                    labels.encode(train_frame[spec.target])
                    if labels
                    else train_frame[spec.target].to_numpy(dtype=float)
                )
                model.fit(x_train, y_train, **fit_kw)
            result["fit_ms"] = round((time.perf_counter() - start) * 1000, 1)
            start = time.perf_counter()
            if spec.task == "clustering":
                assigned = model.predict(x_test)
                space = model[:-1].transform(x_test)
                result["metrics"], result["reasons"] = m.clustering(
                    np.asarray(space, dtype=float), assigned, spec.seed
                )
                result["clusters_found"] = int(len(np.unique(assigned)))
            elif labels is not None:
                y_test = labels.encode(test_frame[spec.target])
                predicted = model.predict(x_test)
                proba = _aligned_proba(model, x_test, len(labels.labels))
                metrics_, reasons, details = m.classification(
                    y_test, predicted, proba, list(range(len(labels.labels)))
                )
                result.update(metrics=metrics_, reasons=reasons, **details)
            else:
                y_test = test_frame[spec.target].to_numpy(dtype=float)
                result["metrics"], result["reasons"] = m.regression(y_test, model.predict(x_test))
            result["predict_ms"] = round((time.perf_counter() - start) * 1000, 1)
        result["library_warnings"] = _library_warnings(caught)
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001  (isolated per model and fold)
        result.update(status="failed", metrics=None, error=_error(exc))
    return result


def _aligned_proba(model: Any, x: pd.DataFrame, n_classes: int) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    raw = model.predict_proba(x)
    out = np.zeros((len(x), n_classes))
    for j, c in enumerate(model.classes_):
        out[:, int(c)] = raw[:, j]
    return out


def _stats(values: list[float | None]) -> dict[str, Any]:
    present = [v for v in values if v is not None]
    return {
        "mean": round(float(np.mean(present)), 6) if present else None,
        "sd": round(float(np.std(present, ddof=1)), 6) if len(present) > 1 else None,
        "min": round(float(np.min(present)), 6) if present else None,
        "max": round(float(np.max(present)), 6) if present else None,
        "n_folds": len(present),
        "values": [None if v is None else round(float(v), 6) for v in values],
    }


def _summarise_candidate(
    candidate: Candidate,
    folds: list[dict[str, Any]],
    spec: ExperimentSpec,
    labels: _Labels | None,
) -> dict[str, Any]:
    ok = [f for f in folds if f["status"] == "ok"]
    metric_names = m.TASK_METRICS["binary" if spec.task == "binary" else spec.task]
    scores: dict[str, Any] = {}
    confusion: dict[str, Any] = {}
    per_class: dict[str, Any] = {}
    model_failures = 0
    for name in spec.models:
        runs = [f["models"].get(name) if f["status"] == "ok" else None for f in folds]
        model_failures += sum(1 for r in runs if r is None or r["status"] != "ok")
        scores[name] = {
            metric: _stats(
                [r["metrics"][metric] if r and r["status"] == "ok" else None for r in runs]
            )
            for metric in metric_names
        }
        scores[name]["fit_ms"] = _stats(
            [r.get("fit_ms") if r and r["status"] == "ok" else None for r in runs]
        )
        if labels is not None:
            matrices = [np.array(r["confusion_matrix"]) for r in runs if r and r["status"] == "ok"]
            if matrices:
                total = np.sum(matrices, axis=0)
                confusion[name] = total.tolist()
                per_class[name] = _per_class(total)
    steps = candidate.validation.steps
    selection = []
    for v in steps:
        if v.spec.op in SELECTION_OPS:
            fold_entries = [e for f in ok for e in f["selection"] if e["step"] == v.index]
            if fold_entries:
                selection.append(
                    stability.summarise(
                        {"index": v.index, "op": v.spec.op, "action": v.step.params.get("action")},
                        fold_entries,
                    )
                )
    failed = [f for f in folds if f["status"] != "ok"]
    status = (
        "completed"
        if not failed and model_failures == 0
        else "failed"
        if len(failed) == len(folds)
        else "partial"
    )
    comparable = not failed
    max_risk = max((v.spec.risk for v in steps), key=("low", "medium", "high").index, default="low")
    training_changes = [
        f["quality"]["training_rows_after"] - f["quality"]["training_rows_before"] for f in ok
    ]
    return {
        "key": candidate.key,
        "label": candidate.label,
        "status": status,
        "comparable": comparable,
        "comparability_reason": None
        if comparable
        else f"{len(failed)} of {len(folds)} folds failed; this candidate is excluded from "
        "comparisons.",
        "max_risk": max_risk,
        "has_destructive": any(v.spec.destructive for v in steps),
        "stages": sorted({v.spec.stage for v in steps}),
        "metrics": scores,
        "confusion_matrix": confusion or None,
        "per_class": per_class or None,
        "data_quality": quality.aggregate([f["quality"] for f in ok]),
        "selection_stability": selection,
        "training_rows_change_per_fold": training_changes,
        "folds": [
            {
                "fold": f["fold"],
                "status": f["status"],
                "error": f["error"],
                "preprocess_ms": f.get("preprocess_ms"),
                "step_warnings": f["warnings"],
                "models": {
                    name: {
                        k: v for k, v in run.items() if k not in ("confusion_matrix", "per_class")
                    }
                    for name, run in f["models"].items()
                },
                "quality": f.get("quality"),
            }
            for f in folds
        ],
        "failures": [{"fold": f["fold"], **f["error"]} for f in failed]
        + [
            {"fold": f["fold"], "model": name, **run["error"]}
            for f in ok
            for name, run in f["models"].items()
            if run["status"] != "ok"
        ],
    }


def _per_class(matrix: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for i in range(matrix.shape[0]):
        tp = float(matrix[i, i])
        predicted = float(matrix[:, i].sum())
        support = float(matrix[i, :].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        out.append(
            {
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "support": int(support),
            }
        )
    return out


def _compare(
    candidate: dict[str, Any], baseline: dict[str, Any], spec: ExperimentSpec
) -> list[dict[str, Any]]:
    out = []
    for model in spec.models:
        for metric, stats in candidate["metrics"][model].items():
            if metric == "fit_ms":
                continue
            base = baseline["metrics"][model][metric]
            pairs = [
                (a, b)
                for a, b in zip(stats["values"], base["values"], strict=True)
                if a is not None and b is not None
            ]
            label, higher, bounded = m.METRICS[metric]
            diffs = [a - b for a, b in pairs]
            entry: dict[str, Any] = {
                "candidate": candidate["key"],
                "baseline": baseline["key"],
                "model": model,
                "metric": metric,
                "per_fold_difference": [round(d, 6) for d in diffs],
                "n_folds": len(diffs),
            }
            if len(diffs) < 2:
                entry.update(
                    mean_difference=None,
                    sd_difference=None,
                    folds_better=None,
                    folds_worse=None,
                    verdict="insufficient_folds",
                    interpretation="Fewer than two folds with both scores.",
                )
                out.append(entry)
                continue
            mean = float(np.mean(diffs))
            sd = float(np.std(diffs, ddof=1))
            sign = 1 if higher else -1
            better = sum(1 for d in diffs if sign * d > 0)
            worse = sum(1 for d in diffs if sign * d < 0)
            base_mean = abs(float(np.mean([b for _, b in pairs])))
            threshold = PRACTICAL_DIFFERENCE if bounded else PRACTICAL_DIFFERENCE * base_mean
            if abs(mean) < threshold:
                verdict = "negligible"
                text = f"The mean difference is below the practical threshold ({threshold:.3g})."
            elif better == len(diffs):
                verdict = "consistently_better"
                text = f"Better than the baseline in every fold ({better}/{len(diffs)})."
            elif worse == len(diffs):
                verdict = "consistently_worse"
                text = f"Worse than the baseline in every fold ({worse}/{len(diffs)})."
            elif abs(mean) < sd:
                verdict = "within_fold_variation"
                text = "The mean difference is smaller than its fold-to-fold variation."
            else:
                verdict = "mixed"
                text = f"Better in {better} and worse in {worse} of {len(diffs)} folds."
            entry.update(
                mean_difference=round(mean, 6),
                sd_difference=round(sd, 6),
                folds_better=better,
                folds_worse=worse,
                practical_threshold=round(threshold, 6),
                verdict=verdict,
                interpretation=text + " Describes these folds only; not a significance test.",
            )
            out.append(entry)
    return out


def _warnings(
    results: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    spec: ExperimentSpec,
    data: pd.DataFrame,
    splits: Splits,
    labels: _Labels | None,
    ignored: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = list(splits.warnings)
    non_baseline = len(results) - 1
    if non_baseline > 2:
        out.append(
            {
                "code": "MULTIPLE_COMPARISONS",
                "message": f"{non_baseline} candidates are "
                "compared with the baseline: with several comparisons, some differences "
                "appear by chance.",
            }
        )
    for r in results:
        if not r["comparable"]:
            out.append(
                {
                    "code": "NOT_DIRECTLY_COMPARABLE",
                    "candidate": r["key"],
                    "message": r["comparability_reason"],
                }
            )
        if any(c != 0 for c in r["training_rows_change_per_fold"]):
            out.append(
                {
                    "code": "ROWSET_DIFFERS",
                    "candidate": r["key"],
                    "message": "This "
                    "candidate trained on a different set of training rows (resampled); "
                    "test rows are identical for every candidate.",
                }
            )
        for model, scores in r["metrics"].items():
            for metric in ("roc_auc", "accuracy", "balanced_accuracy", "r2"):
                mean = scores.get(metric, {}).get("mean")
                if mean is not None and mean > SUSPICIOUS and model != "dummy":
                    out.append(
                        {
                            "code": "SUSPICIOUSLY_HIGH_SCORE",
                            "candidate": r["key"],
                            "model": model,
                            "message": f"Mean {metric} above {SUSPICIOUS}: "
                            "check the leakage findings (a feature may encode the target).",
                        }
                    )
                    break
    if spec.task != "clustering" and "dummy" in spec.models:
        primary = m.PRIMARY[spec.task]
        for r in results:
            dummy = r["metrics"]["dummy"][primary]["mean"]
            others = [r["metrics"][k][primary]["mean"] for k in spec.models if k != "dummy"]
            others = [o for o in others if o is not None]
            if dummy is not None and others and max(others) - dummy < PRACTICAL_DIFFERENCE:
                out.append(
                    {
                        "code": "DUMMY_COMPETITIVE",
                        "candidate": r["key"],
                        "message": f"No model beats the dummy baseline on {primary} by "
                        f"{PRACTICAL_DIFFERENCE}: the features may carry little signal.",
                    }
                )
    if labels is not None and spec.target is not None:
        codes = labels.encode(data[spec.target])
        ids = data[ROW_ID].to_numpy()
        smallest = min(
            int(np.bincount(codes[np.isin(ids, f.test_ids)], minlength=len(labels.labels)).min())
            for f in splits.folds
        )
        if smallest < FEW_PER_CLASS:
            out.append(
                {
                    "code": "FEW_SAMPLES_PER_CLASS_PER_FOLD",
                    "message": f"A test fold has "
                    f"only {smallest} rows of some class: per-fold metrics for it are noisy.",
                }
            )
    if ignored:
        out.append(
            {
                "code": "DATETIME_FEATURES_IGNORED",
                "columns": sorted(ignored),
                "message": "Date and time columns are not used as model inputs.",
            }
        )
    return out

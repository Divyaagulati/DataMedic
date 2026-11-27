"""Supervised training with nested model selection on frozen folds (worker only).

For every outer fold *k* of the frozen split:

1. Outer training rows = cohort minus fold *k*; outer test rows are not looked at until the
   selected settings are refitted.
2. Model selection (``nested`` mode, models with more than one setting): inner folds are built
   by the same splitter on the outer training rows only (groups, duplicate clusters and time
   order respected; seed ``seed + 1000 + k``). The inner assignment is shared by every model and
   setting. For each inner fold the preprocessing is fitted on the inner training rows and
   applied to the inner test rows; each setting is scored on the selection metric and the best
   inner mean wins (ties: the earlier, more regularised setting). If inner folds cannot be
   built (too few groups, a class too rare) the fold falls back to the default settings and the
   reason is recorded; group or time constraints are never weakened.
3. The preprocessing (replayed commits and model-stage steps) is refitted on all outer training
   rows, the selected setting is fitted, and the outer test rows are predicted once.

Scores therefore estimate the *selection procedure*, not one fixed configuration. The runner
checks that every outer test row was scored, records boundary evidence per fold for the
reliability audit, and returns out-of-fold predictions keyed by ``_dm_row_id``.
"""

import time
import warnings
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.ml import explain
from datamedic.engine.ml import metrics as mm
from datamedic.engine.ml.models import build_estimator, output_groups
from datamedic.engine.ml.replay import FoldPreprocessor, Prepared
from datamedic.engine.ml.spec import (
    CLASSIFICATION,
    INNER_FOLDS,
    INNER_SEED_OFFSET,
    METRIC_CONVENTIONS,
    METRICS,
    MIN_CLASS_ROWS_PER_INNER_FOLD,
    MIN_ROWS_FOR_NESTED,
    MODELS,
    PRIMARY,
    SELECTION,
    SELECTION_FALLBACK,
    TASK_METRICS,
    configs,
    metric_info,
)
from datamedic.engine.preprocess.catalog import SAMPLE_WEIGHT
from datamedic.engine.preprocess.experiment.models import feature_kinds, model_matrix
from datamedic.engine.preprocess.experiment.runner import (
    _aligned_proba,
    _error,
    _Labels,
    _library_warnings,
    _stats,
)
from datamedic.engine.preprocess.split import Splits, _blocks, make_splits

PRACTICAL_DIFFERENCE = 0.01
SUSPICIOUS = 0.99
FEW_PER_CLASS = 5

LIMITATIONS = [
    "Scores estimate the whole training procedure (preprocessing, model selection on inner "
    "folds and fitting), not one fixed model: different folds may select different settings.",
    "No independent final holdout exists. Choosing the model with the best score on these "
    "same folds is optimistic; the chosen model's score overstates how it would perform on "
    "new data.",
    "Fold scores are not independent (training rows overlap), so no p-values are given; "
    "differences between models describe these folds only.",
    "Bootstrap intervals resample the evaluated predictions; they ignore variation from "
    "training, tuning and model selection and are not a guarantee.",
    "Every learned preprocessing parameter, every model setting and every model is fitted on "
    "training rows only; test rows are never resampled, removed or used for fitting.",
    "No trained model is stored; results are reproduced by re-running the stored configuration.",
    "The 'weight' role is not used for training or metrics.",
    "Training never changes the dataset: no version is written.",
]


class Cancelled(Exception):
    """The job was cancelled (checked before every outer fold and every model)."""


@dataclass(frozen=True)
class SplitSpec:
    method: str
    target: str | None  # stratification label (classification)
    group: str | None
    time: str | None
    duplicate_columns: tuple[str, ...]


@dataclass(frozen=True)
class TrainingSpec:
    task: str
    target: str
    models: tuple[str, ...]
    mode: str  # nested | defaults (requested)
    seed: int
    split: SplitSpec
    sensitive: frozenset[str] = frozenset()
    sensitive_target: bool = False
    explain: bool = True
    permutation_rows: int = 5_000
    permutation_columns: int = 100
    permutation_repeats: int = 5
    bootstrap_resamples: int = 1_000


@dataclass
class TrainingOutput:
    summary: dict[str, Any]
    models: dict[str, dict[str, Any]]
    oof: pd.DataFrame  # _dm_row_id, fold, model, prediction, p_0..p_{k-1}
    evidence: dict[str, Any] = field(default_factory=dict)


def run_training(
    data: pd.DataFrame,
    splits: Splits,
    pre: FoldPreprocessor,
    spec: TrainingSpec,
    *,
    excluded_missing_target: int = 0,
    on_progress: Callable[[int, int, str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> TrainingOutput:
    with threadpool_limits(limits=1):
        return _Run(data, splits, pre, spec, on_progress, should_cancel).run(
            excluded_missing_target
        )


class _Run:
    def __init__(
        self,
        data: pd.DataFrame,
        splits: Splits,
        pre: FoldPreprocessor,
        spec: TrainingSpec,
        on_progress: Callable[[int, int, str], None] | None,
        should_cancel: Callable[[], bool] | None,
    ) -> None:
        self.data, self.splits, self.pre, self.spec = data, splits, pre, spec
        self.on_progress, self.should_cancel = on_progress, should_cancel
        self.classification = spec.task in CLASSIFICATION
        self.labels = _Labels(data[spec.target]) if self.classification else None
        self.n_classes = len(self.labels.labels) if self.labels else 0
        if self.classification and self.n_classes < 2:
            raise EngineError("TASK_TARGET_MISMATCH", "The target has fewer than two classes.")
        self.y_all = (
            self.labels.encode(data[spec.target])
            if self.labels
            else data[spec.target].to_numpy(dtype=float)
        )
        self.ids = data[ROW_ID].to_numpy()
        self.ignored: set[str] = set()

    # -- helpers ------------------------------------------------------------------------------

    def _cancel(self) -> None:
        if self.should_cancel is not None and self.should_cancel():
            raise Cancelled()

    def _y(self, frame: pd.DataFrame) -> np.ndarray:
        if self.labels is not None:
            return self.labels.encode(frame[self.spec.target])
        return np.asarray(frame[self.spec.target].to_numpy(dtype=float))

    def _matrices(self, prepared: Prepared) -> tuple[Any, pd.DataFrame, pd.DataFrame, Any]:
        executor = prepared.executor
        features = executor.feature_columns(prepared.training.frame)
        kinds = feature_kinds(prepared.training.frame, features)
        self.ignored.update(kinds.ignored)
        x_train = model_matrix(prepared.training.frame, kinds)
        x_test = model_matrix(prepared.held_out.frame, kinds)
        weights = (
            prepared.training.frame[SAMPLE_WEIGHT].to_numpy(dtype=float)
            if SAMPLE_WEIGHT in prepared.training.frame.columns
            else None
        )
        return kinds, x_train, x_test, weights

    def _fit(
        self,
        model: str,
        params: dict[str, Any],
        kinds: Any,
        x: pd.DataFrame,
        y: np.ndarray,
        weights: np.ndarray | None,
    ) -> Any:
        est = build_estimator(model, self.spec.task, kinds, params, self.spec.seed)
        kw = {"model__sample_weight": weights} if weights is not None else {}
        est.fit(x, y, **kw)
        return est

    def _predict(self, est: Any, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        pred = est.predict(x)
        proba = _aligned_proba(est, x, self.n_classes) if self.classification else None
        if self.classification:
            pred = np.asarray(pred, dtype=np.int64)
        else:
            pred = np.asarray(pred, dtype=float)
        return pred, proba

    def _metric(
        self, name: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None
    ) -> float | None:
        m, _, _ = mm.score(self.spec.task, y, pred, proba, self.n_classes)
        return m.get(name)

    # -- selection ----------------------------------------------------------------------------

    def _selection_metric(self, test_codes: list[np.ndarray]) -> str:
        metric = SELECTION[self.spec.task]
        if self.spec.task == "binary" and any(len(np.unique(c)) < 2 for c in test_codes):
            return SELECTION_FALLBACK["binary"]
        return metric

    def _inner(
        self, k: int, train: pd.DataFrame, models: list[str]
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Selected setting per model for outer fold ``k`` and the inner-fold record."""
        tuned = [m for m in models if len(configs(m, self.spec.task, "nested")) > 1]
        chosen = {m: dict(MODELS[m].default(self.spec.task)) for m in models}
        record: dict[str, Any] = {"fold": k, "mode": "nested", "scores": {}}
        if not tuned:
            record.update(mode="defaults", reason="No model with more than one setting.")
            return chosen, record
        s = self.spec.split
        try:
            inner = make_splits(
                train,
                method=s.method,
                n_splits=INNER_FOLDS,
                seed=self.spec.seed + INNER_SEED_OFFSET + k,
                target=s.target,
                group=s.group,
                time=s.time,
                duplicate_columns=s.duplicate_columns,
            )
        except EngineError as exc:
            record.update(
                mode="defaults",
                reason=f"Inner folds could not be built ({exc.code}); default settings used "
                "for this fold. Group and time constraints were kept.",
            )
            return chosen, record
        outer_train_ids = set(train[ROW_ID].tolist())
        test_codes = []
        for f in inner.folds:
            if (
                not set(f.train_ids.tolist()) <= outer_train_ids
                or not set(f.test_ids.tolist()) <= outer_train_ids
            ):
                raise EngineError(
                    "POSTCONDITION_FAILED", "Inner folds left the outer training rows."
                )
            sub = train[train[ROW_ID].isin(f.test_ids)]
            test_codes.append(self._y(sub))
        if self.classification:
            smallest = min(int(np.bincount(c, minlength=self.n_classes).min()) for c in test_codes)
            if smallest < MIN_CLASS_ROWS_PER_INNER_FOLD:
                record.update(
                    mode="defaults",
                    reason=f"An inner fold has fewer than {MIN_CLASS_ROWS_PER_INNER_FOLD} rows "
                    "of some class; default settings used for this fold.",
                )
                return chosen, record
        metric = self._selection_metric(test_codes)
        higher = METRICS[metric][1]
        record["selection_metric"] = metric
        record["inner_test_ids_within_outer_training"] = True
        scores: dict[str, list[list[float | None]]] = {
            m: [[] for _ in configs(m, self.spec.task, "nested")] for m in tuned
        }
        for f in inner.folds:
            itrain, itest = inner.frames(train, f)
            prepared = self.pre.prepare(itrain, itest)
            kinds, x_tr, x_te, w = self._matrices(prepared)
            y_tr, y_te = self._y(prepared.training.frame), self._y(prepared.held_out.frame)
            for m in tuned:
                for i, params in enumerate(configs(m, self.spec.task, "nested")):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            est = self._fit(m, params, kinds, x_tr, y_tr, w)
                            pred, proba = self._predict(est, x_te)
                        scores[m][i].append(self._metric(metric, y_te, pred, proba))
                    except MemoryError:
                        raise
                    except Exception:  # noqa: BLE001  (a failing setting cannot be selected)
                        scores[m][i].append(None)
        for m in tuned:
            means: list[float | None] = []
            for values in scores[m]:
                present = [v for v in values if v is not None]
                means.append(float(np.mean(present)) if len(present) == len(values) else None)
            best = None
            for i, v in enumerate(means):
                if v is None:
                    continue
                if best is None or (v > means[best] if higher else v < means[best]):  # type: ignore[operator]
                    best = i
            grid = configs(m, self.spec.task, "nested")
            if best is not None:
                chosen[m] = dict(grid[best])
            record["scores"][m] = [
                {"params": dict(g), "inner_mean": None if v is None else round(v, 6)}
                for g, v in zip(grid, means, strict=True)
            ]
            if best is None:
                record["scores"][m].append(
                    {"fallback": "No setting could be scored; default used."}
                )
        return chosen, record

    # -- main loop ----------------------------------------------------------------------------

    def run(self, excluded: int) -> TrainingOutput:
        spec = self.spec
        models = list(spec.models)
        mode = spec.mode
        fallback_reason = None
        if mode == "nested" and len(self.data) < MIN_ROWS_FOR_NESTED:
            mode, fallback_reason = (
                "defaults",
                f"The cohort has fewer than {MIN_ROWS_FOR_NESTED} rows; nested selection "
                "would be unreliable, so default settings were used.",
            )
        folds = self.splits.folds
        total = len(folds) * len(models)
        done = 0
        importance_metric = self._importance_metric()
        per_model: dict[str, dict[str, Any]] = {
            m: {"folds": [], "perm": [], "coef": [], "impurity": [], "confusion": None}
            for m in models
        }
        oof_parts: list[pd.DataFrame] = []
        boundaries: list[dict[str, Any]] = []
        inner_records: list[dict[str, Any]] = []
        fold_info: list[dict[str, Any]] = []
        input_columns = [c for c in self.data.columns if c != ROW_ID]
        for fold in folds:
            self._cancel()
            train, test = self.splits.frames(self.data, fold)
            chosen = {m: dict(MODELS[m].default(spec.task)) for m in models}
            if mode == "nested":
                chosen, record = self._inner(fold.index, train, models)
                inner_records.append(record)
                self._cancel()
            try:
                prepared = self.pre.prepare(train, test)
            except MemoryError:
                raise
            except Exception as exc:  # noqa: BLE001  (fold failure, reported without values)
                for m in models:
                    per_model[m]["folds"].append(
                        {"fold": fold.index, "status": "failed", "error": _error(exc)}
                    )
                    done += 1
                fold_info.append({"fold": fold.index, "status": "failed"})
                continue
            tested = prepared.held_out.frame[ROW_ID].to_numpy()
            complete = len(tested) == len(fold.test_ids) and np.array_equal(
                np.sort(tested), fold.test_ids
            )
            synthetic_in_test = bool((tested < 0).any())
            boundaries.append(
                {
                    "fold": fold.index,
                    "n_train": len(train),
                    "fit_rows": sorted(set(prepared.fit_rows)),
                    "fit_rows_equal_training_rows": all(n == len(train) for n in prepared.fit_rows),
                    "test_rows_complete": bool(complete),
                    "synthetic_rows_in_test": synthetic_in_test,
                    "training_rows_after_preprocessing": len(prepared.training.frame),
                    "weights_used_in_scoring": False,
                }
            )
            if not complete:
                for m in models:
                    per_model[m]["folds"].append(
                        {
                            "fold": fold.index,
                            "status": "failed",
                            "error": {
                                "code": "NOT_DIRECTLY_COMPARABLE",
                                "message": "Not every test row of the fold was scored.",
                            },
                        }
                    )
                    done += 1
                continue
            kinds, x_train, x_test, weights = self._matrices(prepared)
            y_train = self._y(prepared.training.frame)
            y_test = self._y(prepared.held_out.frame)
            original_train = (prepared.training.frame[ROW_ID] >= 0).to_numpy()
            origins = explain.origin_map(prepared.lineage, input_columns)
            fold_info.append(
                {
                    "fold": fold.index,
                    "status": "ok",
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_train_model": len(prepared.training.frame),
                    "model_inputs": len(x_train.columns),
                }
            )
            for m in models:
                self._cancel()
                per_model[m]["folds"].append(
                    self._outer_model(
                        m,
                        chosen[m],
                        fold.index,
                        kinds,
                        x_train,
                        x_test,
                        y_train,
                        y_test,
                        weights,
                        original_train,
                        tested,
                        origins,
                        importance_metric,
                        per_model[m],
                        oof_parts,
                    )
                )
                done += 1
                if self.on_progress is not None:
                    self.on_progress(done, total, m)
        oof = (
            pd.concat(oof_parts, ignore_index=True)
            if oof_parts
            else pd.DataFrame({ROW_ID: [], "fold": [], "model": [], "prediction": []})
        )
        results = {m: self._summarise(m, per_model[m], oof, importance_metric) for m in models}
        comparisons = self._comparisons(results)
        coverage = self._coverage()
        summary = {
            "task": spec.task,
            "target": spec.target,
            "models": models,
            "model_labels": {m: MODELS[m].label for m in models},
            "model_notes": {m: MODELS[m].note for m in models},
            "metrics": list(TASK_METRICS[spec.task]),
            "primary_metric": PRIMARY[spec.task],
            "metric_info": metric_info(),
            "metric_conventions": METRIC_CONVENTIONS,
            "classes": self._class_names(),
            "positive_class": self._positive_class(),
            "cohort": {"rows": len(self.data), "excluded_missing_target": excluded},
            "coverage": coverage,
            "folds": fold_info,
            "split": {
                "method": self.splits.method,
                "n_splits": self.splits.n_splits,
                "seed": self.splits.seed,
                "group": spec.split.group,
                "time": spec.split.time,
            },
            "tuning": {
                "requested": spec.mode,
                "used": mode,
                "fallback_reason": fallback_reason,
                "inner_folds": INNER_FOLDS,
                "inner_seed_offset": INNER_SEED_OFFSET,
                "grids": {m: list(configs(m, spec.task, "nested")) for m in models},
                "defaults": {m: MODELS[m].default(spec.task) for m in models},
                "per_fold": inner_records,
                "tie_break": "earlier (more regularised) setting",
            },
            "importance_metric": importance_metric,
            "comparisons": comparisons,
            "warnings": self._warnings(results, mode, fallback_reason, coverage),
            "limitations": LIMITATIONS,
            "explain_limitations": explain.LIMITATIONS,
        }
        evidence = {"boundaries": boundaries, "inner": inner_records}
        return TrainingOutput(summary, results, oof, evidence)

    def _importance_metric(self) -> str:
        if self.spec.task == "regression":
            return "r2"
        if self.spec.task == "binary":
            for f in self.splits.folds:
                codes = self.y_all[np.isin(self.ids, f.test_ids)]
                if len(np.unique(codes)) < 2:
                    return "balanced_accuracy"
            return "roc_auc"
        return "balanced_accuracy"

    def _outer_model(
        self,
        m: str,
        params: dict[str, Any],
        k: int,
        kinds: Any,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
        weights: np.ndarray | None,
        original_train: np.ndarray,
        tested: np.ndarray,
        origins: dict[str, str | None],
        importance_metric: str,
        acc: dict[str, Any],
        oof_parts: list[pd.DataFrame],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {"fold": k, "status": "ok", "error": None, "params": params}
        if m != "dummy" and not kinds.numeric and not kinds.categorical:
            out.update(
                status="failed",
                error={"code": "NO_FEATURES", "message": "No usable feature columns."},
            )
            return out
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = time.perf_counter()
                est = self._fit(m, params, kinds, x_train, y_train, weights)
                out["fit_ms"] = round((time.perf_counter() - start) * 1000, 1)
                start = time.perf_counter()
                pred, proba = self._predict(est, x_test)
                out["predict_ms"] = round((time.perf_counter() - start) * 1000, 1)
                metrics, reasons, details = mm.score(
                    self.spec.task, y_test, pred, proba, self.n_classes
                )
                out.update(metrics=metrics, reasons=reasons)
                if details:
                    cm = np.asarray(details["confusion_matrix"])
                    acc["confusion"] = cm if acc["confusion"] is None else acc["confusion"] + cm
                tr_x = x_train[original_train]
                tr_pred, tr_proba = self._predict(est, tr_x)
                primary = PRIMARY[self.spec.task]
                out["train_score"] = {
                    primary: self._metric(primary, y_train[original_train], tr_pred, tr_proba),
                    importance_metric: self._metric(
                        importance_metric, y_train[original_train], tr_pred, tr_proba
                    ),
                }
                part = pd.DataFrame({ROW_ID: tested, "fold": k, "model": m, "prediction": pred})
                if proba is not None:
                    for c in range(self.n_classes):
                        part[f"p_{c}"] = proba[:, c]
                oof_parts.append(part)
                if self.spec.explain and m != "dummy":
                    self._explain(m, est, kinds, x_test, y_test, origins, importance_metric, k, acc)
            out["library_warnings"] = _library_warnings(caught)
        except MemoryError:
            raise
        except Exception as exc:  # noqa: BLE001  (isolated per model and fold)
            out.update(status="failed", metrics=None, error=_error(exc))
        return out

    def _explain(
        self,
        m: str,
        est: Any,
        kinds: Any,
        x_test: pd.DataFrame,
        y_test: np.ndarray,
        origins: dict[str, str | None],
        metric: str,
        k: int,
        acc: dict[str, Any],
    ) -> None:
        groups = explain.group_inputs(list(x_test.columns), origins)

        def scorer(x: pd.DataFrame) -> float | None:
            index = x_test.index.get_indexer(x.index)
            pred, proba = self._predict(est, x)
            return self._metric(metric, y_test[index], pred, proba)

        seed = self.spec.seed + 7919 * (k + 1) + sorted(MODELS).index(m)
        perm = explain.permutation(
            x_test,
            groups,
            scorer,
            repeats=self.spec.permutation_repeats,
            max_rows=self.spec.permutation_rows,
            max_groups=self.spec.permutation_columns,
            seed=seed,
        )
        acc["perm"].append(perm)
        sources = output_groups(est, kinds)
        coef = explain.linear_importance(est, sources)
        if coef is not None:
            acc["coef"].append(explain.to_original(coef, origins))
        imp = explain.impurity_importance(est, sources)
        if imp is not None:
            acc["impurity"].append(explain.to_original(imp, origins))

    # -- summaries ----------------------------------------------------------------------------

    def _positive_class(self) -> str | None:
        names = self._class_names()
        return names[1] if self.spec.task == "binary" and names else None

    def _class_names(self) -> list[str] | None:
        if self.labels is None:
            return None
        if self.spec.sensitive_target:
            return [f"class {i + 1}" for i in range(self.n_classes)]
        return [str(v) for v in self.labels.labels]

    def _coverage(self) -> dict[str, Any]:
        tested = np.unique(np.concatenate([f.test_ids for f in self.splits.folds]))
        n = len(self.data)
        return {
            "cohort_rows": n,
            "tested_rows": int(len(tested)),
            "untested_rows": int(n - len(tested)),
            "complete": bool(len(tested) == n),
            "note": None
            if len(tested) == n
            else "Time-ordered folds never test the earliest rows; out-of-fold results cover "
            "only the tested rows.",
        }

    def _summarise(
        self, m: str, acc: dict[str, Any], oof: pd.DataFrame, importance_metric: str
    ) -> dict[str, Any]:
        folds = acc["folds"]
        failures = [
            {"fold": f["fold"], **(f.get("error") or {})} for f in folds if f["status"] != "ok"
        ]
        ok = [f for f in folds if f["status"] == "ok"]
        n_folds = len(self.splits.folds)
        status = "completed" if len(ok) == n_folds else ("partial" if ok else "failed")
        comparable = status == "completed"
        metrics: dict[str, Any] = {}
        reasons: dict[str, list[dict[str, Any]]] = {}
        for name in TASK_METRICS[self.spec.task]:
            values = [f["metrics"].get(name) if f["status"] == "ok" else None for f in folds]
            metrics[name] = _stats(values)
            for f in ok:
                if f["metrics"].get(name) is None and f["reasons"].get(name):
                    reasons.setdefault(name, []).append(
                        {"fold": f["fold"], "reason": f["reasons"][name]}
                    )
        rows = oof[oof["model"] == m] if len(oof) else oof
        pooled: dict[str, Any] = {"available": False, "reason": "No predictions."}
        curves: dict[str, Any] = {"available": False, "reason": "Not applicable."}
        calibration: dict[str, Any] = {"available": False, "reason": "Not applicable."}
        residuals: dict[str, Any] = {"available": False, "reason": "Not applicable."}
        bootstrap: dict[str, Any] = {"available": False, "reason": "No predictions."}
        slices: list[dict[str, Any]] = []
        if len(rows):
            order = pd.Index(self.ids).get_indexer(rows[ROW_ID].to_numpy())
            y = self.y_all[order]
            pred = rows["prediction"].to_numpy()
            pred = pred.astype(np.int64) if self.classification else pred.astype(float)
            cols = [f"p_{c}" for c in range(self.n_classes)]
            proba = (
                rows[cols].to_numpy(dtype=float)
                if self.classification and all(c in rows for c in cols)
                else None
            )
            pm, pr = mm.pooled(self.spec.task, y, pred, proba, self.n_classes)
            pooled = {"available": True, "metrics": pm, "reasons": pr, "n": int(len(rows))}
            if self.classification:
                curves = mm.curves(y, proba, self.n_classes)
                calibration = mm.calibration(y, proba, self.n_classes)
                errors = (pred != y).astype(float)
                slice_metric = "error_rate"
            else:
                residuals = mm.residuals(y, pred, self.spec.sensitive_target)
                errors = np.abs(y - pred)
                slice_metric = "mean_absolute_error"
            bootstrap = self._bootstrap(y, pred, proba, order)
            if comparable or ok:
                ranking = explain.aggregate(
                    [p.get("drops", {}) for p in acc["perm"] if p.get("available")], n_folds
                )
                top = [r["column"] for r in ranking if not r["column"].startswith("unresolved:")]
                slices = explain.error_slices(
                    self.data.iloc[order].reset_index(drop=True),
                    errors,
                    top,
                    self.spec.sensitive,
                    slice_metric,
                )
        confusion = acc["confusion"].tolist() if acc["confusion"] is not None else None
        per_class = None
        if confusion is not None:
            cm = np.asarray(confusion)
            per_class = []
            for c in range(cm.shape[0]):
                tp = cm[c, c]
                support = int(cm[c].sum())
                predicted = int(cm[:, c].sum())
                per_class.append(
                    {
                        "support": support,
                        "precision": None if predicted == 0 else round(float(tp / predicted), 6),
                        "recall": None if support == 0 else round(float(tp / support), 6),
                    }
                )
        selected = [f.get("params") for f in folds if f["status"] == "ok"]
        frequency = Counter(tuple(sorted((p or {}).items())) for p in selected)
        perm_ok = [p for p in acc["perm"] if p.get("available")]
        importance = {
            "metric": importance_metric,
            "permutation": explain.aggregate([p["drops"] for p in perm_ok], n_folds),
            "permutation_rows_per_fold": [p["rows"] for p in perm_ok],
            "permutation_sampled": any(p["sampled"] for p in perm_ok),
            "permutation_not_computed_columns": max(
                (p["not_computed"] for p in perm_ok), default=0
            ),
            "repeats": self.spec.permutation_repeats,
            "coefficients": explain.aggregate_simple(acc["coef"], n_folds) if acc["coef"] else None,
            "impurity": explain.aggregate_simple(acc["impurity"], n_folds)
            if acc["impurity"]
            else None,
            "available": bool(perm_ok),
        }
        train_scores = [f.get("train_score") for f in ok]
        return {
            "model": m,
            "status": status,
            "comparable": comparable,
            "comparability_reason": None
            if comparable
            else f"{len(failures)} of {n_folds} folds failed; excluded from comparisons.",
            "metrics": metrics,
            "metric_reasons": reasons,
            "pooled": pooled,
            "bootstrap": bootstrap,
            "fold_results": [
                {k: v for k, v in f.items() if k not in ("metrics", "reasons")}
                | {"metrics": f.get("metrics")}
                for f in folds
            ],
            "train_scores": train_scores,
            "selection": {
                "per_fold": selected,
                "frequency": [{"params": dict(k), "folds": v} for k, v in frequency.most_common()],
            },
            "curves": curves,
            "calibration": calibration,
            "residuals": residuals,
            "confusion": confusion,
            "per_class": per_class,
            "importance": importance,
            "error_slices": slices,
            "failures": failures,
            "timings": {
                "fit_ms": _stats([f.get("fit_ms") for f in ok]),
                "predict_ms": _stats([f.get("predict_ms") for f in ok]),
            },
        }

    def _bootstrap(
        self, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None, order: np.ndarray
    ) -> dict[str, Any]:
        s = self.spec.split
        if s.method == "time":
            blocks = None
        else:
            all_blocks = _blocks(self.data, s.group, list(s.duplicate_columns))
            blocks = all_blocks[order]
        return mm.bootstrap(
            self.spec.task,
            y,
            pred,
            proba,
            self.n_classes,
            blocks,
            self.spec.bootstrap_resamples,
            self.spec.seed,
        )

    def _comparisons(self, results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        keys = [m for m in results if results[m]["comparable"]]
        pairs = [(m, "dummy") for m in keys if m != "dummy" and "dummy" in keys]
        others = [m for m in keys if m != "dummy"]
        pairs += [(a, b) for i, a in enumerate(others) for b in others[i + 1 :]]
        for a, b in pairs:
            metric_names = (
                list(TASK_METRICS[self.spec.task]) if b == "dummy" else [PRIMARY[self.spec.task]]
            )
            for metric in metric_names:
                out.append(
                    paired(
                        a,
                        b,
                        metric,
                        results[a]["metrics"][metric]["values"],
                        results[b]["metrics"][metric]["values"],
                    )
                )
        return out

    def _warnings(
        self,
        results: dict[str, dict[str, Any]],
        mode: str,
        fallback: str | None,
        coverage: dict[str, Any],
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = list(self.splits.warnings)
        primary = PRIMARY[self.spec.task]
        non_dummy = [m for m in results if m != "dummy"]
        if len(non_dummy) > 2:
            out.append(
                {
                    "code": "MULTIPLE_COMPARISONS",
                    "message": f"{len(non_dummy)} models are compared on the same folds: some "
                    "differences appear by chance, and the best of them is optimistic.",
                }
            )
        if fallback:
            out.append({"code": "TUNING_FALLBACK", "message": fallback})
        for m, r in results.items():
            if not r["comparable"]:
                out.append(
                    {
                        "code": "NOT_DIRECTLY_COMPARABLE",
                        "model": m,
                        "message": r["comparability_reason"],
                    }
                )
            if m == "dummy":
                continue
            for metric in ("roc_auc", "balanced_accuracy", "r2"):
                mean = r["metrics"].get(metric, {}).get("mean")
                if mean is not None and mean > SUSPICIOUS:
                    out.append(
                        {
                            "code": "SUSPICIOUSLY_HIGH_SCORE",
                            "model": m,
                            "message": f"Mean {metric} above {SUSPICIOUS}: see the reliability "
                            "audit (a feature may encode the target).",
                        }
                    )
                    break
        if not coverage["complete"]:
            out.append({"code": "PARTIAL_COVERAGE", "message": coverage["note"]})
        if self.labels is not None:
            smallest = min(
                int(
                    np.bincount(
                        self.y_all[np.isin(self.ids, f.test_ids)], minlength=self.n_classes
                    ).min()
                )
                for f in self.splits.folds
            )
            if smallest < FEW_PER_CLASS:
                out.append(
                    {
                        "code": "FEW_SAMPLES_PER_CLASS_PER_FOLD",
                        "message": f"A test fold has only {smallest} rows of some class: "
                        "per-fold metrics for it are noisy.",
                    }
                )
        if self.ignored:
            out.append(
                {
                    "code": "DATETIME_FEATURES_IGNORED",
                    "columns": sorted(self.ignored),
                    "message": "Date and time columns are not used as model inputs.",
                }
            )
        del primary, mode
        return out


def paired(
    a: str, b: str, metric: str, values_a: list[float | None], values_b: list[float | None]
) -> dict[str, Any]:
    """Per-fold differences ``a - b`` on identical test rows with a descriptive verdict (the
    Experiment Lab's rules): negligible, consistently better/worse, within fold variation,
    mixed or insufficient folds. Not a significance test."""
    pairs = [
        (x, y) for x, y in zip(values_a, values_b, strict=True) if x is not None and y is not None
    ]
    label, higher, bounded, _ = METRICS[metric]
    diffs = [x - y for x, y in pairs]
    entry: dict[str, Any] = {
        "model": a,
        "reference": b,
        "metric": metric,
        "per_fold_difference": [round(d, 6) for d in diffs],
        "n_folds": len(diffs),
    }
    if len(diffs) < 2:
        entry.update(
            verdict="insufficient_folds", interpretation="Fewer than two folds with both scores."
        )
        return entry
    mean, sd = float(np.mean(diffs)), float(np.std(diffs, ddof=1))
    sign = 1 if higher else -1
    better = sum(1 for d in diffs if sign * d > 0)
    worse = sum(1 for d in diffs if sign * d < 0)
    base = abs(float(np.mean([y for _, y in pairs])))
    threshold = PRACTICAL_DIFFERENCE if bounded else PRACTICAL_DIFFERENCE * base
    if abs(mean) < threshold:
        verdict, text = (
            "negligible",
            f"The mean difference is below the practical threshold ({threshold:.3g}).",
        )
    elif better == len(diffs):
        verdict, text = "consistently_better", f"Better in every fold ({better}/{len(diffs)})."
    elif worse == len(diffs):
        verdict, text = "consistently_worse", f"Worse in every fold ({worse}/{len(diffs)})."
    elif abs(mean) < sd:
        verdict, text = (
            "within_fold_variation",
            "The mean difference is smaller than its fold-to-fold variation.",
        )
    else:
        verdict, text = "mixed", f"Better in {better} and worse in {worse} of {len(diffs)} folds."
    entry.update(
        mean_difference=round(mean, 6),
        sd_difference=round(sd, 6),
        folds_better=better,
        folds_worse=worse,
        practical_threshold=round(threshold, 6),
        verdict=verdict,
        interpretation=text + " Describes these folds only; not a significance test.",
    )
    del label
    return entry

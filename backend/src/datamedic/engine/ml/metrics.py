"""Metrics, curves, calibration, residual summaries and bootstrap intervals (worker only).

Per-fold metrics reuse the Experiment Lab implementation (``experiment.metrics``) and add
the weighted multiclass averages, multiclass top-label ECE and MSE. Undefined values are
``None`` with a reason, never 0.

Curves, calibration and residual summaries are computed on the *pooled* out-of-fold
predictions (each row predicted once, by the model of the fold in which it was a test row)
and returned as bounded aggregates: no per-row value leaves the worker.
"""

from typing import Any

import numpy as np
from sklearn import metrics as skm

from datamedic.engine.ml.spec import TASK_METRICS
from datamedic.engine.preprocess.experiment import metrics as lab

CURVE_POINTS = 200
CALIBRATION_BINS = 10
GRID_BINS = 40
RESIDUAL_BINS = 30
MAX_CURVE_CLASSES = 10
BOOTSTRAP_METRICS = {
    "binary": ("roc_auc", "balanced_accuracy"),
    "multiclass": ("balanced_accuracy", "accuracy"),
    "regression": ("r2", "rmse"),
}


def _f(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def top_label_ece(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    confidence = proba.max(axis=1)
    correct = (proba.argmax(axis=1) == y_true).astype(float)
    return lab.expected_calibration_error(correct, confidence, bins)


def classification(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, n_classes: int
) -> tuple[dict[str, float | None], dict[str, str], dict[str, Any]]:
    classes = list(range(n_classes))
    out, reasons, details = lab.classification(y_true, y_pred, proba, classes)
    task = "binary" if n_classes == 2 else "multiclass"
    if task == "multiclass":
        kw: dict[str, Any] = {"labels": classes, "average": "weighted", "zero_division": 0}
        out["precision_weighted"] = _f(skm.precision_score(y_true, y_pred, **kw))
        out["recall_weighted"] = _f(skm.recall_score(y_true, y_pred, **kw))
        out["f1_weighted"] = _f(skm.f1_score(y_true, y_pred, **kw))
        if proba is None:
            out["ece"] = None
            reasons["ece"] = "The model gives no probabilities."
        else:
            out["ece"] = _f(top_label_ece(y_true, proba))
    return {k: out.get(k) for k in TASK_METRICS[task]}, reasons, details


def regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[dict[str, float | None], dict[str, str]]:
    out, reasons = lab.regression(y_true, y_pred)
    out["mse"] = _f(skm.mean_squared_error(y_true, y_pred))
    return {k: out.get(k) for k in TASK_METRICS["regression"]}, reasons


def score(
    task: str, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, n_classes: int
) -> tuple[dict[str, float | None], dict[str, str], dict[str, Any]]:
    if task == "regression":
        m, r = regression(y_true, y_pred)
        return m, r, {}
    return classification(y_true, y_pred, proba, n_classes)


# ---- Curves -----------------------------------------------------------------------------------


def _thin(points: list[tuple[float, float]], limit: int = CURVE_POINTS) -> list[list[float]]:
    if len(points) <= limit:
        return [[round(x, 6), round(y, 6)] for x, y in points]
    index = np.unique(np.linspace(0, len(points) - 1, limit).round().astype(int))
    return [[round(points[i][0], 6), round(points[i][1], 6)] for i in index]


def _binary_curves(truth: np.ndarray, score_: np.ndarray) -> dict[str, Any]:
    if len(np.unique(truth)) < 2:
        return {"available": False, "reason": "Both classes are needed among the rows."}
    fpr, tpr, _ = skm.roc_curve(truth, score_)
    precision, recall, _ = skm.precision_recall_curve(truth, score_)
    return {
        "available": True,
        "roc": _thin(list(zip(fpr.tolist(), tpr.tolist(), strict=True))),
        "pr": _thin(list(zip(recall.tolist()[::-1], precision.tolist()[::-1], strict=True))),
        "roc_auc": _f(skm.roc_auc_score(truth, score_)),
        "pr_auc": _f(skm.average_precision_score(truth, score_)),
        "prevalence": round(float(truth.mean()), 6),
        "n": int(len(truth)),
    }


def curves(y_true: np.ndarray, proba: np.ndarray | None, n_classes: int) -> dict[str, Any]:
    """ROC and precision-recall curves from pooled out-of-fold probabilities (binary: the
    positive class; multiclass: one-vs-rest per class for at most 10 classes)."""
    if proba is None:
        return {"available": False, "reason": "The model gives no probabilities."}
    if n_classes == 2:
        return {"per_class": None, **_binary_curves((y_true == 1).astype(int), proba[:, 1])}
    if n_classes > MAX_CURVE_CLASSES:
        return {
            "available": False,
            "reason": f"Per-class curves are shown for at most {MAX_CURVE_CLASSES} classes.",
        }
    return {
        "available": True,
        "per_class": [
            {"class_index": c, **_binary_curves((y_true == c).astype(int), proba[:, c])}
            for c in range(n_classes)
        ],
    }


def calibration(y_true: np.ndarray, proba: np.ndarray | None, n_classes: int) -> dict[str, Any]:
    """Reliability curve on quantile bins: mean predicted probability vs observed frequency
    per bin, with counts. Binary: the positive class; multiclass: top-label confidence vs
    accuracy."""
    if proba is None:
        return {"available": False, "reason": "The model gives no probabilities."}
    if n_classes == 2:
        p, observed = proba[:, 1], (y_true == 1).astype(float)
        kind = "positive_class"
    else:
        p = proba.max(axis=1)
        observed = (proba.argmax(axis=1) == y_true).astype(float)
        kind = "top_label"
    if len(p) < CALIBRATION_BINS * 10:
        return {
            "available": False,
            "reason": f"Fewer than {CALIBRATION_BINS * 10} rows with predictions.",
        }
    edges = np.unique(np.quantile(p, np.linspace(0, 1, CALIBRATION_BINS + 1)))
    if len(edges) < 3:
        return {"available": False, "reason": "Predicted probabilities barely vary."}
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    bins = []
    for b in range(len(edges) - 1):
        mask = index == b
        if mask.any():
            bins.append(
                {
                    "predicted": round(float(p[mask].mean()), 6),
                    "observed": round(float(observed[mask].mean()), 6),
                    "count": int(mask.sum()),
                }
            )
    ece = sum(abs(b["predicted"] - b["observed"]) * b["count"] for b in bins) / len(p)
    return {
        "available": True,
        "kind": kind,
        "bins": bins,
        "ece_quantile_bins": round(float(ece), 6),
        "n": int(len(p)),
    }


def residuals(y_true: np.ndarray, y_pred: np.ndarray, redact_axes: bool) -> dict[str, Any]:
    """Binned actual-vs-predicted and residual-vs-predicted count grids, a residual histogram
    and residual quantiles. With a sensitive target, axis values are withheld (counts only)."""
    if len(y_true) < 2:
        return {"available": False, "reason": "Too few rows with predictions."}
    resid = y_true - y_pred
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    if hi <= lo:
        hi = lo + 1.0
    grid, _, _ = np.histogram2d(y_pred, y_true, bins=GRID_BINS, range=[[lo, hi], [lo, hi]])
    r_lo, r_hi = float(resid.min()), float(resid.max())
    if r_hi <= r_lo:
        r_hi = r_lo + 1.0
    rgrid, _, _ = np.histogram2d(y_pred, resid, bins=GRID_BINS, range=[[lo, hi], [r_lo, r_hi]])
    hist, hist_edges = np.histogram(resid, bins=RESIDUAL_BINS, range=(r_lo, r_hi))
    quantiles = np.quantile(resid, [0.05, 0.25, 0.5, 0.75, 0.95])
    out: dict[str, Any] = {
        "available": True,
        "n": int(len(y_true)),
        "redacted_axes": redact_axes,
        "actual_vs_predicted": grid.astype(int).T.tolist(),  # rows: actual bins, cols: predicted
        "residual_vs_predicted": rgrid.astype(int).T.tolist(),  # rows: residual bins
        "residual_histogram": hist.astype(int).tolist(),
    }
    if not redact_axes:
        out.update(
            value_range=[round(lo, 6), round(hi, 6)],
            residual_range=[round(r_lo, 6), round(r_hi, 6)],
            residual_histogram_edges=[round(float(e), 6) for e in hist_edges],
            residual_quantiles={
                k: round(float(v), 6)
                for k, v in zip(("p05", "p25", "p50", "p75", "p95"), quantiles, strict=True)
            },
            mean_residual=round(float(resid.mean()), 6),
        )
    return out


# ---- Bootstrap --------------------------------------------------------------------------------


def bootstrap(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray | None,
    n_classes: int,
    blocks: np.ndarray | None,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    """Percentile intervals (2.5%, 97.5%) for the headline metrics of pooled out-of-fold
    predictions, resampling whole split groups (``blocks``: rows of one group or duplicate
    cluster share a label). ``blocks=None`` means the dependence structure does not allow a
    supported bootstrap (time-ordered rows), and the intervals are unavailable."""
    names = BOOTSTRAP_METRICS[task]
    if blocks is None:
        return {
            "available": False,
            "reason": "Rows are time-ordered; an independent or group bootstrap would ignore "
            "temporal dependence, and a block bootstrap is not supported.",
        }
    if len(y_true) < 30:
        return {"available": False, "reason": "Fewer than 30 rows with predictions."}
    labels, codes = np.unique(blocks, return_inverse=True)
    if len(labels) < 10:
        return {"available": False, "reason": "Fewer than 10 independent groups."}
    members = [np.flatnonzero(codes == g) for g in range(len(labels))]
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {n: [] for n in names}
    skipped = dict.fromkeys(names, 0)
    for _ in range(resamples):
        chosen = rng.integers(0, len(labels), len(labels))
        rows = np.concatenate([members[g] for g in chosen])
        m = _headline(task, y_true[rows], y_pred[rows], None if proba is None else proba[rows])
        for n in names:
            v = m.get(n)
            if v is None:
                skipped[n] += 1
            else:
                values[n].append(v)
    out: dict[str, Any] = {
        "available": True,
        "method": "percentile bootstrap over split groups (rows of one group or duplicate "
        "cluster are resampled together) of pooled out-of-fold predictions",
        "resamples": resamples,
        "groups": int(len(labels)),
        "metrics": {},
    }
    for n in names:
        vs = values[n]
        out["metrics"][n] = (
            {
                "low": round(float(np.quantile(vs, 0.025)), 6),
                "high": round(float(np.quantile(vs, 0.975)), 6),
                "undefined_resamples": skipped[n],
            }
            if len(vs) >= resamples // 2
            else None
        )
    return out


def _headline(
    task: str, y: np.ndarray, pred: np.ndarray, proba: np.ndarray | None
) -> dict[str, float | None]:
    """The bootstrap metrics only (same definitions as ``score``), computed cheaply."""
    if task == "regression":
        r2 = _f(skm.r2_score(y, pred)) if len(y) > 1 and float(np.var(y)) > 0 else None
        return {"r2": r2, "rmse": _f(np.sqrt(skm.mean_squared_error(y, pred)))}
    out: dict[str, float | None] = {"accuracy": _f(skm.accuracy_score(y, pred))}
    out["balanced_accuracy"] = (
        _f(skm.balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else None
    )
    if task == "binary":
        out["roc_auc"] = (
            _f(skm.roc_auc_score(y, proba[:, 1]))
            if proba is not None and len(np.unique(y)) == 2
            else None
        )
    return out


def pooled(
    task: str, y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, n_classes: int
) -> tuple[dict[str, float | None], dict[str, str]]:
    m, r, _ = score(task, y_true, y_pred, proba, n_classes)
    return m, r

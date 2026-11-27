"""Evaluation metrics on held-out rows (scikit-learn implementations).

Classification: accuracy, balanced accuracy, precision, recall, F1 (binary: for the positive
class, the second label in sorted order; multiclass: macro averages), ROC-AUC and PR-AUC
(average precision; one-vs-rest macro for multiclass), log loss, and for binary targets the
Brier score and expected calibration error (10 equal-width bins). ROC-AUC and PR-AUC need
every class among the test rows; otherwise they are ``None`` for that fold, with a reason.

Regression: MAE, RMSE, R², median absolute error.

Clustering (internal metrics, no ground truth): silhouette (seeded sample of at most 5,000
test rows), Davies-Bouldin, Calinski-Harabasz.
"""

from typing import Any

import numpy as np
from sklearn import metrics as skm

ECE_BINS = 10
SILHOUETTE_MAX_ROWS = 5_000

# name -> (label, higher is better, bounded in [0, 1] or [-1, 1])
METRICS: dict[str, tuple[str, bool, bool]] = {
    "accuracy": ("Accuracy", True, True),
    "balanced_accuracy": ("Balanced accuracy", True, True),
    "precision": ("Precision", True, True),
    "recall": ("Recall", True, True),
    "f1": ("F1", True, True),
    "roc_auc": ("ROC-AUC", True, True),
    "pr_auc": ("PR-AUC (average precision)", True, True),
    "log_loss": ("Log loss", False, False),
    "brier": ("Brier score", False, True),
    "ece": ("Expected calibration error", False, True),
    "mae": ("MAE", False, False),
    "rmse": ("RMSE", False, False),
    "r2": ("R²", True, False),
    "median_ae": ("Median absolute error", False, False),
    "silhouette": ("Silhouette", True, True),
    "davies_bouldin": ("Davies-Bouldin", False, False),
    "calinski_harabasz": ("Calinski-Harabasz", True, False),
}
PRIMARY = {
    "binary": "roc_auc",
    "multiclass": "balanced_accuracy",
    "regression": "r2",
    "clustering": "silhouette",
}
TASK_METRICS = {
    "binary": [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "brier",
        "ece",
    ],
    "multiclass": [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "log_loss",
    ],
    "regression": ["mae", "rmse", "r2", "median_ae"],
    "clustering": ["silhouette", "davies_bouldin", "calinski_harabasz"],
}


def _f(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def expected_calibration_error(y_true: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = index == b
        if mask.any():
            total += mask.mean() * abs(float(y_true[mask].mean()) - float(p[mask].mean()))
    return total


def classification(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, classes: list[Any]
) -> tuple[dict[str, float | None], dict[str, str], dict[str, Any]]:
    """Returns (metrics, reasons for metrics that are None, details: confusion matrix and
    per-class precision/recall/F1/support aligned with ``classes``)."""
    binary = len(classes) == 2
    out: dict[str, float | None] = {}
    reasons: dict[str, str] = {}
    out["accuracy"] = _f(skm.accuracy_score(y_true, y_pred))
    out["balanced_accuracy"] = (
        _f(skm.balanced_accuracy_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None
    )
    if out["balanced_accuracy"] is None:
        reasons["balanced_accuracy"] = "Only one class among the test rows."
    if binary:
        kw: dict[str, Any] = {"pos_label": classes[1], "average": "binary", "zero_division": 0}
    else:
        kw = {"labels": classes, "average": "macro", "zero_division": 0}
    out["precision"] = _f(skm.precision_score(y_true, y_pred, **kw))
    out["recall"] = _f(skm.recall_score(y_true, y_pred, **kw))
    out["f1"] = _f(skm.f1_score(y_true, y_pred, **kw))
    present = set(np.unique(y_true).tolist())
    all_present = present == set(classes)
    if proba is None:
        for m in ("roc_auc", "pr_auc", "log_loss", "brier", "ece"):
            out[m] = None
            reasons[m] = "The model gives no probabilities."
    else:
        if all_present:
            if binary:
                positive = proba[:, 1]
                truth = (y_true == classes[1]).astype(int)
                out["roc_auc"] = _f(skm.roc_auc_score(truth, positive))
                out["pr_auc"] = _f(skm.average_precision_score(truth, positive))
            else:
                out["roc_auc"] = _f(
                    skm.roc_auc_score(
                        y_true, proba, multi_class="ovr", average="macro", labels=classes
                    )
                )
                onehot = np.column_stack([(y_true == c).astype(int) for c in classes])
                out["pr_auc"] = _f(skm.average_precision_score(onehot, proba, average="macro"))
        else:
            out["roc_auc"] = out["pr_auc"] = None
            reasons["roc_auc"] = reasons["pr_auc"] = "Not every class occurs among the test rows."
        clipped = np.clip(proba, 1e-15, 1.0)
        clipped = clipped / clipped.sum(axis=1, keepdims=True)
        out["log_loss"] = _f(skm.log_loss(y_true, clipped, labels=classes))
        if binary:
            truth = (y_true == classes[1]).astype(int)
            out["brier"] = _f(skm.brier_score_loss(truth, proba[:, 1]))
            out["ece"] = _f(expected_calibration_error(truth, proba[:, 1]))
    precision, recall, f1, support = skm.precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    details = {
        "confusion_matrix": skm.confusion_matrix(y_true, y_pred, labels=classes).tolist(),
        "per_class": [
            {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for p, r, f, s in zip(precision, recall, f1, support, strict=True)
        ],
    }
    return {k: out[k] for k in TASK_METRICS["binary" if binary else "multiclass"]}, reasons, details


def regression(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[dict[str, float | None], dict[str, str]]:
    reasons: dict[str, str] = {}
    out: dict[str, float | None] = {
        "mae": _f(skm.mean_absolute_error(y_true, y_pred)),
        "rmse": _f(np.sqrt(skm.mean_squared_error(y_true, y_pred))),
        "median_ae": _f(skm.median_absolute_error(y_true, y_pred)),
    }
    if len(y_true) < 2 or float(np.var(y_true)) == 0.0:
        out["r2"] = None
        reasons["r2"] = "R² is undefined when the test target does not vary."
    else:
        out["r2"] = _f(skm.r2_score(y_true, y_pred))
    return {k: out[k] for k in TASK_METRICS["regression"]}, reasons


def clustering(
    x: np.ndarray, labels: np.ndarray, seed: int
) -> tuple[dict[str, float | None], dict[str, str]]:
    n_labels = len(np.unique(labels))
    if n_labels < 2 or n_labels >= len(labels):
        reason = "Internal cluster metrics need between 2 and n-1 clusters among the test rows."
        return dict.fromkeys(TASK_METRICS["clustering"]), dict.fromkeys(
            TASK_METRICS["clustering"], reason
        )
    sample = min(len(labels), SILHOUETTE_MAX_ROWS)
    out = {
        "silhouette": _f(skm.silhouette_score(x, labels, sample_size=sample, random_state=seed)),
        "davies_bouldin": _f(skm.davies_bouldin_score(x, labels)),
        "calinski_harabasz": _f(skm.calinski_harabasz_score(x, labels)),
    }
    return out, {}

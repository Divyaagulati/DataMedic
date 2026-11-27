"""Model catalogue, tuning grids and metric definitions for supervised training (Phase 5).

Standard library only: the API lists, validates and estimates from this module without loading
pandas or scikit-learn. Estimators are built from these definitions by ``engine.ml.models``
(worker only).

Grids are fixed and small (at most four settings per model). They are ordered from the most
regularised / simplest setting to the least, and ties in selection go to the earlier entry, so
selection is deterministic. ``default`` is the setting used when tuning is off or falls back.
"""

from dataclasses import dataclass
from typing import Any

TASKS = ("binary", "multiclass", "regression")
CLASSIFICATION = ("binary", "multiclass")
TUNING_MODES = ("nested", "defaults")
INNER_FOLDS = 3
MIN_ROWS_FOR_NESTED = 200
MIN_CLASS_ROWS_PER_INNER_FOLD = 3
MAX_CLASSES = 50
MIN_COHORT_ROWS = 50
MIN_REGRESSION_DISTINCT = 10
INNER_SEED_OFFSET = 1000

ENGINE_ML_VERSION = "1.0.0"


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    note: str
    classification_grid: tuple[dict[str, Any], ...]
    regression_grid: tuple[dict[str, Any], ...]
    classification_default: dict[str, Any]
    regression_default: dict[str, Any]
    estimator_classification: str
    estimator_regression: str

    def grid(self, task: str) -> tuple[dict[str, Any], ...]:
        return self.classification_grid if task in CLASSIFICATION else self.regression_grid

    def default(self, task: str) -> dict[str, Any]:
        return self.classification_default if task in CLASSIFICATION else self.regression_default

    def estimator(self, task: str) -> str:
        return (
            self.estimator_classification if task in CLASSIFICATION else self.estimator_regression
        )


def _grid(**axes: tuple[Any, ...]) -> tuple[dict[str, Any], ...]:
    """Cartesian product in the given axis order (first axis varies slowest)."""
    out: list[dict[str, Any]] = [{}]
    for name, values in axes.items():
        out = [{**g, name: v} for g in out for v in values]
    return tuple(out)


MODELS: dict[str, ModelSpec] = {
    "dummy": ModelSpec(
        key="dummy",
        label="Dummy baseline",
        note="Predicts the training-fold class prior (classification) or median "
        "(regression) and ignores every feature. Any useful model should beat it.",
        classification_grid=({},),
        regression_grid=({},),
        classification_default={},
        regression_default={},
        estimator_classification="DummyClassifier(strategy='prior')",
        estimator_regression="DummyRegressor(strategy='median')",
    ),
    "linear": ModelSpec(
        key="linear",
        label="Linear (logistic / ridge)",
        note="Logistic regression (classification) or ridge regression (regression), after "
        "in-fold median imputation and standardisation of numeric columns and most-frequent "
        "imputation plus one-hot encoding (20 most frequent categories) of categorical "
        "columns.",
        classification_grid=_grid(C=(0.01, 0.1, 1.0, 10.0)),
        regression_grid=_grid(alpha=(100.0, 10.0, 1.0, 0.1)),
        classification_default={"C": 1.0},
        regression_default={"alpha": 1.0},
        estimator_classification="LogisticRegression(max_iter=2000)",
        estimator_regression="Ridge",
    ),
    "random_forest": ModelSpec(
        key="random_forest",
        label="Random forest",
        note="100 trees, depth at most 12, one thread; categories as ordinal codes and "
        "missing values imputed in-fold (median / most frequent).",
        classification_grid=_grid(max_depth=(6, 12), min_samples_leaf=(10, 2)),
        regression_grid=_grid(max_depth=(6, 12), min_samples_leaf=(10, 2)),
        classification_default={"max_depth": 12, "min_samples_leaf": 2},
        regression_default={"max_depth": 12, "min_samples_leaf": 2},
        estimator_classification="RandomForestClassifier(n_estimators=100, n_jobs=1)",
        estimator_regression="RandomForestRegressor(n_estimators=100, n_jobs=1)",
    ),
    "hgb": ModelSpec(
        key="hgb",
        label="Histogram gradient boosting",
        note="Gradient boosting with native missing-value support and categories as ordinal "
        "codes; early stopping off (its internal validation split would ignore groups).",
        classification_grid=_grid(max_leaf_nodes=(15, 31), learning_rate=(0.05, 0.1)),
        regression_grid=_grid(max_leaf_nodes=(15, 31), learning_rate=(0.05, 0.1)),
        classification_default={"max_leaf_nodes": 31, "learning_rate": 0.1},
        regression_default={"max_leaf_nodes": 31, "learning_rate": 0.1},
        estimator_classification="HistGradientBoostingClassifier(early_stopping=False)",
        estimator_regression="HistGradientBoostingRegressor(early_stopping=False)",
    ),
}
MODEL_KEYS = tuple(MODELS)

# Metric name -> (label, higher is better, bounded, uses probabilities)
METRICS: dict[str, tuple[str, bool, bool, bool]] = {
    "accuracy": ("Accuracy", True, True, False),
    "balanced_accuracy": ("Balanced accuracy", True, True, False),
    "precision": ("Precision", True, True, False),
    "recall": ("Recall", True, True, False),
    "f1": ("F1", True, True, False),
    "precision_weighted": ("Precision (weighted)", True, True, False),
    "recall_weighted": ("Recall (weighted)", True, True, False),
    "f1_weighted": ("F1 (weighted)", True, True, False),
    "roc_auc": ("ROC-AUC", True, True, True),
    "pr_auc": ("PR-AUC (average precision)", True, True, True),
    "log_loss": ("Log loss", False, False, True),
    "brier": ("Brier score", False, True, True),
    "ece": ("Expected calibration error", False, True, True),
    "mae": ("MAE", False, False, False),
    "mse": ("MSE", False, False, False),
    "rmse": ("RMSE", False, False, False),
    "r2": ("R²", True, False, False),
    "median_ae": ("Median absolute error", False, False, False),
}
TASK_METRICS: dict[str, tuple[str, ...]] = {
    "binary": (
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
    ),
    "multiclass": (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
        "roc_auc",
        "pr_auc",
        "log_loss",
        "ece",
    ),
    "regression": ("mae", "mse", "rmse", "r2", "median_ae"),
}
# Reported headline metric per task.
PRIMARY = {"binary": "roc_auc", "multiclass": "balanced_accuracy", "regression": "r2"}
# Inner-fold selection metric. Binary uses ROC-AUC only when every inner test fold contains
# both classes (decided before any fit); otherwise balanced accuracy. Regression selects on
# RMSE, which is always defined.
SELECTION = {"binary": "roc_auc", "multiclass": "balanced_accuracy", "regression": "rmse"}
SELECTION_FALLBACK = {"binary": "balanced_accuracy"}

METRIC_CONVENTIONS = [
    "Binary precision, recall and F1 are for the positive class: the second class in sorted "
    "order of the original labels (shown as 'class 2' when the target is sensitive).",
    "Multiclass precision, recall and F1 are macro averages (every class weighs the same); "
    "the weighted variants weight classes by their number of test rows.",
    "Multiclass ROC-AUC and PR-AUC are one-vs-rest macro averages and need every class among "
    "the test rows of a fold; ECE for multiclass is the top-label calibration error.",
    "ROC-AUC, PR-AUC, log loss, Brier score and ECE use predicted probabilities; the other "
    "classification metrics use hard labels (the class with the highest probability).",
    "The dummy baseline gives every row the same predicted probability, so all its scores "
    "tie and its ROC-AUC is exactly 0.5 whenever both classes are among the test rows: the "
    "chance reference, not a missing value. ROC-AUC is unavailable (with a reason) only when "
    "a fold's test rows contain a single class.",
    "Metrics are computed on held-out test rows of each outer fold and summarised as mean, "
    "standard deviation, minimum and maximum across folds. A metric that cannot be computed "
    "in a fold is unavailable for that fold, with a reason; it is never replaced by 0.",
]


def metric_info() -> dict[str, dict[str, Any]]:
    return {
        k: {"label": v[0], "higher_is_better": v[1], "bounded": v[2], "probability": v[3]}
        for k, v in METRICS.items()
    }


def configs(model: str, task: str, mode: str) -> tuple[dict[str, Any], ...]:
    spec = MODELS[model]
    if mode == "nested":
        return spec.grid(task)
    return (spec.default(task),)


def estimate_fits(n_outer: int, models: list[str], task: str, mode: str) -> int:
    """Model fits for a run: per outer fold and model, one fit per setting and inner fold
    (when the grid has more than one setting), plus the outer refit. Preprocessing fits and
    audit / explainability work are not counted."""
    total = 0
    for model in models:
        grid = configs(model, task, mode)
        inner = len(grid) * INNER_FOLDS if mode == "nested" and len(grid) > 1 else 0
        total += n_outer * (inner + 1)
    return total


def catalogue() -> list[dict[str, Any]]:
    return [
        {
            "key": m.key,
            "label": m.label,
            "note": m.note,
            "tasks": list(TASKS),
            "grids": {
                "classification": list(m.classification_grid),
                "regression": list(m.regression_grid),
            },
            "defaults": {
                "classification": m.classification_default,
                "regression": m.regression_default,
            },
            "estimators": {
                "classification": m.estimator_classification,
                "regression": m.estimator_regression,
            },
        }
        for m in MODELS.values()
    ]

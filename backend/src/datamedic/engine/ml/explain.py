"""Explainability on held-out rows (worker only).

**Permutation importance** (model-agnostic). For an outer fold, the model is fitted on the
training rows and never sees the test rows. For every *original* column, the model input
columns derived from it (the column itself, its missing indicator, its one-hot or encoded
forms, from the feature lineage) are permuted together, with one shared row permutation, in
the fold's test rows, and the drop in the importance metric is recorded. Repeats use seeded
permutations. Importance is aggregated per original column across folds (mean and sd over
all fold x repeat values); a column removed by a fold's own feature selection has no value in
that fold (reported as the number of folds in which it was a model input), never 0.

Multi-column operations (e.g. KNN imputation) can carry information from one column into
another; their outputs are attributed to the column they were written to. When the lineage
of a model input cannot be resolved to exactly one original column, it is reported under its
own name with ``mapping: unresolved`` rather than guessed.

**Model-specific importances**: linear coefficients (standardised numeric inputs; the L2 norm
of a column's one-hot coefficients; per class for multiclass) and random-forest impurity
importance. Histogram gradient boosting exposes no built-in importance in scikit-learn.

**Error slices**: the error of pooled out-of-fold predictions per bin or category of the most
important original columns, with at least ``MIN_SLICE_ROWS`` rows per slice; sensitive-role
columns are never sliced.

None of this is causal: importance describes how this model, trained on this data, uses its
inputs.
"""

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

MIN_SLICE_ROWS = 30
SLICE_FEATURES = 5
SLICE_CATEGORIES = 10
SLICE_BINS = 5
MAX_LABEL = 64

LIMITATIONS = [
    "Importance describes how these models use their inputs on this data; it is not causal "
    "evidence and not a property of the real-world process.",
    "Correlated columns share or mask each other's importance: permuting one leaves its "
    "information available through the others.",
    "Permutation creates combinations of values that may not occur in reality.",
    "Importance can differ between folds and between models; both are shown.",
    "Computed on held-out test rows after the models were selected and fitted; it never "
    "influences tuning or model selection.",
]


def origin_map(
    lineage: Sequence[Mapping[str, Any]], inputs: Sequence[str]
) -> dict[str, str | None]:
    """Model-input column -> the original (evaluation-source) column it derives from, or None
    when it cannot be resolved to exactly one."""
    parents: dict[str, list[str]] = {}
    for entry in lineage:
        if entry.get("origin") == "added":
            parents[str(entry["column"])] = [str(c) for c in entry.get("derived_from") or []]
    originals = set(inputs)

    def resolve(column: str, depth: int = 0) -> str | None:
        if column in originals and column not in parents:
            return column
        if depth > 50:
            return None
        sources = parents.get(column)
        if not sources:
            return column if column in originals else None
        roots = {resolve(s, depth + 1) for s in sources}
        roots.discard(None)
        return roots.pop() if len(roots) == 1 else None

    return {c: resolve(c) for c in parents} | {c: c for c in originals if c not in parents}


def group_inputs(columns: Sequence[str], origins: Mapping[str, str | None]) -> dict[str, list[str]]:
    """Original column -> model-input columns derived from it. Inputs whose origin cannot be
    resolved form their own group, keyed ``UNRESOLVED_PREFIX + name``."""
    groups: dict[str, list[str]] = defaultdict(list)
    for c in columns:
        root = origins.get(c)
        groups[root if root is not None else UNRESOLVED_PREFIX + c].append(c)
    return dict(groups)


UNRESOLVED_PREFIX = "unresolved:"


def permutation(
    x_test: pd.DataFrame,
    groups: Mapping[str, list[str]],
    scorer: Callable[[pd.DataFrame], float | None],
    *,
    repeats: int,
    max_rows: int,
    max_groups: int,
    seed: int,
) -> dict[str, Any]:
    """Drop in the score for each group of model-input columns permuted together."""
    rng = np.random.default_rng(seed)
    sampled = False
    if len(x_test) > max_rows:
        index = np.sort(rng.choice(len(x_test), max_rows, replace=False))
        x_test = x_test.iloc[index]
        sampled = True
    base = scorer(x_test)
    if base is None:
        return {"available": False, "reason": "The importance metric is undefined in this fold."}
    names = list(groups)[:max_groups]
    out: dict[str, list[float]] = {}
    for name in names:
        cols = [c for c in groups[name] if c in x_test.columns]
        if not cols:
            continue
        drops = []
        for _ in range(repeats):
            order = rng.permutation(len(x_test))
            shuffled = x_test.copy()
            shuffled[cols] = x_test[cols].iloc[order].to_numpy()
            value = scorer(shuffled)
            if value is not None:
                drops.append(base - value)
        out[name] = drops
    return {
        "available": True,
        "baseline_score": base,
        "drops": out,
        "rows": len(x_test),
        "sampled": sampled,
        "not_computed": max(0, len(groups) - len(names)),
    }


def linear_importance(
    pipeline: Any, input_sources: Sequence[str] | None
) -> dict[str, float] | None:
    """L2 norm of coefficients per transformed-table column (per class rows combined)."""
    model = pipeline.named_steps.get("model")
    coef = getattr(model, "coef_", None)
    if coef is None or input_sources is None:
        return None
    coef = np.atleast_2d(np.asarray(coef, dtype=float))
    if coef.shape[1] != len(input_sources):
        return None
    out: dict[str, float] = defaultdict(float)
    for j, source in enumerate(input_sources):
        out[source] += float(np.sum(coef[:, j] ** 2))
    return {k: float(np.sqrt(v)) for k, v in out.items()}


def impurity_importance(
    pipeline: Any, input_sources: Sequence[str] | None
) -> dict[str, float] | None:
    model = pipeline.named_steps.get("model")
    values = getattr(model, "feature_importances_", None)
    if values is None or input_sources is None or len(values) != len(input_sources):
        return None
    out: dict[str, float] = defaultdict(float)
    for j, source in enumerate(input_sources):
        out[source] += float(values[j])
    return dict(out)


def to_original(values: Mapping[str, float], origins: Mapping[str, str | None]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for column, v in values.items():
        root = origins.get(column)
        out[root if root is not None else UNRESOLVED_PREFIX + column] += v
    return dict(out)


def aggregate(
    per_fold: Sequence[Mapping[str, Sequence[float]]], n_folds: int
) -> list[dict[str, Any]]:
    """Per original column: mean and sd over every fold x repeat value, folds present."""
    values: dict[str, list[float]] = defaultdict(list)
    folds_present: dict[str, int] = defaultdict(int)
    for fold in per_fold:
        for column, drops in fold.items():
            if drops:
                values[column].extend(drops)
                folds_present[column] += 1
    rows = [
        {
            "column": c,
            "mean": round(float(np.mean(v)), 6),
            "sd": round(float(np.std(v, ddof=1)), 6) if len(v) > 1 else None,
            "folds_present": folds_present[c],
            "n_folds": n_folds,
        }
        for c, v in values.items()
    ]
    return sorted(rows, key=lambda r: (-r["mean"], r["column"]))


def aggregate_simple(per_fold: Sequence[Mapping[str, float]], n_folds: int) -> list[dict[str, Any]]:
    return aggregate([{k: [v] for k, v in f.items()} for f in per_fold], n_folds)


def _label(value: Any) -> str:
    text = str(value)
    return text if len(text) <= MAX_LABEL else text[: MAX_LABEL - 1] + "…"


def error_slices(
    frame: pd.DataFrame,
    errors: np.ndarray,
    columns: Sequence[str],
    sensitive: frozenset[str],
    metric: str,
) -> list[dict[str, Any]]:
    """``errors``: per row of ``frame`` (error indicator for classification, absolute error
    for regression). Slices with fewer than ``MIN_SLICE_ROWS`` rows are merged into 'other'
    (categories) or withheld (numeric bins)."""
    out = []
    for column in columns[:SLICE_FEATURES]:
        if column in sensitive or column not in frame.columns:
            continue
        s = frame[column]
        slices: list[dict[str, Any]] = []
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            values = s.to_numpy(dtype=float, na_value=np.nan)
            present = ~np.isnan(values)
            if present.sum() >= MIN_SLICE_ROWS:
                edges = np.unique(np.quantile(values[present], np.linspace(0, 1, SLICE_BINS + 1)))
                index = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
                for b in range(len(edges) - 1):
                    mask = present & (index == b)
                    n = int(mask.sum())
                    if n >= MIN_SLICE_ROWS:
                        slices.append(
                            {
                                "label": f"{edges[b]:.4g} – {edges[b + 1]:.4g}",
                                "n": n,
                                "error": round(float(errors[mask].mean()), 6),
                            }
                        )
            missing = ~present
        else:
            text = s.astype(object).where(s.notna(), None)
            counts = text.dropna().value_counts()
            kept = [v for v, n in counts.items() if n >= MIN_SLICE_ROWS][:SLICE_CATEGORIES]
            other = np.ones(len(s), dtype=bool)
            for v in kept:
                mask = (text == v).to_numpy()
                other &= ~mask
                slices.append(
                    {
                        "label": _label(v),
                        "n": int(mask.sum()),
                        "error": round(float(errors[mask].mean()), 6),
                    }
                )
            missing = s.isna().to_numpy()
            other &= ~missing
            if other.sum() >= MIN_SLICE_ROWS:
                slices.append(
                    {
                        "label": "other",
                        "n": int(other.sum()),
                        "error": round(float(errors[other].mean()), 6),
                    }
                )
        if missing.sum() >= MIN_SLICE_ROWS:
            slices.append(
                {
                    "label": "missing",
                    "n": int(missing.sum()),
                    "error": round(float(errors[missing].mean()), 6),
                }
            )
        if slices:
            out.append({"column": column, "metric": metric, "slices": slices})
    return out

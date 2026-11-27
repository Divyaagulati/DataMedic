"""Data-quality changes made by a candidate, measured on the held-out rows of each fold
(before = the raw test rows, after = the same rows transformed by the pipeline fitted on the
training rows) plus the training-table changes made by resampling.

These are descriptions, not scores: fewer missing values or fewer outliers is not
automatically better (imputed values are estimates; capped values are overwritten
observations). Counts only, never values.
"""

from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine.preprocess.catalog import get_op
from datamedic.engine.preprocess.report import StepReport

MIN_SKEW_VALUES = 8


def _numeric(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
    ]


def _categorical(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        c
        for c in columns
        if not pd.api.types.is_numeric_dtype(frame[c])
        and not pd.api.types.is_bool_dtype(frame[c])
        and not pd.api.types.is_datetime64_any_dtype(frame[c])
    ]


def describe(frame: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    numeric = _numeric(frame, columns)
    outliers = 0
    skews = []
    for c in numeric:
        values = frame[c].to_numpy(dtype=float, na_value=np.nan)
        values = values[np.isfinite(values)]
        if values.size >= 4:
            q1, q3 = np.quantile(values, [0.25, 0.75])
            iqr = q3 - q1
            if iqr > 0:
                outliers += int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        if values.size >= MIN_SKEW_VALUES and len(np.unique(values)) > 2:
            skew = pd.Series(values).skew()
            if pd.notna(skew):
                skews.append(abs(float(skew)))
    categorical = _categorical(frame, columns)
    data = frame[columns]
    return {
        "features": len(columns),
        "missing_cells": int(data.isna().sum().sum()) if columns else 0,
        "duplicate_rows": int(data.duplicated().sum()) if columns else 0,
        "outlier_cells_iqr_1_5": outliers,
        "mean_abs_skewness": round(float(np.mean(skews)), 4) if skews else None,
        "constant_columns": int(sum(frame[c].nunique(dropna=True) <= 1 for c in columns)),
        "categorical_columns": len(categorical),
        "category_levels": int(sum(frame[c].nunique(dropna=True) for c in categorical)),
    }


def fold_quality(
    before: pd.DataFrame,
    after: pd.DataFrame,
    features_before: list[str],
    features_after: list[str],
    reports: list[StepReport],
    training_rows: tuple[int, int],
) -> dict[str, Any]:
    b = describe(before, features_before)
    a = describe(after, features_after)
    destructive = {r.op for r in reports if (s := get_op(r.op)) is not None and s.destructive}
    # Destructive steps overwrite observations (capping); others re-express them (log,
    # scaling), which is invertible and not an information loss.
    overwritten = sum(c.observed_changed for r in reports if r.op in destructive for c in r.columns)
    transformed = sum(
        c.observed_changed for r in reports if r.op not in destructive for c in r.columns
    )
    filled = sum(c.cells_filled for r in reports for c in r.columns)
    common = [c for c in features_before if c in after.columns]
    rows_affected = 0
    if common and len(before) == len(after):
        x = before[common].reset_index(drop=True).astype(object)
        y = after[common].reset_index(drop=True).astype(object)
        differ = ~((x == y) | (x.isna() & y.isna()))
        rows_affected = int(differ.any(axis=1).sum())
    return {
        "before": b,
        "after": a,
        "cells_imputed": int(filled),
        "observed_values_overwritten": int(overwritten),
        "observed_values_transformed": int(transformed),
        "columns_added": sorted({c for r in reports for c in r.columns_added}),
        "columns_removed": sorted({c for r in reports for c in r.columns_removed}),
        "rows_affected": rows_affected,
        "test_rows": len(before),
        "training_rows_before": training_rows[0],
        "training_rows_after": training_rows[1],
    }


_NUMERIC_KEYS = (
    "features",
    "missing_cells",
    "duplicate_rows",
    "outlier_cells_iqr_1_5",
    "mean_abs_skewness",
    "constant_columns",
    "categorical_columns",
    "category_levels",
)


def aggregate(folds: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Mean over the folds that ran (counts are per fold, on that fold's test rows)."""
    if not folds:
        return None

    def mean(values: list[Any]) -> float | None:
        present = [float(v) for v in values if v is not None]
        return round(float(np.mean(present)), 4) if present else None

    out: dict[str, Any] = {"folds": len(folds), "before": {}, "after": {}}
    for side in ("before", "after"):
        for key in _NUMERIC_KEYS:
            out[side][key] = mean([f[side][key] for f in folds])
    for key in (
        "cells_imputed",
        "observed_values_overwritten",
        "observed_values_transformed",
        "rows_affected",
        "test_rows",
        "training_rows_before",
        "training_rows_after",
    ):
        out[key] = mean([f[key] for f in folds])
    out["columns_added"] = sorted({c for f in folds for c in f["columns_added"]})
    out["columns_removed"] = sorted({c for f in folds for c in f["columns_removed"]})
    loss = []
    if out["observed_values_overwritten"]:
        loss.append("Observed values were overwritten (e.g. capped).")
    if out["columns_removed"]:
        loss.append("Columns were removed from the model input in some folds.")
    if (out["training_rows_after"] or 0) < (out["training_rows_before"] or 0):
        loss.append("Training rows were discarded (undersampling).")
    if (out["training_rows_after"] or 0) > (out["training_rows_before"] or 0):
        loss.append("Training rows were repeated or synthesised (not new observations).")
    if out["cells_imputed"]:
        loss.append("Missing values were replaced by estimates.")
    out["information_loss_indicators"] = loss
    return out

"""Reference implementations of DataMedic preprocessing steps.

This module is the single implementation of every step's algorithm. The engine's ops call
it, and code generation embeds its source verbatim into ``pipeline.py``, so platform results
and generated code cannot drift apart. It must therefore import only the standard library,
NumPy, pandas, scikit-learn and SciPy (a scikit-learn dependency), never ``datamedic``, and
must not read files or the network. imbalanced-learn is imported inside the resampling
functions only, so generated code needs it only when a pipeline resamples.

Conventions:
- ``fit_*`` functions learn only from the rows they are given and return an in-memory state;
  ``apply_*`` functions never learn. Inputs are never modified.
- Missing means ``Series.isna()`` (None, NaN, pd.NA, NaT). Infinite values are not missing;
  numeric steps refuse them instead of guessing.
- Observed values are never changed: only cells that were missing are written.
"""

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.ensemble import IsolationForest
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (enables the import)
from sklearn.feature_selection import f_regression, mutual_info_classif, mutual_info_regression
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

ROW_ID = "_dm_row_id"
INDICATOR_SUFFIX = "__missing"
OUTLIER_SUFFIX = "__outlier"
ONEHOT_SEPARATOR = "="
ONEHOT_OTHER = "__other__"
MAD_SCALE = 1.4826  # makes the median absolute deviation consistent with a normal SD
MAX_NAME_LENGTH = 200  # same limit as canonical column names
KNN_WORKING_MEMORY_MB = 256  # scikit-learn processes neighbour distances in chunks of this size


class PreprocessError(Exception):
    """A step cannot be performed safely. Messages never contain data values."""

    def __init__(self, code: str, message: str, column: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.column = column


# ---- Helpers ----------------------------------------------------------------------------------


def numeric_matrix(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Float matrix (rows x columns) with NaN for missing values. Refuses infinities."""
    if not columns:
        return np.empty((len(df), 0), dtype="float64")
    parts = []
    for column in columns:
        values = df[column].to_numpy(dtype="float64", na_value=np.nan)
        if np.isinf(values).any():
            raise PreprocessError(
                "NON_FINITE_VALUES",
                "The column contains infinite values. Replace or remove them before this step.",
                column,
            )
        parts.append(values)
    return np.column_stack(parts)


def require_observed(matrix: np.ndarray, columns: list[str]) -> None:
    """Every column to impute needs at least one observed value in the fitting rows."""
    for j, column in enumerate(columns):
        if matrix.shape[0] == 0 or np.isnan(matrix[:, j]).all():
            raise PreprocessError(
                "FIT_COLUMN_EMPTY",
                "The column has no observed value in the fitting rows, so nothing can be "
                "learned from it.",
                column,
            )


def require_finite(values: Any, column: str | None) -> None:
    """Refuse a computed value that is not a finite number. Arithmetic on values near the
    float limit (about 1.8e308) overflows; the result must never be written or reported."""
    with np.errstate(all="ignore"):
        finite = bool(np.all(np.isfinite(np.asarray(values, dtype="float64"))))
    if not finite:
        raise PreprocessError(
            "NUMERIC_OVERFLOW",
            "The computation produced a value that is not a finite number, usually because "
            "values are too large in magnitude. Rescale the column first (e.g. change its unit).",
            column,
        )


def object_values(series: pd.Series) -> np.ndarray:
    """Object array with NaN for every kind of missing value (scikit-learn does not treat
    None or pd.NA as missing in object arrays)."""
    values: np.ndarray = series.astype(object).to_numpy(dtype=object, copy=True)
    values[series.isna().to_numpy()] = np.nan
    return values


def fill_missing(series: pd.Series, value: Any) -> pd.Series:
    """Fill missing cells with ``value`` while keeping the column's storage type:
    nullable integer/float/boolean/string and categorical dtypes are preserved. The one
    documented change: a non-whole value in an integer column makes it float64."""
    if not series.isna().any():
        if pd.api.types.is_integer_dtype(series) and not float(value).is_integer():
            return series.astype("float64")
        return series.copy()
    if isinstance(series.dtype, pd.CategoricalDtype):
        if value not in series.cat.categories:
            series = series.cat.add_categories([value])
        return series.fillna(value)
    if pd.api.types.is_bool_dtype(series):
        return series.astype("boolean").fillna(bool(value))
    if pd.api.types.is_integer_dtype(series):
        if float(value).is_integer():
            return series.astype("Int64").fillna(int(value))
        return series.astype("float64").fillna(float(value))
    if pd.api.types.is_float_dtype(series):
        if isinstance(series.dtype, pd.Float64Dtype | pd.Float32Dtype):
            return series.fillna(float(value))
        return series.astype("float64").fillna(float(value))
    if isinstance(series.dtype, pd.StringDtype):
        return series.fillna(str(value))
    return series.astype(object).where(series.notna(), value)


def with_columns(df: pd.DataFrame, values: dict[str, pd.Series]) -> pd.DataFrame:
    out = df.copy()
    for column, series in values.items():
        out[column] = series
    return out


def _drop_empty_predictors(
    matrix: np.ndarray, names: list[str], n_target: int
) -> tuple[np.ndarray, list[str], list[str]]:
    """Predictors with no observed value in the fitting rows carry no information."""
    keep = [j for j in range(len(names)) if j < n_target or not np.isnan(matrix[:, j]).all()]
    dropped = [names[j] for j in range(len(names)) if j not in keep]
    return matrix[:, keep], [names[j] for j in keep], dropped


def _write_back(
    df: pd.DataFrame, columns: list[str], original: np.ndarray, estimates: np.ndarray
) -> pd.DataFrame:
    filled = {}
    for j, column in enumerate(columns):
        observed = original[:, j]
        require_finite(estimates[np.isnan(observed), j], column)
        values = np.where(np.isnan(observed), estimates[:, j], observed)
        filled[column] = pd.Series(values, index=df.index, dtype="float64")
    return with_columns(df, filled)


# ---- impute.mean / impute.median --------------------------------------------------------------


def fit_simple_numeric(fit_df: pd.DataFrame, columns: list[str], strategy: str) -> dict[str, Any]:
    matrix = numeric_matrix(fit_df, columns)
    require_observed(matrix, columns)
    with np.errstate(all="ignore"):
        imputer = SimpleImputer(strategy=strategy).fit(matrix)
    for column, value in zip(columns, imputer.statistics_, strict=True):
        require_finite(value, column)
    return {"imputer": imputer, "statistics": [float(v) for v in imputer.statistics_]}


def apply_simple_numeric(
    df: pd.DataFrame, columns: list[str], state: dict[str, Any]
) -> pd.DataFrame:
    matrix = numeric_matrix(df, columns)
    statistics = np.asarray(state["statistics"], dtype="float64")
    estimates = np.broadcast_to(statistics, matrix.shape)
    return _write_back(df, columns, matrix, estimates)


# ---- impute.mode ------------------------------------------------------------------------------


def _mode_input(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
        return object_values(series).reshape(-1, 1)
    numbers: np.ndarray = series.to_numpy(dtype="float64", na_value=np.nan)
    return numbers.reshape(-1, 1)


def _native(value: Any, series: pd.Series) -> Any:
    if pd.api.types.is_bool_dtype(series):
        return bool(value)
    if pd.api.types.is_integer_dtype(series):
        return int(value)
    if pd.api.types.is_float_dtype(series):
        return float(value)
    # Text, object and categorical columns: keep the value's own type (a number stored in
    # an object column must not become text).
    return value.item() if isinstance(value, np.generic) else value


def fit_mode(fit_df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    values: list[Any] = []
    ties: list[bool] = []
    for column in columns:
        series = fit_df[column]
        if not series.notna().any():
            raise PreprocessError(
                "FIT_COLUMN_EMPTY",
                "The column has no observed value in the fitting rows, so nothing can be "
                "learned from it.",
                column,
            )
        if (
            pd.api.types.is_float_dtype(series)
            and np.isinf(series.to_numpy(dtype="float64", na_value=np.nan)).any()
        ):
            raise PreprocessError(
                "NON_FINITE_VALUES",
                "The column contains infinite values. Replace or remove them before this step.",
                column,
            )
        imputer = SimpleImputer(strategy="most_frequent", missing_values=np.nan)
        imputer.fit(_mode_input(series))
        values.append(_native(imputer.statistics_[0], series))
        counts = series.dropna().value_counts()
        ties.append(bool((counts == counts.max()).sum() > 1))
    return {"values": values, "ties": ties}


def apply_mode(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    return with_columns(
        df, {c: fill_missing(df[c], v) for c, v in zip(columns, state["values"], strict=True)}
    )


# ---- impute.constant --------------------------------------------------------------------------


def apply_constant(df: pd.DataFrame, columns: list[str], fill_value: Any) -> pd.DataFrame:
    return with_columns(df, {c: fill_missing(df[c], fill_value) for c in columns})


# ---- impute.knn -------------------------------------------------------------------------------


def fit_knn(
    fit_df: pd.DataFrame,
    columns: list[str],
    predictors: list[str],
    n_neighbors: int,
    weights: str,
) -> dict[str, Any]:
    names = list(columns) + [p for p in predictors if p not in columns]
    matrix = numeric_matrix(fit_df, names)
    require_observed(matrix, columns)
    matrix, names, dropped = _drop_empty_predictors(matrix, names, len(columns))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows are allowed
        with np.errstate(all="ignore"):
            mean = np.nanmean(matrix, axis=0)
            std = np.nanstd(matrix, axis=0)
    for j, column in enumerate(names):
        require_finite([mean[j], std[j]], column)
    std = np.where(std > 0, std, 1.0)
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputer.fit((matrix - mean) / std)
    return {
        "imputer": imputer,
        "names": names,
        "dropped_predictors": dropped,
        "mean": [float(v) for v in mean],
        "std": [float(v) for v in std],
        "n_fit_rows": int(matrix.shape[0]),
    }


def apply_knn(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    matrix = numeric_matrix(df, state["names"])
    if matrix.shape[0] == 0:
        return _write_back(df, columns, matrix, matrix)
    mean = np.asarray(state["mean"])
    std = np.asarray(state["std"])
    with config_context(working_memory=KNN_WORKING_MEMORY_MB):
        estimates = state["imputer"].transform((matrix - mean) / std) * std + mean
    return _write_back(df, columns, matrix, estimates)


# ---- impute.iterative -------------------------------------------------------------------------


def fit_iterative(
    fit_df: pd.DataFrame,
    columns: list[str],
    predictors: list[str],
    max_iter: int,
    clip_to_fit_range: bool,
    seed: int,
) -> dict[str, Any]:
    names = list(columns) + [p for p in predictors if p not in columns]
    matrix = numeric_matrix(fit_df, names)
    require_observed(matrix, columns)
    matrix, names, dropped = _drop_empty_predictors(matrix, names, len(columns))
    if clip_to_fit_range:
        low = np.nanmin(matrix, axis=0)
        high = np.nanmax(matrix, axis=0)
        # scikit-learn requires max > min; a column constant in the fitting rows keeps its
        # single value (the bound is widened by one floating-point step).
        high = np.where(high > low, high, np.nextafter(low, np.inf))
    else:
        low = np.full(matrix.shape[1], -np.inf)
        high = np.full(matrix.shape[1], np.inf)
    imputer = IterativeImputer(
        max_iter=max_iter,
        random_state=seed,
        sample_posterior=False,
        min_value=low,
        max_value=high,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        imputer.fit(matrix)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return {
        "imputer": imputer,
        "names": names,
        "dropped_predictors": dropped,
        "low": [float(v) for v in low],
        "high": [float(v) for v in high],
        "converged": converged,
        "n_iter": int(imputer.n_iter_),
    }


def apply_iterative(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    matrix = numeric_matrix(df, state["names"])
    if matrix.shape[0] == 0:
        return _write_back(df, columns, matrix, matrix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        estimates = state["imputer"].transform(matrix)
    return _write_back(df, columns, matrix, estimates)


# ---- missing.indicator ------------------------------------------------------------------------


def apply_indicator(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    added = {}
    for column in columns:
        name = column + INDICATOR_SUFFIX
        if name in df.columns:
            raise PreprocessError(
                "OUTPUT_COLUMN_CONFLICT", "The indicator column name already exists.", column
            )
        added[name] = pd.Series(df[column].isna().to_numpy(dtype=bool), index=df.index)
    return with_columns(df, added)


# ---- Outliers: detection rules (fitted bounds) ------------------------------------------------


def _skewness(values: np.ndarray) -> float | None:
    skew = pd.Series(values).skew()
    return None if pd.isna(skew) else float(skew)


def fit_bounds(
    fit_df: pd.DataFrame,
    columns: list[str],
    method: str,
    params: dict[str, Any],
    min_values: int,
) -> dict[str, Any]:
    """Learn outlier bounds per column from the fitting rows.

    iqr: [Q1 - k*IQR, Q3 + k*IQR]; zscore: mean -/+ z*SD (ddof=1); mad: median -/+
    t*1.4826*MAD; percentile: [p_low, p_high] quantiles (linear interpolation). A column with
    no spread under the rule gets ``None`` (no bounds: nothing is flagged or capped)."""
    bounds: list[list[float] | None] = []
    stats: list[dict[str, Any]] = []
    for column in columns:
        values = numeric_matrix(fit_df, [column])[:, 0]
        values = values[~np.isnan(values)]
        if values.size < min_values:
            raise PreprocessError(
                "FIT_TOO_FEW_VALUES",
                f"The rule needs at least {min_values} observed values in the fitting rows.",
                column,
            )
        info: dict[str, Any] = {"n_values": int(values.size), "skewness": _skewness(values)}
        if method == "iqr":
            q1, q3 = (float(v) for v in np.quantile(values, [0.25, 0.75]))
            spread = q3 - q1
            k = float(params["iqr_multiplier"])
            low, high = q1 - k * spread, q3 + k * spread
            info.update(q1=q1, q3=q3)
        elif method == "zscore":
            centre = float(values.mean())
            spread = float(values.std(ddof=1))
            z = float(params["z_threshold"])
            low, high = centre - z * spread, centre + z * spread
            info.update(mean=centre, sd=spread)
        elif method == "mad":
            centre = float(np.median(values))
            spread = MAD_SCALE * float(np.median(np.abs(values - centre)))
            t = float(params["mad_threshold"])
            low, high = centre - t * spread, centre + t * spread
            info.update(median=centre, scaled_mad=spread)
        elif method == "percentile":
            quantiles = [
                float(params["lower_percentile"]) / 100,
                float(params["upper_percentile"]) / 100,
            ]
            low, high = (float(v) for v in np.quantile(values, quantiles))
            spread = high - low
        else:
            raise PreprocessError("OP_INVALID", "Unknown detection rule.", column)
        # The bounds must be finite. The spread itself may overflow for the percentile rule
        # (bounds near the float limit) without making the bounds invalid.
        require_finite([low, high], column)
        if high > low:
            bounds.append([float(low), float(high)])
        else:
            bounds.append(None)
            info["zero_spread"] = True
        stats.append(info)
    return {"bounds": bounds, "stats": stats}


def outlier_mask(df: pd.DataFrame, column: str, bound: list[float] | None) -> np.ndarray:
    """True where the value lies outside the bounds. Missing values are never outliers."""
    values = numeric_matrix(df, [column])[:, 0]
    if bound is None:
        return np.zeros(len(df), dtype=bool)
    with np.errstate(invalid="ignore"):
        mask: np.ndarray = (values < bound[0]) | (values > bound[1])
    return mask


def apply_outlier_flag(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    added = {}
    for column, bound in zip(columns, state["bounds"], strict=True):
        name = column + OUTLIER_SUFFIX
        if name in df.columns:
            raise PreprocessError(
                "OUTPUT_COLUMN_CONFLICT", "The output column name already exists.", column
            )
        added[name] = pd.Series(outlier_mask(df, column, bound), index=df.index)
    return with_columns(df, added)


def integer_bounds(bound: list[float]) -> tuple[int, int]:
    """Whole-number bounds for integer columns: rounded inward, or outward when no whole
    number lies between them (so the column keeps its integer type)."""
    low, high = math.ceil(bound[0]), math.floor(bound[1])
    if low > high:
        low, high = math.floor(bound[0]), math.ceil(bound[1])
    return low, high


def apply_outlier_cap(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    capped = {}
    for column, bound in zip(columns, state["bounds"], strict=True):
        numeric_matrix(df, [column])  # refuses infinite values
        if bound is None:
            continue
        series = df[column]
        if pd.api.types.is_integer_dtype(series):
            low, high = integer_bounds(bound)
            capped[column] = series.clip(lower=low, upper=high)
        else:
            capped[column] = series.clip(lower=bound[0], upper=bound[1])
    return with_columns(df, capped)


# ---- Outliers: robust transformation ----------------------------------------------------------


def apply_signed_log(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = {}
    for column in columns:
        x = numeric_matrix(df, [column])[:, 0]
        out[column] = pd.Series(np.sign(x) * np.log1p(np.abs(x)), index=df.index, dtype="float64")
    return with_columns(df, out)


# ---- Outliers: Isolation Forest ---------------------------------------------------------------


def fit_isolation_forest(
    fit_df: pd.DataFrame,
    columns: list[str],
    n_estimators: int,
    contamination: float,
    seed: int,
    min_rows: int,
) -> dict[str, Any]:
    """Fit on fitting rows with a value in every selected column (rows with gaps are not
    used: how the trees treat missing values is not a documented anomaly semantics).

    ``contamination`` is the share of fitting rows to flag: the forest ranks rows and this
    share sets the threshold. Columns constant in the fitting rows cannot separate rows and
    are left out; if none vary, there is no model and nothing is flagged."""
    matrix = numeric_matrix(fit_df, columns)
    complete = ~np.isnan(matrix).any(axis=1)
    if int(complete.sum()) < min_rows:
        raise PreprocessError(
            "FIT_TOO_FEW_VALUES",
            f"Isolation Forest needs at least {min_rows} fitting rows with a value in every "
            "selected column.",
        )
    rows = matrix[complete]
    used = [c for j, c in enumerate(columns) if rows[:, j].min() < rows[:, j].max()]
    model = None
    if used:
        model = IsolationForest(
            n_estimators=n_estimators, contamination=contamination, random_state=seed
        )
        # fit() also scores the fitting rows to set the threshold: same memory cap as apply.
        with config_context(working_memory=KNN_WORKING_MEMORY_MB):
            model.fit(rows[:, [columns.index(c) for c in used]])
    return {
        "model": model,
        "used": used,
        "n_complete": int(complete.sum()),
        "offset": None if model is None else float(model.offset_),
    }


def apply_isolation_forest(
    df: pd.DataFrame, columns: list[str], output_column: str, state: dict[str, Any]
) -> pd.DataFrame:
    if output_column in df.columns:
        raise PreprocessError("OUTPUT_COLUMN_CONFLICT", "The output column name already exists.")
    matrix = numeric_matrix(df, columns)
    complete = ~np.isnan(matrix).any(axis=1)
    flags = np.zeros(len(df), dtype=bool)
    if state["model"] is not None and complete.any():
        used = [columns.index(c) for c in state["used"]]
        with config_context(working_memory=KNN_WORKING_MEMORY_MB):
            flags[complete] = state["model"].predict(matrix[complete][:, used]) == -1
    return with_columns(df, {output_column: pd.Series(flags, index=df.index)})


# ---- Encoding ---------------------------------------------------------------------------------


def _present_values(series: pd.Series) -> pd.Series:
    """Observed values as Python objects (no unused categories, no missing markers)."""
    return series[series.notna()].astype(object)


def fit_onehot(
    fit_df: pd.DataFrame, columns: list[str], max_categories: int, name_by: str
) -> dict[str, Any]:
    """Categories per column from the fitting rows: the ``max_categories`` most frequent
    (ties by text), output in text order. Rarer categories are grouped as 'other'. Output
    names are fixed here, so every transform produces the same schema."""
    kept: list[list[Any]] = []
    seen: list[list[Any]] = []
    other: list[bool] = []
    names: list[list[str]] = []
    fallback: list[bool] = []
    for column in columns:
        present = _present_values(fit_df[column])
        if present.empty:
            raise PreprocessError(
                "FIT_COLUMN_EMPTY",
                "The column has no observed value in the fitting rows, so nothing can be "
                "learned from it.",
                column,
            )
        counts = present.value_counts()
        ranked = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        top = [value for value, _ in ranked[:max_categories]]
        kept.append(sorted(top, key=str))
        seen.append(sorted(counts.index.tolist(), key=str))
        other.append(len(ranked) > max_categories)
        column_names, used_fallback = onehot_names(column, kept[-1], other[-1], name_by)
        names.append(column_names)
        fallback.append(used_fallback)
    return {
        "categories": kept,
        "seen": seen,
        "other": other,
        "names": names,
        "fallback": fallback,
    }


def _unsafe_label(label: str, name: str) -> bool:
    return (
        not label.strip()
        or label == ONEHOT_OTHER
        or len(name) > MAX_NAME_LENGTH
        or any(ord(ch) < 32 or ord(ch) == 127 for ch in label)
    )


def onehot_names(
    column: str, categories: list[Any], other: bool, name_by: str
) -> tuple[list[str], bool]:
    """Output names and whether position naming was used as a fallback. Value names are
    used only when every label is safe: no control characters, not blank, not '__other__',
    unique as text, and within the column-name length limit."""
    fallback = False
    if name_by == "value":
        labels = [str(value) for value in categories]
        names = [f"{column}{ONEHOT_SEPARATOR}{label}" for label in labels]
        if len(set(labels)) != len(labels) or any(
            _unsafe_label(label, name) for label, name in zip(labels, names, strict=True)
        ):
            fallback = True
    if name_by == "position" or fallback:
        names = [f"{column}{ONEHOT_SEPARATOR}{i}" for i in range(1, len(categories) + 1)]
    if other:
        names.append(f"{column}{ONEHOT_SEPARATOR}{ONEHOT_OTHER}")
    return names, fallback


def apply_onehot(
    df: pd.DataFrame,
    columns: list[str],
    state: dict[str, Any],
    handle_unknown: str,
) -> pd.DataFrame:
    """Replace each column, in place, by its output columns (stable order, names fixed at
    fit). Categories never seen in the fitting rows raise with 'error'; with 'ignore' they
    go to the 'other' column when one exists, else all zeros. Rare categories that were
    seen go to 'other'."""
    position = {c: j for j, c in enumerate(columns)}
    pieces: list[tuple[str, Any]] = []
    for name in df.columns:
        if name not in position:
            pieces.append((name, df[name]))
            continue
        j = position[name]
        categories, other = state["categories"][j], state["other"][j]
        series = df[name]
        values = series.astype(object)
        present = series.notna().to_numpy()
        known = np.zeros(len(df), dtype=bool)
        hits = []
        for value in categories:
            hit = present & (values == value).to_numpy(dtype=bool)
            hits.append(hit)
            known |= hit
        unseen = present & ~values.isin(state["seen"][j]).to_numpy(dtype=bool)
        if unseen.any() and handle_unknown == "error":
            raise PreprocessError(
                "UNKNOWN_CATEGORY", "A category not seen in the fitting rows was found.", name
            )
        if other:
            hits.append(present & ~known)
        for out_name, hit in zip(state["names"][j], hits, strict=True):
            pieces.append((out_name, pd.Series(hit, index=df.index)))
    all_names = [n for n, _ in pieces]
    if len(set(all_names)) != len(all_names):
        raise PreprocessError(
            "OUTPUT_COLUMN_CONFLICT", "One-hot output names collide with other columns."
        )
    return pd.DataFrame(dict(pieces), index=df.index)


def fit_ordinal(fit_df: pd.DataFrame, column: str, order: list[str]) -> dict[str, Any]:
    """Nothing is learned (the order is given); the fitting rows must not contain a value
    missing from the order, because that means the order is incomplete."""
    present = _present_values(fit_df[column])
    if (~present.isin(order)).any():
        raise PreprocessError(
            "ORDER_INCOMPLETE",
            "The fitting rows contain a category that is not in the given order.",
            column,
        )
    return {}


def apply_ordinal(
    df: pd.DataFrame, column: str, order: list[str], handle_unknown: str
) -> pd.DataFrame:
    series = df[column]
    codes = series.astype(object).map({value: i for i, value in enumerate(order)})
    unknown = series.notna() & codes.isna()
    if unknown.any() and handle_unknown == "error":
        raise PreprocessError("UNKNOWN_CATEGORY", "A value is not in the given order.", column)
    return with_columns(df, {column: codes.astype("float64").astype("Int64")})


def fit_frequency(fit_df: pd.DataFrame, columns: list[str]) -> dict[str, Any]:
    tables: list[dict[Any, float]] = []
    tied: list[int] = []
    for column in columns:
        present = _present_values(fit_df[column])
        if present.empty:
            raise PreprocessError(
                "FIT_COLUMN_EMPTY",
                "The column has no observed value in the fitting rows, so nothing can be "
                "learned from it.",
                column,
            )
        shares = present.value_counts(normalize=True)
        tables.append({key: float(share) for key, share in shares.items()})
        tied.append(int(shares.duplicated(keep=False).sum()))
    return {"tables": tables, "tied": tied}


def apply_frequency(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    out = {}
    for column, table in zip(columns, state["tables"], strict=True):
        series = df[column]
        mapped = series.astype(object).map(table).astype("float64")
        unseen = series.notna().to_numpy() & mapped.isna().to_numpy()
        out[column] = pd.Series(
            np.where(unseen, 0.0, mapped.to_numpy()), index=df.index, dtype="float64"
        )
    return with_columns(df, out)


# ---- Scaling ----------------------------------------------------------------------------------


def fit_scaler(
    fit_df: pd.DataFrame, columns: list[str], kind: str, clip: bool = False
) -> dict[str, Any]:
    matrix = numeric_matrix(fit_df, columns)
    require_observed(matrix, columns)
    scaler: Any
    if kind == "standard":
        scaler = StandardScaler()
    elif kind == "minmax":
        scaler = MinMaxScaler(clip=clip)
    elif kind == "robust":
        scaler = RobustScaler()
    else:
        raise PreprocessError("OP_INVALID", "Unknown scaler.")
    with np.errstate(all="ignore"):
        scaler.fit(matrix)
    for attribute in (
        "mean_",
        "scale_",
        "var_",
        "data_min_",
        "data_max_",
        "data_range_",
        "center_",
    ):
        learned = getattr(scaler, attribute, None)
        if learned is not None:
            for column, value in zip(columns, np.atleast_1d(learned), strict=True):
                require_finite(value, column)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if kind == "robust":
            q1, q3 = np.nanquantile(matrix, [0.25, 0.75], axis=0)
            zero = q3 - q1 == 0
        else:
            zero = np.nanmax(matrix, axis=0) - np.nanmin(matrix, axis=0) == 0
    return {"scaler": scaler, "zero_spread": [bool(v) for v in zero]}


def apply_scaler(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    matrix = numeric_matrix(df, columns)
    with np.errstate(all="ignore"):
        scaled = matrix if matrix.shape[0] == 0 else state["scaler"].transform(matrix)
    for j, column in enumerate(columns):
        require_finite(scaled[~np.isnan(matrix[:, j]), j], column)
    return with_columns(
        df,
        {
            c: pd.Series(scaled[:, j], index=df.index, dtype="float64")
            for j, c in enumerate(columns)
        },
    )


def apply_no_scaling(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy()


# ---- Distribution transforms ------------------------------------------------------------------


def _observed(values: np.ndarray) -> np.ndarray:
    return values[~np.isnan(values)]


def apply_log(df: pd.DataFrame, columns: list[str], base: str) -> pd.DataFrame:
    """log(x) for x > 0. A zero or negative value fails (never becomes a missing value)."""
    out = {}
    for column in columns:
        x = numeric_matrix(df, [column])[:, 0]
        if (_observed(x) <= 0).any():
            raise PreprocessError(
                "VALUE_OUT_OF_DOMAIN",
                "The logarithm is defined only for values greater than zero, and the column "
                "contains zero or negative values. Use log1p or Yeo-Johnson, or shift the column.",
                column,
            )
        with np.errstate(all="ignore"):
            y = np.log(x) if base == "e" else np.log2(x) if base == "2" else np.log10(x)
        out[column] = pd.Series(y, index=df.index, dtype="float64")
    return with_columns(df, out)


def apply_log1p(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """log(1 + x) for x > -1. A value of -1 or below fails."""
    out = {}
    for column in columns:
        x = numeric_matrix(df, [column])[:, 0]
        if (_observed(x) <= -1).any():
            raise PreprocessError(
                "VALUE_OUT_OF_DOMAIN",
                "log(1 + x) is defined only for values greater than -1, and the column contains "
                "values of -1 or below. Use Yeo-Johnson, or shift the column.",
                column,
            )
        with np.errstate(all="ignore"):
            out[column] = pd.Series(np.log1p(x), index=df.index, dtype="float64")
    return with_columns(df, out)


def _require_values(values: np.ndarray, column: str, min_values: int) -> np.ndarray:
    observed = _observed(values)
    if observed.size < min_values:
        raise PreprocessError(
            "FIT_TOO_FEW_VALUES",
            f"This step needs at least {min_values} observed values in the fitting rows.",
            column,
        )
    return observed


def fit_yeo_johnson(fit_df: pd.DataFrame, columns: list[str], min_values: int) -> dict[str, Any]:
    """One lambda per column by maximum likelihood (scikit-learn PowerTransformer, not
    standardised). A column constant in the fitting rows keeps lambda = 1 (the identity)."""
    lambdas: list[float] = []
    constant: list[bool] = []
    n_values: list[int] = []
    fit_range: list[list[float]] = []  # kept in memory only (fitting values; never reported)
    for column in columns:
        observed = _require_values(numeric_matrix(fit_df, [column])[:, 0], column, min_values)
        n_values.append(int(observed.size))
        fit_range.append([float(observed.min()), float(observed.max())])
        if observed.min() == observed.max():
            lambdas.append(1.0)
            constant.append(True)
            continue
        with np.errstate(all="ignore"):
            model = PowerTransformer(method="yeo-johnson", standardize=False)
            model.fit(observed.reshape(-1, 1))
        lam = float(model.lambdas_[0])
        require_finite(lam, column)
        lambdas.append(lam)
        constant.append(False)
    return {"lambdas": lambdas, "constant": constant, "n_values": n_values, "fit_range": fit_range}


def yeo_johnson(x: np.ndarray, lam: float) -> np.ndarray:
    """The Yeo-Johnson transformation with a given lambda (missing values stay missing)."""
    if lam == 1.0:  # the identity on both branches; exact, not up to rounding
        return x.astype("float64", copy=True)
    out = np.full(x.shape, np.nan)
    pos = x >= 0
    neg = x < 0  # NaN is in neither
    eps = np.spacing(1.0)
    with np.errstate(all="ignore"):
        if abs(lam) < eps:
            out[pos] = np.log1p(x[pos])
        else:
            out[pos] = np.expm1(lam * np.log1p(x[pos])) / lam
        if abs(lam - 2) < eps:
            out[neg] = -np.log1p(-x[neg])
        else:
            out[neg] = -np.expm1((2 - lam) * np.log1p(-x[neg])) / (2 - lam)
    return out


def apply_yeo_johnson(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    out = {}
    for column, lam in zip(columns, state["lambdas"], strict=True):
        x = numeric_matrix(df, [column])[:, 0]
        y = yeo_johnson(x, float(lam))
        require_finite(y[~np.isnan(x)], column)
        out[column] = pd.Series(y, index=df.index, dtype="float64")
    return with_columns(df, out)


def normal_output_bounds() -> tuple[float, float]:
    """The ends of scikit-learn's normal quantile output (about -5.2 and 5.2)."""
    from scipy.stats import norm

    threshold = 1e-7 - np.spacing(1)
    return float(norm.ppf(threshold)), float(norm.ppf(1 - threshold))


def fit_quantile(
    fit_df: pd.DataFrame,
    columns: list[str],
    n_quantiles: int,
    output_distribution: str,
    seed: int,
    subsample: int,
    min_values: int,
) -> dict[str, Any]:
    """One QuantileTransformer per column, fitted on that column's observed values. The
    number of quantiles is reduced to the number of observed values when larger."""
    models: list[Any] = []
    constant_value: list[float | None] = []
    used_quantiles: list[int] = []
    n_values: list[int] = []
    fit_range: list[list[float]] = []
    for column in columns:
        observed = _require_values(numeric_matrix(fit_df, [column])[:, 0], column, min_values)
        n_values.append(int(observed.size))
        fit_range.append([float(observed.min()), float(observed.max())])
        if observed.min() == observed.max():
            models.append(None)
            constant_value.append(float(observed[0]))
            used_quantiles.append(0)
            continue
        # scikit-learn estimates at most as many quantiles as rows it samples.
        quantiles = min(n_quantiles, int(observed.size), subsample)
        model = QuantileTransformer(
            n_quantiles=quantiles,
            output_distribution=output_distribution,
            subsample=subsample,
            random_state=seed,
        )
        model.fit(observed.reshape(-1, 1))
        models.append(model)
        constant_value.append(None)
        used_quantiles.append(quantiles)
    return {
        "models": models,
        "constant_value": constant_value,
        "n_quantiles": used_quantiles,
        "n_values": n_values,
        "fit_range": fit_range,
        "output_distribution": output_distribution,
    }


def apply_quantile(df: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:
    if state["output_distribution"] == "normal":
        low, high = normal_output_bounds()
        middle = 0.0
    else:
        low, high, middle = 0.0, 1.0, 0.5
    out = {}
    for j, column in enumerate(columns):
        x = numeric_matrix(df, [column])[:, 0]
        model = state["models"][j]
        if model is None:  # constant in the fitting rows: below, at, above that value
            value = state["constant_value"][j]
            with np.errstate(invalid="ignore"):
                y = np.where(x < value, low, np.where(x > value, high, middle))
            y = np.where(np.isnan(x), np.nan, y)
        elif x.size == 0:
            y = x
        else:
            y = quantile_map(x, model, state["output_distribution"], low, high)
        out[column] = pd.Series(y, index=df.index, dtype="float64")
    return with_columns(df, out)


def quantile_map(
    x: np.ndarray, model: Any, output_distribution: str, low: float, high: float
) -> np.ndarray:
    """Map values with the fitted quantiles. Like scikit-learn, the forward and backward
    interpolations are averaged, so tied values get their middle rank, and values outside
    the fitting range go to the ends. Unlike scikit-learn, values tied with the fitting
    minimum or maximum are not forced to the end: in a column that is 40% zeros, the zeros
    map to their middle rank (about 0.2, or -0.84 with a normal output), not to 0 / -5.2."""
    quantiles = model.quantiles_[:, 0]
    references = model.references_
    with np.errstate(invalid="ignore"):
        u = 0.5 * (
            np.interp(x, quantiles, references) - np.interp(-x, -quantiles[::-1], -references[::-1])
        )
    if output_distribution != "normal":
        return np.asarray(u, dtype="float64")
    from scipy.stats import norm

    with np.errstate(divide="ignore", invalid="ignore"):
        z = norm.ppf(u)
    return np.where(np.isnan(x), np.nan, np.clip(z, low, high))


# ---- Feature selection ------------------------------------------------------------------------


def apply_drop_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove columns (model input only); the remaining columns keep their order."""
    return df.drop(columns=list(columns))


def _decide(
    columns: list[str],
    fails: list[bool],
    keep: list[str],
    role_kept: list[str],
    action: str,
    not_assessed: list[bool] | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Decision per column and the columns removed. Columns in ``keep`` or with a domain role
    and columns not assessed are never removed."""
    decisions: dict[str, str] = {}
    drop: list[str] = []
    for j, column in enumerate(columns):
        if not_assessed is not None and not_assessed[j]:
            decisions[column] = "not_assessed"
        elif not fails[j]:
            decisions[column] = "kept"
        elif column in role_kept:
            decisions[column] = "kept_domain_role"
        elif column in keep:
            decisions[column] = "kept_by_user"
        elif action == "drop":
            decisions[column] = "removed"
            drop.append(column)
        else:
            decisions[column] = "would_remove"
    return decisions, drop


def fit_variance(
    fit_df: pd.DataFrame,
    columns: list[str],
    threshold: float,
    action: str,
    keep: list[str],
    role_kept: list[str],
) -> dict[str, Any]:
    """Population variance (ddof=0) of each column's observed fitting values; true/false
    columns count as 0/1. A column whose observed values are all equal has variance 0."""
    variances: list[float] = []
    n_values: list[int] = []
    for column in columns:
        observed = _observed(numeric_matrix(fit_df, [column])[:, 0])
        if observed.size == 0:
            raise PreprocessError(
                "FIT_COLUMN_EMPTY",
                "The column has no observed value in the fitting rows, so nothing can be "
                "learned from it.",
                column,
            )
        if observed.min() == observed.max():
            variance = 0.0
        else:
            with np.errstate(all="ignore"):
                variance = float(np.var(observed))
            require_finite(variance, column)
        variances.append(variance)
        n_values.append(int(observed.size))
    fails = [v <= threshold for v in variances]
    decisions, drop = _decide(columns, fails, keep, role_kept, action)
    return {"variances": variances, "n_values": n_values, "decisions": decisions, "drop": drop}


def maxabs_scaled(values: np.ndarray) -> np.ndarray:
    """Divide each column by its largest absolute observed value. Correlations, rank tests,
    F statistics and mutual information do not depend on scale, and this keeps sums of
    squares finite for values near 1e300 and non-zero for values near 1e-300."""
    # Explicit shapes: reshape(0, -1) is ambiguous for an empty fold.
    matrix = (values.reshape(-1, 1) if values.ndim == 1 else values).astype("float64", copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if len(matrix):
            top = np.nanmax(np.abs(matrix), axis=0)
        else:
            top = np.zeros(matrix.shape[1])
    top = np.where(np.isfinite(top) & (top > 0), top, 1.0)
    scaled: np.ndarray = (matrix / top).reshape(values.shape)
    return scaled


def pairwise_correlation(matrix: np.ndarray, min_pairs: int, chunk_rows: int = 8192) -> np.ndarray:
    """Pearson correlation of every column pair over the rows where both are observed
    (NaN when fewer than ``min_pairs`` such rows or no spread). Columns are standardised
    first and the sums are accumulated in row chunks, so memory stays small."""
    n, p = matrix.shape
    matrix = maxabs_scaled(matrix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        centre = np.nanmean(matrix, axis=0) if n else np.zeros(p)
        spread = np.nanstd(matrix, axis=0) if n else np.ones(p)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    spread = np.where(np.isfinite(spread) & (spread > 0), spread, 1.0)
    count = np.zeros((p, p))
    sum_x = np.zeros((p, p))  # sum_x[i, j]: sum of z_i over rows where i and j are observed
    sum_xx = np.zeros((p, p))
    sum_xy = np.zeros((p, p))
    for start in range(0, n, chunk_rows):
        block = (matrix[start : start + chunk_rows] - centre) / spread
        mask = (~np.isnan(block)).astype("float64")
        z = np.where(mask > 0, block, 0.0)
        count += mask.T @ mask
        sum_x += z.T @ mask
        sum_xx += (z * z).T @ mask
        sum_xy += z.T @ z
    with np.errstate(all="ignore"):
        cov = sum_xy - sum_x * sum_x.T / count
        var_i = sum_xx - sum_x**2 / count
        var_j = var_i.T
        tiny = 1e-12 * count
        r = cov / np.sqrt(var_i * var_j)
    r = np.where((count >= min_pairs) & (var_i > tiny) & (var_j > tiny), r, np.nan)
    return np.clip(r, -1.0, 1.0)


def fit_correlation(
    fit_df: pd.DataFrame,
    columns: list[str],
    threshold: float,
    method: str,
    keep: list[str],
    role_kept: list[str],
    min_pairs: int,
) -> dict[str, Any]:
    """Pairs with |r| >= threshold and one proposed removal per pair (the column with the
    higher mean |r| to the other columns; a column in ``keep`` or with a domain role is never
    proposed). Spearman ranks each column over its observed values, then correlates ranks.
    Nothing is removed."""
    matrix = numeric_matrix(fit_df, columns)
    if method == "spearman":
        matrix = pd.DataFrame(matrix).rank(method="average").to_numpy(dtype="float64")
    r = pairwise_correlation(matrix, min_pairs)
    p = len(columns)
    absolute = np.abs(r)
    off = ~np.eye(p, dtype=bool)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean_abs = np.array(
            [
                np.nanmean(np.where(off[i], absolute[i], np.nan)) if p > 1 else np.nan
                for i in range(p)
            ]
        )
    pairs = []
    not_assessed = 0
    for i in range(p):
        for j in range(i + 1, p):
            if np.isnan(r[i, j]):
                not_assessed += 1
            elif absolute[i, j] >= threshold:
                pairs.append((i, j, float(r[i, j])))
    pairs.sort(key=lambda t: (-abs(t[2]), t[0], t[1]))
    protected = set(keep) | set(role_kept)
    proposed: list[str] = []
    out_pairs = []
    for i, j, value in pairs:
        a, b = columns[i], columns[j]
        choice = None
        if a not in proposed and b not in proposed:
            candidates = [c for c in (i, j) if columns[c] not in protected]
            if len(candidates) == 1:
                choice = columns[candidates[0]]
            elif candidates:
                mi = -1.0 if np.isnan(mean_abs[i]) else mean_abs[i]
                mj = -1.0 if np.isnan(mean_abs[j]) else mean_abs[j]
                choice = a if mi > mj else b
            if choice is not None:
                proposed.append(choice)
        out_pairs.append({"columns": [a, b], "correlation": value, "proposed_removal": choice})
    return {
        "pairs": out_pairs,
        "proposed": proposed,
        "max_abs_correlation": [
            None if np.all(np.isnan(absolute[i][off[i]])) else float(np.nanmax(absolute[i][off[i]]))
            for i in range(p)
        ]
        if p > 1
        else [None] * p,
        "pairs_not_assessed": not_assessed,
    }


def discrete_codes(series: pd.Series) -> np.ndarray:
    """Integer codes for a discrete column (categories ordered by text); -1 for missing."""
    present = series.notna().to_numpy()
    values = series.astype(object).to_numpy()
    labels = sorted({str(v) for v in values[present]})
    lookup = {label: i for i, label in enumerate(labels)}
    codes = np.full(len(series), -1, dtype="int64")
    codes[present] = [lookup[str(v)] for v in values[present]]
    return codes


def target_values(fit_df: pd.DataFrame, target: str, task: str) -> tuple[np.ndarray, np.ndarray]:
    """(values, observed mask). Classes become integer codes; a numeric target is float."""
    series = fit_df[target]
    if task == "regression":
        y = numeric_matrix(fit_df, [target])[:, 0]
        return maxabs_scaled(y), ~np.isnan(y)
    codes = discrete_codes(series)
    return codes, codes >= 0


def _feature_values(
    fit_df: pd.DataFrame, column: str, discrete: bool
) -> tuple[np.ndarray, np.ndarray]:
    if discrete:
        codes = discrete_codes(fit_df[column])
        return codes, codes >= 0
    x = numeric_matrix(fit_df, [column])[:, 0]
    return maxabs_scaled(x), ~np.isnan(x)


def _rank_and_decide(
    columns: list[str],
    scores: list[float | None],
    order_key: list[tuple[float, float] | None],
    k: int,
    action: str,
    keep: list[str],
    role_kept: list[str],
) -> tuple[list[int | None], dict[str, str], list[str]]:
    """Rank assessed columns by ``order_key`` (smaller first, ties by column order); the k
    best are kept, the others fail the criterion."""
    keyed = [(key, j) for j, key in enumerate(order_key) if key is not None]
    rank: list[int | None] = [None] * len(columns)
    for position, (_, j) in enumerate(sorted(keyed), start=1):
        rank[j] = position
    fails = [(r is not None and r > k) for r in rank]
    not_assessed = [scores[j] is None for j in range(len(columns))]
    decisions, drop = _decide(columns, fails, keep, role_kept, action, not_assessed)
    return rank, decisions, drop


def fit_mutual_information(
    fit_df: pd.DataFrame,
    columns: list[str],
    target: str,
    task: str,
    discrete: list[bool],
    n_neighbors: int,
    k: int,
    action: str,
    keep: list[str],
    role_kept: list[str],
    seed: int,
    min_rows: int,
) -> dict[str, Any]:
    """Mutual information of each column with the target, each on the rows where both are
    observed (scikit-learn's k-nearest-neighbour estimator, seeded)."""
    y, y_observed = target_values(fit_df, target, task)
    scores: list[float | None] = []
    reasons: list[str | None] = []
    n_rows: list[int] = []
    near_perfect: list[bool] = []
    for column, is_discrete in zip(columns, discrete, strict=True):
        x, x_observed = _feature_values(fit_df, column, is_discrete)
        rows = x_observed & y_observed
        n_rows.append(int(rows.sum()))
        xv, yv = x[rows], y[rows]
        reason = None
        if rows.sum() < min_rows:
            reason = "too_few_rows"
        elif np.unique(xv).size < 2:
            reason = "constant"
        elif task != "regression" and np.unique(yv).size < 2:
            reason = "single_class"
        elif task == "regression" and np.unique(yv).size < 2:
            reason = "constant_target"
        if reason is not None:
            scores.append(None)
            reasons.append(reason)
            near_perfect.append(False)
            continue
        feature = xv.reshape(-1, 1).astype("float64")
        if task == "regression":
            score = mutual_info_regression(
                feature,
                yv,
                discrete_features=[is_discrete],
                n_neighbors=n_neighbors,
                random_state=seed,
            )[0]
        else:
            score = mutual_info_classif(
                feature,
                yv,
                discrete_features=[is_discrete],
                n_neighbors=n_neighbors,
                random_state=seed,
            )[0]
        value = float(max(score, 0.0))
        scores.append(value)
        reasons.append(None)
        if task == "regression":
            near_perfect.append(False)
        else:  # MI close to the target's entropy: the column almost determines the target
            shares = np.bincount(yv) / len(yv)
            shares = shares[shares > 0]
            entropy = float(-(shares * np.log(shares)).sum())
            near_perfect.append(entropy > 0 and value >= 0.9 * entropy)
    order_key = [None if s is None else (-s, 0.0) for s in scores]
    rank, decisions, drop = _rank_and_decide(columns, scores, order_key, k, action, keep, role_kept)
    return {
        "scores": scores,
        "reasons": reasons,
        "near_perfect": near_perfect,
        "n_rows": n_rows,
        "rank": rank,
        "decisions": decisions,
        "drop": drop,
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (same order as the input)."""
    m = len(p_values)
    if m == 0:
        return []
    order = np.argsort(p_values, kind="stable")
    ranked = np.asarray(p_values, dtype="float64")[order] * m / np.arange(1, m + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adjusted, 1.0)
    return [float(v) for v in out]


def _skew(values: np.ndarray) -> float:
    s = pd.Series(values).skew()
    return 0.0 if pd.isna(s) else float(s)


def univariate_test(
    test: str, x: np.ndarray, y: np.ndarray, max_levels: int
) -> tuple[float | None, float | None, str | None, list[str]]:
    """(statistic, p-value, reason not assessed, doubtful assumptions) for one column on the
    rows where it and the target are observed. Classes are integer codes."""
    from scipy import stats

    doubts: list[str] = []
    if test in ("anova_f", "kruskal"):
        groups = [x[y == c] for c in np.unique(y)]
        groups = [g for g in groups if g.size > 0]
        if len(groups) < 2:
            return None, None, "single_class", doubts
        if np.unique(x).size < 2:
            return None, None, "constant", doubts
        if test == "anova_f":
            if all(np.unique(g).size < 2 for g in groups):
                return None, None, "no_within_class_variation", doubts
            if any(g.size >= 8 and abs(_skew(g)) > 2 for g in groups):
                doubts.append("normality")
            variances = [float(np.var(g, ddof=1)) for g in groups if g.size >= 2]
            if variances and (min(variances) == 0 or max(variances) / min(variances) > 4):
                doubts.append("equal_variances")
            with np.errstate(all="ignore"):
                result = stats.f_oneway(*groups)
        else:
            result = stats.kruskal(*groups)
        return float(result.statistic), float(result.pvalue), None, doubts
    if test == "chi2":
        levels = np.unique(x)
        if levels.size < 2:
            return None, None, "constant", doubts
        if levels.size > max_levels:
            return None, None, "too_many_levels", doubts
        if np.unique(y).size < 2:
            return None, None, "single_class", doubts
        table = pd.crosstab(x, y).to_numpy()
        result = stats.chi2_contingency(table)
        expected = np.asarray(result.expected_freq)
        if (expected < 1).any() or (expected < 5).mean() > 0.2:
            doubts.append("expected_counts")
        return float(result.statistic), float(result.pvalue), None, doubts
    # Regression tests: numeric column, numeric target.
    if np.unique(x).size < 2:
        return None, None, "constant", doubts
    if np.unique(y).size < 2:
        return None, None, "constant_target", doubts
    if test == "f_regression":
        if abs(_skew(x)) > 2 or abs(_skew(y)) > 2:
            doubts.append("linearity_outliers")
        statistic, p_value = f_regression(x.reshape(-1, 1), y)
        return float(statistic[0]), float(p_value[0]), None, doubts
    result = stats.spearmanr(x, y)
    return float(abs(result.statistic)), float(result.pvalue), None, doubts


def fit_univariate(
    fit_df: pd.DataFrame,
    columns: list[str],
    target: str,
    task: str,
    test: str,
    k: int,
    action: str,
    keep: list[str],
    role_kept: list[str],
    min_rows: int,
    max_levels: int,
) -> dict[str, Any]:
    y, y_observed = target_values(fit_df, target, task)
    statistics: list[float | None] = []
    p_values: list[float | None] = []
    reasons: list[str | None] = []
    doubts: list[list[str]] = []
    n_rows: list[int] = []
    for column in columns:
        x, x_observed = _feature_values(fit_df, column, test == "chi2")
        rows = x_observed & y_observed
        n_rows.append(int(rows.sum()))
        statistic: float | None
        p_value: float | None
        reason: str | None
        doubt: list[str]
        if rows.sum() < min_rows:
            statistic, p_value, reason, doubt = None, None, "too_few_rows", []
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                statistic, p_value, reason, doubt = univariate_test(
                    test, x[rows], y[rows], max_levels
                )
        if reason is None and (
            statistic is None
            or p_value is None
            or not math.isfinite(statistic)
            or not math.isfinite(p_value)
        ):
            statistic, p_value, reason = None, None, "undefined"
        statistics.append(statistic)
        p_values.append(p_value)
        reasons.append(reason)
        doubts.append(doubt)
    assessed = [(j, p) for j, p in enumerate(p_values) if p is not None]
    adjusted_list = benjamini_hochberg([p for _, p in assessed])
    adjusted: list[float | None] = [None] * len(columns)
    for (j, _), value in zip(assessed, adjusted_list, strict=True):
        adjusted[j] = value
    order_key: list[tuple[float, float] | None] = []
    for p_value_j, statistic_j in zip(p_values, statistics, strict=True):
        if p_value_j is None or statistic_j is None:
            order_key.append(None)
        else:
            order_key.append((p_value_j, -statistic_j))
    rank, decisions, drop = _rank_and_decide(
        columns, statistics, order_key, k, action, keep, role_kept
    )
    return {
        "statistics": statistics,
        "p_values": p_values,
        "adjusted_p_values": adjusted,
        "reasons": reasons,
        "doubts": doubts,
        "n_rows": n_rows,
        "rank": rank,
        "decisions": decisions,
        "drop": drop,
    }


# ---- Class imbalance (training rows only) ------------------------------------------------------


def _native_label(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def class_labels(df: pd.DataFrame, target: str) -> tuple[list[Any], np.ndarray]:
    """(classes in a stable order, integer code per row). Missing labels are refused."""
    series = df[target]
    if series.isna().any():
        raise PreprocessError(
            "TARGET_MISSING",
            "The target is missing in some training rows; resampling and class weights need a "
            "label for every training row. Remove or label those rows first.",
            target,
        )
    values = [_native_label(v) for v in series.astype(object).to_numpy()]
    unique = list(dict.fromkeys(values))
    try:
        classes = sorted(unique)
    except TypeError:
        classes = sorted(unique, key=lambda v: (type(v).__name__, str(v)))
    lookup = {c: i for i, c in enumerate(classes)}
    codes = np.fromiter((lookup[v] for v in values), dtype="int64", count=len(values))
    return classes, codes


def fit_class_counts(fit_df: pd.DataFrame, target: str) -> dict[str, Any]:
    classes, codes = class_labels(fit_df, target)
    if len(classes) < 2:
        raise PreprocessError(
            "TARGET_SINGLE_CLASS",
            "The training rows contain a single class; there is no imbalance to correct.",
            target,
        )
    counts = np.bincount(codes, minlength=len(classes))
    return {"classes": classes, "counts": [int(c) for c in counts]}


def class_weights(counts: list[int]) -> list[float]:
    """'balanced' weights: n_rows / (n_classes x class count)."""
    total, k = sum(counts), len(counts)
    return [total / (k * c) for c in counts]


def resample_class_weight(
    df: pd.DataFrame, target: str, state: dict[str, Any], weight_column: str
) -> pd.DataFrame:
    if weight_column in df.columns:
        raise PreprocessError("OUTPUT_COLUMN_CONFLICT", "The weight column name already exists.")
    classes, codes = class_labels(df, target)
    lookup = dict(zip(state["classes"], class_weights(state["counts"]), strict=True))
    weights = np.array([lookup[c] for c in classes], dtype="float64")[codes]
    return with_columns(df, {weight_column: pd.Series(weights, index=df.index, dtype="float64")})


IMBLEARN_STRATEGY = {
    "not_majority": "not majority",
    "minority": "minority",
    "not_minority": "not minority",
    "majority": "majority",
}


def expected_rows(counts: list[int], method: str, strategy: str) -> int:
    largest, smallest = max(counts), min(counts)
    if method in ("oversample", "smote"):
        if strategy == "minority":
            return sum(counts) + largest - smallest
        return largest * len(counts)
    if strategy == "majority":
        return sum(counts) - largest + smallest
    return smallest * len(counts)


def oversampled_classes(counts: list[int], strategy: str) -> list[int]:
    largest = max(counts)
    if strategy == "minority":
        return [int(np.argmin(counts))]
    return [j for j, c in enumerate(counts) if c < largest]


def check_output_size(rows: int, columns: int, max_rows: int, max_cells: int) -> None:
    """Refuse before resampling when the training table would be too large."""
    if rows > max_rows:
        raise PreprocessError(
            "OP_LIMIT_EXCEEDED", f"Resampling would produce more than {max_rows:,} training rows."
        )
    if rows * columns > max_cells:
        raise PreprocessError(
            "OP_LIMIT_EXCEEDED",
            f"Resampling would produce more than {max_cells:,} cells (rows x columns). Select "
            "fewer columns or use class weights.",
        )


def _nullable(series: pd.Series) -> pd.Series:
    """Integer and boolean columns as nullable types, so added rows can be missing."""
    if pd.api.types.is_bool_dtype(series) and not isinstance(series.dtype, pd.BooleanDtype):
        return series.astype("boolean")
    if pd.api.types.is_integer_dtype(series) and not isinstance(series.dtype, pd.Int64Dtype):
        return series.astype("Int64")
    return series


def _added_row_ids(n_added: int) -> np.ndarray:
    return -np.arange(1, n_added + 1, dtype="int64")


def resample_random(
    df: pd.DataFrame,
    target: str,
    method: str,
    strategy: str,
    seed: int,
    max_output_rows: int,
    max_output_cells: int,
) -> pd.DataFrame:
    """Random over- or undersampling of whole rows (imbalanced-learn, seeded). Original rows
    keep their identifier and order; added duplicates follow with negative identifiers."""
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler

    classes, codes = class_labels(df, target)
    counts = [int(c) for c in np.bincount(codes, minlength=len(classes))]
    if len(classes) < 2:
        raise PreprocessError(
            "TARGET_SINGLE_CLASS", "The training rows contain a single class.", target
        )
    check_output_size(
        expected_rows(counts, method, strategy), df.shape[1], max_output_rows, max_output_cells
    )
    sampler: Any
    if method == "oversample":
        sampler = RandomOverSampler(
            sampling_strategy=IMBLEARN_STRATEGY[strategy], random_state=seed
        )
    else:
        sampler = RandomUnderSampler(
            sampling_strategy=IMBLEARN_STRATEGY[strategy], random_state=seed
        )
    positions = np.arange(len(df)).reshape(-1, 1)
    sampler.fit_resample(positions, codes)
    chosen = np.asarray(sampler.sample_indices_, dtype="int64")
    if method == "undersample":
        return df.iloc[np.sort(chosen)].reset_index(drop=True)
    seen = np.zeros(len(df), dtype=bool)
    added = []
    for i in chosen:
        if seen[i]:
            added.append(i)
        seen[i] = True
    if not seen.all():
        raise PreprocessError("POSTCONDITION_FAILED", "Oversampling lost an original row.")
    extra = df.iloc[added].copy()
    extra[ROW_ID] = _added_row_ids(len(added))
    return pd.concat([df, extra], ignore_index=True)


def resample_smote(
    df: pd.DataFrame,
    target: str,
    passthrough: list[str],
    strategy: str,
    k_neighbors: int,
    seed: int,
    max_features: int,
    max_output_rows: int,
    max_output_cells: int,
) -> pd.DataFrame:
    """SMOTE on every feature column (all columns except the row identifier, the target and
    the pass-through columns). Features must be numeric (not true/false) with no missing
    values. Synthetic rows follow the original rows with negative identifiers; their
    pass-through columns are missing."""
    from imblearn.over_sampling import SMOTE

    features = [c for c in df.columns if c not in (ROW_ID, target) and c not in passthrough]
    if not features:
        raise PreprocessError("NO_FEATURES", "SMOTE needs at least one feature column.")
    if len(features) > max_features:
        raise PreprocessError(
            "OP_LIMIT_EXCEEDED", f"SMOTE uses at most {max_features} feature columns."
        )
    for column in features:
        series = df[column]
        if pd.api.types.is_bool_dtype(series) or not pd.api.types.is_numeric_dtype(series):
            raise PreprocessError(
                "FEATURE_NOT_NUMERIC",
                "SMOTE interpolates between rows, so every feature column must be numeric. "
                "Encode categorical columns numerically or use random oversampling.",
                column,
            )
    matrix = numeric_matrix(df, features)
    for j, column in enumerate(features):
        if np.isnan(matrix[:, j]).any():
            raise PreprocessError(
                "FEATURE_MISSING_VALUES",
                "SMOTE cannot interpolate missing values; impute the column first.",
                column,
            )
    classes, codes = class_labels(df, target)
    counts = [int(c) for c in np.bincount(codes, minlength=len(classes))]
    if len(classes) < 2:
        raise PreprocessError(
            "TARGET_SINGLE_CLASS", "The training rows contain a single class.", target
        )
    for j in oversampled_classes(counts, strategy):
        if counts[j] <= k_neighbors:
            raise PreprocessError(
                "SMOTE_TOO_FEW_SAMPLES",
                f"A class to be oversampled has {counts[j]} training rows; SMOTE with "
                f"k_neighbors={k_neighbors} needs more than {k_neighbors}. Lower k_neighbors "
                "or use random oversampling.",
                target,
            )
    check_output_size(
        expected_rows(counts, "smote", strategy), df.shape[1], max_output_rows, max_output_cells
    )
    sampler = SMOTE(
        sampling_strategy=IMBLEARN_STRATEGY[strategy], k_neighbors=k_neighbors, random_state=seed
    )
    with config_context(working_memory=KNN_WORKING_MEMORY_MB):
        resampled, labels = sampler.fit_resample(matrix, codes)
    n = len(df)
    if not np.array_equal(resampled[:n], matrix) or not np.array_equal(labels[:n], codes):
        raise PreprocessError("POSTCONDITION_FAILED", "SMOTE changed an original training row.")
    new, new_codes = resampled[n:], labels[n:]
    require_finite(new, None)
    originals = df.copy()
    for column in features:
        originals[column] = originals[column].astype("float64")
    for column in passthrough:
        originals[column] = _nullable(originals[column])
    synthetic = originals.iloc[:0].reindex(range(len(new)))
    for j, column in enumerate(features):
        synthetic[column] = new[:, j]
    synthetic[ROW_ID] = _added_row_ids(len(new))
    synthetic[target] = pd.Series(
        [classes[c] for c in new_codes], dtype=object, index=synthetic.index
    ).astype(df[target].dtype)
    for column in synthetic.columns:
        synthetic[column] = synthetic[column].astype(originals[column].dtype)
        if synthetic[column].dtype == object and column != target:
            # Missing text is None in canonical tables (NaN would differ after a round trip).
            synthetic[column] = pd.Series([None] * len(new), index=synthetic.index, dtype=object)
    return pd.concat([originals, synthetic], ignore_index=True)

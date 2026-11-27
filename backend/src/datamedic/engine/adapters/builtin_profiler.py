"""BuiltinProfiler: descriptive per-column and dataset statistics with pandas/NumPy.

Descriptive only. It reports facts (counts, distributions, correlations) and never judges
severity; diagnostic findings with severities are a later phase (check framework).

Privacy: frequency tables are the only place cell values appear. They are capped
(``MAX_COMPLETE_VALUES`` values, each at most ``MAX_VALUE_CHARS`` characters) and omitted
for identifier-like and high-cardinality columns.
"""

import math
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.preview import json_cell
from datamedic.engine.table import (
    BOOLEAN,
    CATEGORICAL,
    DATETIME,
    IDENTIFIER,
    NUMERIC,
    SEMANTIC_TYPES,
    TEXT,
)

HISTOGRAM_BINS = 20
MAX_COMPLETE_VALUES = 50  # columns with <= this many distinct values get a full table
TOP_VALUES = 10  # otherwise, at most this many (only for low-cardinality-ratio columns)
MAX_VALUE_CHARS = 64
TOP_VALUES_MAX_UNIQUE_RATIO = 0.5
CORR_MAX_COLUMNS = 50
CORR_MAX_ROWS = 200_000
CORR_MIN_PERIODS = 3
TOP_PAIRS = 20

BASE_LIMITATIONS = [
    "Column types are inferred heuristically. Dates are recognised only in ISO-8601 form "
    "(e.g. 2024-03-01); other date formats are treated as text.",
    "Duplicate rows are exact matches across all columns (floating-point values must be "
    "identical). The hidden row identifier is ignored.",
    "Skewness and kurtosis are bias-corrected sample estimates and are unreliable for small "
    "samples.",
    "Correlations are pairwise over complete pairs and cover numeric and boolean columns only "
    "(Pearson: linear; Spearman: monotonic). They do not imply causation. Associations "
    "between categorical columns are not computed yet.",
    "Frequency tables keep at most 50 values per column (each cut to 64 characters). Columns "
    "with many distinct values show the top 10 values or none.",
]


def _num(value: Any) -> float | None:
    """JSON-safe float: NaN and infinities become None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pct(part: int | float, whole: int | float) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def as_float(series: pd.Series) -> np.ndarray:
    """Numeric or boolean column as a float array (missing -> NaN)."""
    return np.asarray(
        series.astype("Float64").to_numpy(dtype="float64", na_value=np.nan), dtype=np.float64
    )


def _value_label(value: Any) -> Any:
    cell = json_cell(value)
    if isinstance(cell, str) and len(cell) > MAX_VALUE_CHARS:
        return cell[: MAX_VALUE_CHARS - 1] + "…"
    return cell


def _frequencies(
    series: pd.Series, semantic: str, n_present: int, n_unique: int
) -> dict[str, Any] | None:
    if n_present == 0:
        return None
    if semantic == IDENTIFIER:
        return {
            "values": [],
            "complete": False,
            "other_count": n_present,
            "reason": "Identifier-like column: values are not summarised.",
        }
    if n_unique <= MAX_COMPLETE_VALUES:
        limit = n_unique
    elif semantic in (CATEGORICAL, TEXT) and n_unique / n_present <= TOP_VALUES_MAX_UNIQUE_RATIO:
        limit = TOP_VALUES
    else:
        return {
            "values": [],
            "complete": False,
            "other_count": n_present,
            "reason": "Too many distinct values to summarise.",
        }
    counts = series.value_counts(dropna=True)
    table = sorted(
        ((value, int(count)) for value, count in counts.items()),
        key=lambda item: (-item[1], str(item[0])),
    )[:limit]
    listed = sum(count for _, count in table)
    return {
        "values": [
            {"value": _value_label(v), "count": c, "pct": _pct(c, n_present)} for v, c in table
        ],
        "complete": limit >= n_unique,
        "other_count": n_present - listed,
        "reason": None,
    }


def _numeric_stats(series: pd.Series) -> tuple[dict[str, Any], dict[str, Any] | None]:
    values = as_float(series)
    values = values[~np.isnan(values)]
    finite = values[np.isfinite(values)]
    stats: dict[str, Any] = {
        "count": int(finite.size),
        "n_infinite": int(values.size - finite.size),
        "n_zeros": int((finite == 0).sum()),
        "n_negative": int((finite < 0).sum()),
        "mean": None,
        "std": None,
        "min": None,
        "q1": None,
        "median": None,
        "q3": None,
        "max": None,
        "iqr": None,
        "skewness": None,
        "kurtosis": None,
        "skewness_note": None,
    }
    if finite.size == 0:
        stats["skewness_note"] = "No finite values."
        return stats, None
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    s = pd.Series(finite)
    stats.update(
        mean=_num(finite.mean()),
        std=_num(s.std(ddof=1)) if finite.size > 1 else None,
        min=_num(finite.min()),
        q1=_num(q1),
        median=_num(median),
        q3=_num(q3),
        max=_num(finite.max()),
        iqr=_num(q3 - q1),
    )
    if finite.size < 3:
        stats["skewness_note"] = "At least 3 values are needed for skewness."
    elif float(finite.min()) == float(finite.max()):
        stats["skewness_note"] = "All values are equal, so skewness is undefined."
    else:
        stats["skewness"] = _num(s.skew())
        stats["kurtosis"] = _num(s.kurt()) if finite.size >= 4 else None
    histogram = None
    if float(finite.min()) != float(finite.max()):
        counts, edges = np.histogram(finite, bins=HISTOGRAM_BINS)
        histogram = {
            "kind": "numeric",
            "edges": [_num(e) for e in edges],
            "counts": [int(c) for c in counts],
        }
    return stats, histogram


def _datetime_stats(series: pd.Series) -> tuple[dict[str, Any], dict[str, Any] | None]:
    present = series.dropna()
    if getattr(present.dtype, "tz", None) is not None:
        present = present.dt.tz_convert("UTC").dt.tz_localize(None)
    if present.empty:
        return {"min": None, "max": None}, None
    lo, hi = present.min(), present.max()
    stats = {"min": lo.isoformat(), "max": hi.isoformat()}
    if lo == hi:
        return stats, None
    # Integer ticks are in the column's own unit (ns, us, ms or s: Parquet sources keep theirs),
    # so they are converted back with that same unit.
    unit = present.dt.unit
    ticks = present.astype("int64").to_numpy()
    counts, edges = np.histogram(ticks, bins=HISTOGRAM_BINS)
    return stats, {
        "kind": "datetime",
        "edges": [pd.Timestamp(int(e), unit=unit).isoformat() for e in edges],
        "counts": [int(c) for c in counts],
    }


def _text_stats(series: pd.Series) -> dict[str, Any]:
    lengths = series.dropna().astype(str).str.len()
    if lengths.empty:
        return {"min_length": None, "mean_length": None, "max_length": None}
    return {
        "min_length": int(lengths.min()),
        "mean_length": _num(round(float(lengths.mean()), 2)),
        "max_length": int(lengths.max()),
    }


def _correlations(
    data: pd.DataFrame, columns: list[dict[str, Any]], seed: int
) -> tuple[dict[str, Any], list[str]]:
    eligible = [
        c["name"]
        for c in columns
        if c["inferred_type"] in (NUMERIC, BOOLEAN) and not c["is_constant"] and not c["is_empty"]
    ]
    notes: list[str] = []
    result: dict[str, Any] = {
        "status": "not_run",
        "reason": None,
        "columns": [],
        "pearson": [],
        "spearman": [],
        "top_pairs": [],
        "n_rows_used": 0,
        "sampled": False,
        "seed": seed,
        "truncated_columns": 0,
    }
    if len(eligible) < 2:
        result["reason"] = (
            "At least two non-constant numeric or boolean columns are needed "
            f"(found {len(eligible)})."
        )
        return result, notes
    if len(eligible) > CORR_MAX_COLUMNS:
        result["truncated_columns"] = len(eligible) - CORR_MAX_COLUMNS
        eligible = eligible[:CORR_MAX_COLUMNS]
        notes.append(
            f"Correlations cover the first {CORR_MAX_COLUMNS} eligible columns; "
            f"{result['truncated_columns']} were left out."
        )
    sub = data[eligible]
    if len(sub) > CORR_MAX_ROWS:
        sub = sub.sample(n=CORR_MAX_ROWS, random_state=seed)
        result["sampled"] = True
        notes.append(
            f"Correlations were computed on a random sample of {CORR_MAX_ROWS:,} rows "
            f"(seed {seed})."
        )
    numeric = pd.DataFrame({name: as_float(sub[name]) for name in eligible})
    pearson = numeric.corr(method="pearson", min_periods=CORR_MIN_PERIODS)
    spearman = numeric.corr(method="spearman", min_periods=CORR_MIN_PERIODS)

    def matrix(m: pd.DataFrame) -> list[list[float | None]]:
        return [[_num(round(float(v), 6)) for v in row] for row in m.to_numpy()]

    pairs = []
    for i, a in enumerate(eligible):
        for b in eligible[i + 1 :]:
            p = _num(pearson.at[a, b])
            if p is None:
                continue
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "pearson": round(p, 6),
                    "spearman": _num(round(float(spearman.at[a, b]), 6)),
                }
            )
    pairs.sort(key=lambda pair: -abs(pair["pearson"]))
    result.update(
        status="ok",
        columns=eligible,
        pearson=matrix(pearson),
        spearman=matrix(spearman),
        top_pairs=pairs[:TOP_PAIRS],
        n_rows_used=len(numeric),
    )
    return result, notes


class BuiltinProfiler:
    name = "builtin"
    version = "1.0.0"

    def profile(
        self, frame: pd.DataFrame, schema: list[dict[str, Any]], *, seed: int = 0
    ) -> dict[str, Any]:
        data = frame.drop(columns=[ROW_ID])
        n_rows, n_cols = data.shape
        semantic = {c["name"]: c["inferred_type"] for c in schema}
        dtypes = {c["name"]: c["dtype"] for c in schema}
        columns: list[dict[str, Any]] = []
        for name in data.columns:
            series = data[name]
            sem = semantic.get(name, TEXT)
            n_missing = int(series.isna().sum())
            n_present = n_rows - n_missing
            n_unique = int(series.nunique(dropna=True))
            counts = series.value_counts(dropna=True) if n_present else None
            top_pct = (
                _pct(int(counts.iloc[0]), n_present) if counts is not None and len(counts) else None
            )
            col: dict[str, Any] = {
                "name": name,
                "inferred_type": sem,
                "dtype": dtypes.get(name, "string"),
                "n_missing": n_missing,
                "missing_pct": _pct(n_missing, n_rows),
                "n_present": n_present,
                "n_unique": n_unique,
                "unique_pct": _pct(n_unique, n_present),
                "is_empty": n_present == 0,
                "is_constant": n_unique == 1,
                "top_value_pct": top_pct,
                "numeric": None,
                "datetime": None,
                "text": None,
                "histogram": None,
                "frequencies": None,
            }
            is_numeric_storage = pd.api.types.is_numeric_dtype(
                series
            ) and not pd.api.types.is_bool_dtype(series)
            if sem == IDENTIFIER:
                # Identifiers are labels, not measurements: no mean, spread or histogram, even
                # when they are stored as integers. Only counts and uniqueness are reported.
                if n_present and not is_numeric_storage:
                    col["text"] = _text_stats(series)
            elif is_numeric_storage and n_present:
                col["numeric"], col["histogram"] = _numeric_stats(series)
            elif pd.api.types.is_datetime64_any_dtype(series) and n_present:
                col["datetime"], col["histogram"] = _datetime_stats(series)
            elif sem in (CATEGORICAL, TEXT) and n_present:
                col["text"] = _text_stats(series)
            if sem != DATETIME:
                col["frequencies"] = _frequencies(series, sem, n_present, n_unique)
            columns.append(col)

        correlations, corr_notes = _correlations(data, columns, seed)
        n_cells = n_rows * n_cols
        missing_cells = int(data.isna().to_numpy().sum())
        dup_first = int(data.duplicated(keep="first").sum()) if n_cols else 0
        dup_all = int(data.duplicated(keep=False).sum()) if n_cols else 0
        rows_with_missing = int(data.isna().any(axis=1).sum()) if n_cols else 0
        type_counts = {t: 0 for t in SEMANTIC_TYPES}
        for c in columns:
            type_counts[c["inferred_type"]] = type_counts.get(c["inferred_type"], 0) + 1
        summary = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_cells": n_cells,
            "missing_cells": missing_cells,
            "missing_cells_pct": _pct(missing_cells, n_cells),
            "rows_with_missing": rows_with_missing,
            "rows_with_missing_pct": _pct(rows_with_missing, n_rows),
            "duplicate_rows": dup_first,
            "duplicate_rows_pct": _pct(dup_first, n_rows),
            "rows_in_duplicate_groups": dup_all,
            "constant_columns": [c["name"] for c in columns if c["is_constant"]],
            "empty_columns": [c["name"] for c in columns if c["is_empty"]],
            "type_counts": type_counts,
        }
        return {
            "profiler": self.name,
            "profiler_version": self.version,
            "summary": summary,
            "columns": columns,
            "correlations": correlations,
            "sampling": {
                "statistics": "exact (all rows)",
                "correlations": "sampled" if correlations["sampled"] else "exact (all rows)",
                "seed": seed,
            },
            "limitations": BASE_LIMITATIONS + corr_notes,
        }

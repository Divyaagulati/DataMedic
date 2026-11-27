"""Canonical table construction: column names, missing tokens and type inference.

Readers produce raw values. This module turns them into the canonical version-0 frame:
unique non-empty column names (originals kept), typed columns, and ``_dm_row_id``.

Type inference is deliberately conservative and recorded per column, so later phases can
explain (and the user can override) every decision:
- numbers with leading zeros (``007``) or 16+ digits stay text (identifiers, codes);
- dates are recognised only in unambiguous ISO-8601 form (``2024-03-01``); ``01/03/2024``
  stays text because day/month order cannot be known;
- a fixed, documented list of missing-value tokens is applied (``MISSING_TOKENS``).
"""

import datetime as dt
import json
import math
import re
import warnings
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID

MISSING_TOKENS = frozenset({"", "NA", "N/A", "n/a", "NaN", "nan", "NULL", "null", "#N/A", "<NA>"})
"""Cell texts treated as missing (after trimming whitespace). Deliberately excludes "None",
"-", "?" and sentinels such as -999, which can be meaningful; detecting disguised missing
values is a diagnostic check, not a parsing rule."""

BOOL_WORDS = {"true": True, "false": False, "yes": True, "no": False}
BOOL_LETTER_PAIRS = ({"t": True, "f": False}, {"y": True, "n": False})
"""Words are always boolean. Single letters are boolean only when both letters of one pair
occur: a column holding only "F" is far more likely a code (e.g. sex) than all-false."""

MAX_NAME_LENGTH = 200
RESERVED_PREFIX = "_dm_"

# Semantic types (``inferred_type``)
NUMERIC = "numeric"
BOOLEAN = "boolean"
DATETIME = "datetime"
CATEGORICAL = "categorical"
TEXT = "text"
IDENTIFIER = "identifier"
EMPTY = "empty"
SEMANTIC_TYPES = (NUMERIC, BOOLEAN, DATETIME, CATEGORICAL, TEXT, IDENTIFIER, EMPTY)

CATEGORICAL_MAX_UNIQUE = 50
IDENTIFIER_MIN_ROWS = 20

_LEADING_ZERO = re.compile(r"^[+-]?0\d")
_LONG_DIGITS = re.compile(r"^[+-]?\d{16,}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")
_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


@dataclass
class ColumnInfo:
    name: str
    original_name: str
    dtype: str
    inferred_type: str
    n_missing: int
    n_unique: int
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "original_name": self.original_name,
            "dtype": self.dtype,
            "inferred_type": self.inferred_type,
            "n_missing": self.n_missing,
            "n_unique": self.n_unique,
            "notes": list(self.notes),
        }


# ---- Cell conversion ------------------------------------------------------------------------


def to_text(value: Any) -> str | None:
    """Convert one raw cell (from Excel, JSON or Parquet) to text, or None if missing."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float | np.floating):
        if math.isnan(value):
            return None
        return repr(float(value))
    if isinstance(value, bool | np.bool_):
        return "true" if value else "false"
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, pd.Timestamp | dt.datetime | dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, dict | list | tuple):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=False, default=str)
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Decimal):
        return str(value)
    return str(value)


def normalise_text_column(series: pd.Series) -> pd.Series:
    """Return an object Series of ``str`` or ``None`` with missing tokens applied."""
    if pd.api.types.infer_dtype(series, skipna=True) not in ("string", "empty"):
        series = series.map(to_text)
    series = series.astype(object)
    stripped = series.str.strip()
    missing = series.isna() | stripped.isin(MISSING_TOKENS)
    return series.where(~missing, None)


# ---- Column names ---------------------------------------------------------------------------


def canonical_names(raw_names: list[Any]) -> list[tuple[str, str]]:
    """Return ``(name, original_name)`` pairs: unique, non-empty, no control characters,
    never using the reserved ``_dm_`` prefix."""
    used: set[str] = {ROW_ID}
    out: list[tuple[str, str]] = []
    for i, raw in enumerate(raw_names):
        original = to_text(raw) or ""
        base = " ".join(_CONTROL.sub(" ", original).split())[:MAX_NAME_LENGTH]
        if not base:
            base = f"column_{i + 1}"
        if base.startswith(RESERVED_PREFIX):
            base = f"col{base}"
        name = base
        k = 2
        while name in used:
            name = f"{base}_{k}"
            k += 1
        used.add(name)
        out.append((name, original))
    return out


# ---- Type inference -------------------------------------------------------------------------


def _string_semantic(values: pd.Series) -> str:
    """Semantic type of a non-empty column of strings."""
    n = len(values)
    n_unique = int(values.nunique())
    if n >= IDENTIFIER_MIN_ROWS and n_unique == n and float(values.str.len().mean()) <= 40:
        return IDENTIFIER
    if n_unique <= CATEGORICAL_MAX_UNIQUE and (n < IDENTIFIER_MIN_ROWS or n_unique / n <= 0.5):
        return CATEGORICAL
    return TEXT


def _try_numeric(present: pd.Series) -> pd.Series | None:
    if present.str.match(_LEADING_ZERO).any() or present.str.match(_LONG_DIGITS).any():
        return None
    nums = pd.to_numeric(present, errors="coerce")
    if nums.isna().any() or not np.isfinite(nums.to_numpy(dtype=float)).all():
        return None
    return nums


def _try_datetime(present: pd.Series) -> pd.Series | None:
    if not present.str.match(_ISO_DATE).all():
        return None
    has_tz = present.str.contains(_TZ_SUFFIX)
    if has_tz.any() and not has_tz.all():
        return None  # naive mixed with zoned: converting would invent a time zone
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(
                present, format="ISO8601", errors="coerce", utc=bool(has_tz.all())
            )
        except (ValueError, OverflowError, TypeError):
            return None
    if not pd.api.types.is_datetime64_any_dtype(parsed) or parsed.isna().any():
        return None
    return parsed


def infer_text_column(series: pd.Series, *, allow_retyping: bool = True) -> tuple[pd.Series, str]:
    """Infer the type of a normalised text column (``str`` or ``None`` values).

    Returns the typed Series and its semantic type. ``allow_retyping=False`` keeps strings as
    strings (used for typed sources such as Parquet, where the file declared a string type).
    """
    mask = series.notna()
    if not mask.any():
        return series.astype(object), EMPTY
    present = series[mask].str.strip()

    if allow_retyping:
        nums = _try_numeric(present)
        if nums is not None:
            target = "Int64" if pd.api.types.is_integer_dtype(nums) else "float64"
            out = nums.astype(target).reindex(series.index)
            return out, numeric_semantic(out)

        lowered = present.str.lower()
        mapping = _boolean_mapping(set(lowered.unique()))
        if mapping is not None:
            out = lowered.map(mapping).astype("boolean").reindex(series.index)
            return out, BOOLEAN

        dates = _try_datetime(present)
        if dates is not None:
            return dates.reindex(series.index), DATETIME

    return series.astype(object), _string_semantic(series[mask])


def _boolean_mapping(tokens: set[str]) -> dict[str, bool] | None:
    letters = tokens - BOOL_WORDS.keys()
    if not letters:
        return dict(BOOL_WORDS)
    for pair in BOOL_LETTER_PAIRS:
        # Only letters of this pair, and both of them present.
        if letters <= pair.keys() and pair.keys() <= tokens:
            return {**BOOL_WORDS, **pair}
    return None


IDENTIFIER_MAX_SPAN_RATIO = 2.0


def numeric_semantic(series: pd.Series) -> str:
    """Integer columns look like identifiers when, over many rows, every value is distinct AND
    the values are packed into a dense range (e.g. 1001-1500 for 500 rows), as row or record
    numbers are. Distinct integers spread over a wide range (incomes, step counts) stay
    measurements."""
    present = series.dropna()
    n = len(present)
    if (
        pd.api.types.is_integer_dtype(series)
        and n >= IDENTIFIER_MIN_ROWS
        and int(present.nunique()) == n
        and (int(present.max()) - int(present.min()) + 1) <= IDENTIFIER_MAX_SPAN_RATIO * n
    ):
        return IDENTIFIER
    return NUMERIC if n else EMPTY


def dtype_label(series: pd.Series) -> str:
    """Stable, human-readable storage type."""
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_integer_dtype(series):
        return "integer"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        tz = getattr(series.dtype, "tz", None)
        return "datetime_utc" if tz is not None else "datetime"
    return "string"


# ---- Canonical frame ------------------------------------------------------------------------


@dataclass
class CanonicalTable:
    frame: pd.DataFrame  # includes ROW_ID as the first column
    columns: list[ColumnInfo]

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    @property
    def n_cols(self) -> int:
        return len(self.columns)


@dataclass
class RawColumns:
    """Reader output: header cells and raw column values, positionally aligned."""

    names: list[Any]
    columns: list[pd.Series]
    typed: set[int] = field(default_factory=set)  # positions already typed by the reader
    notes: dict[int, list[str]] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return len(self.columns[0]) if self.columns else 0


def _typed_semantic(series: pd.Series) -> tuple[pd.Series, str]:
    if pd.api.types.is_bool_dtype(series):
        return series, BOOLEAN if series.notna().any() else EMPTY
    if pd.api.types.is_datetime64_any_dtype(series):
        return series, DATETIME if series.notna().any() else EMPTY
    if pd.api.types.is_numeric_dtype(series):
        return series, numeric_semantic(series)
    return infer_text_column(normalise_text_column(series), allow_retyping=False)


def build_canonical(
    raw: RawColumns, *, drop_empty: bool = False
) -> tuple[CanonicalTable, dict[str, int]]:
    """Build the canonical frame.

    Untyped columns (CSV, Excel, JSON) are normalised to text and then type-inferred. Typed
    columns (Parquet) keep their type; their strings stay strings.

    ``drop_empty`` (spreadsheet-like sources) drops rows in which every cell is missing and
    columns that have neither a header nor any value. Both counts are reported.
    """
    prepared = [
        col.reset_index(drop=True)
        if pos in raw.typed
        else normalise_text_column(col.reset_index(drop=True))
        for pos, col in enumerate(raw.columns)
    ]
    names = list(raw.names)
    notes = dict(raw.notes)
    stats = {"dropped_empty_rows": 0, "dropped_empty_columns": 0}
    if drop_empty and prepared:
        empty_rows = np.logical_and.reduce([c.isna().to_numpy() for c in prepared])
        if empty_rows.any():
            keep = ~empty_rows
            prepared = [c[keep].reset_index(drop=True) for c in prepared]
            stats["dropped_empty_rows"] = int(empty_rows.sum())
        kept = [
            pos
            for pos, col in enumerate(prepared)
            if (to_text(names[pos]) or "").strip() or col.notna().any()
        ]
        stats["dropped_empty_columns"] = len(prepared) - len(kept)
        prepared = [prepared[p] for p in kept]
        notes = {new: notes.get(old, []) for new, old in enumerate(kept)}
        typed = {new for new, old in enumerate(kept) if old in raw.typed}
        names = [names[p] for p in kept]
    else:
        typed = set(raw.typed)

    data: dict[str, pd.Series] = {}
    infos: list[ColumnInfo] = []
    for pos, (name, original) in enumerate(canonical_names(names)):
        col_notes = list(notes.get(pos, []))
        if pos in typed:
            series, semantic = _typed_semantic(prepared[pos])
        else:
            series, semantic = infer_text_column(prepared[pos])
        if original != name:
            col_notes.append("renamed")
        data[name] = series
        infos.append(
            ColumnInfo(
                name=name,
                original_name=original,
                dtype=dtype_label(series),
                inferred_type=semantic,
                n_missing=int(series.isna().sum()),
                n_unique=int(series.nunique(dropna=True)),
                notes=col_notes,
            )
        )
    frame = pd.DataFrame(data)
    frame.insert(0, ROW_ID, np.arange(len(frame), dtype="int64"))
    return CanonicalTable(frame=frame, columns=infos), stats

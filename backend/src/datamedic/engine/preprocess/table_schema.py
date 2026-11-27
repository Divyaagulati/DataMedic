"""Schema entries for a transformed table.

Semantic types of existing columns carry over from the parent schema (they were decided at
ingest, possibly by heuristics the user has reviewed). Columns added by steps get a type
from their dtype. Counts are recomputed from the data.
"""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.table import BOOLEAN, CATEGORICAL, DATETIME, NUMERIC, TEXT, dtype_label


def _semantic_from_dtype(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series):
        return BOOLEAN
    if pd.api.types.is_numeric_dtype(series):
        return NUMERIC
    if pd.api.types.is_datetime64_any_dtype(series):
        return DATETIME
    unique = series.nunique(dropna=True)
    return CATEGORICAL if unique <= 50 else TEXT


def schema_of(frame: pd.DataFrame, known_types: Mapping[str, str]) -> list[dict[str, Any]]:
    out = []
    for name in frame.columns:
        if name == ROW_ID:
            continue
        series = frame[name]
        out.append(
            {
                "name": name,
                "original_name": name,
                "dtype": dtype_label(series),
                "inferred_type": known_types.get(name) or _semantic_from_dtype(series),
                "n_missing": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return out

"""Owner-only preview of the first rows of a version (stored as an artifact, never logged)."""

import math
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID

PREVIEW_ROWS = 20
MAX_CELL_CHARS = 200


def json_cell(value: Any) -> Any:
    """A JSON-safe rendering of one cell. Long text is truncated (and marked with "…")."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        f = float(value)
        if math.isnan(f):
            return None
        return f if math.isfinite(f) else str(f)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[: MAX_CELL_CHARS - 1] + "…"


def build_preview(
    frame: pd.DataFrame, columns: list[dict[str, Any]], n: int = PREVIEW_ROWS
) -> dict[str, Any]:
    head = frame.drop(columns=[ROW_ID]).head(n)
    rows = [[json_cell(v) for v in row] for row in head.itertuples(index=False, name=None)]
    return {
        "columns": [
            {"name": c["name"], "dtype": c["dtype"], "inferred_type": c["inferred_type"]}
            for c in columns
        ],
        "rows": rows,
        "n_rows_total": len(frame),
        "truncated": len(frame) > n,
    }

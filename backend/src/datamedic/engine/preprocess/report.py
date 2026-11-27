"""Per-step reports: what a step did to a table (PHASE4_PLAN §9.2).

Reports contain counts, dtypes and learned parameters; never row values. Learned parameters
of sensitive columns are replaced by ``{"redacted": true}`` because they are derived from
cell values (e.g. a most-frequent category).
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.table import dtype_label

REDACTED = {"redacted": True}


@dataclass
class ColumnChange:
    column: str
    missing_before: int
    missing_after: int
    cells_filled: int  # missing before, present after
    observed_changed: int  # present before, different after (0 for imputation)
    dtype_before: str
    dtype_after: str


@dataclass
class StepReport:
    index: int
    op: str
    op_version: str
    step_hash: str
    rows_in: int
    rows_out: int
    columns_added: list[str]
    columns_removed: list[str]
    columns: list[ColumnChange]
    learned: dict[str, Any]
    warnings: list[dict[str, Any]]
    n_fit_rows: int
    duration_ms: float
    effects: dict[str, Any] = field(default_factory=dict)  # counts only, never values
    rows_removed: int = field(init=False)

    def __post_init__(self) -> None:
        self.rows_removed = self.rows_in - self.rows_out

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _changed_count(before: pd.Series, after: pd.Series) -> int:
    """Cells present before whose value differs after (compared as Python objects, so an
    integer 3 and a float 3.0 are equal)."""
    present = before.notna().to_numpy() & after.notna().to_numpy()
    if not present.any():
        return int((before.notna().to_numpy() & after.isna().to_numpy()).sum())
    lost = before.notna().to_numpy() & after.isna().to_numpy()
    fast = _vector_values(before, after)
    if fast is not None:  # same result as the object comparison, without a Python loop
        b_values, a_values = fast
        return int((b_values[present] != a_values[present]).sum() + lost.sum())
    b = before.astype(object).to_numpy()[present]
    a = after.astype(object).to_numpy()[present]
    differs = np.fromiter((x != y for x, y in zip(b, a, strict=True)), dtype=bool, count=len(b))
    return int(differs.sum() + lost.sum())


def _vector_values(before: pd.Series, after: pd.Series) -> tuple[np.ndarray, np.ndarray] | None:
    """Arrays that compare like Python objects, for boolean, integer and float columns
    (missing cells are excluded by the caller). Integers stay int64 so values above 2**53 are
    compared exactly; other combinations use the Python comparison."""

    def kind(series: pd.Series) -> str | None:
        if pd.api.types.is_bool_dtype(series):
            return "bool"
        if pd.api.types.is_integer_dtype(series):
            return "int"
        if pd.api.types.is_float_dtype(series):
            return "float"
        return None

    kinds = {kind(before), kind(after)}
    dtype: str
    na: object
    if None in kinds:
        return None
    if kinds == {"bool"}:
        dtype, na = "bool", False
    elif kinds == {"int"}:
        dtype, na = "int64", 0
    elif "bool" in kinds:
        return None  # True == 1 as Python objects; keep the exact object semantics
    else:
        dtype, na = "float64", np.nan
    return (
        before.to_numpy(dtype=dtype, na_value=na),
        after.to_numpy(dtype=dtype, na_value=na),
    )


def column_changes(
    before: pd.DataFrame, after: pd.DataFrame, columns: list[str]
) -> list[ColumnChange]:
    out = []
    aligned = before.index.equals(after.index)
    for column in columns:
        if column not in before.columns or column not in after.columns:
            continue
        b, a = before[column], after[column]
        missing_before, missing_after = int(b.isna().sum()), int(a.isna().sum())
        if aligned:
            filled = int((b.isna().to_numpy() & a.notna().to_numpy()).sum())
            changed = _changed_count(b, a)
        else:  # rows were removed: compare surviving rows by row id
            joined = before[[ROW_ID, column]].merge(
                after[[ROW_ID, column]], on=ROW_ID, suffixes=("_b", "_a")
            )
            bb, aa = joined[f"{column}_b"], joined[f"{column}_a"]
            filled = int((bb.isna().to_numpy() & aa.notna().to_numpy()).sum())
            changed = _changed_count(bb, aa)
        out.append(
            ColumnChange(
                column=column,
                missing_before=missing_before,
                missing_after=missing_after,
                cells_filled=filled,
                observed_changed=changed,
                dtype_before=dtype_label(b),
                dtype_after=dtype_label(a),
            )
        )
    return out


def redact(learned: dict[str, Any], sensitive: frozenset[str]) -> dict[str, Any]:
    """Replace per-column learned values of sensitive columns, and the value of any column
    pair (e.g. a correlation) that involves a sensitive column. Decisions (kept, removed)
    are not values and stay visible."""
    if not sensitive:
        return learned
    out = dict(learned)
    columns = learned.get("columns")
    if isinstance(columns, dict):
        out["columns"] = {c: (REDACTED if c in sensitive else v) for c, v in columns.items()}
    pairs = learned.get("pairs")
    if isinstance(pairs, list):
        out["pairs"] = [
            {"columns": pair.get("columns"), **REDACTED}
            if isinstance(pair, dict) and sensitive.intersection(pair.get("columns") or [])
            else pair
            for pair in pairs
        ]
    return out


def redact_effects(effects: dict[str, Any], sensitive: frozenset[str]) -> dict[str, Any]:
    """Effects are counts, except distribution statistics, which describe values."""
    distribution = effects.get("distribution")
    if not sensitive or not isinstance(distribution, dict):
        return effects
    return {
        **effects,
        "distribution": {c: (REDACTED if c in sensitive else v) for c, v in distribution.items()},
    }


def feature_lineage(input_columns: list[str], steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What happened to every column through the steps. ``steps``: dicts with ``index``,
    ``op``, ``columns`` (selected), ``added``, ``removed`` and ``decisions`` (column ->
    selection decision). Each entry: column, origin (input | added), derived_from, final
    (present | removed) and events (step, op, action, decision)."""
    entries: dict[str, dict[str, Any]] = {
        c: {"column": c, "origin": "input", "derived_from": [], "events": []} for c in input_columns
    }
    for step in steps:
        selected = list(step["columns"])
        decisions = step.get("decisions") or {}
        for column in step["added"]:
            sources = [c for c in selected if column.startswith(c)] or selected
            entries[column] = {
                "column": column,
                "origin": "added",
                "derived_from": sources,
                "events": [{"step": step["index"], "op": step["op"], "action": "added"}],
            }
        for column in selected:
            if column not in entries:
                continue
            if column in step["removed"]:
                action = "removed"
            elif column in decisions:
                action = "assessed"
            else:
                action = "transformed"
            event = {"step": step["index"], "op": step["op"], "action": action}
            if column in decisions:
                event["decision"] = decisions[column]
            entries[column]["events"].append(event)
        for column in step["removed"]:
            if column in entries and column not in selected:
                entries[column]["events"].append(
                    {"step": step["index"], "op": step["op"], "action": "removed"}
                )
    out = []
    for entry in entries.values():
        removed = any(e["action"] == "removed" for e in entry["events"])
        out.append(
            {
                **entry,
                "final": "removed" if removed else "present",
                "feature": not str(entry["column"]).startswith("_dm_"),
            }
        )
    return out

"""Committing a pipeline into a new dataset version (Phase 4 Part E).

A pipeline step carries a *chosen stage*:
- ``data`` steps are materialised into the new version through
  ``PreprocessExecutor.commit_transform`` (fitted on every row of the source version, so the
  version is flagged ``fitted_on_full_data`` when any step learns from data);
- ``model`` steps are never executed here. They are stored with the version as its
  ``model_pipeline_spec`` and run only inside model training folds (owner decision D1).

Ops whose catalogue stage is ``data`` or ``model`` must keep it; ``either`` ops need an
explicit choice, which is never changed silently. Data-stage steps must all come before the
first model-stage step, so the committed table is exactly what the model-stage steps were
validated against.

The impact preview runs the same ``commit_transform`` call as the commit, then replays the
fitted steps one by one to describe each step's effect. It records the content hash of the
result; the commit refuses to write a version whose content differs from the previewed one.
Preview rows are bounded (``max_rows`` per step, chosen deterministically by row id) and
values of sensitive columns are redacted.
"""

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.hashing import content_hash
from datamedic.engine.io.parse import write_frame
from datamedic.engine.preprocess.executor import POSTCONDITION_FAILED, PreprocessExecutor
from datamedic.engine.preprocess.report import REDACTED
from datamedic.engine.preprocess.stages import (
    CommitPlan,
    PlannedStep,
    default_stage,
    plan,
)
from datamedic.engine.preprocess.validate import ValidationResult

MAX_PREVIEW_ROWS = 1_000
MAX_PREVIEW_COLUMNS = 20
MAX_TEXT = 64
REDACTED_VALUE = "[redacted]"

PREVIEW_LIMITATIONS = [
    "Counts describe the whole source version; the listed rows are at most "
    f"{MAX_PREVIEW_ROWS:,} per step (lowest row ids first).",
    "Fewer missing values or extreme values is not evidence of better predictive "
    "performance; evaluate in the Experiment Lab.",
    "Steps that learn from data are fitted on every row of the source version. A version "
    "built this way must not be used directly for leakage-controlled model evaluation: "
    "evaluate with "
    "the steps inside training folds instead.",
    "Model-stage steps are stored with the version as its modelling specification and do not "
    "change the saved data.",
]

__all__ = ["CommitPlan", "PlannedStep", "default_stage", "materialise", "plan", "preview"]


# ---- Materialisation --------------------------------------------------------------------------


@dataclass
class Materialised:
    frame: pd.DataFrame  # as read back from the Parquet bytes (what the version stores)
    parquet: bytes
    content_hash: bytes
    reports: list[dict[str, Any]]
    lineage: list[dict[str, Any]]
    fitted_on_full_data: bool


def finalise(frame: pd.DataFrame) -> tuple[bytes, pd.DataFrame, bytes]:
    """Parquet bytes, the frame read back from them, and its content hash. Both the preview
    and the commit (and generated code) hash the read-back frame, so they agree exactly."""
    buf = io.BytesIO()
    write_frame(frame, buf)
    data = buf.getvalue()
    back = pd.read_parquet(io.BytesIO(data), engine="pyarrow")
    return data, back, content_hash(back)


def _check_rows(before: pd.DataFrame, after: pd.DataFrame) -> None:
    ids_before, ids_after = before[ROW_ID], after[ROW_ID]
    if (
        not ids_after.is_unique
        or len(after) != len(before)
        or not ids_after.isin(ids_before).all()
        or (ids_after < 0).any()
    ):
        raise EngineError(
            POSTCONDITION_FAILED,
            "A committed version must keep exactly the "
            "rows of its source version with their row identifiers.",
        )


def materialise(frame: pd.DataFrame, data: ValidationResult, seed: int = 0) -> Materialised:
    executor = PreprocessExecutor(data, seed=seed)
    result = executor.commit_transform(frame)
    _check_rows(frame, result.frame)
    parquet, back, digest = finalise(result.frame)
    return Materialised(
        frame=back,
        parquet=parquet,
        content_hash=digest,
        reports=result.report_dicts(),
        lineage=result.lineage,
        fitted_on_full_data=any(v.spec.kind == "fitted" for v in data.steps),
    )


# ---- Impact preview ---------------------------------------------------------------------------


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        f = float(value)
        return None if math.isnan(f) else (f if math.isfinite(f) else str(f))
    if isinstance(value, pd.Timestamp | np.datetime64):
        return pd.Timestamp(value).isoformat()
    text = str(value)
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"


def _numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _diff_mask(before: pd.Series, after: pd.Series) -> np.ndarray:
    """Rows whose value or missingness differs (aligned frames)."""
    b_na, a_na = before.isna().to_numpy(), after.isna().to_numpy()
    both = ~b_na & ~a_na
    if _numeric(before) and _numeric(after):
        values = before.to_numpy(dtype=float, na_value=np.nan) != after.to_numpy(
            dtype=float, na_value=np.nan
        )
    else:
        values = before.astype(object).to_numpy() != after.astype(object).to_numpy()
    return np.asarray((b_na != a_na) | (both & values), dtype=bool)


def _summary(series: pd.Series) -> dict[str, Any]:
    present = series.dropna()
    out: dict[str, Any] = {"count": int(len(present)), "missing": int(series.isna().sum())}
    if _numeric(series) and len(present):
        values = present.to_numpy(dtype=float)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        out.update(
            mean=_json_value(values.mean()),
            std=_json_value(values.std(ddof=1)) if len(values) > 1 else None,
            min=_json_value(values.min()),
            q1=_json_value(q1),
            median=_json_value(med),
            q3=_json_value(q3),
            max=_json_value(values.max()),
        )
    else:
        out["distinct"] = int(present.nunique())
    return out


def _step_impact(
    index: int,
    v: Any,
    before: pd.DataFrame,
    after: pd.DataFrame,
    report: dict[str, Any],
    sensitive: frozenset[str],
    max_rows: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    spec = v.spec
    added = [c for c in after.columns if c not in before.columns]
    removed = [c for c in before.columns if c not in after.columns]
    changes = {c["column"]: c for c in report["columns"]}
    mask = np.zeros(len(before), dtype=bool)
    columns: list[dict[str, Any]] = []
    shown: list[str] = []
    for name in v.step.columns:
        if name not in before.columns or name not in after.columns:
            continue
        diff = _diff_mask(before[name], after[name])
        mask |= diff
        ch = changes.get(name, {})
        filled = int(ch.get("cells_filled", 0))
        changed = int(ch.get("observed_changed", 0))
        redacted = name in sensitive
        columns.append(
            {
                "column": name,
                "rows_changed": int(diff.sum()),
                "missing_before": int(ch.get("missing_before", before[name].isna().sum())),
                "missing_after": int(ch.get("missing_after", after[name].isna().sum())),
                "estimates_filled": filled,
                "observed_overwritten": changed if spec.destructive else 0,
                "observed_re_expressed": 0 if spec.destructive else changed,
                "dtype_before": ch.get("dtype_before"),
                "dtype_after": ch.get("dtype_after"),
                "before": REDACTED if redacted else _summary(before[name]),
                "after": REDACTED if redacted else _summary(after[name]),
            }
        )
        if diff.any():
            shown.append(name)
    for name in added:
        values = after[name]
        marks = (
            values.fillna(False).astype(bool).to_numpy()
            if pd.api.types.is_bool_dtype(values)
            else values.notna().to_numpy()
        )
        mask |= marks
        shown.append(name)
    affected = int(mask.sum())
    ids = np.sort(before[ROW_ID].to_numpy()[mask])
    sample_ids = ids[:max_rows]
    shown = shown[:MAX_PREVIEW_COLUMNS]
    loss = []
    if any(c["estimates_filled"] for c in columns):
        loss.append("Missing values replaced by estimates.")
    if any(c["observed_overwritten"] for c in columns):
        loss.append("Observed values overwritten (cannot be recovered from this version).")
    if any(c["observed_re_expressed"] for c in columns):
        loss.append("Observed values re-expressed (an invertible change of scale or shape).")
    if removed:
        loss.append("Columns removed.")
    if added:
        loss.append("Columns created.")
    summary = {
        "index": index,
        "op": spec.op,
        "op_title": spec.title,
        "risk": spec.risk,
        "destructive": spec.destructive,
        "kind": spec.kind,
        "fitted": spec.kind == "fitted",
        "rows_before": len(before),
        "rows_after": len(after),
        "columns_added": added,
        "columns_removed": removed,
        "columns": columns,
        "rows_affected": affected,
        "rows_listed": int(len(sample_ids)),
        "rows_truncated": affected > len(sample_ids),
        "information_loss": loss,
        "warnings": report["warnings"],
    }
    position = pd.Series(np.arange(len(before)), index=before[ROW_ID].to_numpy())
    rows = []
    if len(sample_ids):
        at = position.loc[sample_ids].to_numpy()
        b_part, a_part = before.iloc[at], after.iloc[at]
        for k, row_id in enumerate(sample_ids.tolist()):
            rows.append(
                {
                    "row_id": int(row_id),
                    "before": {
                        c: (REDACTED_VALUE if c in sensitive else _json_value(b_part[c].iat[k]))
                        for c in shown
                        if c in before.columns
                    },
                    "after": {
                        c: (REDACTED_VALUE if c in sensitive else _json_value(a_part[c].iat[k]))
                        for c in shown
                        if c in after.columns
                    },
                }
            )
    return summary, {
        "index": index,
        "columns": shown,
        "rows": rows,
        "redacted_columns": sorted(c for c in shown if c in sensitive),
    }


def preview(
    frame: pd.DataFrame,
    plan_: CommitPlan,
    *,
    sensitive: frozenset[str],
    seed: int = 0,
    max_rows: int = MAX_PREVIEW_ROWS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """(summary, affected rows per step). The summary includes the content hash the commit
    must reproduce."""
    if not plan_.valid:
        raise EngineError("OP_INVALID", "The pipeline has validation errors.")
    executor = PreprocessExecutor(plan_.data, seed=seed)
    result = executor.commit_transform(frame)
    _check_rows(frame, result.frame)
    current = frame
    steps, rows = [], []
    for i, (v, t, report) in enumerate(
        zip(executor.steps, executor.transformers(), result.report_dicts(), strict=True)
    ):
        after = t.transform(current)
        summary, listed = _step_impact(i, v, current, after, report, sensitive, max_rows)
        steps.append(summary)
        rows.append(listed)
        current = after
    if not current.equals(result.frame):
        raise EngineError(
            POSTCONDITION_FAILED, "The step-by-step preview does not reproduce the commit result."
        )
    _, back, digest = finalise(result.frame)
    model = [
        {
            "op": s.step.op,
            "columns": list(s.step.columns),
            "note": "Stored as the version's modelling specification; does not change the "
            "saved data.",
        }
        for s in plan_.model_steps
    ]
    total = {
        "rows": len(frame),
        "columns_before": len([c for c in frame.columns if c != ROW_ID]),
        "columns_after": len([c for c in back.columns if c != ROW_ID]),
        "estimates_filled": sum(c["estimates_filled"] for s in steps for c in s["columns"]),
        "observed_overwritten": sum(c["observed_overwritten"] for s in steps for c in s["columns"]),
        "observed_re_expressed": sum(
            c["observed_re_expressed"] for s in steps for c in s["columns"]
        ),
        "columns_added": sorted({c for s in steps for c in s["columns_added"]}),
        "columns_removed": sorted({c for s in steps for c in s["columns_removed"]}),
    }
    summary = {
        "spec_hash": plan_.spec_hash,
        "expected_content_hash": digest.hex(),
        "fitted_on_full_data": any(v.spec.kind == "fitted" for v in plan_.data.steps),
        "totals": total,
        "steps": steps,
        "model_steps": model,
        "max_rows_per_step": max_rows,
        "limitations": PREVIEW_LIMITATIONS,
    }
    return summary, rows

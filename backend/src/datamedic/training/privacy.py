"""Sensitive-column protection for training results (API side; standard library and
SQLAlchemy only).

A training run stores the sensitive columns of its roles snapshot, and the engine never slices
them. Roles can change after a run: a column can become sensitive later, and a committed step
can derive a new column (for example a missing indicator) from a sensitive one. Historical
results are therefore filtered again whenever they are served. The stored rows are never
rewritten: training provenance and numerical results stay exactly as recorded.

Conservative policy (PHASE5_PLAN §17, owner closure 2026-11-26). A column is protected when
it was sensitive in the run's training-time roles, when it has a sensitive role on any version
of the dataset now, or when committed lineage derives it from such a column. For a protected
column, category-level error slices are withheld. When the target is protected, class names
are replaced by "class 1", "class 2", … and regression residual axis values are withheld,
exactly as at training time for a sensitive target. Aggregate importances (one number per
column) are not values of the column and are kept.
"""

import copy
import uuid
from collections.abc import Iterable
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from datamedic.db.models import ColumnRole, TrainingRun, TransformationRun
from datamedic.preprocessing.context import role_specs

AT_TRAINING = "sensitive_at_training"
NOW = "sensitive_now"
DERIVED = "derived_from_sensitive"
MAX_PASSES = 50
RESIDUAL_AXIS_KEYS = (
    "value_range",
    "residual_range",
    "residual_histogram_edges",
    "residual_quantiles",
    "mean_residual",
)
POLICY = (
    "Columns that were sensitive when the run was trained, are sensitive now on any version "
    "of the dataset, or are derived from such a column by committed steps are never shown as "
    "category-level error slices. When the target is protected, class names and residual "
    "axis values are withheld. Stored results are not changed; they are filtered when shown."
)


def sensitive_role_keys() -> frozenset[str]:
    return frozenset(k for k, r in role_specs().items() if r.is_sensitive)


def currently_sensitive(db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID) -> set[str]:
    """Columns with a sensitive role on any version of the dataset, now."""
    keys = sensitive_role_keys()
    if not keys:
        return set()
    return set(
        db.scalars(
            select(ColumnRole.column_name).where(
                ColumnRole.owner_id == owner_id,
                ColumnRole.dataset_id == dataset_id,
                ColumnRole.role.in_(sorted(keys)),
            )
        )
    )


def derived_edges(db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID) -> list[tuple[str, str]]:
    """(source column, added column) pairs from every committed transformation of the dataset,
    using the same attribution rule as the preprocessing feature lineage: an added column
    derives from the selected columns whose name it extends, otherwise from all of them."""
    edges: list[tuple[str, str]] = []
    for reports in db.scalars(
        select(TransformationRun.step_reports)
        .where(
            TransformationRun.owner_id == owner_id,
            TransformationRun.dataset_id == dataset_id,
            TransformationRun.result_version_id.is_not(None),
        )
        .order_by(TransformationRun.created_at)
    ):
        for step in reports or []:
            if not isinstance(step, dict):
                continue
            inputs = [
                str(c["column"])
                for c in step.get("columns") or []
                if isinstance(c, dict) and "column" in c
            ]
            for added in step.get("columns_added") or []:
                added = str(added)
                sources = [c for c in inputs if added.startswith(c)] or inputs
                edges.extend((s, added) for s in sources)
    return edges


def propagate(roots: Iterable[str], edges: list[tuple[str, str]]) -> set[str]:
    """Columns reachable from ``roots`` through ``edges`` (excluding the roots)."""
    marked = set(roots)
    start = set(marked)
    for _ in range(MAX_PASSES):
        grown = {b for a, b in edges if a in marked} - marked
        if not grown:
            break
        marked |= grown
    return marked - start


def protected_columns(
    db: Session,
    owner_id: uuid.UUID,
    dataset_id: uuid.UUID,
    at_training: Iterable[str] = (),
) -> dict[str, list[str]]:
    """Column -> reasons it is protected (``AT_TRAINING``, ``NOW``, ``DERIVED``)."""
    reasons: dict[str, list[str]] = {}
    for c in sorted(set(at_training)):
        reasons.setdefault(c, []).append(AT_TRAINING)
    for c in sorted(currently_sensitive(db, owner_id, dataset_id)):
        reasons.setdefault(c, []).append(NOW)
    for c in sorted(propagate(list(reasons), derived_edges(db, owner_id, dataset_id))):
        reasons.setdefault(c, []).append(DERIVED)
    return reasons


def derived_from(
    db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID, roots: Iterable[str]
) -> set[str]:
    return propagate(roots, derived_edges(db, owner_id, dataset_id))


def for_run(db: Session, run: TrainingRun) -> dict[str, list[str]]:
    return protected_columns(
        db, run.owner_id, run.dataset_id, run.config.get("sensitive_columns", [])
    )


# ---- Redaction of served results (copies; stored rows are never modified) --------------------


def redact_details(
    details: dict[str, Any], protected: dict[str, list[str]], target_protected: bool
) -> tuple[dict[str, Any], list[str]]:
    out = copy.deepcopy(details)
    withheld: list[str] = []
    kept = []
    for s in out.get("error_slices") or []:
        if str(s.get("column")) in protected:
            withheld.append(str(s.get("column")))
        else:
            kept.append(s)
    if "error_slices" in out:
        out["error_slices"] = kept
    residuals = out.get("residuals")
    if target_protected and isinstance(residuals, dict) and residuals.get("available"):
        for k in RESIDUAL_AXIS_KEYS:
            residuals.pop(k, None)
        residuals["redacted_axes"] = True
    return out, withheld


def redact_summary(summary: dict[str, Any] | None, target_protected: bool) -> dict[str, Any] | None:
    if summary is None or not target_protected:
        return summary
    out = copy.deepcopy(summary)
    classes = out.get("classes")
    if isinstance(classes, list):
        names = [f"class {i + 1}" for i in range(len(classes))]
        out["classes"] = names
        if out.get("positive_class") is not None and len(names) == 2:
            out["positive_class"] = names[1]
    return out


def disclosure(
    protected: dict[str, list[str]], withheld: Iterable[str], target: str
) -> dict[str, Any]:
    return {
        "policy": POLICY,
        "protected_columns": [{"column": c, "reasons": r} for c, r in sorted(protected.items())],
        "withheld_error_slices": sorted(set(withheld)),
        "target_values_withheld": target in protected,
    }

"""Roles, domains and analysis runs. Every lookup is scoped to the owner (and RLS)."""

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from datamedic.analyses.semantics import roles_snapshot, semantics_hash
from datamedic.config import Settings
from datamedic.datasets.service import get_owned
from datamedic.db.models import AnalysisRun, ColumnRole, Dataset, DatasetVersion, FindingRow, Job
from datamedic.domains.bootstrap import build_registry
from datamedic.engine.checks.spec import TASK_TYPES
from datamedic.errors import Conflict, NotFound, ValidationFailed
from datamedic.jobs.registry import ACTIVE_STATUSES
from datamedic.jobs.service import create_job
from datamedic.versions.service import get_owned_version

MULTICLASS_MAX = 50


# ---- Domain ---------------------------------------------------------------------------------


def update_dataset(
    db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID, name: str | None, domain: str | None
) -> Dataset:
    ds = get_owned(db, owner_id, dataset_id, for_update=True)
    if domain is not None:
        keys = build_registry().domain_keys()
        if domain not in keys:
            raise ValidationFailed(
                f"Unknown domain. Choose one of: {', '.join(keys)}.",
                errors=[{"pointer": "/domain", "code": "UNKNOWN_DOMAIN"}],
            )
        if domain != ds.domain:
            ds.domain = domain
            ds.domain_changed_at = datetime.now(UTC)
    if name is not None:
        ds.name = name.strip()
    ds.updated_at = datetime.now(UTC)
    db.commit()
    return ds


# ---- Roles ----------------------------------------------------------------------------------


def current_roles(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> list[ColumnRole]:
    return list(
        db.scalars(
            select(ColumnRole)
            .where(ColumnRole.owner_id == owner_id, ColumnRole.dataset_version_id == version_id)
            .order_by(ColumnRole.column_name, ColumnRole.role)
        )
    )


def _validate(
    domain: str, schema: list[dict[str, Any]], roles: list[dict[str, Any]]
) -> list[dict[str, str]]:
    registry = build_registry()
    available = {r.key: r for r in registry.roles_for(domain)}
    columns = {c["name"]: c for c in schema}
    errors: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    counts: dict[str, int] = {}

    def err(i: int, code: str, message: str) -> None:
        errors.append({"pointer": f"/roles/{i}", "code": code, "message": message})

    for i, item in enumerate(roles):
        column, role, task = item["column"], item["role"], item.get("task_type")
        info = columns.get(column)
        spec = available.get(role)
        if info is None:
            err(i, "COLUMN_NOT_FOUND", "This column does not exist in the version.")
            continue
        if spec is None:
            other = registry.role(role)
            err(
                i,
                "ROLE_NOT_AVAILABLE",
                f"This role belongs to the '{other.domain}' domain; change the dataset domain "
                "first."
                if other
                else "Unknown role.",
            )
            continue
        if (column, role) in seen:
            err(i, "DUPLICATE_ROLE", "The same role is assigned twice to this column.")
            continue
        seen.add((column, role))
        counts[role] = counts.get(role, 0) + 1
        if spec.max_per_version is not None and counts[role] > spec.max_per_version:
            err(i, "TOO_MANY", f"At most {spec.max_per_version} column(s) can have this role.")
        kind = info["inferred_type"]
        if kind not in spec.allowed_types:
            err(
                i,
                "ROLE_TYPE_MISMATCH",
                f"This role needs a column of type {', '.join(sorted(spec.allowed_types))}; "
                f"the column was detected as {kind}.",
            )
        if role == "target":
            n_unique = int(info.get("n_unique", 0))
            if task not in TASK_TYPES:
                err(i, "TASK_TYPE_REQUIRED", "Choose binary, multiclass or regression.")
            elif task == "binary" and n_unique != 2:
                err(
                    i,
                    "TASK_TYPE_MISMATCH",
                    f"A binary target needs exactly 2 values; this column has {n_unique}.",
                )
            elif task == "multiclass" and not 3 <= n_unique <= MULTICLASS_MAX:
                err(
                    i,
                    "TASK_TYPE_MISMATCH",
                    f"A multiclass target needs 3 to {MULTICLASS_MAX} "
                    f"values; this column has {n_unique}.",
                )
            elif task == "regression" and kind != "numeric":
                err(i, "TASK_TYPE_MISMATCH", "A regression target must be numeric.")
        elif task is not None:
            err(i, "TASK_TYPE_NOT_ALLOWED", "Only the target has a task type.")
    return errors


def replace_roles(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID, roles: list[dict[str, Any]]
) -> tuple[DatasetVersion, Dataset]:
    version, ds = get_owned_version(db, owner_id, version_id)
    db.execute(select(Dataset.id).where(Dataset.id == ds.id).with_for_update())
    errors = _validate(ds.domain, version.schema, roles)
    if errors:
        raise ValidationFailed("Some roles are not valid for this dataset.", errors=errors)
    db.execute(
        delete(ColumnRole).where(
            ColumnRole.owner_id == owner_id, ColumnRole.dataset_version_id == version.id
        )
    )
    db.flush()
    for item in roles:
        db.add(
            ColumnRole(
                owner_id=owner_id,
                dataset_id=ds.id,
                dataset_version_id=version.id,
                column_name=item["column"],
                role=item["role"],
                task_type=item.get("task_type"),
                created_by=owner_id,
            )
        )
    db.commit()
    return version, ds


def current_semantics_hash(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID, domain: str
) -> bytes:
    return semantics_hash(domain, roles_snapshot(current_roles(db, owner_id, version_id)))


# ---- Analysis runs --------------------------------------------------------------------------


def _latest_analysis_job(
    db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID, version_id: uuid.UUID
) -> Job | None:
    return db.scalar(
        select(Job)
        .where(
            Job.owner_id == owner_id,
            Job.type == "analysis",
            Job.dataset_id == dataset_id,
            Job.params["version_id"].astext == str(version_id),
        )
        .order_by(Job.queued_at.desc())
        .limit(1)
    )


def request_analysis(
    db: Session, settings: Settings, owner_id: uuid.UUID, version_id: uuid.UUID
) -> Job:
    version, ds = get_owned_version(db, owner_id, version_id)
    db.execute(select(Dataset.id).where(Dataset.id == ds.id).with_for_update())
    current = _latest_analysis_job(db, owner_id, ds.id, version.id)
    if current is not None and current.status in ACTIVE_STATUSES:
        raise Conflict("Diagnostics are already queued or running for this version.")
    job = create_job(db, settings, owner_id, "analysis", ds.id, {"version_id": str(version.id)})
    db.commit()
    return job


def latest_analysis(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID
) -> tuple[AnalysisRun | None, list[FindingRow], Job | None, bytes, Dataset]:
    version, ds = get_owned_version(db, owner_id, version_id)
    run = db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.owner_id == owner_id,
            AnalysisRun.dataset_version_id == version.id,
            AnalysisRun.kind == "diagnostics",
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )
    findings = _findings(db, owner_id, run.id) if run else []
    job = _latest_analysis_job(db, owner_id, ds.id, version.id)
    return run, findings, job, current_semantics_hash(db, owner_id, version.id, ds.domain), ds


def get_analysis(
    db: Session, owner_id: uuid.UUID, analysis_id: uuid.UUID
) -> tuple[AnalysisRun, list[FindingRow], bytes]:
    run = db.scalar(
        select(AnalysisRun).where(
            AnalysisRun.id == analysis_id,
            AnalysisRun.owner_id == owner_id,
            AnalysisRun.kind == "diagnostics",  # audits are read through their training run
        )
    )
    if run is None:
        raise NotFound()
    version, ds = get_owned_version(db, owner_id, run.dataset_version_id)
    return (
        run,
        _findings(db, owner_id, run.id),
        current_semantics_hash(db, owner_id, version.id, ds.domain),
    )


def _findings(db: Session, owner_id: uuid.UUID, run_id: uuid.UUID) -> list[FindingRow]:
    return list(
        db.scalars(
            select(FindingRow)
            .where(FindingRow.owner_id == owner_id, FindingRow.analysis_run_id == run_id)
            .order_by(FindingRow.position)
        )
    )

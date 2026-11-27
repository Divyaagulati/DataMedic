"""Roles and validation context of a version, shared by recommendations and experiments.
Standard library and SQLAlchemy only (the API must not load pandas)."""

import uuid
from typing import Any

from sqlalchemy.orm import Session

from datamedic.analyses.service import current_roles
from datamedic.db.models import DatasetVersion
from datamedic.domains.bootstrap import build_registry
from datamedic.engine.checks.spec import RoleSpec
from datamedic.engine.preprocess.state import ValidationContext


def role_specs() -> dict[str, RoleSpec]:
    return {r.key: r for r in build_registry().all_roles()}


def roles_snapshot(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> dict[str, Any]:
    """``{"roles": {role: [columns]}, "task_type": ...}``, sorted for stable hashing."""
    roles: dict[str, list[str]] = {}
    task_type = None
    for r in current_roles(db, owner_id, version_id):
        roles.setdefault(r.role, []).append(r.column_name)
        if r.role == "target":
            task_type = r.task_type
    return {"roles": {k: sorted(v) for k, v in sorted(roles.items())}, "task_type": task_type}


def validation_context(version: DatasetVersion, snapshot: dict[str, Any]) -> ValidationContext:
    return ValidationContext.from_schema(
        version.schema,
        snapshot["roles"],
        role_specs(),
        n_rows=int(version.n_rows),
        task_type=snapshot.get("task_type"),
    )

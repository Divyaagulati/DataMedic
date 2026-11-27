"""Version lineage helpers shared by the API and the worker (no data libraries).

- ``replay_chain``: the committed transformations that reproduce a version from its original
  file. A ``revert`` version replays the chain of the version it restored, so transformations
  that were undone are never replayed.
- ``copy_roles``: carry column roles to a new version; roles whose column disappeared or no
  longer has an allowed type are reported as lost (never silently dropped).
- ``render_code``: the version's ``pipeline.py``.
"""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from datamedic.db.models import (
    ColumnRole,
    Dataset,
    DatasetVersion,
    OriginalFile,
    TransformationRun,
)
from datamedic.engine.preprocess.codegen import CodegenError
from datamedic.engine.preprocess.pipeline_code import Commit, render_pipeline
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.state import ValidationContext
from datamedic.engine.preprocess.validate import ValidatedStep, validate_steps
from datamedic.errors import Conflict
from datamedic.preprocessing.context import role_specs

MAX_CHAIN = 1_000


def replay_chain(
    db: Session, version: DatasetVersion
) -> tuple[DatasetVersion, list[TransformationRun]]:
    runs: list[TransformationRun] = []
    current: DatasetVersion | None = version
    for _ in range(MAX_CHAIN):
        assert current is not None
        if current.kind == "ingest":
            return current, list(reversed(runs))
        if current.kind == "revert":
            current = db.get(DatasetVersion, current.restored_from_version_id)
        elif current.kind == "transform":
            run = db.get(TransformationRun, current.transformation_run_id)
            if run is None:
                raise Conflict("The version's transformation run is missing.")
            runs.append(run)
            current = db.get(DatasetVersion, current.parent_version_id)
        else:
            raise Conflict("This version kind cannot be replayed.")
        if current is None:
            raise Conflict("The version lineage is incomplete.")
    raise Conflict("The version lineage is too long.")


def roles_of(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> dict[str, Any]:
    roles: dict[str, list[str]] = {}
    task = None
    for r in db.scalars(
        select(ColumnRole).where(
            ColumnRole.owner_id == owner_id, ColumnRole.dataset_version_id == version_id
        )
    ):
        roles.setdefault(r.role, []).append(r.column_name)
        if r.role == "target":
            task = r.task_type
    return {"roles": {k: sorted(v) for k, v in sorted(roles.items())}, "task_type": task}


def copy_roles(
    db: Session,
    owner_id: uuid.UUID,
    dataset_id: uuid.UUID,
    from_version_id: uuid.UUID,
    to_version_id: uuid.UUID,
    schema: list[dict[str, Any]],
) -> dict[str, Any]:
    columns = {c["name"]: c for c in schema}
    specs = role_specs()
    kept, lost = [], []
    for r in db.scalars(
        select(ColumnRole)
        .where(ColumnRole.owner_id == owner_id, ColumnRole.dataset_version_id == from_version_id)
        .order_by(ColumnRole.column_name, ColumnRole.role)
    ):
        info = columns.get(r.column_name)
        spec = specs.get(r.role)
        if info is None:
            lost.append(
                {
                    "column": r.column_name,
                    "role": r.role,
                    "reason": "The column is not in the new version.",
                }
            )
            continue
        if spec is not None and info["inferred_type"] not in spec.allowed_types:
            lost.append(
                {
                    "column": r.column_name,
                    "role": r.role,
                    "reason": "The column's type is no longer allowed for this role.",
                }
            )
            continue
        db.add(
            ColumnRole(
                owner_id=owner_id,
                dataset_id=dataset_id,
                dataset_version_id=to_version_id,
                column_name=r.column_name,
                role=r.role,
                task_type=r.task_type,
                source=r.source,
                created_by=r.created_by,
            )
        )
        kept.append({"column": r.column_name, "role": r.role})
    return {"kept": kept, "lost": lost}


def _context(
    schema: list[dict[str, Any]], snapshot: dict[str, Any], n_rows: int
) -> ValidationContext:
    return ValidationContext.from_schema(
        schema, snapshot["roles"], role_specs(), n_rows=n_rows, task_type=snapshot.get("task_type")
    )


def _revalidate(
    steps: list[dict[str, Any]], ctx: ValidationContext, hashes: list[str] | None
) -> list[ValidatedStep]:
    """Committed data steps must reproduce their recorded hashes exactly. The modelling
    specification (hashes None) is validated against the version's own schema, where
    resolved parameters can legitimately differ from those resolved at commit time."""
    result = validate_steps(
        [Step(s["op"], tuple(s["columns"]), dict(s["params"])) for s in steps], ctx
    )
    if not result.valid or (hashes is not None and [v.step_hash for v in result.steps] != hashes):
        raise CodegenError(
            "A committed step no longer validates to its recorded hash "
            "(engine or catalogue changed)."
        )
    return result.steps


def render_code(db: Session, version: DatasetVersion) -> str:
    root, runs = replay_chain(db, version)
    ds = db.get(Dataset, version.dataset_id)
    original = db.scalar(select(OriginalFile).where(OriginalFile.dataset_id == version.dataset_id))
    if ds is None or original is None or original.detected_format is None:
        raise Conflict("The original file is not available.")
    commits = []
    for run in runs:
        source = db.get(DatasetVersion, run.source_version_id)
        snap = run.spec_snapshot
        if source is None:
            raise Conflict("The version lineage is incomplete.")
        ctx = _context(source.schema, snap["roles"], int(source.n_rows))
        steps = _revalidate(snap["data_steps"], ctx, snap["data_step_hashes"])
        result_version = db.get(DatasetVersion, run.result_version_id)
        commits.append(Commit(result_version.version_number if result_version else 0, steps))
    model: list[ValidatedStep] = []
    spec = version.model_pipeline_spec or []
    if spec:
        content = version
        if version.kind == "revert":
            content = db.get(DatasetVersion, version.restored_from_version_id) or version
        content_run = (
            db.get(TransformationRun, content.transformation_run_id)
            if content.transformation_run_id
            else None
        )
        snapshot = (
            content_run.spec_snapshot["roles"]
            if content_run
            else roles_of(db, version.owner_id, version.id)
        )
        ctx = _context(content.schema, snapshot, int(content.n_rows))
        model = _revalidate(spec, ctx, None)
    seeds = {int((r.seeds or {}).get("seed", 0)) for r in runs} or {0}
    if len(seeds) > 1:
        raise CodegenError("Committed transformations used different seeds.")
    return render_pipeline(
        version_number=version.version_number,
        fmt=original.detected_format,
        declared_extension=ds.declared_extension,
        parse_options=root.parse_options or {},
        commits=commits,
        model_steps=model,
        seed=seeds.pop(),
        libraries={k: str(v) for k, v in (version.library_versions or {}).items()},
    )

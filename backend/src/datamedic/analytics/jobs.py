"""Automated insights job (worker, queue ``analysis``): stores an ``analysis_runs`` row of kind
``insights``. Never writes versions, roles or findings."""

import hashlib
import io
import uuid

from sqlalchemy import select

from datamedic.analyses.semantics import roles_snapshot, semantics_hash
from datamedic.db.models import AnalysisRun, ColumnRole, Dataset, DatasetVersion
from datamedic.errors import INTEGRITY_FAILED
from datamedic.storage import ObjectNotFound
from datamedic.training import privacy
from datamedic.workers.runner import HandlerContext, JobOutcome, register_handler


def _read(storage, version: DatasetVersion) -> bytes:  # type: ignore[no-untyped-def]
    with storage.open_reader(version.object_key) as fh:
        raw: bytes = fh.read()
    if hashlib.sha256(raw).digest() != version.file_sha256:
        raise ValueError("integrity")
    return raw


@register_handler("insights")
def insights(hctx: HandlerContext) -> JobOutcome:
    db, job, storage = hctx.db, hctx.job, hctx.app.storage
    try:
        version_id = uuid.UUID(str(job.params.get("version_id")))
    except ValueError:
        return JobOutcome(False, error_code="VALIDATION_FAILED", error_message="Bad job params.")
    version = db.get(DatasetVersion, version_id)
    ds = db.get(Dataset, version.dataset_id) if version else None
    if version is None or ds is None or ds.deleted_at is not None or version.deleted_at:
        return JobOutcome(True, result={"skipped": "version deleted"})
    try:
        raw = _read(storage, version)
        parent = (
            db.get(DatasetVersion, version.parent_version_id) if version.parent_version_id else None
        )
        parent_raw = _read(storage, parent) if parent is not None else None
    except ObjectNotFound:
        return JobOutcome(False, error_code="NOT_FOUND", error_message="Version data missing.")
    except ValueError:
        return JobOutcome(
            False, error_code=INTEGRITY_FAILED, error_message="Stored version does not match."
        )

    import pandas as pd

    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.analytics.insights import INSIGHTS_VERSION, generate
    from datamedic.engine.versions import library_versions

    rows = list(db.scalars(select(ColumnRole).where(ColumnRole.dataset_version_id == version.id)))
    roles: dict[str, list[str]] = {}
    task_type = None
    for r in rows:
        roles.setdefault(r.role, []).append(r.column_name)
        if r.role == "target":
            task_type = r.task_type
    protected = set(privacy.protected_columns(db, version.owner_id, ds.id))
    bio = db.scalar(
        select(AnalysisRun)
        .where(AnalysisRun.dataset_version_id == version.id, AnalysisRun.kind == "biomedical")
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )
    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    parent_frame = pd.read_parquet(io.BytesIO(parent_raw), engine="pyarrow") if parent_raw else None
    summary = generate(
        frame,
        version.schema,
        roles,
        task_type,
        ds.domain,
        protected,
        ds.domain == "biomedical" or bool(protected),
        parent=parent_frame,
        bio_summary=bio.summary if bio else None,
    )
    snapshot = roles_snapshot(rows)
    run = AnalysisRun(
        owner_id=version.owner_id,
        dataset_id=ds.id,
        dataset_version_id=version.id,
        job_id=job.id,
        kind="insights",
        domain=ds.domain,
        check_set_hash=hashlib.sha256(f"insights@{INSIGHTS_VERSION}".encode()).digest(),
        semantics_hash=semantics_hash(ds.domain, snapshot),
        roles_snapshot=snapshot,
        config={
            "insights_version": INSIGHTS_VERSION,
            "parent_version_id": str(parent.id) if parent else None,
        },
        summary=summary,
        engine_version=ENGINE_VERSION,
        library_versions=library_versions(),
    )
    db.add(run)
    db.flush()
    return JobOutcome(
        True, result={"analysis_run_id": str(run.id), "insights": len(summary["insights"])}
    )

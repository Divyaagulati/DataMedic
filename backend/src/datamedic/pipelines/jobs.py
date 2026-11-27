"""Part E job handlers (worker): impact preview, commit, revert, version comparison.

Commit and revert are the only writers of new versions (DEVELOPMENT_GUIDE.md invariant 9):
1. idempotency: a job whose result already exists (its transformation run or version) ends
   as succeeded without doing anything (``claim_job`` already prevents parallel runs);
2. ``commit_readiness`` (the same checks as the API) runs before any data is read;
3. the source object is verified against its SHA-256; the transformation runs through
   ``commit_transform`` (model-stage steps are refused there and stored as the version's
   modelling specification instead); the result must equal the previewed content hash;
4. the new object is written under a fresh key (never overwriting anything);
5. under the dataset row lock every check runs again (the pipeline, its approvals or the
   head may have changed while the data was processed), then the version, its run, the event
   and the copied roles are inserted and the head moves, all in the job's transaction.

A failure at any point leaves no version, no run and the head unchanged. An object written
before a failure is unreferenced and removed later by orphan reconciliation. Cancellation is
honoured up to the final transaction; a committed version is never reported as cancelled.
Profiling and diagnostics of the new version are queued as follow-up jobs; their failure
never affects the committed version.
"""

import hashlib
import io
import json
import logging
import uuid
from typing import Any

from sqlalchemy import func, select

from datamedic.db.models import (
    Artifact,
    Dataset,
    DatasetVersion,
    Job,
    Pipeline,
    TransformationRun,
    VersionEvent,
)
from datamedic.db.session import bind_user, session_scope
from datamedic.errors import INTEGRITY_FAILED, JOB_CANCELLED, DomainError
from datamedic.jobs.service import create_job
from datamedic.pipelines import service
from datamedic.pipelines.versioning import copy_roles, roles_of
from datamedic.storage import ObjectNotFound, artifact_object_key, version_object_key
from datamedic.workers.runner import HandlerContext, JobOutcome, register_handler

log = logging.getLogger(__name__)


def _fail(code: str, message: str) -> JobOutcome:
    return JobOutcome(False, error_code=code, error_message=message)


def as_outcome(exc: DomainError) -> JobOutcome:
    """A failed check in the worker: the same stable code the API would return."""
    return _fail(exc.code, exc.detail or exc.title)


def _read_version(hctx: HandlerContext, version: DatasetVersion) -> bytes:
    with hctx.app.storage.open_reader(version.object_key) as fh:
        raw = fh.read()
    if hashlib.sha256(raw).digest() != version.file_sha256:
        raise _Integrity()
    return raw


class _Integrity(Exception):
    pass


def _store(hctx: HandlerContext, key: str, data: bytes) -> Any:
    writer = hctx.app.storage.open_writer(key)
    try:
        writer.write(data)
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def _artifact(
    hctx: HandlerContext,
    owner_id: uuid.UUID,
    dataset_id: uuid.UUID,
    owner_type: str,
    owner_ref: uuid.UUID,
    kind: str,
    payload: Any,
) -> Artifact:
    artifact_id = uuid.uuid4()
    data = (
        payload
        if isinstance(payload, bytes)
        else json.dumps(payload, allow_nan=False).encode("utf-8")
    )
    stored = _store(hctx, artifact_object_key(owner_id, dataset_id, artifact_id, "json"), data)
    art = Artifact(
        id=artifact_id,
        owner_id=owner_id,
        dataset_id=dataset_id,
        owner_type=owner_type,
        owner_ref_id=owner_ref,
        kind=kind,
        object_key=stored.key,
        sha256=stored.sha256,
        size_bytes=stored.size_bytes,
    )
    hctx.db.add(art)
    hctx.db.flush()
    return art


def _cancel_requested(hctx: HandlerContext) -> bool:
    with session_scope(hctx.app.sessions) as db:
        bind_user(db, hctx.job.owner_id)
        return db.scalar(select(Job.status).where(Job.id == hctx.job.id)) == "cancel_requested"


def _follow_ups(
    hctx: HandlerContext, ds: Dataset, version_id: uuid.UUID
) -> list[tuple[uuid.UUID, str]]:
    out = []
    for job_type in ("profile", "analysis"):
        job = create_job(
            hctx.db,
            hctx.app.settings,
            ds.owner_id,
            job_type,
            ds.id,
            {"version_id": str(version_id)},
            enforce_limits=False,
        )
        out.append((job.id, job_type))
    return out


# ---- Impact preview ---------------------------------------------------------------------------


@register_handler("impact_preview")
def impact_preview(hctx: HandlerContext) -> JobOutcome:
    db, job = hctx.db, hctx.job
    pipeline = db.get(Pipeline, uuid.UUID(str(job.params["pipeline_id"])))
    if pipeline is None:
        return _fail("NOT_FOUND", "The pipeline no longer exists.")
    expected = str(job.params["spec_hash"])
    steps = service.steps_of(db, pipeline)
    plan = service.plan_for(db, pipeline, steps)
    if not plan.valid or plan.spec_hash != expected:
        return _fail("APPROVAL_STALE", "The pipeline changed before the preview ran; run it again.")
    version = db.get(DatasetVersion, pipeline.source_version_id)
    assert version is not None
    try:
        raw = _read_version(hctx, version)
    except (ObjectNotFound, _Integrity):
        return _fail(INTEGRITY_FAILED, "The source version could not be read intact.")

    import pandas as pd

    from datamedic.engine.errors import EngineError
    from datamedic.engine.preprocess.commit import preview

    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    try:
        summary, rows = preview(
            frame,
            plan,
            sensitive=plan.full.sensitive,
            max_rows=hctx.app.settings.impact_preview_max_rows,
        )
    except EngineError as exc:
        return _fail(exc.code, exc.message)
    art = _artifact(
        hctx,
        pipeline.owner_id,
        pipeline.dataset_id,
        "pipeline",
        pipeline.id,
        "impact_preview",
        {"summary": summary, "rows": rows, "source_version_id": str(version.id)},
    )
    db.refresh(pipeline, with_for_update=True)
    attached = pipeline.spec_hash == expected and pipeline.state in service.EDITABLE
    if attached:
        pipeline.impact_preview_artifact_id = art.id
        pipeline.impact_preview_spec_hash = expected
    db.flush()
    return JobOutcome(
        True,
        result={"artifact_id": str(art.id), "attached": attached, "pipeline_id": str(pipeline.id)},
    )


# ---- Commit -----------------------------------------------------------------------------------


@register_handler("commit")
def commit(hctx: HandlerContext) -> JobOutcome:
    db, job, settings = hctx.db, hctx.job, hctx.app.settings
    p = job.params
    run_id, version_id = uuid.UUID(p["transformation_run_id"]), uuid.UUID(p["version_id"])
    done = db.scalar(select(TransformationRun.id).where(TransformationRun.job_id == job.id))
    if done is not None:
        return JobOutcome(
            True, result={"skipped": "already committed", "transformation_run_id": str(done)}
        )
    pipeline = db.get(Pipeline, uuid.UUID(p["pipeline_id"]))
    if pipeline is None:
        return _fail("NOT_FOUND", "The pipeline no longer exists.")
    try:
        plan, _ = service.commit_readiness(db, settings, pipeline, p["spec_hash"])
    except DomainError as exc:
        return as_outcome(exc)
    if str(pipeline.source_version_id) != p["source_version_id"]:
        return _fail("CONFLICT", "The pipeline's source changed after the commit was queued.")
    source = db.get(DatasetVersion, pipeline.source_version_id)
    ds = db.get(Dataset, pipeline.dataset_id)
    assert source is not None and ds is not None
    payload = service.read_preview(hctx.app.storage, db, pipeline)
    if payload is None:
        return _fail("APPROVAL_STALE", "The impact preview is no longer available.")
    try:
        raw = _read_version(hctx, source)
    except (ObjectNotFound, _Integrity):
        return _fail(INTEGRITY_FAILED, "The source version could not be read intact.")

    import pandas as pd

    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.errors import EngineError
    from datamedic.engine.preprocess.commit import materialise
    from datamedic.engine.preprocess.table_schema import schema_of
    from datamedic.engine.preview import build_preview
    from datamedic.engine.versions import library_versions

    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    try:
        result = materialise(frame, plan.data, seed=0)
    except EngineError as exc:
        return _fail(exc.code, exc.message)
    del frame
    if result.content_hash.hex() != payload["summary"]["expected_content_hash"]:
        return _fail(
            "PREVIEW_MISMATCH",
            "The committed result differs from the approved "
            "preview; nothing was written. Run the preview again.",
        )
    if _cancel_requested(hctx):
        return _fail(JOB_CANCELLED, "Cancelled by user before the version was written.")
    known = {c["name"]: c["inferred_type"] for c in source.schema}
    notes = {c["name"]: list(c.get("notes", [])) for c in source.schema}
    schema = [{**c, "notes": notes.get(c["name"], [])} for c in schema_of(result.frame, known)]
    stored = _store(hctx, version_object_key(ds.owner_id, ds.id, version_id), result.parquet)

    # Final checks under the dataset lock, immediately before the version is created.
    db.refresh(ds, with_for_update=True)
    db.refresh(pipeline, with_for_update=True)
    try:
        plan_now, ready = service.commit_readiness(db, settings, pipeline, p["spec_hash"])
    except DomainError as exc:
        return as_outcome(exc)
    if plan_now.spec_hash != plan.spec_hash:
        return _fail("APPROVAL_STALE", "The pipeline changed while it was being committed.")
    number = (
        int(
            db.scalar(
                select(func.max(DatasetVersion.version_number)).where(
                    DatasetVersion.dataset_id == ds.id
                )
            )
            or 0
        )
        + 1
    )
    data_steps = [s for s in plan.steps if s.stage == "data"]
    snapshot = {
        "steps": [{**s.step.as_dict(), "stage": s.stage} for s in plan.steps],
        "roles": roles_of(db, ds.owner_id, source.id),
        "data_steps": [s.step.as_dict() for s in data_steps],
        "data_step_hashes": [v.step_hash for v in plan.data.steps],
        "data_ops": [
            {
                "op": v.spec.op,
                "op_version": v.spec.version,
                "risk": v.spec.risk,
                "destructive": v.spec.destructive,
                "kind": v.spec.kind,
            }
            for v in plan.data.steps
        ],
        "preview_artifact_id": str(pipeline.impact_preview_artifact_id),
    }
    model_spec = plan.model_spec()
    # Insert-only rows: the version first (its run reference is a deferred key), then the
    # copied roles, then the run with its final roles report.
    version = DatasetVersion(
        id=version_id,
        owner_id=ds.owner_id,
        dataset_id=ds.id,
        version_number=number,
        kind="transform",
        parent_version_id=source.id,
        transformation_run_id=run_id,
        object_key=stored.key,
        file_sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        content_hash=result.content_hash,
        n_rows=len(result.frame),
        n_cols=len(schema),
        schema=schema,
        fitted_on_full_data=result.fitted_on_full_data or bool(source.fitted_on_full_data),
        model_pipeline_spec=model_spec or None,
        engine_version=ENGINE_VERSION,
        library_versions=library_versions(),
        created_by=ds.owner_id,
    )
    db.add(version)
    db.flush()
    roles = copy_roles(db, ds.owner_id, ds.id, source.id, version.id, schema)
    run = TransformationRun(
        id=run_id,
        owner_id=ds.owner_id,
        dataset_id=ds.id,
        pipeline_id=pipeline.id,
        job_id=job.id,
        source_version_id=source.id,
        result_version_id=version_id,
        spec_snapshot=snapshot,
        spec_hash=p["spec_hash"],
        approval_ids=ready.approval_ids,
        step_reports=result.reports,
        model_pipeline_spec=model_spec,
        roles_report=roles,
        seeds={"seed": 0},
        engine_version=ENGINE_VERSION,
        library_versions=library_versions(),
        executed_by=ds.owner_id,
    )
    db.add(run)
    db.flush()
    _artifact(
        hctx,
        ds.owner_id,
        ds.id,
        "dataset_version",
        version.id,
        "preview",
        build_preview(result.frame, schema),
    )
    db.add(
        VersionEvent(
            owner_id=ds.owner_id,
            dataset_id=ds.id,
            event="created",
            version_id=version.id,
            related_version_id=source.id,
            actor_id=ds.owner_id,
            details={"transformation_run_id": str(run_id), "roles": roles},
        )
    )
    ds.head_version_id = version.id
    pipeline.state = "committed"
    pipeline.committed_version_id = version.id
    db.flush()
    follow = _follow_ups(hctx, ds, version.id)
    return JobOutcome(
        True,
        result={
            "version_id": str(version.id),
            "version_number": number,
            "transformation_run_id": str(run_id),
            "roles_lost": len(roles["lost"]),
        },
        follow_ups=follow,
    )


# ---- Revert -----------------------------------------------------------------------------------


@register_handler("revert")
def revert(hctx: HandlerContext) -> JobOutcome:
    db, job, settings = hctx.db, hctx.job, hctx.app.settings
    p = job.params
    version_id = uuid.UUID(p["version_id"])
    if db.get(DatasetVersion, version_id) is not None:
        return JobOutcome(
            True, result={"skipped": "already reverted", "version_id": str(version_id)}
        )
    target = db.get(DatasetVersion, uuid.UUID(p["target_version_id"]))
    if target is None or target.deleted_at is not None:
        return _fail("NOT_FOUND", "The version to restore no longer exists.")
    ds = db.get(Dataset, target.dataset_id)
    if ds is None or ds.deleted_at is not None:
        return JobOutcome(True, result={"skipped": "dataset deleted"})
    if str(ds.head_version_id) != p["expected_head"]:
        return _fail("CONFLICT", "The dataset changed after the revert was requested.")
    try:
        raw = _read_version(hctx, target)
    except (ObjectNotFound, _Integrity):
        return _fail(INTEGRITY_FAILED, "The version to restore could not be read intact.")
    if _cancel_requested(hctx):
        return _fail(JOB_CANCELLED, "Cancelled by user before the version was written.")
    stored = _store(hctx, version_object_key(ds.owner_id, ds.id, version_id), raw)
    preview = db.scalar(
        select(Artifact).where(
            Artifact.owner_type == "dataset_version",
            Artifact.owner_ref_id == target.id,
            Artifact.kind == "preview",
        )
    )
    preview_bytes = None
    if preview is not None:
        with hctx.app.storage.open_reader(preview.object_key) as fh:
            preview_bytes = fh.read()

    db.refresh(ds, with_for_update=True)
    if str(ds.head_version_id) != p["expected_head"]:
        return _fail("CONFLICT", "The dataset changed while the revert ran.")
    try:
        service.check_quota(db, settings, ds.id)
    except DomainError as exc:
        return as_outcome(exc)
    head = db.get(DatasetVersion, ds.head_version_id)
    assert head is not None
    number = (
        int(
            db.scalar(
                select(func.max(DatasetVersion.version_number)).where(
                    DatasetVersion.dataset_id == ds.id
                )
            )
            or 0
        )
        + 1
    )
    version = DatasetVersion(
        id=version_id,
        owner_id=ds.owner_id,
        dataset_id=ds.id,
        version_number=number,
        kind="revert",
        parent_version_id=head.id,
        restored_from_version_id=target.id,
        object_key=stored.key,
        file_sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        content_hash=target.content_hash,
        n_rows=target.n_rows,
        n_cols=target.n_cols,
        schema=target.schema,
        fitted_on_full_data=target.fitted_on_full_data,
        model_pipeline_spec=target.model_pipeline_spec,
        engine_version=target.engine_version,
        library_versions=target.library_versions,
        created_by=ds.owner_id,
    )
    db.add(version)
    db.flush()
    roles = copy_roles(db, ds.owner_id, ds.id, target.id, version.id, target.schema)
    if preview_bytes is not None:
        _artifact(hctx, ds.owner_id, ds.id, "dataset_version", version.id, "preview", preview_bytes)
    db.add(
        VersionEvent(
            owner_id=ds.owner_id,
            dataset_id=ds.id,
            event="reverted",
            version_id=version.id,
            related_version_id=target.id,
            actor_id=ds.owner_id,
            details={"previous_head": str(head.id), "roles": roles},
        )
    )
    ds.head_version_id = version.id
    db.flush()
    follow = _follow_ups(hctx, ds, version.id)
    return JobOutcome(
        True,
        result={
            "version_id": str(version.id),
            "version_number": number,
            "restored_from": str(target.id),
        },
        follow_ups=follow,
    )


# ---- Comparison -------------------------------------------------------------------------------


@register_handler("compare_versions")
def compare_versions(hctx: HandlerContext) -> JobOutcome:
    db, job = hctx.db, hctx.job
    va = db.get(DatasetVersion, uuid.UUID(job.params["version_a"]))
    vb = db.get(DatasetVersion, uuid.UUID(job.params["version_b"]))
    if va is None or vb is None:
        return _fail("NOT_FOUND", "A version no longer exists.")
    try:
        raw_a, raw_b = _read_version(hctx, va), _read_version(hctx, vb)
    except (ObjectNotFound, _Integrity):
        return _fail(INTEGRITY_FAILED, "A version could not be read intact.")

    import pandas as pd

    from datamedic.engine.preprocess.compare import compare

    a = pd.read_parquet(io.BytesIO(raw_a), engine="pyarrow")
    b = pd.read_parquet(io.BytesIO(raw_b), engine="pyarrow")
    result = compare(a, b)
    lo, hi = sorted((va.version_number, vb.version_number))
    lineage = [
        {
            "version_number": v.version_number,
            "kind": v.kind,
            "id": str(v.id),
            "fitted_on_full_data": v.fitted_on_full_data,
            "restored_from_version_id": str(v.restored_from_version_id)
            if v.restored_from_version_id
            else None,
            "transformation_run_id": str(v.transformation_run_id)
            if v.transformation_run_id
            else None,
        }
        for v in db.scalars(
            select(DatasetVersion)
            .where(
                DatasetVersion.dataset_id == va.dataset_id,
                DatasetVersion.version_number.between(lo, hi),
            )
            .order_by(DatasetVersion.version_number)
        )
    ]
    result.update(
        version_a=str(va.id),
        version_b=str(vb.id),
        version_numbers=[va.version_number, vb.version_number],
        lineage=lineage,
    )
    art = _artifact(
        hctx, vb.owner_id, vb.dataset_id, "dataset_version", vb.id, "version_comparison", result
    )
    return JobOutcome(
        True, result={"artifact_id": str(art.id), "version_a": str(va.id), "version_b": str(vb.id)}
    )

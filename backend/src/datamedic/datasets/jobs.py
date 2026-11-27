"""Background job handlers for datasets. They run under RLS bound to the job's owner.

ingest  : integrity + content validation, sandboxed parse, version 0, preview, then chains
          a profile job.
profile : BuiltinProfiler over a version; the result is stored in version_profiles.
purge   : delete every stored object and row of a soft-deleted dataset, then verify.
"""

import hashlib
import io
import json
import logging
import tempfile
import uuid
from pathlib import Path
from typing import Any, BinaryIO

from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.datasets.file_validation import validate_file
from datamedic.datasets.parsing import copy_original, run_parse_sandbox
from datamedic.db.models import (
    AnalysisRun,
    Approval,
    Artifact,
    ColumnRole,
    Dataset,
    DatasetVersion,
    Experiment,
    ExperimentCandidate,
    ExperimentResult,
    FindingRow,
    Job,
    ModelEvaluation,
    OriginalFile,
    Pipeline,
    PipelineStep,
    Recommendation,
    RecommendationFinding,
    RecommendationSet,
    SplitDefinition,
    TrainingRun,
    TransformationRun,
    VersionEvent,
    VersionProfile,
)
from datamedic.errors import INTEGRITY_FAILED
from datamedic.jobs.service import create_job
from datamedic.storage import (
    ObjectNotFound,
    ObjectStorage,
    StoredObject,
    artifact_object_key,
    dataset_prefix,
    version_object_key,
)
from datamedic.storage.keys import AREAS
from datamedic.workers.runner import (
    HandlerContext,
    JobOutcome,
    register_failure_hook,
    register_handler,
)

log = logging.getLogger(__name__)

_CHUNK = 1024 * 1024
# Parse failures that mean "this file is not acceptable" (rejected) rather than "this file
# could not be parsed with the current options" (parse_failed).
_REJECT_CODES = frozenset({"LIMIT_EXCEEDED", "MALICIOUS_FILE", "UNSUPPORTED_FORMAT"})


def _sha256_of(fh: BinaryIO) -> bytes:
    digest = hashlib.sha256()
    while chunk := fh.read(_CHUNK):
        digest.update(chunk)
    return digest.digest()


def _store_file(storage: ObjectStorage, key: str, path: Path) -> StoredObject:
    writer = storage.open_writer(key)
    try:
        with open(path, "rb") as fh:
            while chunk := fh.read(_CHUNK):
                writer.write(chunk)
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def _store_bytes(storage: ObjectStorage, key: str, data: bytes) -> StoredObject:
    writer = storage.open_writer(key)
    try:
        writer.write(data)
        return writer.commit()
    except BaseException:
        writer.abort()
        raise


def _parse_into_version(
    db: Session,
    settings: Settings,
    storage: ObjectStorage,
    ds: Dataset,
    original: OriginalFile,
    job: Job,
) -> JobOutcome:
    fmt = original.detected_format
    assert fmt is not None
    with tempfile.TemporaryDirectory(prefix="dm-parse-") as tmp:
        workdir = Path(tmp)
        source = workdir / "original"
        copy_original(storage, original.object_key, source)
        outcome = run_parse_sandbox(settings, source, workdir, fmt, ds.declared_extension)
        if not outcome.ok:
            ds.status = "rejected" if outcome.code in _REJECT_CODES else "parse_failed"
            log.info("parse failed", extra={"job_id": str(job.id), "code": outcome.code})
            return JobOutcome(False, error_code=outcome.code, error_message=outcome.message)
        assert outcome.result is not None and outcome.parquet_path is not None
        result: dict[str, Any] = outcome.result

        version_id = uuid.uuid4()
        stored = _store_file(
            storage, version_object_key(ds.owner_id, ds.id, version_id), outcome.parquet_path
        )

    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.versions import library_versions

    # Re-check under a row lock right before creating version 0: a retried or late-finishing
    # duplicate ingest must never create a second version (the unique version number would
    # also reject it, but this way it ends cleanly). The lock is held only to commit.
    db.refresh(ds, with_for_update=True)
    if ds.head_version_id is not None or ds.deleted_at is not None:
        log.info("ingest result discarded: dataset already parsed", extra={"job_id": str(job.id)})
        return JobOutcome(True, result={"skipped": "already parsed"})

    version = DatasetVersion(
        id=version_id,
        owner_id=ds.owner_id,
        dataset_id=ds.id,
        version_number=0,
        kind="ingest",
        parse_options=result["parse_options"],
        parse_report=result["report"],
        object_key=stored.key,
        file_sha256=stored.sha256,
        size_bytes=stored.size_bytes,
        content_hash=bytes.fromhex(result["content_hash"]),
        n_rows=int(result["n_rows"]),
        n_cols=int(result["n_cols"]),
        schema=result["schema"],
        engine_version=ENGINE_VERSION,
        library_versions=library_versions(),
        created_by=ds.owner_id,
    )
    db.add(version)
    db.flush()  # the head-version trigger needs the version row to exist

    artifact_id = uuid.uuid4()
    preview = _store_bytes(
        storage,
        artifact_object_key(ds.owner_id, ds.id, artifact_id, "json"),
        json.dumps(result["preview"]).encode("utf-8"),
    )
    db.add(
        Artifact(
            id=artifact_id,
            owner_id=ds.owner_id,
            dataset_id=ds.id,
            owner_type="dataset_version",
            owner_ref_id=version.id,
            kind="preview",
            object_key=preview.key,
            sha256=preview.sha256,
            size_bytes=preview.size_bytes,
        )
    )
    ds.head_version_id = version.id
    ds.status = "ready"
    profile_job = create_job(
        db,
        settings,
        ds.owner_id,
        "profile",
        ds.id,
        {"version_id": str(version.id)},
        enforce_limits=False,  # chained system job; the ingest already passed admission
    )
    analysis_job = create_job(
        db,
        settings,
        ds.owner_id,
        "analysis",
        ds.id,
        {"version_id": str(version.id)},
        enforce_limits=False,
    )
    return JobOutcome(
        True,
        result={
            "detected_format": fmt,
            "version_id": str(version.id),
            "n_rows": version.n_rows,
            "n_cols": version.n_cols,
            "profile_job_id": str(profile_job.id),
            "analysis_job_id": str(analysis_job.id),
        },
        follow_ups=[(profile_job.id, "profile"), (analysis_job.id, "analysis")],
    )


@register_handler("ingest")
def ingest(hctx: HandlerContext) -> JobOutcome:
    db, job, storage = hctx.db, hctx.job, hctx.app.storage
    ds = db.get(Dataset, job.dataset_id) if job.dataset_id else None
    if ds is None or ds.deleted_at is not None:
        return JobOutcome(True, result={"skipped": "dataset deleted"})
    if ds.head_version_id is not None:
        return JobOutcome(True, result={"skipped": "already parsed"})  # duplicate delivery
    original = db.scalar(select(OriginalFile).where(OriginalFile.dataset_id == ds.id))
    if original is None:
        return JobOutcome(False, error_code="NOT_FOUND", error_message="No uploaded file.")

    try:
        with storage.open_reader(original.object_key) as fh:
            if _sha256_of(fh) != original.sha256:
                ds.status = "rejected"
                return JobOutcome(
                    False,
                    error_code=INTEGRITY_FAILED,
                    error_message="Stored file does not match its recorded hash.",
                )
            result = validate_file(fh, ds.declared_extension, original.size_bytes)
    except ObjectNotFound:
        ds.status = "rejected"
        return JobOutcome(False, error_code="NOT_FOUND", error_message="Uploaded file missing.")

    if not result.ok:
        ds.status = "rejected"
        log.info("upload rejected", extra={"job_id": str(job.id), "code": result.code})
        return JobOutcome(False, error_code=result.code, error_message=result.reason)
    if original.detected_format is None:
        original.detected_format = result.detected_format
    return _parse_into_version(db, hctx.app.settings, storage, ds, original, job)


@register_failure_hook("ingest")
def ingest_crashed(db: Session, job: Job) -> None:
    if job.dataset_id is None:
        return
    db.execute(
        update(Dataset)
        .where(Dataset.id == job.dataset_id, Dataset.status == "validating")
        .values(status="parse_failed")
    )


@register_handler("profile")
def profile(hctx: HandlerContext) -> JobOutcome:
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
        with storage.open_reader(version.object_key) as fh:
            raw = fh.read()
    except ObjectNotFound:
        return JobOutcome(False, error_code="NOT_FOUND", error_message="Version data missing.")
    if hashlib.sha256(raw).digest() != version.file_sha256:
        return JobOutcome(
            False,
            error_code=INTEGRITY_FAILED,
            error_message="Stored version does not match its recorded hash.",
        )

    import pandas as pd

    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.adapters.builtin_profiler import BuiltinProfiler
    from datamedic.engine.versions import library_versions

    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    profiler = BuiltinProfiler()
    result = profiler.profile(frame, version.schema, seed=0)
    record = VersionProfile(
        owner_id=version.owner_id,
        dataset_id=version.dataset_id,
        dataset_version_id=version.id,
        job_id=job.id,
        profiler=result["profiler"],
        profiler_version=result["profiler_version"],
        summary=result["summary"],
        columns=result["columns"],
        correlations=result["correlations"],
        sampling=result["sampling"],
        limitations=result["limitations"],
        engine_version=ENGINE_VERSION,
        library_versions=library_versions(),
    )
    db.add(record)
    db.flush()
    return JobOutcome(True, result={"version_id": str(version.id), "profile_id": str(record.id)})


@register_handler("purge")
def purge(hctx: HandlerContext) -> JobOutcome:
    """Delete every stored object and row of a soft-deleted dataset, then verify."""
    db, job, storage = hctx.db, hctx.job, hctx.app.storage
    dataset_id = job.dataset_id
    if dataset_id is None:
        return JobOutcome(True, result={"objects_deleted": 0, "already_purged": True})
    ds = db.get(Dataset, dataset_id)
    if ds is None:
        return JobOutcome(True, result={"objects_deleted": 0, "already_purged": True})
    if ds.deleted_at is None:
        return JobOutcome(False, error_code="CONFLICT", error_message="Dataset is not deleted.")
    deleted = 0
    for area in AREAS:
        deleted += storage.delete_prefix(dataset_prefix(area, ds.owner_id, ds.id))
    leftovers = sum(len(storage.list_prefix(dataset_prefix(a, ds.owner_id, ds.id))) for a in AREAS)
    if leftovers:
        return JobOutcome(
            False,
            error_code="PURGE_INCOMPLETE",
            error_message="Some stored objects could not be deleted.",
            retryable=True,
        )
    # Rows in foreign-key order; the head pointer goes first.
    ds.head_version_id = None
    db.flush()
    # Phase 5 rows first: audits reference training runs, which reference splits,
    # artifacts and versions.
    audit_runs = select(AnalysisRun.id).where(
        AnalysisRun.dataset_id == ds.id, AnalysisRun.kind == "reliability_audit"
    )
    db.execute(delete(FindingRow).where(FindingRow.analysis_run_id.in_(audit_runs)))
    db.execute(
        delete(AnalysisRun).where(
            AnalysisRun.dataset_id == ds.id, AnalysisRun.kind == "reliability_audit"
        )
    )
    db.execute(delete(ModelEvaluation).where(ModelEvaluation.dataset_id == ds.id))
    db.execute(delete(TrainingRun).where(TrainingRun.dataset_id == ds.id))
    # Part E rows next (versions and runs reference each other through deferred keys).
    db.execute(delete(VersionEvent).where(VersionEvent.dataset_id == ds.id))
    db.execute(delete(Approval).where(Approval.dataset_id == ds.id))
    db.execute(delete(TransformationRun).where(TransformationRun.dataset_id == ds.id))
    db.execute(delete(PipelineStep).where(PipelineStep.dataset_id == ds.id))
    db.execute(delete(Pipeline).where(Pipeline.dataset_id == ds.id))
    # Part D rows next: they reference findings, runs, profiles, artifacts and versions.
    db.execute(delete(ExperimentResult).where(ExperimentResult.dataset_id == ds.id))
    db.execute(delete(ExperimentCandidate).where(ExperimentCandidate.dataset_id == ds.id))
    db.execute(delete(Experiment).where(Experiment.dataset_id == ds.id))
    db.execute(delete(SplitDefinition).where(SplitDefinition.dataset_id == ds.id))
    db.execute(delete(RecommendationFinding).where(RecommendationFinding.dataset_id == ds.id))
    db.execute(delete(Recommendation).where(Recommendation.dataset_id == ds.id))
    db.execute(delete(RecommendationSet).where(RecommendationSet.dataset_id == ds.id))
    db.execute(delete(FindingRow).where(FindingRow.dataset_id == ds.id))
    db.execute(delete(AnalysisRun).where(AnalysisRun.dataset_id == ds.id))
    db.execute(delete(ColumnRole).where(ColumnRole.dataset_id == ds.id))
    db.execute(delete(VersionProfile).where(VersionProfile.dataset_id == ds.id))
    db.execute(delete(Artifact).where(Artifact.dataset_id == ds.id))
    db.execute(delete(DatasetVersion).where(DatasetVersion.dataset_id == ds.id))
    db.execute(delete(OriginalFile).where(OriginalFile.dataset_id == ds.id))
    db.delete(ds)  # jobs.dataset_id is SET NULL by the foreign key
    return JobOutcome(True, result={"objects_deleted": deleted, "verified_empty": True})

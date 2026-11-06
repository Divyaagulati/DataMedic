"""Background job handlers for datasets. They run under RLS bound to the job's owner."""

import hashlib
import logging

from sqlalchemy import delete, select

from datamedic.datasets.file_validation import validate_file
from datamedic.db.models import Dataset, OriginalFile
from datamedic.errors import INTEGRITY_FAILED
from datamedic.storage import ObjectNotFound, dataset_prefix
from datamedic.storage.keys import AREAS
from datamedic.workers.runner import HandlerContext, JobOutcome, register_handler

log = logging.getLogger(__name__)

_CHUNK = 1024 * 1024


@register_handler("ingest")
def ingest(hctx: HandlerContext) -> JobOutcome:
    """Phase 1 ingest = integrity + content validation. Parsing to version 0 is Phase 3."""
    db, job, storage = hctx.db, hctx.job, hctx.app.storage
    ds = db.get(Dataset, job.dataset_id) if job.dataset_id else None
    if ds is None or ds.deleted_at is not None:
        return JobOutcome(True, result={"skipped": "dataset deleted"})
    original = db.scalar(select(OriginalFile).where(OriginalFile.dataset_id == ds.id))
    if original is None:
        return JobOutcome(False, error_code="NOT_FOUND", error_message="No uploaded file.")

    job.progress = {"phase": "validating", "pct": 10}
    try:
        with storage.open_reader(original.object_key) as fh:
            digest = hashlib.sha256()
            while chunk := fh.read(_CHUNK):
                digest.update(chunk)
            if digest.digest() != original.sha256:
                ds.status = "rejected"
                return JobOutcome(
                    False,
                    error_code=INTEGRITY_FAILED,
                    error_message="Stored file does not match its recorded hash.",
                )
            result = validate_file(fh, ds.declared_extension, original.size_bytes)
    except ObjectNotFound:
        return JobOutcome(False, error_code="NOT_FOUND", error_message="Uploaded file missing.")

    if not result.ok:
        ds.status = "rejected"
        log.info("upload rejected", extra={"job_id": str(job.id), "code": result.code})
        return JobOutcome(False, error_code=result.code, error_message=result.reason)
    original.detected_format = result.detected_format
    ds.status = "validated"
    return JobOutcome(True, result={"detected_format": result.detected_format})


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
    db.execute(delete(OriginalFile).where(OriginalFile.dataset_id == ds.id))
    db.delete(ds)  # jobs.dataset_id is SET NULL by the foreign key
    return JobOutcome(True, result={"objects_deleted": deleted, "verified_empty": True})

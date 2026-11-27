"""Export job (worker): writes a CSV of one dataset version as an owner-only artifact.

- The source version is read and verified against its hash; it is never modified.
- Internal ``_dm_`` columns are dropped. Sensitive columns (sensitive on any version of the
  dataset, or derived from one) are dropped unless the request explicitly included them with an
  acknowledgement (checked by the API and again here).
- Spreadsheet formula injection: text cells starting with ``= + - @``, a tab or a carriage
  return are prefixed with an apostrophe (OWASP CSV injection guidance). Numbers are unchanged.
"""

import hashlib
import io
import uuid

from datamedic.db.models import Artifact, Dataset, DatasetVersion
from datamedic.errors import INTEGRITY_FAILED
from datamedic.storage import ObjectNotFound, artifact_object_key
from datamedic.training import privacy
from datamedic.workers.runner import HandlerContext, JobOutcome, register_handler

FORMULA_PREFIXES = ("=", "+", "-", "@", "\t", "\r")


def neutralise(value: object) -> object:
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


@register_handler("export")
def export(hctx: HandlerContext) -> JobOutcome:
    db, job, storage = hctx.db, hctx.job, hctx.app.storage
    params = job.params or {}
    try:
        version_id = uuid.UUID(str(params.get("version_id")))
    except ValueError:
        return JobOutcome(False, error_code="VALIDATION_FAILED", error_message="Bad job params.")
    if params.get("format") != "csv" or params.get("acknowledged") is not True:
        return JobOutcome(False, error_code="VALIDATION_FAILED", error_message="Bad job params.")
    include_sensitive = params.get("include_sensitive") is True
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
            False, error_code=INTEGRITY_FAILED, error_message="Stored version does not match."
        )

    import pandas as pd

    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    sensitive = set(privacy.protected_columns(db, version.owner_id, ds.id))
    drop = [c for c in frame.columns if str(c).startswith("_dm_")]
    excluded = [] if include_sensitive else sorted(c for c in frame.columns if c in sensitive)
    frame = frame.drop(columns=drop + excluded)
    for c in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c]):
            frame[c] = frame[c].map(neutralise, na_action="ignore")
    payload = frame.to_csv(index=False).encode("utf-8")
    artifact_id = uuid.uuid4()
    key = artifact_object_key(version.owner_id, ds.id, artifact_id, "csv")
    writer = storage.open_writer(key)
    try:
        writer.write(payload)
        stored = writer.commit()
    except BaseException:
        writer.abort()
        raise
    db.add(
        Artifact(
            id=artifact_id,
            owner_id=version.owner_id,
            dataset_id=ds.id,
            owner_type="dataset_version",
            owner_ref_id=version.id,
            kind="export",
            object_key=key,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
        )
    )
    db.flush()
    return JobOutcome(
        True,
        result={
            "artifact_id": str(artifact_id),
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "excluded_sensitive_columns": len(excluded),
            "sensitive_included": include_sensitive and bool(sensitive & set(frame.columns)),
        },
    )

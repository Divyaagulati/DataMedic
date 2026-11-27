"""Biomedical summary job (worker, queue ``analysis``): a descriptive cohort summary stored as an
``analysis_runs`` row of kind ``biomedical``. Never writes versions, roles or findings."""

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


@register_handler("biomedical_summary")
def biomedical_summary(hctx: HandlerContext) -> JobOutcome:
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
            False, error_code=INTEGRITY_FAILED, error_message="Stored version does not match."
        )

    import pandas as pd

    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.bio.summary import SUMMARY_VERSION, summarise
    from datamedic.engine.ml.audit import _key_like_repeats
    from datamedic.engine.versions import library_versions

    roles_rows = list(
        db.scalars(select(ColumnRole).where(ColumnRole.dataset_version_id == version.id))
    )
    roles: dict[str, list[str]] = {}
    task_type = None
    for r in roles_rows:
        roles.setdefault(r.role, []).append(r.column_name)
        if r.role == "target":
            task_type = r.task_type
    # Conservative sensitivity (owner policy): sensitive on any version, or derived from one.
    sensitive = set(privacy.protected_columns(db, version.owner_id, ds.id))
    numeric = [c["name"] for c in version.schema if c.get("inferred_type") == "numeric"]
    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    declared = {c for cols in roles.values() for c in cols}
    key_like = []
    for c in frame.columns:
        if c.startswith("_dm_") or c in declared:
            continue
        rows_per = _key_like_repeats(frame[c])
        if rows_per is not None:
            key_like.append({"column": c, "rows_per_value": rows_per})
    summary = summarise(frame, roles, task_type, sensitive, numeric, key_like)
    snapshot = roles_snapshot(roles_rows)
    run = AnalysisRun(
        owner_id=version.owner_id,
        dataset_id=ds.id,
        dataset_version_id=version.id,
        job_id=job.id,
        kind="biomedical",
        domain=ds.domain,
        check_set_hash=hashlib.sha256(f"biomedical_summary@{SUMMARY_VERSION}".encode()).digest(),
        semantics_hash=semantics_hash(ds.domain, snapshot),
        roles_snapshot=snapshot,
        config={"summary_version": SUMMARY_VERSION, "min_cell": summary["min_cell"]},
        summary=summary,
        engine_version=ENGINE_VERSION,
        library_versions=library_versions(),
    )
    db.add(run)
    db.flush()
    return JobOutcome(True, result={"analysis_run_id": str(run.id)})

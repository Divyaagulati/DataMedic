"""Dataset operations. Every lookup is scoped to the authenticated owner (and RLS)."""

import uuid
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.datasets.file_validation import extension_of
from datamedic.db.models import Dataset, DatasetVersion, Job, OriginalFile
from datamedic.errors import (
    Conflict,
    NotFound,
    QuotaExceeded,
    UnsupportedFormat,
    UploadTooLarge,
)
from datamedic.jobs.service import create_job, ensure_job_capacity
from datamedic.storage import StoredObject


def get_owned(
    db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID, *, for_update: bool = False
) -> Dataset:
    q = select(Dataset).where(
        Dataset.id == dataset_id, Dataset.owner_id == owner_id, Dataset.deleted_at.is_(None)
    )
    if for_update:
        q = q.with_for_update()
    ds = db.scalar(q)
    if ds is None:
        raise NotFound()
    return ds


def detected_format(db: Session, dataset_id: uuid.UUID) -> str | None:
    return db.scalar(
        select(OriginalFile.detected_format).where(OriginalFile.dataset_id == dataset_id)
    )


def create_dataset(
    db: Session, settings: Settings, owner_id: uuid.UUID, name: str, filename: str, size_bytes: int
) -> Dataset:
    ext = extension_of(filename)
    if ext is None:
        raise UnsupportedFormat(
            "Supported extensions: csv, tsv, txt, xlsx, xls, parquet, json, jsonl."
        )
    if size_bytes > settings.max_upload_bytes:
        raise UploadTooLarge(f"Maximum upload size is {settings.max_upload_bytes} bytes.")
    count = (
        db.scalar(
            select(func.count())
            .select_from(Dataset)
            .where(Dataset.owner_id == owner_id, Dataset.deleted_at.is_(None))
        )
        or 0
    )
    if count >= settings.max_datasets_per_user:
        raise QuotaExceeded(f"Dataset limit reached ({settings.max_datasets_per_user}).")
    ds = Dataset(
        owner_id=owner_id,
        name=name.strip(),
        declared_filename=filename,
        declared_extension=ext,
        declared_size_bytes=size_bytes,
    )
    db.add(ds)
    db.commit()
    return ds


def begin_upload(
    db: Session, settings: Settings, owner_id: uuid.UUID, dataset_id: uuid.UUID
) -> Dataset:
    """Claim the dataset for a single upload (short transaction; no lock held while streaming)."""
    ds = get_owned(db, owner_id, dataset_id, for_update=True)
    if ds.status != "upload_pending":
        raise Conflict("Content was already uploaded for this dataset.")
    # Admission control before accepting bytes, so a full job queue does not waste an upload.
    ensure_job_capacity(db, settings, owner_id)
    ds.status = "validating"
    ds.updated_at = datetime.now(UTC)
    db.commit()
    return ds


def abort_upload(db: Session, ds: Dataset) -> None:
    db.rollback()
    ds.status = "upload_pending"
    db.commit()


def register_upload(db: Session, settings: Settings, ds: Dataset, stored: StoredObject) -> Job:
    db.add(
        OriginalFile(
            owner_id=ds.owner_id,
            dataset_id=ds.id,
            object_key=stored.key,
            sha256=stored.sha256,
            size_bytes=stored.size_bytes,
            original_filename=ds.declared_filename,
        )
    )
    job = create_job(db, settings, ds.owner_id, "ingest", ds.id)
    db.commit()
    return job


def head_version(db: Session, ds: Dataset) -> DatasetVersion | None:
    if ds.head_version_id is None:
        return None
    return db.scalar(
        select(DatasetVersion).where(
            DatasetVersion.id == ds.head_version_id, DatasetVersion.owner_id == ds.owner_id
        )
    )


def list_datasets(
    db: Session, owner_id: uuid.UUID
) -> list[tuple[Dataset, str | None, DatasetVersion | None]]:
    rows = db.execute(
        select(Dataset, OriginalFile.detected_format, DatasetVersion)
        .outerjoin(OriginalFile, OriginalFile.dataset_id == Dataset.id)
        .outerjoin(DatasetVersion, DatasetVersion.id == Dataset.head_version_id)
        .where(Dataset.owner_id == owner_id, Dataset.deleted_at.is_(None))
        .order_by(Dataset.created_at.desc())
    ).all()
    return [(ds, fmt, head) for ds, fmt, head in rows]


def queue_parse(
    db: Session, settings: Settings, owner_id: uuid.UUID, dataset_id: uuid.UUID
) -> tuple[Dataset, Job]:
    """Queue parsing again for a stored upload that has no version yet: a dataset validated
    before parsing existed, or one whose parse failed (e.g. after a worker crash)."""
    ds = get_owned(db, owner_id, dataset_id, for_update=True)
    if ds.status not in ("validated", "parse_failed") or ds.head_version_id is not None:
        raise Conflict("Parsing can only be queued for an uploaded dataset without a version.")
    ensure_job_capacity(db, settings, owner_id)
    ds.status = "validating"
    ds.updated_at = datetime.now(UTC)
    job = create_job(db, settings, owner_id, "ingest", ds.id)
    db.commit()
    return ds, job


def soft_delete(db: Session, settings: Settings, owner_id: uuid.UUID, dataset_id: uuid.UUID) -> Job:
    ds = get_owned(db, owner_id, dataset_id, for_update=True)
    ds.deleted_at = datetime.now(UTC)
    ds.status = "deleting"
    job = create_job(db, settings, owner_id, "purge", ds.id, enforce_limits=False)
    db.commit()
    return job

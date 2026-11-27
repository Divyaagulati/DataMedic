"""Version operations. Every lookup is scoped to the authenticated owner (and RLS)."""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.db.models import Artifact, Dataset, DatasetVersion, Job, VersionProfile
from datamedic.errors import Conflict, NotFound
from datamedic.jobs.registry import ACTIVE_STATUSES
from datamedic.jobs.service import create_job
from datamedic.storage import ObjectNotFound, ObjectStorage

MAX_PREVIEW_BYTES = 32 * 1024 * 1024


def get_owned_version(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID
) -> tuple[DatasetVersion, Dataset]:
    row = db.execute(
        select(DatasetVersion, Dataset)
        .join(Dataset, Dataset.id == DatasetVersion.dataset_id)
        .where(
            DatasetVersion.id == version_id,
            DatasetVersion.owner_id == owner_id,
            DatasetVersion.deleted_at.is_(None),
            Dataset.owner_id == owner_id,
            Dataset.deleted_at.is_(None),
        )
    ).one_or_none()
    if row is None:
        raise NotFound()
    return row[0], row[1]


def list_versions(db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID) -> list[DatasetVersion]:
    return list(
        db.scalars(
            select(DatasetVersion)
            .where(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.owner_id == owner_id,
                DatasetVersion.deleted_at.is_(None),
            )
            .order_by(DatasetVersion.version_number)
        )
    )


def read_preview(
    db: Session, storage: ObjectStorage, owner_id: uuid.UUID, version_id: uuid.UUID
) -> dict[str, Any]:
    version, _ = get_owned_version(db, owner_id, version_id)
    artifact = db.scalar(
        select(Artifact)
        .where(
            Artifact.owner_id == owner_id,
            Artifact.owner_type == "dataset_version",
            Artifact.owner_ref_id == version.id,
            Artifact.kind == "preview",
        )
        .order_by(Artifact.created_at.desc())
        .limit(1)
    )
    if artifact is None or artifact.size_bytes > MAX_PREVIEW_BYTES:
        raise NotFound("No preview is available for this version.")
    try:
        with storage.open_reader(artifact.object_key) as fh:
            raw = fh.read(MAX_PREVIEW_BYTES + 1)
    except ObjectNotFound as exc:
        raise NotFound("No preview is available for this version.") from exc
    if hashlib.sha256(raw).digest() != artifact.sha256:
        raise NotFound("The stored preview failed its integrity check.")
    preview: dict[str, Any] = json.loads(raw)
    return preview


def latest_profile(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID
) -> tuple[VersionProfile | None, Job | None]:
    version, ds = get_owned_version(db, owner_id, version_id)
    profile = db.scalar(
        select(VersionProfile)
        .where(VersionProfile.owner_id == owner_id, VersionProfile.dataset_version_id == version.id)
        .order_by(VersionProfile.created_at.desc())
        .limit(1)
    )
    return profile, _latest_profile_job(db, owner_id, ds.id, version.id)


def _latest_profile_job(
    db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID, version_id: uuid.UUID
) -> Job | None:
    return db.scalar(
        select(Job)
        .where(
            Job.owner_id == owner_id,
            Job.type == "profile",
            Job.dataset_id == dataset_id,
            Job.params["version_id"].astext == str(version_id),
        )
        .order_by(Job.queued_at.desc())
        .limit(1)
    )


def request_profile(
    db: Session, settings: Settings, owner_id: uuid.UUID, version_id: uuid.UUID
) -> Job:
    version, ds = get_owned_version(db, owner_id, version_id)
    # Serialise concurrent requests for the same dataset.
    db.execute(select(Dataset.id).where(Dataset.id == ds.id).with_for_update())
    current = _latest_profile_job(db, owner_id, ds.id, version.id)
    if current is not None and current.status in ACTIVE_STATUSES:
        raise Conflict("Profiling is already queued or running for this version.")
    job = create_job(db, settings, owner_id, "profile", ds.id, {"version_id": str(version.id)})
    db.commit()
    return job

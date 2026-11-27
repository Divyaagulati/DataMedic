import uuid

from fastapi import APIRouter, status

from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut
from datamedic.versions import service
from datamedic.versions.schemas import (
    PreviewOut,
    ProfileAcceptedOut,
    ProfileOut,
    ProfileStateOut,
    VersionOut,
)

router = APIRouter(prefix="/api/v1/versions", tags=["versions"])


@router.get("/{version_id}")
def get_version(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> VersionOut:
    version, ds = service.get_owned_version(db, principal.user.id, version_id)
    return VersionOut.build(version, is_head=ds.head_version_id == version.id)


@router.get("/{version_id}/preview")
def get_preview(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> PreviewOut:
    """First rows of the version (owner only; responses are never cached)."""
    data = service.read_preview(db, ctx.storage, principal.user.id, version_id)
    return PreviewOut(version_id=version_id, **data)


@router.get("/{version_id}/profile")
def get_profile(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> ProfileStateOut:
    profile, job = service.latest_profile(db, principal.user.id, version_id)
    return ProfileStateOut(
        profile=ProfileOut.from_record(profile) if profile else None,
        job=JobOut.from_job(job) if job else None,
    )


@router.post("/{version_id}/profile", status_code=status.HTTP_202_ACCEPTED)
def request_profile(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> ProfileAcceptedOut:
    """Queue a new profiling run (e.g. after a failure). Results are stored, never replaced."""
    job = service.request_profile(db, ctx.settings, principal.user.id, version_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return ProfileAcceptedOut(job=JobOut.from_job(job))

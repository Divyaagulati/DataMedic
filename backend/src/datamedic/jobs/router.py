import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, Query, status

from datamedic.api.deps import DbDep, PrincipalDep
from datamedic.jobs import service
from datamedic.jobs.schemas import JobListOut, JobOut

router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])

StatusFilter = Literal["queued", "running", "succeeded", "failed", "cancel_requested", "cancelled"]


@router.get("")
def list_jobs(
    principal: PrincipalDep,
    db: DbDep,
    status_: Annotated[StatusFilter | None, Query(alias="status")] = None,
    type_: Annotated[Literal["ingest", "purge"] | None, Query(alias="type")] = None,
    dataset_id: uuid.UUID | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    cursor: Annotated[str | None, Query(max_length=200)] = None,
) -> JobListOut:
    rows, next_cursor = service.list_jobs(
        db,
        principal.user.id,
        status=status_,
        job_type=type_,
        dataset_id=dataset_id,
        limit=limit,
        cursor=cursor,
    )
    return JobListOut(items=[JobOut.from_job(j) for j in rows], next_cursor=next_cursor)


@router.get("/{job_id}")
def get_job(job_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> JobOut:
    return JobOut.from_job(service.get_owned_job(db, principal.user.id, job_id))


@router.post("/{job_id}/cancel", status_code=status.HTTP_202_ACCEPTED)
def cancel_job(job_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> JobOut:
    return JobOut.from_job(service.request_cancel(db, principal.user.id, job_id))

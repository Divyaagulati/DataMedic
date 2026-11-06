import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from datamedic.db.models import Job
from datamedic.jobs.registry import ACTIVE_STATUSES

POLL_AFTER_MS = 2000


class JobErrorOut(BaseModel):
    code: str
    message: str
    retryable: bool


class JobOut(BaseModel):
    id: uuid.UUID
    type: str
    status: str
    dataset_id: uuid.UUID | None
    progress: dict[str, Any] | None
    result: dict[str, Any] | None
    error: JobErrorOut | None
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    poll_after_ms: int

    @classmethod
    def from_job(cls, job: Job) -> "JobOut":
        return cls(
            id=job.id,
            type=job.type,
            status=job.status,
            dataset_id=job.dataset_id,
            progress=job.progress,
            result=job.result,
            error=JobErrorOut(**job.error) if job.error else None,
            queued_at=job.queued_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            poll_after_ms=POLL_AFTER_MS if job.status in ACTIVE_STATUSES else 0,
        )


class JobListOut(BaseModel):
    items: list[JobOut]
    next_cursor: str | None

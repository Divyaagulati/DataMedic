import base64
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, func, or_, select
from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.db.models import Job
from datamedic.errors import Conflict, JobLimitExceeded, NotFound, ValidationFailed
from datamedic.jobs.registry import ACTIVE_STATUSES, JOB_TYPES


def ensure_job_capacity(db: Session, settings: Settings, owner_id: uuid.UUID) -> None:
    """Admission control (PRD §6.13): cap active (queued + running) jobs per user."""
    active = (
        db.scalar(
            select(func.count())
            .select_from(Job)
            .where(Job.owner_id == owner_id, Job.status.in_(ACTIVE_STATUSES))
        )
        or 0
    )
    if active >= settings.max_queued_jobs + settings.max_running_jobs:
        raise JobLimitExceeded(
            "Too many active jobs. Wait for some to finish.", headers={"Retry-After": "30"}
        )


def create_job(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    job_type: str,
    dataset_id: uuid.UUID | None,
    params: dict[str, Any] | None = None,
    *,
    enforce_limits: bool = True,
) -> Job:
    """Insert a queued job (not committed). System jobs such as purge skip admission limits."""
    jt = JOB_TYPES[job_type]
    if enforce_limits:
        ensure_job_capacity(db, settings, owner_id)
    job = Job(
        owner_id=owner_id,
        type=jt.name,
        queue=jt.queue,
        dataset_id=dataset_id,
        params=params or {},
        time_limit_seconds=jt.time_limit_seconds,
    )
    db.add(job)
    db.flush()
    return job


def get_owned_job(db: Session, owner_id: uuid.UUID, job_id: uuid.UUID) -> Job:
    # Explicit owner filter in addition to RLS (defence in depth).
    job = db.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id))
    if job is None:
        raise NotFound()
    return job


def _encode_cursor(job: Job) -> str:
    raw = f"{job.queued_at.isoformat()}|{job.id}"
    return base64.urlsafe_b64encode(raw.encode()).decode()


def _decode_cursor(cursor: str) -> tuple[datetime, uuid.UUID]:
    try:
        ts, jid = base64.urlsafe_b64decode(cursor.encode()).decode().split("|")
        return datetime.fromisoformat(ts), uuid.UUID(jid)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValidationFailed("Invalid cursor.") from exc


def list_jobs(
    db: Session,
    owner_id: uuid.UUID,
    *,
    status: str | None,
    job_type: str | None,
    dataset_id: uuid.UUID | None,
    limit: int,
    cursor: str | None,
) -> tuple[list[Job], str | None]:
    q = select(Job).where(Job.owner_id == owner_id)
    if status:
        q = q.where(Job.status == status)
    if job_type:
        q = q.where(Job.type == job_type)
    if dataset_id:
        q = q.where(Job.dataset_id == dataset_id)
    if cursor:
        ts, jid = _decode_cursor(cursor)
        q = q.where(or_(Job.queued_at < ts, and_(Job.queued_at == ts, Job.id < jid)))
    rows = list(db.scalars(q.order_by(Job.queued_at.desc(), Job.id.desc()).limit(limit + 1)))
    next_cursor = _encode_cursor(rows[limit - 1]) if len(rows) > limit else None
    return rows[:limit], next_cursor


def request_cancel(db: Session, owner_id: uuid.UUID, job_id: uuid.UUID) -> Job:
    job = db.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id).with_for_update())
    if job is None:
        raise NotFound()
    if job.status == "queued":
        job.status = "cancelled"
        job.finished_at = datetime.now(UTC)
        job.error = {
            "code": "JOB_CANCELLED",
            "message": "Cancelled before start.",
            "retryable": False,
        }
    elif job.status == "running":
        job.status = "cancel_requested"
    elif job.status != "cancel_requested":
        raise Conflict("The job has already finished.")
    db.commit()
    return job

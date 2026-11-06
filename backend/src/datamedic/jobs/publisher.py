"""Publishing jobs to workers. A broker message contains exactly ``{"job_id": "<uuid>"}``."""

import logging
import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Protocol

from celery import Celery
from sqlalchemy import update
from sqlalchemy.orm import Session

from datamedic.db.models import Job
from datamedic.jobs.registry import JOB_TYPES

if TYPE_CHECKING:
    from datamedic.context import AppContext

log = logging.getLogger(__name__)


class JobPublisher(Protocol):
    def publish(self, job_id: uuid.UUID, job_type: str) -> None: ...


def job_message(job_id: uuid.UUID) -> dict[str, str]:
    """The only payload ever sent through the broker."""
    return {"job_id": str(job_id)}


class CeleryJobPublisher:
    def __init__(self, app: Celery) -> None:
        self._app = app

    def publish(self, job_id: uuid.UUID, job_type: str) -> None:
        jt = JOB_TYPES[job_type]
        self._app.send_task(jt.task_name, args=(), kwargs=job_message(job_id), queue=jt.queue)


class EagerJobPublisher:
    """Development/test only: runs the job synchronously in-process (no broker)."""

    def __init__(self, ctx: "AppContext") -> None:
        self._ctx = ctx

    def publish(self, job_id: uuid.UUID, job_type: str) -> None:
        from datamedic.workers.runner import run_job

        run_job(self._ctx, **job_message(job_id))


def publish_committed(
    db: Session, publisher: JobPublisher, job_id: uuid.UUID, job_type: str
) -> bool:
    """Publish a job whose row is already committed. On broker failure the job stays queued
    and the requeue task republishes it later."""
    try:
        publisher.publish(job_id, job_type)
    except Exception:  # noqa: BLE001
        log.warning("job publish failed; will be requeued", extra={"job_id": str(job_id)})
        return False
    db.execute(update(Job).where(Job.id == job_id).values(last_published_at=datetime.now(UTC)))
    db.commit()
    return True

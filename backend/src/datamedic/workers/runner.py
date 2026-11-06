"""Worker-side job execution.

1. ``claim_job()`` (SECURITY DEFINER) atomically moves queued → running and returns the owner.
2. The session is bound to that owner, so every subsequent read/write is under RLS.
3. The handler runs; its outcome is persisted in the same transaction as its writes.
"""

import logging
import os
import socket
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from datamedic.context import AppContext
from datamedic.db.models import Job
from datamedic.db.session import bind_user, session_scope
from datamedic.errors import INTERNAL, JOB_CANCELLED

log = logging.getLogger(__name__)


@dataclass
class JobOutcome:
    succeeded: bool
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False


@dataclass
class HandlerContext:
    db: Session
    job: Job
    app: AppContext
    extra: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[HandlerContext], JobOutcome]

_HANDLERS: dict[str, Handler] = {}


def register_handler(job_type: str) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        _HANDLERS[job_type] = fn
        return fn

    return deco


def _handlers() -> dict[str, Handler]:
    if not _HANDLERS:
        import datamedic.datasets.jobs  # noqa: F401  (registers handlers)
    return _HANDLERS


def worker_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def claim(ctx: AppContext, job_id: uuid.UUID) -> tuple[uuid.UUID, str] | None:
    with session_scope(ctx.sessions) as db:
        row = db.execute(
            text("SELECT owner_id, job_type FROM claim_job(:id, :worker)"),
            {"id": job_id, "worker": worker_id()},
        ).one_or_none()
    return (row[0], row[1]) if row else None


def _finish(job: Job, outcome: JobOutcome) -> None:
    job.finished_at = datetime.now(UTC)
    job.heartbeat_at = job.finished_at
    if outcome.succeeded:
        job.status = "succeeded"
        job.result = outcome.result
        job.progress = {"phase": "done", "pct": 100}
    else:
        job.status = "cancelled" if outcome.error_code == JOB_CANCELLED else "failed"
        job.result = outcome.result
        job.error = {
            "code": outcome.error_code or INTERNAL,
            "message": outcome.error_message or "Job failed.",
            "retryable": outcome.retryable,
        }


def run_job(ctx: AppContext, job_id: str) -> None:
    jid = uuid.UUID(job_id)
    claimed = claim(ctx, jid)
    if claimed is None:
        log.info(
            "job not claimable (already claimed, finished or cancelled)", extra={"job_id": job_id}
        )
        return
    owner_id, job_type = claimed
    handler = _handlers()[job_type]
    try:
        with session_scope(ctx.sessions) as db:
            bind_user(db, owner_id)
            job = db.get(Job, jid)
            if job is None:  # cannot happen under a consistent DB; fail safe
                return
            if job.status == "cancel_requested":
                outcome = JobOutcome(
                    False, error_code=JOB_CANCELLED, error_message="Cancelled by user."
                )
            else:
                outcome = handler(HandlerContext(db=db, job=job, app=ctx))
            _finish(job, outcome)
    except Exception:
        log.exception("job crashed", extra={"job_id": job_id, "job_type": job_type})
        with session_scope(ctx.sessions) as db:
            bind_user(db, owner_id)
            job = db.get(Job, jid)
            if job is not None and job.status in ("running", "cancel_requested"):
                _finish(
                    job,
                    JobOutcome(
                        False,
                        error_code=INTERNAL,
                        error_message="Unexpected error while running the job.",
                        retryable=True,
                    ),
                )

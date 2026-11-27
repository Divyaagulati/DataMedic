"""Worker-side job execution.

1. ``claim_job()`` (SECURITY DEFINER) atomically moves queued → running and returns the owner.
2. The session is bound to that owner, so every subsequent read/write is under RLS.
3. The handler runs; its outcome is persisted in the same transaction as its writes.
4. Follow-up jobs created by the handler (e.g. ingest -> profile) are published only after
   that transaction commits, so a worker can never pick up a job row that does not exist.
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
from datamedic.errors import INTERNAL, JOB_CANCELLED, LIMIT_EXCEEDED

log = logging.getLogger(__name__)


@dataclass
class JobOutcome:
    succeeded: bool
    result: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    follow_ups: list[tuple[uuid.UUID, str]] = field(default_factory=list)


@dataclass
class HandlerContext:
    db: Session
    job: Job
    app: AppContext
    extra: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[HandlerContext], JobOutcome]
FailureHook = Callable[[Session, Job], None]

_HANDLERS: dict[str, Handler] = {}
_FAILURE_HOOKS: dict[str, FailureHook] = {}


def register_handler(job_type: str) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        _HANDLERS[job_type] = fn
        return fn

    return deco


def register_failure_hook(job_type: str) -> Callable[[FailureHook], FailureHook]:
    """Hook run (in a fresh transaction) when a handler of ``job_type`` crashes, so subject
    state such as a dataset's status is never left mid-transition."""

    def deco(fn: FailureHook) -> FailureHook:
        _FAILURE_HOOKS[job_type] = fn
        return fn

    return deco


def _handlers() -> dict[str, Handler]:
    # Always import every handler module (imports are cached): a module imported earlier on its
    # own must not stop the others from registering.
    import datamedic.analyses.jobs  # noqa: F401  (registers handlers)
    import datamedic.analytics.jobs  # noqa: F401
    import datamedic.biomedical.jobs  # noqa: F401
    import datamedic.datasets.jobs  # noqa: F401
    import datamedic.experiments.jobs  # noqa: F401
    import datamedic.exports.jobs  # noqa: F401
    import datamedic.pipelines.jobs  # noqa: F401
    import datamedic.training.jobs  # noqa: F401

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
    follow_ups: list[tuple[uuid.UUID, str]] = []
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
            follow_ups = outcome.follow_ups
    except MemoryError:
        # The per-process memory cap (workers/limits.py) was reached: a deterministic,
        # non-retryable limit, not a crash.
        log.error("job exceeded memory limit", extra={"job_id": job_id, "job_type": job_type})
        _fail_after_crash(
            ctx,
            owner_id,
            jid,
            job_type,
            JobOutcome(
                False,
                error_code=LIMIT_EXCEEDED,
                error_message="The job needed more memory than this worker allows.",
            ),
        )
        return
    except Exception:
        log.exception("job crashed", extra={"job_id": job_id, "job_type": job_type})
        _fail_after_crash(
            ctx,
            owner_id,
            jid,
            job_type,
            JobOutcome(
                False,
                error_code=INTERNAL,
                error_message="Unexpected error while running the job.",
                retryable=True,
            ),
        )
        return
    _publish_follow_ups(ctx, owner_id, follow_ups)


def _fail_after_crash(
    ctx: AppContext, owner_id: uuid.UUID, jid: uuid.UUID, job_type: str, outcome: JobOutcome
) -> None:
    """Record the failure in a fresh transaction (the handler's was rolled back) and let the
    job type restore its subject's state (e.g. a dataset stuck in 'validating')."""
    with session_scope(ctx.sessions) as db:
        bind_user(db, owner_id)
        job = db.get(Job, jid)
        if job is not None and job.status in ("running", "cancel_requested"):
            _finish(job, outcome)
            hook = _FAILURE_HOOKS.get(job_type)
            if hook is not None:
                hook(db, job)


def _publish_follow_ups(
    ctx: AppContext, owner_id: uuid.UUID, follow_ups: list[tuple[uuid.UUID, str]]
) -> None:
    if not follow_ups:
        return
    from datamedic.jobs.publisher import publish_committed

    publisher = ctx.require_publisher()
    for job_id, job_type in follow_ups:
        with session_scope(ctx.sessions) as db:
            bind_user(db, owner_id)
            publish_committed(db, publisher, job_id, job_type)

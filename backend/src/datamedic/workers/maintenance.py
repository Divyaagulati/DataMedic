"""Scheduled maintenance. Uses the dm_maintenance role (BYPASSRLS) because it spans all users.
Touches job bookkeeping only, never dataset content."""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select, text, update

from datamedic.context import AppContext
from datamedic.db.models import Job
from datamedic.db.session import session_scope

log = logging.getLogger(__name__)


def requeue_stuck(ctx: AppContext, limit: int = 100) -> int:
    """Republish queued jobs whose publish failed or was lost. Safe: claim_job() is atomic,
    so duplicate deliveries are no-ops."""
    cutoff = datetime.now(UTC) - timedelta(seconds=ctx.settings.job_requeue_after_seconds)
    publisher = ctx.require_publisher()
    with session_scope(ctx.maintenance_sessions) as db:
        rows = db.execute(
            select(Job.id, Job.type)
            .where(
                Job.status == "queued",
                or_(Job.last_published_at.is_(None), Job.last_published_at < cutoff),
                Job.queued_at < cutoff,
            )
            .order_by(Job.queued_at)
            .limit(limit)
        ).all()
    count = 0
    for job_id, job_type in rows:
        try:
            publisher.publish(job_id, job_type)
        except Exception:  # noqa: BLE001
            log.warning("requeue publish failed", extra={"job_id": str(job_id)})
            continue
        with session_scope(ctx.maintenance_sessions) as db:
            db.execute(
                update(Job).where(Job.id == job_id).values(last_published_at=datetime.now(UTC))
            )
        count += 1
    return count


def reap_stale(ctx: AppContext) -> int:
    """Fail running jobs whose heartbeat is older than their hard limit plus grace."""
    with session_scope(ctx.maintenance_sessions) as db:
        result = db.execute(
            text(
                """
                UPDATE jobs
                   SET status = 'failed', finished_at = now(),
                       error = jsonb_build_object('code', 'JOB_STALE',
                                                  'message', 'The job stopped responding.',
                                                  'retryable', true)
                 WHERE status IN ('running', 'cancel_requested')
                   AND heartbeat_at < now() - make_interval(secs => time_limit_seconds + :grace)
                """
            ),
            {"grace": ctx.settings.job_stale_grace_seconds},
        )
        return int(result.rowcount)  # type: ignore[attr-defined]

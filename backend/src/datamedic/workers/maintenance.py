"""Scheduled maintenance. Uses the dm_maintenance role (BYPASSRLS) because it spans all users.
Touches job bookkeeping only, never dataset content."""

import logging
from datetime import UTC, datetime, timedelta

from sqlalchemy import or_, select, text, update

from datamedic.context import AppContext
from datamedic.db.models import Artifact, Dataset, DatasetVersion, Job
from datamedic.db.session import session_scope
from datamedic.storage import ObjectNotFound, dataset_prefix

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


_ACTIVE_INGEST = """
    EXISTS (SELECT 1 FROM jobs j
             WHERE j.dataset_id = d.id AND j.type = 'ingest'
               AND j.status IN ('queued', 'running', 'cancel_requested'))
"""


def recover_stuck_datasets(ctx: AppContext) -> dict[str, int]:
    """Move datasets out of 'validating' when nothing is processing them any more.

    Runs after reap_stale() so jobs it just failed count as finished. A dataset with a queued,
    running or cancel-requested ingest job is never touched. Transitions:
    - no stored original, older than upload_stale_after_seconds: the upload was interrupted
      -> 'upload_pending' (the owner can upload again);
    - original stored, version 0 exists: a late-finishing job already succeeded -> 'ready';
    - original stored, no version: the ingest was abandoned (stale, cancelled, crashed)
      -> 'parse_failed' (the owner can retry; the ingest handler re-checks under a row lock,
      so a retry can never create a second version 0).
    """
    counts: dict[str, int] = {}
    with session_scope(ctx.maintenance_sessions) as db:
        counts["upload_pending"] = int(
            db.execute(  # type: ignore[attr-defined]
                text(
                    f"""
                    UPDATE datasets d SET status = 'upload_pending', updated_at = now()
                     WHERE d.status = 'validating' AND d.deleted_at IS NULL
                       AND NOT EXISTS (SELECT 1 FROM original_files f WHERE f.dataset_id = d.id)
                       AND NOT {_ACTIVE_INGEST}
                       AND d.updated_at < now() - make_interval(secs => :grace)
                    """  # noqa: S608  (constant SQL fragment, no user input)
                ),
                {"grace": ctx.settings.upload_stale_after_seconds},
            ).rowcount
        )
        counts["ready"] = int(
            db.execute(  # type: ignore[attr-defined]
                text(
                    f"""
                    UPDATE datasets d SET status = 'ready', updated_at = now()
                     WHERE d.status = 'validating' AND d.deleted_at IS NULL
                       AND d.head_version_id IS NOT NULL AND NOT {_ACTIVE_INGEST}
                    """  # noqa: S608
                )
            ).rowcount
        )
        counts["parse_failed"] = int(
            db.execute(  # type: ignore[attr-defined]
                text(
                    f"""
                    UPDATE datasets d SET status = 'parse_failed', updated_at = now()
                     WHERE d.status = 'validating' AND d.deleted_at IS NULL
                       AND d.head_version_id IS NULL
                       AND EXISTS (SELECT 1 FROM original_files f WHERE f.dataset_id = d.id)
                       AND NOT {_ACTIVE_INGEST}
                    """  # noqa: S608
                )
            ).rowcount
        )
    if any(counts.values()):
        log.info("recovered stuck datasets", extra={"recovered": counts})
    return counts


def reconcile_orphans(ctx: AppContext) -> int:
    """Delete derived objects (version files, artifacts) that no row references and that are
    older than ``orphan_grace_seconds`` (default 24 h, far beyond any job's time limit). They
    are left behind only when a commit or revert crashed after writing its object but before
    its transaction committed. Originals (``raw/``) are never touched, and nothing referenced
    by a version or artifact row is ever deleted."""
    grace = ctx.settings.orphan_grace_seconds
    now = datetime.now(UTC).timestamp()
    with session_scope(ctx.maintenance_sessions) as db:
        datasets = db.execute(
            select(Dataset.id, Dataset.owner_id).where(Dataset.deleted_at.is_(None))
        ).all()
        referenced = set(db.scalars(select(DatasetVersion.object_key))) | set(
            db.scalars(select(Artifact.object_key))
        )
    storage = ctx.storage
    deleted = 0
    for dataset_id, owner_id in datasets:
        for area in ("versions", "artifacts"):
            for key in storage.list_prefix(dataset_prefix(area, owner_id, dataset_id)):
                if key in referenced:
                    continue
                try:
                    age = now - storage.modified_at(key)
                except ObjectNotFound:
                    continue
                if age > grace and storage.delete(key):
                    deleted += 1
    if deleted:
        log.info("orphan objects removed", extra={"deleted": deleted})
    return deleted

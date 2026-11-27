"""Celery entry points. Start with:
    celery -A datamedic.workers.tasks worker -Q ingest,analysis,evaluate,execute,maintenance
    celery -A datamedic.workers.tasks beat
Every task receives only ``job_id`` (or nothing, for maintenance).
"""

from functools import lru_cache
from typing import Any

from celery.signals import setup_logging, worker_process_init

from datamedic.config import get_settings
from datamedic.context import AppContext, build_context
from datamedic.jobs.registry import JOB_TYPES
from datamedic.logging_setup import configure_logging
from datamedic.workers import maintenance
from datamedic.workers.celery_app import create_celery
from datamedic.workers.limits import apply_worker_memory_limit
from datamedic.workers.runner import run_job


@setup_logging.connect
def _keep_datamedic_logging(**_: Any) -> None:
    """Celery replaces the root log handlers at worker start unless a receiver is connected to
    ``setup_logging``. Without this, worker logs bypass the scrubber and the formatter that
    drops tracebacks (which can quote dataset values). See logging_setup.py."""
    configure_logging()


@worker_process_init.connect
def _cap_worker_memory(**_: Any) -> None:
    """Runs in every pool process (prefork child) before it executes tasks."""
    apply_worker_memory_limit(get_settings().worker_memory_limit_mb)


configure_logging()
celery_app = create_celery(get_settings())


@lru_cache
def worker_context() -> AppContext:
    return build_context(get_settings())


_INGEST = JOB_TYPES["ingest"]
_PROFILE = JOB_TYPES["profile"]
_ANALYSIS = JOB_TYPES["analysis"]
_PURGE = JOB_TYPES["purge"]
_EXPERIMENT = JOB_TYPES["experiment"]
_PREVIEW = JOB_TYPES["impact_preview"]
_COMMIT = JOB_TYPES["commit"]
_REVERT = JOB_TYPES["revert"]
_COMPARE = JOB_TYPES["compare_versions"]
_TRAIN = JOB_TYPES["train"]
_AUDIT = JOB_TYPES["reliability_audit"]
_BIO = JOB_TYPES["biomedical_summary"]
_EXPORT = JOB_TYPES["export"]
_INSIGHTS = JOB_TYPES["insights"]


@celery_app.task(
    name=_INSIGHTS.task_name,
    soft_time_limit=_INSIGHTS.time_limit_seconds - 30,
    time_limit=_INSIGHTS.time_limit_seconds,
)
def insights(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_BIO.task_name,
    soft_time_limit=_BIO.time_limit_seconds - 30,
    time_limit=_BIO.time_limit_seconds,
)
def biomedical_summary(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_EXPORT.task_name,
    soft_time_limit=_EXPORT.time_limit_seconds - 30,
    time_limit=_EXPORT.time_limit_seconds,
)
def export(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_INGEST.task_name,
    soft_time_limit=_INGEST.time_limit_seconds - 30,
    time_limit=_INGEST.time_limit_seconds,
)
def ingest(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_PROFILE.task_name,
    soft_time_limit=_PROFILE.time_limit_seconds - 30,
    time_limit=_PROFILE.time_limit_seconds,
)
def profile(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_ANALYSIS.task_name,
    soft_time_limit=_ANALYSIS.time_limit_seconds - 30,
    time_limit=_ANALYSIS.time_limit_seconds,
)
def analysis(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_EXPERIMENT.task_name,
    soft_time_limit=_EXPERIMENT.time_limit_seconds - 30,
    time_limit=_EXPERIMENT.time_limit_seconds,
)
def experiment(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_TRAIN.task_name,
    soft_time_limit=_TRAIN.time_limit_seconds - 30,
    time_limit=_TRAIN.time_limit_seconds,
)
def train(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_AUDIT.task_name,
    soft_time_limit=_AUDIT.time_limit_seconds - 30,
    time_limit=_AUDIT.time_limit_seconds,
)
def reliability_audit(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_PREVIEW.task_name,
    soft_time_limit=_PREVIEW.time_limit_seconds - 30,
    time_limit=_PREVIEW.time_limit_seconds,
)
def impact_preview(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_COMMIT.task_name,
    soft_time_limit=_COMMIT.time_limit_seconds - 30,
    time_limit=_COMMIT.time_limit_seconds,
)
def commit(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_REVERT.task_name,
    soft_time_limit=_REVERT.time_limit_seconds - 30,
    time_limit=_REVERT.time_limit_seconds,
)
def revert(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_COMPARE.task_name,
    soft_time_limit=_COMPARE.time_limit_seconds - 30,
    time_limit=_COMPARE.time_limit_seconds,
)
def compare_versions(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(
    name=_PURGE.task_name,
    soft_time_limit=_PURGE.time_limit_seconds - 30,
    time_limit=_PURGE.time_limit_seconds,
)
def purge(job_id: str) -> None:
    run_job(worker_context(), job_id)


@celery_app.task(name="datamedic.maintenance.requeue_stuck")
def requeue_stuck() -> int:
    return maintenance.requeue_stuck(worker_context())


@celery_app.task(name="datamedic.maintenance.reconcile_orphans")
def reconcile_orphans() -> int:
    return maintenance.reconcile_orphans(worker_context())


@celery_app.task(name="datamedic.maintenance.reap_stale")
def reap_stale() -> int:
    ctx = worker_context()
    reaped = maintenance.reap_stale(ctx)
    maintenance.recover_stuck_datasets(ctx)  # after reaping, so reaped jobs count as finished
    return reaped

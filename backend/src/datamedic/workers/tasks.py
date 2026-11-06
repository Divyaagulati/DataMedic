"""Celery entry points. Start with:
    celery -A datamedic.workers.tasks worker -Q ingest,maintenance
    celery -A datamedic.workers.tasks beat
Every task receives only ``job_id`` (or nothing, for maintenance).
"""

from functools import lru_cache

from datamedic.config import get_settings
from datamedic.context import AppContext, build_context
from datamedic.jobs.registry import JOB_TYPES
from datamedic.logging_setup import configure_logging
from datamedic.workers import maintenance
from datamedic.workers.celery_app import create_celery
from datamedic.workers.runner import run_job

configure_logging()
celery_app = create_celery(get_settings())


@lru_cache
def worker_context() -> AppContext:
    return build_context(get_settings())


_INGEST = JOB_TYPES["ingest"]
_PURGE = JOB_TYPES["purge"]


@celery_app.task(
    name=_INGEST.task_name,
    soft_time_limit=_INGEST.time_limit_seconds - 30,
    time_limit=_INGEST.time_limit_seconds,
)
def ingest(job_id: str) -> None:
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


@celery_app.task(name="datamedic.maintenance.reap_stale")
def reap_stale() -> int:
    return maintenance.reap_stale(worker_context())

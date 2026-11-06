"""Celery configuration. Celery is transport only: job state lives in the ``jobs`` table."""

from celery import Celery

from datamedic.config import Settings
from datamedic.jobs.registry import JOB_TYPES

# Longer than the longest hard time limit, so ack-late messages are not redelivered mid-run.
VISIBILITY_TIMEOUT_SECONDS = 2 * 3600


def create_celery(settings: Settings) -> Celery:
    app = Celery("datamedic", broker=settings.redis_url)
    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        task_ignore_result=True,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        worker_max_tasks_per_child=50,
        worker_max_memory_per_child=1_000_000,  # KiB
        broker_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT_SECONDS},
        broker_connection_retry_on_startup=True,
        task_default_queue="maintenance",
        task_routes={jt.task_name: {"queue": jt.queue} for jt in JOB_TYPES.values()},
        timezone="UTC",
        beat_schedule={
            "requeue-stuck-jobs": {
                "task": "datamedic.maintenance.requeue_stuck",
                "schedule": 60.0,
                "options": {"queue": "maintenance"},
            },
            "reap-stale-jobs": {
                "task": "datamedic.maintenance.reap_stale",
                "schedule": 60.0,
                "options": {"queue": "maintenance"},
            },
        },
    )
    return app

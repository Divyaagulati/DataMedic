"""Job types: queue routing, Celery task names and hard time limits (PRD §6.13)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class JobType:
    name: str
    queue: str
    task_name: str
    time_limit_seconds: int


JOB_TYPES: dict[str, JobType] = {
    "ingest": JobType("ingest", "ingest", "datamedic.jobs.ingest", 600),
    "purge": JobType("purge", "maintenance", "datamedic.jobs.purge", 600),
}

ACTIVE_STATUSES = ("queued", "running", "cancel_requested")
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")

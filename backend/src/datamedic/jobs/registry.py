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
    "profile": JobType("profile", "analysis", "datamedic.jobs.profile", 1200),
    "analysis": JobType("analysis", "analysis", "datamedic.jobs.analysis", 1200),
    "experiment": JobType("experiment", "evaluate", "datamedic.jobs.experiment", 3600),
    "impact_preview": JobType("impact_preview", "execute", "datamedic.jobs.impact_preview", 1800),
    "commit": JobType("commit", "execute", "datamedic.jobs.commit", 1800),
    "revert": JobType("revert", "execute", "datamedic.jobs.revert", 600),
    "compare_versions": JobType(
        "compare_versions", "analysis", "datamedic.jobs.compare_versions", 1200
    ),
    "train": JobType("train", "evaluate", "datamedic.jobs.train", 3600),
    "reliability_audit": JobType(
        "reliability_audit", "evaluate", "datamedic.jobs.reliability_audit", 1200
    ),
    "biomedical_summary": JobType(
        "biomedical_summary", "analysis", "datamedic.jobs.biomedical_summary", 1200
    ),
    "export": JobType("export", "analysis", "datamedic.jobs.export", 1200),
    "insights": JobType("insights", "analysis", "datamedic.jobs.insights", 1200),
    "purge": JobType("purge", "maintenance", "datamedic.jobs.purge", 600),
}

ACTIVE_STATUSES = ("queued", "running", "cancel_requested")
TERMINAL_STATUSES = ("succeeded", "failed", "cancelled")

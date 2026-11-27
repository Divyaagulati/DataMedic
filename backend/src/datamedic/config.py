"""Application settings, read only from environment variables (or a local .env file)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATAMEDIC_",
        env_file=(".env", "../.env"),
        extra="ignore",
    )

    environment: Literal["development", "test", "production"] = "development"

    # PostgreSQL. The app/worker role is subject to Row-Level Security; the maintenance role
    # (BYPASSRLS) is used only by scheduled maintenance tasks (requeue, reaper).
    database_url: SecretStr
    maintenance_database_url: SecretStr

    # Background jobs
    redis_url: str = "redis://localhost:6379/0"
    jobs_eager: bool = Field(
        default=False,
        description="Development only: run jobs in-process instead of via the broker.",
    )
    job_requeue_after_seconds: int = 120
    job_stale_grace_seconds: int = 300

    # Private storage (development filesystem adapter)
    storage_root: Path = Path("./var/storage")

    # Web / cookies
    web_origin: str = "http://localhost:3000"
    public_web_url: str = "http://localhost:3000"
    cookie_secure: bool = True
    session_idle_days: int = 7
    session_absolute_days: int = 30

    # Email
    email_backend: Literal["smtp", "memory", "console"] = "smtp"
    smtp_host: str = "localhost"
    smtp_port: int = 1025
    smtp_from: str = "DataMedic <no-reply@datamedic.local>"
    # Authenticated SMTP for production providers (credentials stay server-side).
    smtp_security: Literal["none", "starttls", "ssl"] = "none"
    smtp_username: str | None = None
    smtp_password: SecretStr | None = None

    # Limits (PRD §6.13)
    max_upload_bytes: int = 200 * 1024 * 1024
    max_datasets_per_user: int = 25
    max_running_jobs: int = 2
    max_queued_jobs: int = 10
    max_cells: int = 20_000_000
    max_columns: int = 2_000
    max_json_depth: int = 10  # FR-2.4: deeper nesting is rejected

    # Parse sandbox (ARCHITECTURE §7.2): a child process with rlimits on Linux
    parse_timeout_seconds: int = 480  # wall clock and CPU; below the 600 s ingest job limit
    parse_memory_limit_mb: int = 2500  # RLIMIT_AS for the child (README "Resource limits")
    parse_max_output_bytes: int = 4 * 1024 * 1024 * 1024

    # In-process job memory cap (RLIMIT_DATA per Celery worker process, Linux). 0 = no cap.
    # Analysis jobs (profiling, diagnostics) run in the worker process, not in the sandbox.
    worker_memory_limit_mb: int = 0

    # Experiment Lab (Phase 4 Part D). Checked in the API before a job is queued; the worker
    # memory cap still applies. Fits = (candidates + baseline) x folds x models.
    experiment_max_rows: int = 200_000
    experiment_max_cells: int = 5_000_000
    experiment_max_candidates: int = 6
    experiment_max_fits: int = 300

    # Versions per dataset (Part E, owner decision E2). Commit and revert refuse beyond it with
    # QUOTA_EXCEEDED; older versions are never deleted to make room.
    max_versions_per_dataset: int = 20
    # Impact preview: affected rows listed per step (E1); counts always cover every row.
    impact_preview_max_rows: int = 1_000
    # Orphan reconciliation: unreferenced derived objects older than this are deleted.
    orphan_grace_seconds: int = 86_400

    # Supervised training and reliability audits (Phase 5). Checked in the API before a job is
    # queued and again in the worker. Fits = outer folds x (settings x inner folds + 1) per model.
    training_max_rows: int = 200_000
    training_max_cells: int = 5_000_000
    training_max_fits: int = 400
    training_max_features: int = 500
    training_max_runs_per_dataset: int = 50
    # Worker memory pre-check: estimated bytes per model-matrix cell (all copies, all models).
    training_memory_budget_mb: int = 900
    permutation_max_rows_per_fold: int = 5_000
    permutation_max_columns: int = 100
    permutation_repeats: int = 5
    bootstrap_resamples: int = 1_000
    audit_max_rows: int = 50_000
    # Analytics Studio (Phase 8): interactive queries run in the API on a bounded,
    # hash-verified in-memory copy of one version (docs/ARCHITECTURE.md "Analytics Studio").
    analytics_max_rows: int = 200_000
    analytics_max_cells: int = 5_000_000
    analytics_cache_entries: int = 2
    analytics_max_concurrency: int = 2
    analytics_max_dashboards_per_dataset: int = 25
    analytics_max_widgets_per_dashboard: int = 24
    audit_single_feature_max_columns: int = 200

    # Maintenance: a dataset left in 'validating' with no stored original for this long is
    # treated as an interrupted upload and reopened for upload.
    upload_stale_after_seconds: int = 3600

    # Login throttling (progressive delay, no hard lockout)
    login_free_attempts: int = 5
    login_max_delay_seconds: int = 900

    @model_validator(mode="after")
    def _production_guards(self) -> "Settings":
        if self.environment == "production":
            if not self.cookie_secure:
                raise ValueError("cookie_secure must be true in production")
            if self.jobs_eager:
                raise ValueError("jobs_eager is not allowed in production")
            if self.email_backend != "smtp":
                raise ValueError("email_backend must be smtp in production")
            if self.smtp_security == "none":
                raise ValueError("smtp_security must be starttls or ssl in production")
        return self

    @property
    def session_cookie_name(self) -> str:
        # The __Host- prefix requires the Secure attribute; plain-HTTP local development
        # (cookie_secure=false) falls back to an unprefixed name.
        return "__Host-dm_session" if self.cookie_secure else "dm_session"


@lru_cache
def get_settings() -> Settings:
    return Settings()

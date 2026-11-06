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

    # Limits (PRD §6.13)
    max_upload_bytes: int = 200 * 1024 * 1024
    max_datasets_per_user: int = 25
    max_running_jobs: int = 2
    max_queued_jobs: int = 10

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
        return self

    @property
    def session_cookie_name(self) -> str:
        # The __Host- prefix requires the Secure attribute; plain-HTTP local development
        # (cookie_secure=false) falls back to an unprefixed name.
        return "__Host-dm_session" if self.cookie_secure else "dm_session"


@lru_cache
def get_settings() -> Settings:
    return Settings()

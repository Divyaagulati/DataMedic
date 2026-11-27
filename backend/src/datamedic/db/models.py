"""SQLAlchemy models (see docs/DATABASE_SCHEMA.md).

Tables with an ``owner_id`` are protected by Row-Level Security (migrations 0001, 0002).
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING)


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _now() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


DATASET_STATUSES = (
    "upload_pending",
    "validating",
    "validated",
    "parsing",
    "parse_failed",
    "ready",
    "rejected",
    "deleting",
)
JOB_STATUSES = ("queued", "running", "succeeded", "failed", "cancel_requested", "cancelled")
JOB_TYPES = (
    "ingest",
    "profile",
    "analysis",
    "experiment",
    "impact_preview",
    "commit",
    "revert",
    "compare_versions",
    "train",
    "reliability_audit",
    "biomedical_summary",
    "export",
    "insights",
    "purge",
)
MUTATION_JOB_TYPES = ("commit", "revert")
VERSION_KINDS = ("ingest", "reparse", "transform", "revert")
ARTIFACT_KINDS = (
    "preview",
    "fold_assignment",
    "impact_preview",
    "version_comparison",
    "oof_predictions",
    "export",
)
ARTIFACT_OWNER_TYPES = ("dataset_version", "split_definition", "pipeline", "training_run")
ANALYSIS_KINDS = ("diagnostics", "reliability_audit", "biomedical", "insights")
TRAINING_TASKS = ("binary", "multiclass", "regression")
EVALUATION_MODES = ("unbiased_direct", "unbiased_replay", "exploratory")
TRAINING_MODELS = ("dummy", "linear", "random_forest", "hgb")
RECOMMENDATION_STATUSES = ("open", "used", "dismissed")
EXPERIMENT_TASKS = ("binary", "multiclass", "regression", "clustering")
RESULT_STATUSES = ("completed", "partial", "failed")
SPLIT_METHODS = ("kfold", "stratified", "group", "stratified_group", "time")
FORMATS = ("csv", "xlsx", "xls", "parquet", "json")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)  # stored lower-case
    email_verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str | None] = mapped_column(Text)
    plan: Mapped[str] = mapped_column(Text, nullable=False, server_default="free")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    failed_login_count: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    last_failed_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("status", ("active", "deleting")), name="status"),
        CheckConstraint("email = lower(email)", name="email_lower"),
    )


class UserSession(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    token_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, unique=True)
    csrf_token: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _now()
    last_seen_at: Mapped[datetime] = _now()
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    user_agent: Mapped[str | None] = mapped_column(Text)


class AuthToken(Base):
    __tablename__ = "auth_tokens"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    purpose: Mapped[str] = mapped_column(Text, nullable=False)
    token_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("purpose", ("verify_email", "reset_password")), name="purpose"),
    )


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(Text, nullable=False, server_default="generic")
    domain_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="upload_pending")
    declared_filename: Mapped[str] = mapped_column(Text, nullable=False)
    declared_extension: Mapped[str] = mapped_column(Text, nullable=False)
    declared_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    head_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "dataset_versions.id",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
            name="fk_datasets_head_version_id_dataset_versions",
        ),
    )
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("status", DATASET_STATUSES), name="status"),
        CheckConstraint("declared_size_bytes > 0", name="size_positive"),
        CheckConstraint("char_length(name) BETWEEN 1 AND 200", name="name_length"),
        Index("ix_datasets_owner_created", "owner_id", "deleted_at", "created_at"),
    )


class OriginalFile(Base):
    __tablename__ = "original_files"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    detected_format: Mapped[str | None] = mapped_column(Text)
    original_filename: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(
            f"detected_format IS NULL OR {_in('detected_format', FORMATS)}", name="format"
        ),
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False
    )
    type: Mapped[str] = mapped_column(Text, nullable=False)
    queue: Mapped[str] = mapped_column(Text, nullable=False)
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="SET NULL")
    )
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, server_default="{}")
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="queued")
    progress: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    idempotency_key: Mapped[str | None] = mapped_column(Text)
    worker_id: Mapped[str | None] = mapped_column(Text)
    time_limit_seconds: Mapped[int] = mapped_column(Integer, nullable=False)
    queued_at: Mapped[datetime] = _now()
    last_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(_in("status", JOB_STATUSES), name="status"),
        CheckConstraint(_in("type", JOB_TYPES), name="type"),
        Index("ix_jobs_owner_status", "owner_id", "status"),
        Index("ix_jobs_status_heartbeat", "status", "heartbeat_at"),
        Index("ix_jobs_status_published", "status", "last_published_at"),
        Index(
            "uq_jobs_owner_idempotency",
            "owner_id",
            "idempotency_key",
            unique=True,
            postgresql_where="idempotency_key IS NOT NULL",
        ),
        # At most one queued or running commit/revert per dataset (Part E).
        Index(
            "uq_jobs_one_mutation_per_dataset",
            "dataset_id",
            unique=True,
            postgresql_where="type IN ('commit', 'revert') "
            "AND status IN ('queued', 'running', 'cancel_requested')",
        ),
    )


class DatasetVersion(Base):
    """Immutable version of a dataset (insert-only except ``deleted_at``)."""

    __tablename__ = "dataset_versions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    parent_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id")
    )
    restored_from_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id")
    )
    transformation_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "transformation_runs.id",
            use_alter=True,
            deferrable=True,
            initially="DEFERRED",
            name="fk_dataset_versions_transformation_run_id_transformation_runs",
        ),
    )
    # Model-stage steps stored with the version (D1): never applied to its data.
    model_pipeline_spec: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)
    parse_options: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    parse_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    file_sha256: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    content_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    n_rows: Mapped[int] = mapped_column(BigInteger, nullable=False)
    n_cols: Mapped[int] = mapped_column(Integer, nullable=False)
    schema: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    diff_summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    fitted_on_full_data: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    library_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("dataset_id", "version_number"),
        CheckConstraint(_in("kind", VERSION_KINDS), name="kind"),
        CheckConstraint("(kind = 'ingest') = (parent_version_id IS NULL)", name="parent"),
        CheckConstraint(
            "(kind = 'revert') = (restored_from_version_id IS NOT NULL)", name="revert"
        ),
        CheckConstraint(
            "(kind = 'transform') = (transformation_run_id IS NOT NULL)", name="transform"
        ),
        CheckConstraint(
            "(kind IN ('ingest', 'reparse')) = (parse_options IS NOT NULL)",
            name="parse_options",
        ),
        CheckConstraint("version_number >= 0", name="number"),
        CheckConstraint("n_rows >= 0 AND n_cols >= 0", name="shape"),
    )


class Artifact(Base):
    """Stored derived object (e.g. the preview). Insert-only."""

    __tablename__ = "artifacts"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    owner_type: Mapped[str] = mapped_column(Text, nullable=False)
    owner_ref_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    object_key: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("kind", ARTIFACT_KINDS), name="kind"),
        CheckConstraint(_in("owner_type", ARTIFACT_OWNER_TYPES), name="owner_type"),
        Index("ix_artifacts_owner_ref", "owner_type", "owner_ref_id", "kind"),
    )


class VersionProfile(Base):
    """Result of one successful profiling job (insert-only). Status lives in ``jobs``."""

    __tablename__ = "version_profiles"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, unique=True
    )
    profiler: Mapped[str] = mapped_column(Text, nullable=False)
    profiler_version: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    columns: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    correlations: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    sampling: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    limitations: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    library_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        Index("ix_version_profiles_version_created", "dataset_version_id", "created_at"),
    )


class ColumnRole(Base):
    """Declared semantic role of a column in a version (replaceable as a set)."""

    __tablename__ = "column_roles"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    column_name: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text, nullable=False, server_default="user")
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "column_name", "role"),
        CheckConstraint("(role = 'target') = (task_type IS NOT NULL)", name="task_type"),
        CheckConstraint(
            "task_type IS NULL OR " + _in("task_type", ("binary", "multiclass", "regression")),
            name="task_type_value",
        ),
        CheckConstraint(_in("source", ("user", "suggested_accepted")), name="source"),
        Index(
            "uq_column_roles_one_target",
            "dataset_version_id",
            unique=True,
            postgresql_where="role = 'target'",
        ),
    )


class AnalysisRun(Base):
    """Inputs and summary of one successful diagnostics job (insert-only)."""

    __tablename__ = "analysis_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, unique=True
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False, server_default="diagnostics")
    # Phase 5: a reliability audit belongs to a training run; a diagnostics run never does.
    training_run_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("training_runs.id", ondelete="CASCADE")
    )
    domain: Mapped[str] = mapped_column(Text, nullable=False)
    check_set_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    semantics_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    roles_snapshot: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    summary: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    library_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("kind", ANALYSIS_KINDS), name="kind"),
        CheckConstraint(
            "(kind = 'reliability_audit') = (training_run_id IS NOT NULL)", name="audit_run"
        ),
        Index("ix_analysis_runs_version_created", "dataset_version_id", "created_at"),
        Index("ix_analysis_runs_training_run", "training_run_id"),
    )


class FindingRow(Base):
    """One diagnostic result (insert-only)."""

    __tablename__ = "findings"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    check_id: Mapped[str] = mapped_column(Text, nullable=False)
    check_version: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    severity: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    affected_rows: Mapped[int | None] = mapped_column(BigInteger)
    affected_pct: Mapped[float | None] = mapped_column(Float)
    recommendation: Mapped[str | None] = mapped_column(Text)
    limitations: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    not_assessed_reason: Mapped[str | None] = mapped_column(Text)
    error_code: Mapped[str | None] = mapped_column(Text)
    sampled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    sample_info: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(
            _in("status", ("detected", "not_detected", "not_assessed", "error")), name="status"
        ),
        CheckConstraint(
            _in("severity", ("critical", "high", "medium", "low", "info")), name="severity"
        ),
        CheckConstraint(
            "status <> 'not_assessed' OR not_assessed_reason IS NOT NULL",
            name="not_assessed_reason",
        ),
        CheckConstraint("status <> 'error' OR error_code IS NOT NULL", name="error_code"),
        CheckConstraint(
            "status <> 'detected' OR (evidence <> '{}'::jsonb AND recommendation IS NOT NULL)",
            name="detected_evidence",
        ),
        CheckConstraint("jsonb_array_length(limitations) > 0", name="limitations"),
        Index("ix_findings_run_position", "analysis_run_id", "position"),
    )


# ---- Phase 4 Part D: recommendations and experiments (migration 0005) ----------------------


def _owner() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)


def _dataset() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False
    )


def _version() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )


class RecommendationSet(Base):
    """One generation of recommendations for a version (insert-only). Regenerated only when
    an input changes (analysis run, profile, roles/domain or rules version)."""

    __tablename__ = "recommendation_sets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    dataset_version_id: Mapped[uuid.UUID] = _version()
    analysis_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    profile_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("version_profiles.id", ondelete="CASCADE")
    )
    semantics_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    input_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    rules_version: Mapped[str] = mapped_column(Text, nullable=False)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str | None] = mapped_column(Text)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "input_hash"),
        Index("ix_recommendation_sets_version_created", "dataset_version_id", "created_at"),
    )


class Recommendation(Base):
    """One recommendation of a set. Only ``status`` (and ``updated_at``) may change."""

    __tablename__ = "recommendations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    set_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recommendation_sets.id", ondelete="CASCADE"),
        nullable=False,
    )
    rec_key: Mapped[str] = mapped_column(Text, nullable=False)  # deterministic engine id
    position: Mapped[int] = mapped_column(Integer, nullable=False)  # rank (1 = first)
    rule_id: Mapped[str] = mapped_column(Text, nullable=False)
    rule_version: Mapped[str] = mapped_column(Text, nullable=False)
    op: Mapped[str] = mapped_column(Text, nullable=False)
    op_version: Mapped[str] = mapped_column(Text, nullable=False)
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    risk: Mapped[str] = mapped_column(Text, nullable=False)
    destructive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="open")
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("set_id", "rec_key"),
        CheckConstraint(_in("status", RECOMMENDATION_STATUSES), name="status"),
        CheckConstraint(_in("risk", ("low", "medium", "high")), name="risk"),
        CheckConstraint(_in("stage", ("data", "model", "either")), name="stage"),
        CheckConstraint("NOT destructive OR risk = 'high'", name="destructive_high"),
    )


class RecommendationFinding(Base):
    """Finding cited as evidence by a recommendation (insert-only)."""

    __tablename__ = "recommendation_findings"

    recommendation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("recommendations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    finding_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("findings.id", ondelete="CASCADE"), primary_key=True
    )
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()


class SplitDefinition(Base):
    """A frozen fold assignment (insert-only), reused by every experiment with the same
    configuration on the same version. The assignment itself is the ``fold_assignment``
    artifact; ``assignment_sha256`` proves which rows each fold contained."""

    __tablename__ = "split_definitions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    dataset_version_id: Mapped[uuid.UUID] = _version()
    config_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    method: Mapped[str] = mapped_column(Text, nullable=False)
    n_splits: Mapped[int] = mapped_column(Integer, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    target_column: Mapped[str | None] = mapped_column(Text)
    group_column: Mapped[str | None] = mapped_column(Text)
    time_column: Mapped[str | None] = mapped_column(Text)
    duplicate_columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    fold_assignment_artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=False
    )
    assignment_sha256: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    cohort: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    fold_stats: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    warnings: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    library_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "config_hash"),
        CheckConstraint(_in("method", SPLIT_METHODS), name="method"),
        CheckConstraint("n_splits BETWEEN 2 AND 20", name="n_splits"),
    )


class Experiment(Base):
    """An Experiment Lab run: its full reproducible configuration (written by the API) and,
    once the job succeeds, its split and summary (written once by the worker). Status lives
    in ``jobs``. Experiments never write dataset versions."""

    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    dataset_version_id: Mapped[uuid.UUID] = _version()
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, unique=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(Text, nullable=False)
    target_column: Mapped[str | None] = mapped_column(Text)
    models: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    source_content_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    recommendation_set_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recommendation_sets.id", ondelete="SET NULL")
    )
    rerun_of_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="SET NULL")
    )
    split_definition_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("split_definitions.id")
    )
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    library_versions: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = _now()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(_in("task_type", EXPERIMENT_TASKS), name="task_type"),
        CheckConstraint("char_length(name) BETWEEN 1 AND 200", name="name_length"),
        CheckConstraint("(completed_at IS NULL) = (summary IS NULL)", name="completed"),
        Index("ix_experiments_version_created", "dataset_version_id", "created_at"),
    )


class ExperimentCandidate(Base):
    """A candidate pipeline of an experiment (insert-only). Risk, destructive flag and stages
    are derived from the catalogue, never taken from the request."""

    __tablename__ = "experiment_candidates"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    key: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    is_baseline: Mapped[bool] = mapped_column(Boolean, nullable=False)
    steps: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    spec_hash: Mapped[str] = mapped_column(Text, nullable=False)
    max_risk: Mapped[str] = mapped_column(Text, nullable=False)
    has_destructive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    stages: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    recommendation_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("experiment_id", "position"),
        UniqueConstraint("experiment_id", "key"),
        CheckConstraint(_in("max_risk", ("low", "medium", "high")), name="max_risk"),
        CheckConstraint("NOT is_baseline OR jsonb_array_length(steps) = 0", name="baseline"),
    )


class ExperimentResult(Base):
    """Results of one candidate (insert-only, written in the job's final transaction)."""

    __tablename__ = "experiment_results"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    candidate_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiment_candidates.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    comparable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comparability_reason: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    fold_results: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    data_quality: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    selection_stability: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    failures: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("status", RESULT_STATUSES), name="status"),
        CheckConstraint("comparable OR comparability_reason IS NOT NULL", name="comparable"),
        Index("ix_experiment_results_experiment", "experiment_id"),
    )


# ---- Phase 4 Part E: pipelines, approvals, commits, reverts (migration 0006) ---------------

PIPELINE_STATES = ("draft", "validated", "committed", "archived")


class Pipeline(Base):
    """An editable preprocessing pipeline against a source version. ``spec_hash`` is
    recomputed by the service on every step change, which resets ``state`` to draft and
    makes every earlier approval and preview inapplicable."""

    __tablename__ = "pipelines"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    source_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    state: Mapped[str] = mapped_column(Text, nullable=False, server_default="draft")
    spec_hash: Mapped[str] = mapped_column(Text, nullable=False)
    validation_report: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    impact_preview_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id")
    )
    impact_preview_spec_hash: Mapped[str | None] = mapped_column(Text)
    impact_preview_job_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id")
    )
    from_experiment_candidate_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiment_candidates.id", ondelete="SET NULL")
    )
    committed_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id")
    )
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("state", PIPELINE_STATES), name="state"),
        CheckConstraint("char_length(name) BETWEEN 1 AND 200", name="name_length"),
        CheckConstraint(
            "(state = 'committed') = (committed_version_id IS NOT NULL)", name="committed"
        ),
        Index("ix_pipelines_dataset_created", "dataset_id", "created_at"),
    )


class PipelineStep(Base):
    """A step of a pipeline. Risk, destructive flag and kind come from the catalogue; the
    chosen stage (data | model) is explicit and part of ``step_hash``."""

    __tablename__ = "pipeline_steps"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    pipeline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipelines.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    op: Mapped[str] = mapped_column(Text, nullable=False)
    op_version: Mapped[str] = mapped_column(Text, nullable=False)
    params: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    columns: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    stage: Mapped[str] = mapped_column(Text, nullable=False)
    risk: Mapped[str] = mapped_column(Text, nullable=False)
    destructive: Mapped[bool] = mapped_column(Boolean, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    step_hash: Mapped[str | None] = mapped_column(Text)  # null while the pipeline is invalid
    from_recommendation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("recommendations.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("pipeline_id", "position", deferrable=True, initially="DEFERRED"),
        CheckConstraint(_in("stage", ("data", "model")), name="stage"),
        CheckConstraint(_in("risk", ("low", "medium", "high")), name="risk"),
        CheckConstraint("NOT destructive OR risk = 'high'", name="destructive_high"),
        CheckConstraint("position >= 0", name="position"),
    )


class Approval(Base):
    """An approval exactly as displayed (insert-only). Valid only while its hashes equal the
    pipeline's current ones and its preview is the current preview."""

    __tablename__ = "approvals"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    pipeline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipelines.id", ondelete="CASCADE"), nullable=False
    )
    # No foreign key: approval history outlives deleted steps.
    pipeline_step_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    scope: Mapped[str] = mapped_column(Text, nullable=False)
    approved_hash: Mapped[str] = mapped_column(Text, nullable=False)
    spec_hash: Mapped[str] = mapped_column(Text, nullable=False)
    source_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    preview_artifact_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id"), nullable=False
    )
    impact_shown: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    acknowledge_destructive: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default="false"
    )
    approved_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("scope", ("step", "pipeline")), name="scope"),
        CheckConstraint("(scope = 'step') = (pipeline_step_id IS NOT NULL)", name="step"),
        Index("ix_approvals_pipeline", "pipeline_id", "created_at"),
    )


class TransformationRun(Base):
    """The committed transformation behind a ``transform`` version (insert-only, written in
    the same transaction as its version)."""

    __tablename__ = "transformation_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    pipeline_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("pipelines.id"), nullable=False
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, unique=True
    )
    source_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id"), nullable=False
    )
    result_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "dataset_versions.id",
            deferrable=True,
            initially="DEFERRED",
            name="fk_transformation_runs_result_version_id_dataset_versions",
        ),
        nullable=False,
        unique=True,
    )
    spec_snapshot: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    spec_hash: Mapped[str] = mapped_column(Text, nullable=False)
    approval_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    step_reports: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    model_pipeline_spec: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    roles_report: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    seeds: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    library_versions: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    executed_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    created_at: Mapped[datetime] = _now()


class VersionEvent(Base):
    """Version history events (insert-only): versions created by commit or revert."""

    __tablename__ = "version_events"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    event: Mapped[str] = mapped_column(Text, nullable=False)
    version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    related_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE")
    )
    actor_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("event", ("created", "reverted")), name="event"),
        Index("ix_version_events_dataset_created", "dataset_id", "created_at"),
    )


# ---- Phase 5: supervised training and reliability audits (migration 0007) -------------------


class TrainingRun(Base):
    """A supervised training run: its complete reproducible configuration (written by the API)
    and, once the job succeeds, its summary (written once by the worker; a trigger forbids any
    later change). Status lives in ``jobs``. Training never writes dataset versions.

    ``evaluation_version_id`` is the version whose rows are evaluated: the selected version
    itself, or its eligible ancestor when committed full-data-fitted steps are replayed inside
    the folds (``evaluation_mode = 'unbiased_replay'``). ``exploratory`` runs evaluate a
    full-data-fitted version as stored and stay labelled as such."""

    __tablename__ = "training_runs"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    dataset_version_id: Mapped[uuid.UUID] = _version()
    evaluation_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id"), nullable=False, unique=True
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    task_type: Mapped[str] = mapped_column(Text, nullable=False)
    target_column: Mapped[str] = mapped_column(Text, nullable=False)
    evaluation_mode: Mapped[str] = mapped_column(Text, nullable=False)
    eligibility: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    replay: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    split_config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    split_definition_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("split_definitions.id")
    )
    models: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    tuning: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    preprocess_steps: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    spec_hash: Mapped[str] = mapped_column(Text, nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    config_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    source_content_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    rerun_of_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("training_runs.id", ondelete="SET NULL")
    )
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    oof_artifact_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("artifacts.id")
    )
    engine_version: Mapped[str] = mapped_column(Text, nullable=False)
    library_versions: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    created_by: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    created_at: Mapped[datetime] = _now()
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(_in("task_type", TRAINING_TASKS), name="task_type"),
        CheckConstraint(_in("evaluation_mode", EVALUATION_MODES), name="evaluation_mode"),
        CheckConstraint(
            "(evaluation_mode = 'unbiased_replay') = (evaluation_version_id <> dataset_version_id)",
            name="replay_source",
        ),
        CheckConstraint("char_length(name) BETWEEN 1 AND 200", name="name_length"),
        CheckConstraint("(completed_at IS NULL) = (summary IS NULL)", name="completed"),
        CheckConstraint("seed >= 0", name="seed"),
        Index("ix_training_runs_dataset_created", "dataset_id", "created_at"),
    )


class ModelEvaluation(Base):
    """Results of one model in a training run (insert-only, written in the job's final
    transaction)."""

    __tablename__ = "model_evaluations"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    training_run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("training_runs.id", ondelete="CASCADE"), nullable=False
    )
    model_key: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    comparable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    comparability_reason: Mapped[str | None] = mapped_column(Text)
    metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    fold_results: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    selection: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    importance: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    failures: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()

    __table_args__ = (
        UniqueConstraint("training_run_id", "model_key", name="uq_model_evaluations_run_model"),
        CheckConstraint(_in("model_key", TRAINING_MODELS), name="model_key"),
        CheckConstraint(_in("status", RESULT_STATUSES), name="status"),
        CheckConstraint("comparable OR comparability_reason IS NOT NULL", name="comparable"),
        CheckConstraint("NOT comparable OR status = 'completed'", name="complete"),
        Index("ix_model_evaluations_run", "training_run_id"),
    )


WIDGET_KINDS = ("chart", "kpi", "pivot")
WIDGET_SIZES = ("small", "medium", "large")


class Dashboard(Base):
    """A saved Analytics Studio dashboard, bound to one immutable dataset version. It stores
    validated widget specifications only (never data) and is never switched silently to
    another version."""

    __tablename__ = "dashboards"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dataset_versions.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default="")
    filters: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    source_dashboard_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dashboards.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint("char_length(name) BETWEEN 1 AND 200", name="name_length"),
        CheckConstraint("char_length(description) <= 2000", name="description_length"),
        CheckConstraint("jsonb_typeof(filters) = 'array'", name="filters_array"),
        Index("ix_dashboards_dataset_created", "dataset_id", "created_at"),
    )


class DashboardWidget(Base):
    """One chart, KPI card or pivot table of a dashboard (a validated query specification)."""

    __tablename__ = "dashboard_widgets"

    id: Mapped[uuid.UUID] = _uuid_pk()
    owner_id: Mapped[uuid.UUID] = _owner()
    dataset_id: Mapped[uuid.UUID] = _dataset()
    dashboard_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dashboards.id", ondelete="CASCADE"), nullable=False
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    size: Mapped[str] = mapped_column(Text, nullable=False, server_default="medium")
    spec: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = _now()

    __table_args__ = (
        CheckConstraint(_in("kind", WIDGET_KINDS), name="kind"),
        CheckConstraint(_in("size", WIDGET_SIZES), name="size"),
        CheckConstraint("char_length(title) BETWEEN 1 AND 200", name="title_length"),
        CheckConstraint("position >= 0", name="position"),
        CheckConstraint("jsonb_typeof(spec) = 'object'", name="spec_object"),
        Index("ix_dashboard_widgets_dashboard_position", "dashboard_id", "position"),
    )

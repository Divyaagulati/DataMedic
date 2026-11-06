"""Phase 1 foundation: users, sessions, auth tokens, datasets, original files, jobs.

Includes Row-Level Security, owner-consistency and immutability triggers, grants for the
dm_app / dm_maintenance roles, and the claim_job() SECURITY DEFINER function.

Revision ID: 0001
Revises:
Create Date: 2026-11-06
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

UUID = pg.UUID(as_uuid=True)
TS = sa.DateTime(timezone=True)
NOW = sa.text("now()")

RLS_TABLES = ("datasets", "original_files", "jobs")
OWNER_EXPR = "NULLIF(current_setting('app.user_id', true), '')::uuid"


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("email", sa.Text, nullable=False),
        sa.Column("email_verified_at", TS),
        sa.Column("password_hash", sa.Text, nullable=False),
        sa.Column("display_name", sa.Text),
        sa.Column("plan", sa.Text, nullable=False, server_default="free"),
        sa.Column("status", sa.Text, nullable=False, server_default="active"),
        sa.Column("failed_login_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_failed_login_at", TS),
        sa.Column("deleted_at", TS),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("updated_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
        sa.CheckConstraint(_in("status", ("active", "deleting")), name="ck_users_status"),
        sa.CheckConstraint("email = lower(email)", name="ck_users_email_lower"),
    )

    op.create_table(
        "sessions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("token_hash", sa.LargeBinary, nullable=False),
        sa.Column("csrf_token", sa.Text, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("last_seen_at", TS, nullable=False, server_default=NOW),
        sa.Column("expires_at", TS, nullable=False),
        sa.Column("revoked_at", TS),
        sa.Column("user_agent", sa.Text),
        sa.PrimaryKeyConstraint("id", name="pk_sessions"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE", name="fk_sessions_user_id_users"
        ),
        sa.UniqueConstraint("token_hash", name="uq_sessions_token_hash"),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    op.create_table(
        "auth_tokens",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("user_id", UUID, nullable=False),
        sa.Column("purpose", sa.Text, nullable=False),
        sa.Column("token_hash", sa.LargeBinary, nullable=False),
        sa.Column("expires_at", TS, nullable=False),
        sa.Column("used_at", TS),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_auth_tokens"),
        sa.ForeignKeyConstraint(
            ["user_id"], ["users.id"], ondelete="CASCADE", name="fk_auth_tokens_user_id_users"
        ),
        sa.UniqueConstraint("token_hash", name="uq_auth_tokens_token_hash"),
        sa.CheckConstraint(
            _in("purpose", ("verify_email", "reset_password")), name="ck_auth_tokens_purpose"
        ),
    )
    op.create_index("ix_auth_tokens_user_id", "auth_tokens", ["user_id"])

    dataset_statuses = (
        "upload_pending",
        "validating",
        "validated",
        "parsing",
        "parse_failed",
        "ready",
        "rejected",
        "deleting",
    )
    op.create_table(
        "datasets",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("domain", sa.Text, nullable=False, server_default="generic"),
        sa.Column("domain_changed_at", TS),
        sa.Column("status", sa.Text, nullable=False, server_default="upload_pending"),
        sa.Column("declared_filename", sa.Text, nullable=False),
        sa.Column("declared_extension", sa.Text, nullable=False),
        sa.Column("declared_size_bytes", sa.BigInteger, nullable=False),
        sa.Column("deleted_at", TS),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("updated_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_datasets"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name="fk_datasets_owner_id_users"),
        sa.CheckConstraint(_in("status", dataset_statuses), name="ck_datasets_status"),
        sa.CheckConstraint("declared_size_bytes > 0", name="ck_datasets_size_positive"),
        sa.CheckConstraint("char_length(name) BETWEEN 1 AND 200", name="ck_datasets_name_length"),
    )
    op.create_index(
        "ix_datasets_owner_created", "datasets", ["owner_id", "deleted_at", "created_at"]
    )

    op.create_table(
        "original_files",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("object_key", sa.Text, nullable=False),
        sa.Column("sha256", sa.LargeBinary, nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("detected_format", sa.Text),
        sa.Column("original_filename", sa.Text),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_original_files"),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_original_files_owner_id_users"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            ondelete="CASCADE",
            name="fk_original_files_dataset_id_datasets",
        ),
        sa.UniqueConstraint("dataset_id", name="uq_original_files_dataset_id"),
        sa.CheckConstraint(
            "detected_format IS NULL OR "
            + _in("detected_format", ("csv", "xlsx", "xls", "parquet", "json")),
            name="ck_original_files_format",
        ),
    )
    op.create_index("ix_original_files_owner_id", "original_files", ["owner_id"])

    op.create_table(
        "jobs",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("type", sa.Text, nullable=False),
        sa.Column("queue", sa.Text, nullable=False),
        sa.Column("dataset_id", UUID),
        sa.Column("params", pg.JSONB, nullable=False, server_default="{}"),
        sa.Column("status", sa.Text, nullable=False, server_default="queued"),
        sa.Column("progress", pg.JSONB),
        sa.Column("result", pg.JSONB),
        sa.Column("error", pg.JSONB),
        sa.Column("attempt", sa.Integer, nullable=False, server_default="0"),
        sa.Column("idempotency_key", sa.Text),
        sa.Column("worker_id", sa.Text),
        sa.Column("time_limit_seconds", sa.Integer, nullable=False),
        sa.Column("queued_at", TS, nullable=False, server_default=NOW),
        sa.Column("last_published_at", TS),
        sa.Column("started_at", TS),
        sa.Column("heartbeat_at", TS),
        sa.Column("finished_at", TS),
        sa.PrimaryKeyConstraint("id", name="pk_jobs"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name="fk_jobs_owner_id_users"),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.id"], ondelete="SET NULL", name="fk_jobs_dataset_id_datasets"
        ),
        sa.CheckConstraint(
            _in(
                "status",
                ("queued", "running", "succeeded", "failed", "cancel_requested", "cancelled"),
            ),
            name="ck_jobs_status",
        ),
        sa.CheckConstraint(_in("type", ("ingest", "purge")), name="ck_jobs_type"),
    )
    op.create_index("ix_jobs_owner_status", "jobs", ["owner_id", "status"])
    op.create_index("ix_jobs_status_heartbeat", "jobs", ["status", "heartbeat_at"])
    op.create_index("ix_jobs_status_published", "jobs", ["status", "last_published_at"])
    op.create_index(
        "uq_jobs_owner_idempotency",
        "jobs",
        ["owner_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )

    # ---- Row-Level Security -------------------------------------------------------------
    for table in RLS_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY owner_isolation ON {table} "
            f"USING (owner_id = {OWNER_EXPR}) WITH CHECK (owner_id = {OWNER_EXPR})"
        )

    # ---- Owner consistency: a child row's owner must equal its dataset's owner -----------
    op.execute(
        """
        CREATE FUNCTION enforce_dataset_owner() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.dataset_id IS NOT NULL AND NOT EXISTS (
               SELECT 1 FROM datasets d WHERE d.id = NEW.dataset_id AND d.owner_id = NEW.owner_id
             ) THEN
            RAISE EXCEPTION 'owner_id does not match dataset owner' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    for table in ("original_files", "jobs"):
        op.execute(
            f"CREATE TRIGGER {table}_dataset_owner BEFORE INSERT OR UPDATE OF owner_id, dataset_id "
            f"ON {table} FOR EACH ROW EXECUTE FUNCTION enforce_dataset_owner()"
        )

    # ---- Original files are immutable (R3); detected_format may be set once --------------
    op.execute(
        """
        CREATE FUNCTION original_files_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.object_key IS DISTINCT FROM OLD.object_key
             OR NEW.sha256 IS DISTINCT FROM OLD.sha256
             OR NEW.size_bytes IS DISTINCT FROM OLD.size_bytes
             OR NEW.owner_id IS DISTINCT FROM OLD.owner_id
             OR NEW.dataset_id IS DISTINCT FROM OLD.dataset_id
             OR (OLD.detected_format IS NOT NULL
                 AND NEW.detected_format IS DISTINCT FROM OLD.detected_format) THEN
            RAISE EXCEPTION 'original_files rows are immutable' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER original_files_no_update BEFORE UPDATE ON original_files "
        "FOR EACH ROW EXECUTE FUNCTION original_files_immutable()"
    )

    # ---- Job claiming without exposing other users' jobs ----------------------------------
    op.execute(
        """
        CREATE FUNCTION claim_job(p_job_id uuid, p_worker text)
        RETURNS TABLE (owner_id uuid, job_type text)
        LANGUAGE sql SECURITY DEFINER SET search_path = public, pg_temp AS $$
          UPDATE jobs
             SET status = 'running', attempt = attempt + 1, worker_id = p_worker,
                 started_at = now(), heartbeat_at = now()
           WHERE id = p_job_id AND status = 'queued'
          RETURNING jobs.owner_id, jobs.type
        $$
        """
    )

    # ---- Grants ---------------------------------------------------------------------------
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON users, sessions, auth_tokens TO dm_app")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON datasets TO dm_app")
    op.execute("GRANT SELECT, INSERT, DELETE ON original_files TO dm_app")
    op.execute("GRANT UPDATE (detected_format) ON original_files TO dm_app")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON jobs TO dm_app")
    op.execute("GRANT SELECT, UPDATE ON jobs TO dm_maintenance")
    op.execute("REVOKE ALL ON FUNCTION claim_job(uuid, text) FROM PUBLIC")
    op.execute("GRANT EXECUTE ON FUNCTION claim_job(uuid, text) TO dm_app")
    # The new owner needs CREATE on the schema only for the ownership transfer itself.
    op.execute("GRANT CREATE ON SCHEMA public TO dm_maintenance")
    op.execute("ALTER FUNCTION claim_job(uuid, text) OWNER TO dm_maintenance")
    op.execute("REVOKE CREATE ON SCHEMA public FROM dm_maintenance")


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS claim_job(uuid, text)")
    op.drop_table("jobs")
    op.drop_table("original_files")
    op.drop_table("datasets")
    op.drop_table("auth_tokens")
    op.drop_table("sessions")
    op.drop_table("users")
    op.execute("DROP FUNCTION IF EXISTS original_files_immutable()")
    op.execute("DROP FUNCTION IF EXISTS enforce_dataset_owner()")

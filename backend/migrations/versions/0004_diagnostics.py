"""Phase 3: column roles, analysis runs and findings (diagnostics).

- column_roles: per version, replaceable (insert/delete), one target per version, task type
  required iff role = target.
- analysis_runs, findings: insert-only. Status lives in jobs; a run row is written when the
  analysis job succeeds.
- findings enforce the honesty rules: not_assessed needs a reason, error needs a code,
  detected needs evidence and a suggested action.
- jobs.type gains 'analysis'.

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-09
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

UUID = pg.UUID(as_uuid=True)
TS = sa.DateTime(timezone=True)
NOW = sa.text("now()")
OWNER_EXPR = "NULLIF(current_setting('app.user_id', true), '')::uuid"
NEW_TABLES = ("column_roles", "analysis_runs", "findings")
JOB_TYPES_OLD = ("ingest", "profile", "purge")
JOB_TYPES_NEW = ("ingest", "profile", "analysis", "purge")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def _owner_dataset_fks(table: str) -> list[sa.ForeignKeyConstraint]:
    return [
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name=f"fk_{table}_owner_id_users"),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            ondelete="CASCADE",
            name=f"fk_{table}_dataset_id_datasets",
        ),
    ]


def upgrade() -> None:
    op.create_table(
        "column_roles",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("column_name", sa.Text, nullable=False),
        sa.Column("role", sa.Text, nullable=False),
        sa.Column("task_type", sa.Text),
        sa.Column("source", sa.Text, nullable=False, server_default="user"),
        sa.Column("created_by", UUID),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_column_roles"),
        *_owner_dataset_fks("column_roles"),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            ondelete="CASCADE",
            name="fk_column_roles_dataset_version_id_dataset_versions",
        ),
        sa.UniqueConstraint(
            "dataset_version_id",
            "column_name",
            "role",
            name="uq_column_roles_dataset_version_id_column_name_role",
        ),
        sa.CheckConstraint(
            "(role = 'target') = (task_type IS NOT NULL)", name="ck_column_roles_task_type"
        ),
        sa.CheckConstraint(
            "task_type IS NULL OR " + _in("task_type", ("binary", "multiclass", "regression")),
            name="ck_column_roles_task_type_value",
        ),
        sa.CheckConstraint(
            _in("source", ("user", "suggested_accepted")), name="ck_column_roles_source"
        ),
    )
    op.create_index("ix_column_roles_owner_id", "column_roles", ["owner_id"])
    op.create_index(
        "uq_column_roles_one_target",
        "column_roles",
        ["dataset_version_id"],
        unique=True,
        postgresql_where=sa.text("role = 'target'"),
    )

    op.create_table(
        "analysis_runs",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("job_id", UUID, nullable=False),
        sa.Column("kind", sa.Text, nullable=False, server_default="diagnostics"),
        sa.Column("domain", sa.Text, nullable=False),
        sa.Column("check_set_hash", sa.LargeBinary, nullable=False),
        sa.Column("semantics_hash", sa.LargeBinary, nullable=False),
        sa.Column("roles_snapshot", pg.JSONB, nullable=False),
        sa.Column("config", pg.JSONB, nullable=False),
        sa.Column("summary", pg.JSONB, nullable=False),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("library_versions", pg.JSONB, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_analysis_runs"),
        *_owner_dataset_fks("analysis_runs"),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            ondelete="CASCADE",
            name="fk_analysis_runs_dataset_version_id_dataset_versions",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name="fk_analysis_runs_job_id_jobs"),
        sa.UniqueConstraint("job_id", name="uq_analysis_runs_job_id"),
        sa.CheckConstraint(_in("kind", ("diagnostics",)), name="ck_analysis_runs_kind"),
    )
    op.create_index("ix_analysis_runs_owner_id", "analysis_runs", ["owner_id"])
    op.create_index(
        "ix_analysis_runs_version_created", "analysis_runs", ["dataset_version_id", "created_at"]
    )

    op.create_table(
        "findings",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("analysis_run_id", UUID, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("check_id", sa.Text, nullable=False),
        sa.Column("check_version", sa.Text, nullable=False),
        sa.Column("domain", sa.Text, nullable=False),
        sa.Column("category", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("severity", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("explanation", sa.Text, nullable=False),
        sa.Column("evidence", pg.JSONB, nullable=False),
        sa.Column("columns", pg.JSONB, nullable=False),
        sa.Column("affected_rows", sa.BigInteger),
        sa.Column("affected_pct", sa.Float),
        sa.Column("recommendation", sa.Text),
        sa.Column("limitations", pg.JSONB, nullable=False),
        sa.Column("not_assessed_reason", sa.Text),
        sa.Column("error_code", sa.Text),
        sa.Column("sampled", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("sample_info", pg.JSONB),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_findings"),
        *_owner_dataset_fks("findings"),
        sa.ForeignKeyConstraint(
            ["analysis_run_id"],
            ["analysis_runs.id"],
            ondelete="CASCADE",
            name="fk_findings_analysis_run_id_analysis_runs",
        ),
        sa.CheckConstraint(
            _in("status", ("detected", "not_detected", "not_assessed", "error")),
            name="ck_findings_status",
        ),
        sa.CheckConstraint(
            _in("severity", ("critical", "high", "medium", "low", "info")),
            name="ck_findings_severity",
        ),
        sa.CheckConstraint(
            "status <> 'not_assessed' OR not_assessed_reason IS NOT NULL",
            name="ck_findings_not_assessed_reason",
        ),
        sa.CheckConstraint(
            "status <> 'error' OR error_code IS NOT NULL", name="ck_findings_error_code"
        ),
        sa.CheckConstraint(
            "status <> 'detected' OR (evidence <> '{}'::jsonb AND recommendation IS NOT NULL)",
            name="ck_findings_detected_evidence",
        ),
        sa.CheckConstraint("jsonb_array_length(limitations) > 0", name="ck_findings_limitations"),
    )
    op.create_index("ix_findings_owner_id", "findings", ["owner_id"])
    op.create_index("ix_findings_run_position", "findings", ["analysis_run_id", "position"])

    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", JOB_TYPES_NEW))

    for table in NEW_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY owner_isolation ON {table} "
            f"USING (owner_id = {OWNER_EXPR}) WITH CHECK (owner_id = {OWNER_EXPR})"
        )
        op.execute(
            f"CREATE TRIGGER {table}_dataset_owner BEFORE INSERT OR UPDATE OF owner_id, dataset_id "
            f"ON {table} FOR EACH ROW EXECUTE FUNCTION enforce_dataset_owner()"
        )
    for table in ("column_roles", "analysis_runs"):
        op.execute(
            f"CREATE TRIGGER {table}_version BEFORE INSERT ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION enforce_version_in_dataset()"
        )
    op.execute(
        """
        CREATE FUNCTION enforce_finding_run() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NOT EXISTS (
               SELECT 1 FROM analysis_runs r
                WHERE r.id = NEW.analysis_run_id
                  AND r.dataset_id = NEW.dataset_id AND r.owner_id = NEW.owner_id) THEN
            RAISE EXCEPTION 'finding does not belong to this run, dataset and owner'
              USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER findings_run BEFORE INSERT ON findings "
        "FOR EACH ROW EXECUTE FUNCTION enforce_finding_run()"
    )

    op.execute("GRANT SELECT, INSERT, DELETE ON column_roles TO dm_app")
    op.execute("GRANT SELECT, INSERT, DELETE ON analysis_runs TO dm_app")
    op.execute("GRANT SELECT, INSERT, DELETE ON findings TO dm_app")


def downgrade() -> None:
    op.drop_table("findings")
    op.drop_table("analysis_runs")
    op.drop_table("column_roles")
    op.execute("DROP FUNCTION IF EXISTS enforce_finding_run()")
    op.execute("DELETE FROM jobs WHERE type = 'analysis'")
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", JOB_TYPES_OLD))

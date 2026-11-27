"""Phase 5: supervised training runs, model evaluations and reliability audits.

- training_runs: the complete reproducible configuration (written by the API) and, once the
  job succeeds, the summary (written once by the worker; a trigger forbids any later change);
- model_evaluations: one insert-only row per model and run;
- analysis_runs gain kind 'reliability_audit' and a training_run_id (an audit always belongs to
  a training run; a diagnostics run never does); audit findings reuse the findings table;
- jobs.type gains train and reliability_audit; artifacts gain kind oof_predictions and owner
  type training_run;
- RLS, owner, parent-consistency and version-in-dataset triggers on every new table.

Expand-only: existing rows are untouched.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.schema import SchemaItem

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None

UUID = pg.UUID(as_uuid=True)
TS = sa.DateTime(timezone=True)
NOW = sa.text("now()")
OWNER_EXPR = "NULLIF(current_setting('app.user_id', true), '')::uuid"
NEW_TABLES = ("training_runs", "model_evaluations")
JOB_TYPES_OLD = (
    "ingest",
    "profile",
    "analysis",
    "experiment",
    "impact_preview",
    "commit",
    "revert",
    "compare_versions",
    "purge",
)
JOB_TYPES_NEW = (*JOB_TYPES_OLD[:-1], "train", "reliability_audit", "purge")
KINDS_OLD = ("preview", "fold_assignment", "impact_preview", "version_comparison")
KINDS_NEW = (*KINDS_OLD, "oof_predictions")
OWNER_TYPES_OLD = ("dataset_version", "split_definition", "pipeline")
OWNER_TYPES_NEW = (*OWNER_TYPES_OLD, "training_run")
MODES = ("unbiased_direct", "unbiased_replay", "exploratory")
TASKS = ("binary", "multiclass", "regression")
RESULT_COLUMNS = (
    "split_definition_id",
    "replay",
    "summary",
    "oof_artifact_id",
    "library_versions",
    "completed_at",
)


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def _fk(
    table: str, column: str, target: str, ondelete: str | None = None
) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        [column], [f"{target}.id"], ondelete=ondelete, name=f"fk_{table}_{column}_{target}"
    )


def _base(table: str) -> list[SchemaItem]:
    return [
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.PrimaryKeyConstraint("id", name=f"pk_{table}"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name=f"fk_{table}_owner_id_users"),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            ondelete="CASCADE",
            name=f"fk_{table}_dataset_id_datasets",
        ),
    ]


def upgrade() -> None:
    t = "training_runs"
    op.create_table(
        t,
        *_base(t),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("evaluation_version_id", UUID, nullable=False),
        sa.Column("job_id", UUID, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("task_type", sa.Text, nullable=False),
        sa.Column("target_column", sa.Text, nullable=False),
        sa.Column("evaluation_mode", sa.Text, nullable=False),
        sa.Column("eligibility", pg.JSONB, nullable=False),
        sa.Column("replay", pg.JSONB),
        sa.Column("split_config", pg.JSONB, nullable=False),
        sa.Column("split_definition_id", UUID),
        sa.Column("models", pg.JSONB, nullable=False),
        sa.Column("tuning", pg.JSONB, nullable=False),
        sa.Column("preprocess_steps", pg.JSONB, nullable=False),
        sa.Column("spec_hash", sa.Text, nullable=False),
        sa.Column("config", pg.JSONB, nullable=False),
        sa.Column("config_hash", sa.LargeBinary, nullable=False),
        sa.Column("source_content_hash", sa.LargeBinary, nullable=False),
        sa.Column("seed", sa.Integer, nullable=False),
        sa.Column("rerun_of_id", UUID),
        sa.Column("summary", pg.JSONB),
        sa.Column("oof_artifact_id", UUID),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("library_versions", pg.JSONB),
        sa.Column("created_by", UUID),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("completed_at", TS),
        _fk(t, "dataset_version_id", "dataset_versions", "CASCADE"),
        _fk(t, "evaluation_version_id", "dataset_versions", "CASCADE"),
        _fk(t, "job_id", "jobs"),
        _fk(t, "split_definition_id", "split_definitions"),
        _fk(t, "rerun_of_id", "training_runs", "SET NULL"),
        _fk(t, "oof_artifact_id", "artifacts"),
        sa.UniqueConstraint("job_id", name=f"uq_{t}_job_id"),
        sa.CheckConstraint(_in("task_type", TASKS), name=f"ck_{t}_task_type"),
        sa.CheckConstraint(_in("evaluation_mode", MODES), name=f"ck_{t}_evaluation_mode"),
        sa.CheckConstraint(
            "(evaluation_mode = 'unbiased_replay') = (evaluation_version_id <> dataset_version_id)",
            name=f"ck_{t}_replay_source",
        ),
        sa.CheckConstraint("char_length(name) BETWEEN 1 AND 200", name=f"ck_{t}_name_length"),
        sa.CheckConstraint("(completed_at IS NULL) = (summary IS NULL)", name=f"ck_{t}_completed"),
        sa.CheckConstraint("seed >= 0", name=f"ck_{t}_seed"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_dataset_created", t, ["dataset_id", "created_at"])

    t = "model_evaluations"
    op.create_table(
        t,
        *_base(t),
        sa.Column("training_run_id", UUID, nullable=False),
        sa.Column("model_key", sa.Text, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("comparable", sa.Boolean, nullable=False),
        sa.Column("comparability_reason", sa.Text),
        sa.Column("metrics", pg.JSONB, nullable=False),
        sa.Column("details", pg.JSONB, nullable=False),
        sa.Column("fold_results", pg.JSONB, nullable=False),
        sa.Column("selection", pg.JSONB, nullable=False),
        sa.Column("importance", pg.JSONB, nullable=False),
        sa.Column("failures", pg.JSONB, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        _fk(t, "training_run_id", "training_runs", "CASCADE"),
        sa.UniqueConstraint("training_run_id", "model_key", name=f"uq_{t}_run_model"),
        sa.CheckConstraint(
            _in("model_key", ("dummy", "linear", "random_forest", "hgb")), name=f"ck_{t}_model_key"
        ),
        sa.CheckConstraint(
            _in("status", ("completed", "partial", "failed")), name=f"ck_{t}_status"
        ),
        sa.CheckConstraint(
            "comparable OR comparability_reason IS NOT NULL", name=f"ck_{t}_comparable"
        ),
        sa.CheckConstraint("NOT comparable OR status = 'completed'", name=f"ck_{t}_complete"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_run", t, ["training_run_id"])

    # ---- analysis_runs: reliability audits -------------------------------------------------
    op.add_column("analysis_runs", sa.Column("training_run_id", UUID))
    op.create_foreign_key(
        "fk_analysis_runs_training_run_id_training_runs",
        "analysis_runs",
        "training_runs",
        ["training_run_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.drop_constraint("ck_analysis_runs_kind", "analysis_runs", type_="check")
    op.create_check_constraint(
        "ck_analysis_runs_kind", "analysis_runs", _in("kind", ("diagnostics", "reliability_audit"))
    )
    op.create_check_constraint(
        "ck_analysis_runs_audit_run",
        "analysis_runs",
        "(kind = 'reliability_audit') = (training_run_id IS NOT NULL)",
    )
    op.create_index("ix_analysis_runs_training_run", "analysis_runs", ["training_run_id"])
    op.execute(
        "CREATE TRIGGER analysis_runs_training_run_id_parent BEFORE INSERT OR UPDATE OF "
        "training_run_id ON analysis_runs FOR EACH ROW EXECUTE FUNCTION "
        "enforce_same_dataset_parent('training_runs', 'training_run_id')"
    )

    # ---- Enumerations ----------------------------------------------------------------------
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", JOB_TYPES_NEW))
    op.drop_constraint("ck_artifacts_kind", "artifacts", type_="check")
    op.create_check_constraint("ck_artifacts_kind", "artifacts", _in("kind", KINDS_NEW))
    op.drop_constraint("ck_artifacts_owner_type", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifacts_owner_type", "artifacts", _in("owner_type", OWNER_TYPES_NEW)
    )

    # ---- RLS, owner and parent consistency ---------------------------------------------------
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
    op.execute(
        "CREATE TRIGGER training_runs_version BEFORE INSERT ON training_runs "
        "FOR EACH ROW EXECUTE FUNCTION enforce_version_in_dataset()"
    )
    parents = [
        ("training_runs", "dataset_versions", "evaluation_version_id"),
        ("training_runs", "split_definitions", "split_definition_id"),
        ("training_runs", "training_runs", "rerun_of_id"),
        ("training_runs", "artifacts", "oof_artifact_id"),
        ("model_evaluations", "training_runs", "training_run_id"),
    ]
    for table, parent, column in parents:
        op.execute(
            f"CREATE TRIGGER {table}_{column}_parent BEFORE INSERT OR UPDATE OF {column} "
            f"ON {table} FOR EACH ROW EXECUTE FUNCTION "
            f"enforce_same_dataset_parent('{parent}', '{column}')"
        )
    # The split belongs to the evaluated (source) version.
    op.execute(
        """
        CREATE FUNCTION training_runs_split_version() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.split_definition_id IS NOT NULL AND NOT EXISTS (
               SELECT 1 FROM split_definitions s WHERE s.id = NEW.split_definition_id
                  AND s.dataset_version_id = NEW.evaluation_version_id) THEN
            RAISE EXCEPTION 'split belongs to another version' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER training_runs_split_version BEFORE INSERT OR UPDATE OF "
        "split_definition_id ON training_runs FOR EACH ROW EXECUTE FUNCTION "
        "training_runs_split_version()"
    )
    # Configuration is immutable; results are written exactly once.
    op.execute(
        """
        CREATE FUNCTION training_runs_complete_once() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF EXISTS (SELECT 1 FROM datasets d WHERE d.id = OLD.dataset_id
                       AND d.deleted_at IS NOT NULL) THEN
            RETURN NEW;  -- dataset purge (ON DELETE SET NULL of rerun_of_id)
          END IF;
          IF OLD.completed_at IS NOT NULL THEN
            RAISE EXCEPTION 'a completed training run cannot change' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER training_runs_complete_once BEFORE UPDATE ON training_runs "
        "FOR EACH ROW EXECUTE FUNCTION training_runs_complete_once()"
    )

    # ---- Grants ------------------------------------------------------------------------------
    op.execute("GRANT SELECT, INSERT, DELETE ON training_runs, model_evaluations TO dm_app")
    op.execute(f"GRANT UPDATE ({', '.join(RESULT_COLUMNS)}) ON training_runs TO dm_app")


def downgrade() -> None:
    op.execute(
        "DELETE FROM findings WHERE analysis_run_id IN (SELECT id FROM analysis_runs "
        "WHERE kind = 'reliability_audit')"
    )
    op.execute("DELETE FROM analysis_runs WHERE kind = 'reliability_audit'")
    op.execute("DROP TRIGGER IF EXISTS analysis_runs_training_run_id_parent ON analysis_runs")
    op.drop_index("ix_analysis_runs_training_run", "analysis_runs")
    op.drop_constraint("ck_analysis_runs_audit_run", "analysis_runs", type_="check")
    op.drop_constraint("ck_analysis_runs_kind", "analysis_runs", type_="check")
    op.create_check_constraint(
        "ck_analysis_runs_kind", "analysis_runs", _in("kind", ("diagnostics",))
    )
    op.drop_constraint(
        "fk_analysis_runs_training_run_id_training_runs", "analysis_runs", type_="foreignkey"
    )
    op.drop_column("analysis_runs", "training_run_id")
    for table in reversed(NEW_TABLES):
        op.drop_table(table)
    for fn in ("training_runs_split_version", "training_runs_complete_once"):
        op.execute(f"DROP FUNCTION IF EXISTS {fn}()")
    op.execute("DELETE FROM jobs WHERE type IN ('train', 'reliability_audit')")
    op.execute("DELETE FROM artifacts WHERE kind = 'oof_predictions'")
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", JOB_TYPES_OLD))
    op.drop_constraint("ck_artifacts_kind", "artifacts", type_="check")
    op.create_check_constraint("ck_artifacts_kind", "artifacts", _in("kind", KINDS_OLD))
    op.drop_constraint("ck_artifacts_owner_type", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifacts_owner_type", "artifacts", _in("owner_type", OWNER_TYPES_OLD)
    )

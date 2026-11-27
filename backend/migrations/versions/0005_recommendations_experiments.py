"""Phase 4 Part D: recommendations and the Experiment Lab.

- recommendation_sets (insert-only), recommendations (only status/updated_at change),
  recommendation_findings (insert-only): deterministic recommendations and their evidence.
- split_definitions (insert-only): frozen fold assignments, stored as fold_assignment
  artifacts and reused by every experiment with the same configuration.
- experiments: configuration written by the API; split, summary and completion written once
  by the worker. experiment_candidates and experiment_results are insert-only.
- jobs.type gains 'experiment'; artifacts gain kind 'fold_assignment' and owner type
  'split_definition'.
- RLS, owner consistency, version-in-dataset and parent-consistency triggers on every table.

Experiments never write dataset versions; no table here references a version as output.

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-24
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.schema import SchemaItem

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None

UUID = pg.UUID(as_uuid=True)
TS = sa.DateTime(timezone=True)
NOW = sa.text("now()")
OWNER_EXPR = "NULLIF(current_setting('app.user_id', true), '')::uuid"
NEW_TABLES = (
    "recommendation_sets",
    "recommendations",
    "recommendation_findings",
    "split_definitions",
    "experiments",
    "experiment_candidates",
    "experiment_results",
)
JOB_TYPES_OLD = ("ingest", "profile", "analysis", "purge")
JOB_TYPES_NEW = ("ingest", "profile", "analysis", "experiment", "purge")
SPLIT_METHODS = ("kfold", "stratified", "group", "stratified_group", "time")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def _base(table: str) -> list[SchemaItem]:
    """id, owner_id, dataset_id with their keys (every Part D table except the link table)."""
    return [
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.PrimaryKeyConstraint("id", name=f"pk_{table}"),
        *_owner_dataset_fks(table),
    ]


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


def _fk(
    table: str, column: str, target: str, ondelete: str | None = None
) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        [column], [f"{target}.id"], ondelete=ondelete, name=f"fk_{table}_{column}_{target}"
    )


def _version_col(table: str) -> list[SchemaItem]:
    return [
        sa.Column("dataset_version_id", UUID, nullable=False),
        _fk(table, "dataset_version_id", "dataset_versions", "CASCADE"),
    ]


def upgrade() -> None:
    t = "recommendation_sets"
    op.create_table(
        t,
        *_base(t),
        *_version_col(t),
        sa.Column("analysis_run_id", UUID, nullable=False),
        sa.Column("profile_id", UUID),
        sa.Column("semantics_hash", sa.LargeBinary, nullable=False),
        sa.Column("input_hash", sa.LargeBinary, nullable=False),
        sa.Column("rules_version", sa.Text, nullable=False),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("domain", sa.Text, nullable=False),
        sa.Column("task_type", sa.Text),
        sa.Column("content", pg.JSONB, nullable=False),
        sa.Column("created_by", UUID),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        _fk(t, "analysis_run_id", "analysis_runs", "CASCADE"),
        _fk(t, "profile_id", "version_profiles", "CASCADE"),
        sa.UniqueConstraint(
            "dataset_version_id", "input_hash", name=f"uq_{t}_dataset_version_id_input_hash"
        ),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_version_created", t, ["dataset_version_id", "created_at"])

    t = "recommendations"
    op.create_table(
        t,
        *_base(t),
        sa.Column("set_id", UUID, nullable=False),
        sa.Column("rec_key", sa.Text, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("rule_id", sa.Text, nullable=False),
        sa.Column("rule_version", sa.Text, nullable=False),
        sa.Column("op", sa.Text, nullable=False),
        sa.Column("op_version", sa.Text, nullable=False),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("risk", sa.Text, nullable=False),
        sa.Column("destructive", sa.Boolean, nullable=False),
        sa.Column("columns", pg.JSONB, nullable=False),
        sa.Column("params", pg.JSONB, nullable=False),
        sa.Column("payload", pg.JSONB, nullable=False),
        sa.Column("status", sa.Text, nullable=False, server_default="open"),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("updated_at", TS, nullable=False, server_default=NOW),
        _fk(t, "set_id", "recommendation_sets", "CASCADE"),
        sa.UniqueConstraint("set_id", "rec_key", name=f"uq_{t}_set_id_rec_key"),
        sa.CheckConstraint(_in("status", ("open", "used", "dismissed")), name=f"ck_{t}_status"),
        sa.CheckConstraint(_in("risk", ("low", "medium", "high")), name=f"ck_{t}_risk"),
        sa.CheckConstraint(_in("stage", ("data", "model", "either")), name=f"ck_{t}_stage"),
        sa.CheckConstraint("NOT destructive OR risk = 'high'", name=f"ck_{t}_destructive_high"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])

    t = "recommendation_findings"
    op.create_table(
        t,
        sa.Column("recommendation_id", UUID, nullable=False),
        sa.Column("finding_id", UUID, nullable=False),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.PrimaryKeyConstraint("recommendation_id", "finding_id", name=f"pk_{t}"),
        *_owner_dataset_fks(t),
        _fk(t, "recommendation_id", "recommendations", "CASCADE"),
        _fk(t, "finding_id", "findings", "CASCADE"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])

    t = "split_definitions"
    op.create_table(
        t,
        *_base(t),
        *_version_col(t),
        sa.Column("config_hash", sa.LargeBinary, nullable=False),
        sa.Column("method", sa.Text, nullable=False),
        sa.Column("n_splits", sa.Integer, nullable=False),
        sa.Column("seed", sa.Integer, nullable=False),
        sa.Column("target_column", sa.Text),
        sa.Column("group_column", sa.Text),
        sa.Column("time_column", sa.Text),
        sa.Column("duplicate_columns", pg.JSONB, nullable=False),
        sa.Column("fold_assignment_artifact_id", UUID, nullable=False),
        sa.Column("assignment_sha256", sa.LargeBinary, nullable=False),
        sa.Column("cohort", pg.JSONB, nullable=False),
        sa.Column("fold_stats", pg.JSONB, nullable=False),
        sa.Column("warnings", pg.JSONB, nullable=False),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("library_versions", pg.JSONB, nullable=False),
        sa.Column("created_by", UUID),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        _fk(t, "fold_assignment_artifact_id", "artifacts"),
        sa.UniqueConstraint(
            "dataset_version_id", "config_hash", name=f"uq_{t}_dataset_version_id_config_hash"
        ),
        sa.CheckConstraint(_in("method", SPLIT_METHODS), name=f"ck_{t}_method"),
        sa.CheckConstraint("n_splits BETWEEN 2 AND 20", name=f"ck_{t}_n_splits"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])

    t = "experiments"
    op.create_table(
        t,
        *_base(t),
        *_version_col(t),
        sa.Column("job_id", UUID, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("task_type", sa.Text, nullable=False),
        sa.Column("target_column", sa.Text),
        sa.Column("models", pg.JSONB, nullable=False),
        sa.Column("seed", sa.Integer, nullable=False),
        sa.Column("config", pg.JSONB, nullable=False),
        sa.Column("config_hash", sa.LargeBinary, nullable=False),
        sa.Column("source_content_hash", sa.LargeBinary, nullable=False),
        sa.Column("recommendation_set_id", UUID),
        sa.Column("rerun_of_id", UUID),
        sa.Column("split_definition_id", UUID),
        sa.Column("summary", pg.JSONB),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("library_versions", pg.JSONB),
        sa.Column("created_by", UUID),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("completed_at", TS),
        _fk(t, "job_id", "jobs"),
        _fk(t, "recommendation_set_id", "recommendation_sets", "SET NULL"),
        _fk(t, "rerun_of_id", "experiments", "SET NULL"),
        _fk(t, "split_definition_id", "split_definitions"),
        sa.UniqueConstraint("job_id", name=f"uq_{t}_job_id"),
        sa.CheckConstraint(
            _in("task_type", ("binary", "multiclass", "regression", "clustering")),
            name=f"ck_{t}_task_type",
        ),
        sa.CheckConstraint("char_length(name) BETWEEN 1 AND 200", name=f"ck_{t}_name_length"),
        sa.CheckConstraint("(completed_at IS NULL) = (summary IS NULL)", name=f"ck_{t}_completed"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_version_created", t, ["dataset_version_id", "created_at"])

    t = "experiment_candidates"
    op.create_table(
        t,
        *_base(t),
        sa.Column("experiment_id", UUID, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("key", sa.Text, nullable=False),
        sa.Column("label", sa.Text, nullable=False),
        sa.Column("is_baseline", sa.Boolean, nullable=False),
        sa.Column("steps", pg.JSONB, nullable=False),
        sa.Column("spec_hash", sa.Text, nullable=False),
        sa.Column("max_risk", sa.Text, nullable=False),
        sa.Column("has_destructive", sa.Boolean, nullable=False),
        sa.Column("stages", pg.JSONB, nullable=False),
        sa.Column("recommendation_ids", pg.JSONB, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        _fk(t, "experiment_id", "experiments", "CASCADE"),
        sa.UniqueConstraint("experiment_id", "position", name=f"uq_{t}_experiment_id_position"),
        sa.UniqueConstraint("experiment_id", "key", name=f"uq_{t}_experiment_id_key"),
        sa.CheckConstraint(_in("max_risk", ("low", "medium", "high")), name=f"ck_{t}_max_risk"),
        sa.CheckConstraint(
            "NOT is_baseline OR jsonb_array_length(steps) = 0", name=f"ck_{t}_baseline"
        ),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])

    t = "experiment_results"
    op.create_table(
        t,
        *_base(t),
        sa.Column("experiment_id", UUID, nullable=False),
        sa.Column("candidate_id", UUID, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("comparable", sa.Boolean, nullable=False),
        sa.Column("comparability_reason", sa.Text),
        sa.Column("metrics", pg.JSONB, nullable=False),
        sa.Column("fold_results", pg.JSONB, nullable=False),
        sa.Column("data_quality", pg.JSONB),
        sa.Column("selection_stability", pg.JSONB, nullable=False),
        sa.Column("details", pg.JSONB, nullable=False),
        sa.Column("failures", pg.JSONB, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        _fk(t, "experiment_id", "experiments", "CASCADE"),
        _fk(t, "candidate_id", "experiment_candidates", "CASCADE"),
        sa.UniqueConstraint("candidate_id", name=f"uq_{t}_candidate_id"),
        sa.CheckConstraint(
            _in("status", ("completed", "partial", "failed")), name=f"ck_{t}_status"
        ),
        sa.CheckConstraint(
            "comparable OR comparability_reason IS NOT NULL", name=f"ck_{t}_comparable"
        ),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_experiment", t, ["experiment_id"])

    # ---- Widened enumerations -----------------------------------------------------------
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", JOB_TYPES_NEW))
    op.drop_constraint("ck_artifacts_kind", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifacts_kind", "artifacts", _in("kind", ("preview", "fold_assignment"))
    )
    op.drop_constraint("ck_artifacts_owner_type", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifacts_owner_type",
        "artifacts",
        _in("owner_type", ("dataset_version", "split_definition")),
    )

    # ---- RLS and owner consistency ------------------------------------------------------
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
    for table in ("recommendation_sets", "split_definitions", "experiments"):
        op.execute(
            f"CREATE TRIGGER {table}_version BEFORE INSERT ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION enforce_version_in_dataset()"
        )

    # A referenced parent row must belong to the same dataset and owner (TG_ARGV: parent
    # table, referencing column). Parents are read under RLS, so another user's row is
    # invisible and fails the check too.
    op.execute(
        """
        CREATE FUNCTION enforce_same_dataset_parent() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE parent uuid; ok boolean;
        BEGIN
          EXECUTE format('SELECT ($1).%I', TG_ARGV[1]) INTO parent USING NEW;
          IF parent IS NULL THEN RETURN NEW; END IF;
          EXECUTE format(
            'SELECT EXISTS (SELECT 1 FROM %I p WHERE p.id = $1 AND p.dataset_id = $2 '
            'AND p.owner_id = $3)', TG_ARGV[0])
            INTO ok USING parent, NEW.dataset_id, NEW.owner_id;
          IF NOT ok THEN
            RAISE EXCEPTION 'referenced % does not belong to this dataset and owner', TG_ARGV[0]
              USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    parents = [
        ("recommendation_sets", "analysis_runs", "analysis_run_id"),
        ("recommendation_sets", "version_profiles", "profile_id"),
        ("recommendations", "recommendation_sets", "set_id"),
        ("recommendation_findings", "recommendations", "recommendation_id"),
        ("recommendation_findings", "findings", "finding_id"),
        ("split_definitions", "artifacts", "fold_assignment_artifact_id"),
        ("experiments", "recommendation_sets", "recommendation_set_id"),
        ("experiments", "experiments", "rerun_of_id"),
        ("experiments", "split_definitions", "split_definition_id"),
        ("experiment_candidates", "experiments", "experiment_id"),
        ("experiment_results", "experiments", "experiment_id"),
        ("experiment_results", "experiment_candidates", "candidate_id"),
    ]
    for table, parent, column in parents:
        op.execute(
            f"CREATE TRIGGER {table}_{column}_parent BEFORE INSERT OR UPDATE OF {column} "
            f"ON {table} FOR EACH ROW EXECUTE FUNCTION "
            f"enforce_same_dataset_parent('{parent}', '{column}')"
        )

    # Same-version references: an experiment's split and recommendations, and a set's
    # analysis run, describe the experiment's (set's) own version.
    op.execute(
        """
        CREATE FUNCTION enforce_same_version_refs() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF TG_TABLE_NAME = 'experiments' THEN
            IF NEW.split_definition_id IS NOT NULL AND NOT EXISTS (
                 SELECT 1 FROM split_definitions s WHERE s.id = NEW.split_definition_id
                    AND s.dataset_version_id = NEW.dataset_version_id) THEN
              RAISE EXCEPTION 'split belongs to another version' USING ERRCODE = '42501';
            END IF;
            IF NEW.recommendation_set_id IS NOT NULL AND NOT EXISTS (
                 SELECT 1 FROM recommendation_sets r WHERE r.id = NEW.recommendation_set_id
                    AND r.dataset_version_id = NEW.dataset_version_id) THEN
              RAISE EXCEPTION 'recommendations belong to another version'
                USING ERRCODE = '42501';
            END IF;
          ELSIF TG_TABLE_NAME = 'recommendation_sets' THEN
            IF NOT EXISTS (SELECT 1 FROM analysis_runs a WHERE a.id = NEW.analysis_run_id
                              AND a.dataset_version_id = NEW.dataset_version_id) THEN
              RAISE EXCEPTION 'analysis run belongs to another version' USING ERRCODE = '42501';
            END IF;
          ELSIF TG_TABLE_NAME = 'experiment_results' THEN
            IF NOT EXISTS (SELECT 1 FROM experiment_candidates c WHERE c.id = NEW.candidate_id
                              AND c.experiment_id = NEW.experiment_id) THEN
              RAISE EXCEPTION 'candidate belongs to another experiment' USING ERRCODE = '42501';
            END IF;
          END IF;
          RETURN NEW;
        END $$
        """
    )
    for table in ("experiments", "recommendation_sets", "experiment_results"):
        op.execute(
            f"CREATE TRIGGER {table}_same_version BEFORE INSERT OR UPDATE ON {table} "
            "FOR EACH ROW EXECUTE FUNCTION enforce_same_version_refs()"
        )

    # An experiment's outcome is written exactly once.
    op.execute(
        """
        CREATE FUNCTION experiments_complete_once() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF OLD.completed_at IS NOT NULL THEN
            RAISE EXCEPTION 'a completed experiment cannot change' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER experiments_complete_once BEFORE UPDATE ON experiments "
        "FOR EACH ROW EXECUTE FUNCTION experiments_complete_once()"
    )

    # ---- Grants (DELETE: dataset purge) ---------------------------------------------------
    for table in NEW_TABLES:
        op.execute(f"GRANT SELECT, INSERT, DELETE ON {table} TO dm_app")
    op.execute("GRANT UPDATE (status, updated_at) ON recommendations TO dm_app")
    op.execute(
        "GRANT UPDATE (split_definition_id, summary, library_versions, completed_at) "
        "ON experiments TO dm_app"
    )


def downgrade() -> None:
    for table in reversed(NEW_TABLES):
        op.drop_table(table)
    for fn in (
        "enforce_same_dataset_parent",
        "enforce_same_version_refs",
        "experiments_complete_once",
    ):
        op.execute(f"DROP FUNCTION IF EXISTS {fn}()")
    op.execute("DELETE FROM jobs WHERE type = 'experiment'")
    op.execute("DELETE FROM artifacts WHERE kind = 'fold_assignment'")
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", JOB_TYPES_OLD))
    op.drop_constraint("ck_artifacts_kind", "artifacts", type_="check")
    op.create_check_constraint("ck_artifacts_kind", "artifacts", _in("kind", ("preview",)))
    op.drop_constraint("ck_artifacts_owner_type", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifacts_owner_type", "artifacts", _in("owner_type", ("dataset_version",))
    )

"""Phase 4 Part E: pipelines, approvals, commits, reverts and version comparison.

- pipelines, pipeline_steps (editable by the owner; UNIQUE(pipeline_id, position) is
  DEFERRABLE INITIALLY DEFERRED so steps can be reordered in one transaction);
- approvals, transformation_runs, version_events: insert-only history;
- dataset_versions: FK transformation_run_id -> transformation_runs (deferrable: a version
  and its run are inserted in one transaction) and nullable model_pipeline_spec (D1);
- jobs.type gains impact_preview, commit, revert, compare_versions, with a partial unique
  index allowing one queued or running commit/revert per dataset;
- artifacts gain kinds impact_preview, version_comparison and owner type pipeline;
- RLS, owner, parent-consistency (0005's enforce_same_dataset_parent) and version-in-dataset
  triggers on every new table; dm_maintenance may read object keys for orphan reconciliation.

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-25
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg
from sqlalchemy.schema import SchemaItem

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None

UUID = pg.UUID(as_uuid=True)
TS = sa.DateTime(timezone=True)
NOW = sa.text("now()")
OWNER_EXPR = "NULLIF(current_setting('app.user_id', true), '')::uuid"
NEW_TABLES = ("pipelines", "pipeline_steps", "approvals", "transformation_runs", "version_events")
JOB_TYPES_OLD = ("ingest", "profile", "analysis", "experiment", "purge")
JOB_TYPES_NEW = (
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
KINDS_OLD = ("preview", "fold_assignment")
KINDS_NEW = ("preview", "fold_assignment", "impact_preview", "version_comparison")
OWNER_TYPES_OLD = ("dataset_version", "split_definition")
OWNER_TYPES_NEW = ("dataset_version", "split_definition", "pipeline")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def _fk(
    table: str, column: str, target: str, ondelete: str | None = None, **kw: object
) -> sa.ForeignKeyConstraint:
    return sa.ForeignKeyConstraint(
        [column], [f"{target}.id"], ondelete=ondelete, name=f"fk_{table}_{column}_{target}", **kw
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
    t = "pipelines"
    op.create_table(
        t,
        *_base(t),
        sa.Column("source_version_id", UUID, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("state", sa.Text, nullable=False, server_default="draft"),
        sa.Column("spec_hash", sa.Text, nullable=False),
        sa.Column("validation_report", pg.JSONB),
        sa.Column("impact_preview_artifact_id", UUID),
        sa.Column("impact_preview_spec_hash", sa.Text),
        sa.Column("impact_preview_job_id", UUID),
        sa.Column("from_experiment_candidate_id", UUID),
        sa.Column("committed_version_id", UUID),
        sa.Column("created_by", UUID),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("updated_at", TS, nullable=False, server_default=NOW),
        _fk(t, "source_version_id", "dataset_versions", "CASCADE"),
        _fk(t, "impact_preview_artifact_id", "artifacts"),
        _fk(t, "impact_preview_job_id", "jobs"),
        _fk(t, "from_experiment_candidate_id", "experiment_candidates", "SET NULL"),
        _fk(t, "committed_version_id", "dataset_versions"),
        sa.CheckConstraint(
            _in("state", ("draft", "validated", "committed", "archived")), name=f"ck_{t}_state"
        ),
        sa.CheckConstraint("char_length(name) BETWEEN 1 AND 200", name=f"ck_{t}_name_length"),
        sa.CheckConstraint(
            "(state = 'committed') = (committed_version_id IS NOT NULL)", name=f"ck_{t}_committed"
        ),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_dataset_created", t, ["dataset_id", "created_at"])

    t = "pipeline_steps"
    op.create_table(
        t,
        *_base(t),
        sa.Column("pipeline_id", UUID, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("op", sa.Text, nullable=False),
        sa.Column("op_version", sa.Text, nullable=False),
        sa.Column("params", pg.JSONB, nullable=False),
        sa.Column("columns", pg.JSONB, nullable=False),
        sa.Column("stage", sa.Text, nullable=False),
        sa.Column("risk", sa.Text, nullable=False),
        sa.Column("destructive", sa.Boolean, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("step_hash", sa.Text),
        sa.Column("from_recommendation_id", UUID),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("updated_at", TS, nullable=False, server_default=NOW),
        _fk(t, "pipeline_id", "pipelines", "CASCADE"),
        _fk(t, "from_recommendation_id", "recommendations", "SET NULL"),
        sa.UniqueConstraint(
            "pipeline_id",
            "position",
            name=f"uq_{t}_pipeline_id_position",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.CheckConstraint(_in("stage", ("data", "model")), name=f"ck_{t}_stage"),
        sa.CheckConstraint(_in("risk", ("low", "medium", "high")), name=f"ck_{t}_risk"),
        sa.CheckConstraint("NOT destructive OR risk = 'high'", name=f"ck_{t}_destructive_high"),
        sa.CheckConstraint("position >= 0", name=f"ck_{t}_position"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])

    t = "approvals"
    op.create_table(
        t,
        *_base(t),
        sa.Column("pipeline_id", UUID, nullable=False),
        sa.Column("pipeline_step_id", UUID),
        sa.Column("scope", sa.Text, nullable=False),
        sa.Column("approved_hash", sa.Text, nullable=False),
        sa.Column("spec_hash", sa.Text, nullable=False),
        sa.Column("source_version_id", UUID, nullable=False),
        sa.Column("preview_artifact_id", UUID, nullable=False),
        sa.Column("impact_shown", pg.JSONB, nullable=False),
        sa.Column("statement", sa.Text, nullable=False),
        sa.Column("acknowledge_destructive", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("approved_by", UUID, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        _fk(t, "pipeline_id", "pipelines", "CASCADE"),
        _fk(t, "source_version_id", "dataset_versions", "CASCADE"),
        _fk(t, "preview_artifact_id", "artifacts"),
        sa.CheckConstraint(_in("scope", ("step", "pipeline")), name=f"ck_{t}_scope"),
        sa.CheckConstraint(
            "(scope = 'step') = (pipeline_step_id IS NOT NULL)", name=f"ck_{t}_step"
        ),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_pipeline", t, ["pipeline_id", "created_at"])

    t = "transformation_runs"
    op.create_table(
        t,
        *_base(t),
        sa.Column("pipeline_id", UUID, nullable=False),
        sa.Column("job_id", UUID, nullable=False),
        sa.Column("source_version_id", UUID, nullable=False),
        sa.Column("result_version_id", UUID, nullable=False),
        sa.Column("spec_snapshot", pg.JSONB, nullable=False),
        sa.Column("spec_hash", sa.Text, nullable=False),
        sa.Column("approval_ids", pg.JSONB, nullable=False),
        sa.Column("step_reports", pg.JSONB, nullable=False),
        sa.Column("model_pipeline_spec", pg.JSONB, nullable=False),
        sa.Column("roles_report", pg.JSONB, nullable=False),
        sa.Column("seeds", pg.JSONB, nullable=False),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("library_versions", pg.JSONB, nullable=False),
        sa.Column("executed_by", UUID, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        _fk(t, "pipeline_id", "pipelines"),
        _fk(t, "job_id", "jobs"),
        _fk(t, "source_version_id", "dataset_versions"),
        _fk(t, "result_version_id", "dataset_versions", deferrable=True, initially="DEFERRED"),
        sa.UniqueConstraint("job_id", name=f"uq_{t}_job_id"),
        sa.UniqueConstraint("result_version_id", name=f"uq_{t}_result_version_id"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])

    t = "version_events"
    op.create_table(
        t,
        *_base(t),
        sa.Column("event", sa.Text, nullable=False),
        sa.Column("version_id", UUID, nullable=False),
        sa.Column("related_version_id", UUID),
        sa.Column("actor_id", UUID, nullable=False),
        sa.Column("details", pg.JSONB, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        _fk(t, "version_id", "dataset_versions", "CASCADE"),
        _fk(t, "related_version_id", "dataset_versions", "CASCADE"),
        sa.CheckConstraint(_in("event", ("created", "reverted")), name=f"ck_{t}_event"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_dataset_created", t, ["dataset_id", "created_at"])

    # ---- dataset_versions -----------------------------------------------------------------
    # The immutability trigger compares whole rows except deleted_at: adding a column with no
    # default changes nothing for existing rows.
    op.add_column("dataset_versions", sa.Column("model_pipeline_spec", pg.JSONB))
    op.create_foreign_key(
        "fk_dataset_versions_transformation_run_id_transformation_runs",
        "dataset_versions",
        "transformation_runs",
        ["transformation_run_id"],
        ["id"],
        deferrable=True,
        initially="DEFERRED",
    )

    # ---- Enumerations and the one-mutation-per-dataset rule -------------------------------
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", JOB_TYPES_NEW))
    op.create_index(
        "uq_jobs_one_mutation_per_dataset",
        "jobs",
        ["dataset_id"],
        unique=True,
        postgresql_where=sa.text(
            "type IN ('commit', 'revert') AND status IN ('queued', 'running', 'cancel_requested')"
        ),
    )
    op.drop_constraint("ck_artifacts_kind", "artifacts", type_="check")
    op.create_check_constraint("ck_artifacts_kind", "artifacts", _in("kind", KINDS_NEW))
    op.drop_constraint("ck_artifacts_owner_type", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifacts_owner_type", "artifacts", _in("owner_type", OWNER_TYPES_NEW)
    )

    # ---- RLS, owner and parent consistency ------------------------------------------------
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
    parents = [
        ("pipelines", "dataset_versions", "source_version_id"),
        ("pipelines", "artifacts", "impact_preview_artifact_id"),
        ("pipelines", "experiment_candidates", "from_experiment_candidate_id"),
        ("pipelines", "dataset_versions", "committed_version_id"),
        ("pipeline_steps", "pipelines", "pipeline_id"),
        ("pipeline_steps", "recommendations", "from_recommendation_id"),
        ("approvals", "pipelines", "pipeline_id"),
        ("approvals", "dataset_versions", "source_version_id"),
        ("approvals", "artifacts", "preview_artifact_id"),
        ("transformation_runs", "pipelines", "pipeline_id"),
        ("transformation_runs", "dataset_versions", "source_version_id"),
        ("version_events", "dataset_versions", "version_id"),
        ("version_events", "dataset_versions", "related_version_id"),
    ]
    for table, parent, column in parents:
        op.execute(
            f"CREATE TRIGGER {table}_{column}_parent BEFORE INSERT OR UPDATE OF {column} "
            f"ON {table} FOR EACH ROW EXECUTE FUNCTION "
            f"enforce_same_dataset_parent('{parent}', '{column}')"
        )
    # The result version is inserted after its run (deferred FK), so its dataset is checked
    # at commit time by a deferred constraint trigger.
    op.execute(
        """
        CREATE FUNCTION enforce_run_result_version() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NOT EXISTS (
               SELECT 1 FROM dataset_versions v
                WHERE v.id = NEW.result_version_id AND v.dataset_id = NEW.dataset_id
                  AND v.owner_id = NEW.owner_id AND v.transformation_run_id = NEW.id
                  AND v.parent_version_id = NEW.source_version_id) THEN
            RAISE EXCEPTION 'result version does not belong to this run, dataset and owner'
              USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE CONSTRAINT TRIGGER transformation_runs_result_version AFTER INSERT "
        "ON transformation_runs DEFERRABLE INITIALLY DEFERRED FOR EACH ROW "
        "EXECUTE FUNCTION enforce_run_result_version()"
    )
    # A step's pipeline cannot change once committed or archived.
    op.execute(
        """
        CREATE FUNCTION pipeline_steps_editable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE pid uuid;
        BEGIN
          IF TG_OP = 'DELETE' AND EXISTS (SELECT 1 FROM datasets d WHERE d.id = OLD.dataset_id
                                            AND d.deleted_at IS NOT NULL) THEN
            RETURN OLD;  -- dataset purge
          END IF;
          pid := CASE WHEN TG_OP = 'DELETE' THEN OLD.pipeline_id ELSE NEW.pipeline_id END;
          IF EXISTS (SELECT 1 FROM pipelines p WHERE p.id = pid
                       AND p.state IN ('committed', 'archived')) THEN
            RAISE EXCEPTION 'a committed or archived pipeline cannot change'
              USING ERRCODE = '42501';
          END IF;
          RETURN CASE WHEN TG_OP = 'DELETE' THEN OLD ELSE NEW END;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER pipeline_steps_editable BEFORE INSERT OR UPDATE OR DELETE "
        "ON pipeline_steps FOR EACH ROW EXECUTE FUNCTION pipeline_steps_editable()"
    )

    # ---- Grants ---------------------------------------------------------------------------
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON pipelines, pipeline_steps TO dm_app")
    for table in ("approvals", "transformation_runs", "version_events"):
        op.execute(f"GRANT SELECT, INSERT, DELETE ON {table} TO dm_app")  # DELETE: purge only
    # Orphan-object reconciliation (maintenance, BYPASSRLS) reads which objects are referenced.
    op.execute("GRANT SELECT (id, dataset_id, object_key) ON dataset_versions TO dm_maintenance")
    op.execute("GRANT SELECT (id, dataset_id, object_key) ON artifacts TO dm_maintenance")


def downgrade() -> None:
    op.execute("REVOKE SELECT (id, dataset_id, object_key) ON dataset_versions FROM dm_maintenance")
    op.execute("REVOKE SELECT (id, dataset_id, object_key) ON artifacts FROM dm_maintenance")
    op.drop_constraint(
        "fk_dataset_versions_transformation_run_id_transformation_runs",
        "dataset_versions",
        type_="foreignkey",
    )
    op.drop_column("dataset_versions", "model_pipeline_spec")
    for table in reversed(NEW_TABLES):
        op.drop_table(table)
    for fn in ("enforce_run_result_version", "pipeline_steps_editable"):
        op.execute(f"DROP FUNCTION IF EXISTS {fn}()")
    op.drop_index("uq_jobs_one_mutation_per_dataset", "jobs")
    op.execute(
        "DELETE FROM jobs WHERE type IN ('impact_preview', 'commit', 'revert', 'compare_versions')"
    )
    op.execute("DELETE FROM artifacts WHERE kind IN ('impact_preview', 'version_comparison')")
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", JOB_TYPES_OLD))
    op.drop_constraint("ck_artifacts_kind", "artifacts", type_="check")
    op.create_check_constraint("ck_artifacts_kind", "artifacts", _in("kind", KINDS_OLD))
    op.drop_constraint("ck_artifacts_owner_type", "artifacts", type_="check")
    op.create_check_constraint(
        "ck_artifacts_owner_type", "artifacts", _in("owner_type", OWNER_TYPES_OLD)
    )

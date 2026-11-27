"""Phase 8: Analytics Studio — saved dashboards and stored automated insights.

- dashboards: owner-scoped, bound to one immutable dataset version (never switched silently);
- dashboard_widgets: validated widget specifications (charts, KPI cards, pivot tables), no data;
- analysis_runs gain kind 'insights' and jobs.type gains 'insights' (stored, evidence-based
  automated insights computed by the worker).
- RLS, owner, version-in-dataset and parent-consistency triggers on both new tables.

Expand-only: existing rows are untouched.

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-27
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None

UUID = pg.UUID(as_uuid=True)
TS = sa.DateTime(timezone=True)
NOW = sa.text("now()")
OWNER_EXPR = "NULLIF(current_setting('app.user_id', true), '')::uuid"
NEW_TABLES = ("dashboards", "dashboard_widgets")
JOB_TYPES_OLD = (
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
    "purge",
)
JOB_TYPES_NEW = (*JOB_TYPES_OLD[:-1], "insights", "purge")
ANALYSIS_OLD = ("diagnostics", "reliability_audit", "biomedical")
ANALYSIS_NEW = (*ANALYSIS_OLD, "insights")
WIDGET_KINDS = ("chart", "kpi", "pivot")
WIDGET_SIZES = ("small", "medium", "large")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def _swap(name: str, table: str, column: str, values: tuple[str, ...]) -> None:
    op.drop_constraint(name, table, type_="check")
    op.create_check_constraint(name, table, _in(column, values))


def upgrade() -> None:
    t = "dashboards"
    op.create_table(
        t,
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=False, server_default=""),
        sa.Column("filters", pg.JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_dashboard_id", UUID),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("updated_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name=f"pk_{t}"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name=f"fk_{t}_owner_id_users"),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.id"], ondelete="CASCADE", name=f"fk_{t}_dataset_id_datasets"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            ondelete="CASCADE",
            name=f"fk_{t}_dataset_version_id_dataset_versions",
        ),
        sa.ForeignKeyConstraint(
            ["source_dashboard_id"],
            ["dashboards.id"],
            ondelete="SET NULL",
            name=f"fk_{t}_source_dashboard_id_dashboards",
        ),
        sa.CheckConstraint("char_length(name) BETWEEN 1 AND 200", name=f"ck_{t}_name_length"),
        sa.CheckConstraint("char_length(description) <= 2000", name=f"ck_{t}_description_length"),
        sa.CheckConstraint("jsonb_typeof(filters) = 'array'", name=f"ck_{t}_filters_array"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_dataset_created", t, ["dataset_id", "created_at"])

    t = "dashboard_widgets"
    op.create_table(
        t,
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("dashboard_id", UUID, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("title", sa.Text, nullable=False),
        sa.Column("size", sa.Text, nullable=False, server_default="medium"),
        sa.Column("spec", pg.JSONB, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.Column("updated_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name=f"pk_{t}"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name=f"fk_{t}_owner_id_users"),
        sa.ForeignKeyConstraint(
            ["dataset_id"], ["datasets.id"], ondelete="CASCADE", name=f"fk_{t}_dataset_id_datasets"
        ),
        sa.ForeignKeyConstraint(
            ["dashboard_id"],
            ["dashboards.id"],
            ondelete="CASCADE",
            name=f"fk_{t}_dashboard_id_dashboards",
        ),
        sa.CheckConstraint(_in("kind", WIDGET_KINDS), name=f"ck_{t}_kind"),
        sa.CheckConstraint(_in("size", WIDGET_SIZES), name=f"ck_{t}_size"),
        sa.CheckConstraint("char_length(title) BETWEEN 1 AND 200", name=f"ck_{t}_title_length"),
        sa.CheckConstraint("position >= 0", name=f"ck_{t}_position"),
        sa.CheckConstraint("jsonb_typeof(spec) = 'object'", name=f"ck_{t}_spec_object"),
    )
    op.create_index(f"ix_{t}_owner_id", t, ["owner_id"])
    op.create_index(f"ix_{t}_dashboard_position", t, ["dashboard_id", "position"])

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
    # A dashboard's version belongs to its dataset (also on an explicit version switch).
    op.execute(
        "CREATE TRIGGER dashboards_version BEFORE INSERT OR UPDATE OF dataset_version_id "
        "ON dashboards FOR EACH ROW EXECUTE FUNCTION enforce_version_in_dataset()"
    )
    for table, parent, column in (
        ("dashboards", "dashboards", "source_dashboard_id"),
        ("dashboard_widgets", "dashboards", "dashboard_id"),
    ):
        op.execute(
            f"CREATE TRIGGER {table}_{column}_parent BEFORE INSERT OR UPDATE OF {column} "
            f"ON {table} FOR EACH ROW EXECUTE FUNCTION "
            f"enforce_same_dataset_parent('{parent}', '{column}')"
        )
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE ON dashboards, dashboard_widgets TO dm_app")

    _swap("ck_jobs_type", "jobs", "type", JOB_TYPES_NEW)
    _swap("ck_analysis_runs_kind", "analysis_runs", "kind", ANALYSIS_NEW)


def downgrade() -> None:
    op.execute("DELETE FROM analysis_runs WHERE kind = 'insights'")
    op.execute("DELETE FROM jobs WHERE type = 'insights'")
    _swap("ck_analysis_runs_kind", "analysis_runs", "kind", ANALYSIS_OLD)
    _swap("ck_jobs_type", "jobs", "type", JOB_TYPES_OLD)
    op.drop_table("dashboard_widgets")
    op.drop_table("dashboards")

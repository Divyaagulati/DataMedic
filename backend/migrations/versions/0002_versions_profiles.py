"""Phase 2: dataset versions (v0 from parsing), artifacts (preview) and version profiles.

- dataset_versions: immutable (insert-only except deleted_at), RLS, owner consistency.
- datasets.head_version_id: deferrable FK; a trigger keeps it inside the same dataset.
- artifacts, version_profiles: insert-only, RLS, owner and version consistency.
- jobs.type gains 'profile'.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-09
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql as pg

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

UUID = pg.UUID(as_uuid=True)
TS = sa.DateTime(timezone=True)
NOW = sa.text("now()")
OWNER_EXPR = "NULLIF(current_setting('app.user_id', true), '')::uuid"
NEW_TABLES = ("dataset_versions", "artifacts", "version_profiles")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("version_number", sa.Integer, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("parent_version_id", UUID),
        sa.Column("restored_from_version_id", UUID),
        sa.Column("transformation_run_id", UUID),  # FK added with transformation_runs
        sa.Column("parse_options", pg.JSONB),
        sa.Column("parse_report", pg.JSONB),
        sa.Column("object_key", sa.Text, nullable=False),
        sa.Column("file_sha256", sa.LargeBinary, nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("content_hash", sa.LargeBinary, nullable=False),
        sa.Column("n_rows", sa.BigInteger, nullable=False),
        sa.Column("n_cols", sa.Integer, nullable=False),
        sa.Column("schema", pg.JSONB, nullable=False),
        sa.Column("diff_summary", pg.JSONB),
        sa.Column("fitted_on_full_data", sa.Boolean, nullable=False, server_default="false"),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("library_versions", pg.JSONB, nullable=False),
        sa.Column("created_by", UUID),
        sa.Column("deleted_at", TS),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_dataset_versions"),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_dataset_versions_owner_id_users"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            ondelete="CASCADE",
            name="fk_dataset_versions_dataset_id_datasets",
        ),
        sa.ForeignKeyConstraint(
            ["parent_version_id"],
            ["dataset_versions.id"],
            name="fk_dataset_versions_parent_version_id_dataset_versions",
        ),
        sa.ForeignKeyConstraint(
            ["restored_from_version_id"],
            ["dataset_versions.id"],
            name="fk_dataset_versions_restored_from_version_id_dataset_versions",
        ),
        sa.UniqueConstraint(
            "dataset_id", "version_number", name="uq_dataset_versions_dataset_id_version_number"
        ),
        sa.CheckConstraint(
            _in("kind", ("ingest", "reparse", "transform", "revert")),
            name="ck_dataset_versions_kind",
        ),
        sa.CheckConstraint(
            "(kind = 'ingest') = (parent_version_id IS NULL)", name="ck_dataset_versions_parent"
        ),
        sa.CheckConstraint(
            "(kind = 'revert') = (restored_from_version_id IS NOT NULL)",
            name="ck_dataset_versions_revert",
        ),
        sa.CheckConstraint(
            "(kind = 'transform') = (transformation_run_id IS NOT NULL)",
            name="ck_dataset_versions_transform",
        ),
        sa.CheckConstraint(
            "(kind IN ('ingest', 'reparse')) = (parse_options IS NOT NULL)",
            name="ck_dataset_versions_parse_options",
        ),
        sa.CheckConstraint("version_number >= 0", name="ck_dataset_versions_number"),
        sa.CheckConstraint("n_rows >= 0 AND n_cols >= 0", name="ck_dataset_versions_shape"),
    )
    op.create_index("ix_dataset_versions_owner_id", "dataset_versions", ["owner_id"])

    op.add_column("datasets", sa.Column("head_version_id", UUID))
    op.create_foreign_key(
        "fk_datasets_head_version_id_dataset_versions",
        "datasets",
        "dataset_versions",
        ["head_version_id"],
        ["id"],
        deferrable=True,
        initially="DEFERRED",
    )

    op.create_table(
        "artifacts",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("owner_type", sa.Text, nullable=False),
        sa.Column("owner_ref_id", UUID, nullable=False),
        sa.Column("kind", sa.Text, nullable=False),
        sa.Column("object_key", sa.Text, nullable=False),
        sa.Column("sha256", sa.LargeBinary, nullable=False),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("expires_at", TS),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_artifacts"),
        sa.ForeignKeyConstraint(["owner_id"], ["users.id"], name="fk_artifacts_owner_id_users"),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            ondelete="CASCADE",
            name="fk_artifacts_dataset_id_datasets",
        ),
        sa.CheckConstraint(_in("kind", ("preview",)), name="ck_artifacts_kind"),
        sa.CheckConstraint(_in("owner_type", ("dataset_version",)), name="ck_artifacts_owner_type"),
    )
    op.create_index("ix_artifacts_owner_id", "artifacts", ["owner_id"])
    op.create_index("ix_artifacts_owner_ref", "artifacts", ["owner_type", "owner_ref_id", "kind"])

    op.create_table(
        "version_profiles",
        sa.Column("id", UUID, primary_key=True),
        sa.Column("owner_id", UUID, nullable=False),
        sa.Column("dataset_id", UUID, nullable=False),
        sa.Column("dataset_version_id", UUID, nullable=False),
        sa.Column("job_id", UUID, nullable=False),
        sa.Column("profiler", sa.Text, nullable=False),
        sa.Column("profiler_version", sa.Text, nullable=False),
        sa.Column("summary", pg.JSONB, nullable=False),
        sa.Column("columns", pg.JSONB, nullable=False),
        sa.Column("correlations", pg.JSONB, nullable=False),
        sa.Column("sampling", pg.JSONB, nullable=False),
        sa.Column("limitations", pg.JSONB, nullable=False),
        sa.Column("engine_version", sa.Text, nullable=False),
        sa.Column("library_versions", pg.JSONB, nullable=False),
        sa.Column("created_at", TS, nullable=False, server_default=NOW),
        sa.PrimaryKeyConstraint("id", name="pk_version_profiles"),
        sa.ForeignKeyConstraint(
            ["owner_id"], ["users.id"], name="fk_version_profiles_owner_id_users"
        ),
        sa.ForeignKeyConstraint(
            ["dataset_id"],
            ["datasets.id"],
            ondelete="CASCADE",
            name="fk_version_profiles_dataset_id_datasets",
        ),
        sa.ForeignKeyConstraint(
            ["dataset_version_id"],
            ["dataset_versions.id"],
            ondelete="CASCADE",
            name="fk_version_profiles_dataset_version_id_dataset_versions",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["jobs.id"], name="fk_version_profiles_job_id_jobs"),
        sa.UniqueConstraint("job_id", name="uq_version_profiles_job_id"),
    )
    op.create_index("ix_version_profiles_owner_id", "version_profiles", ["owner_id"])
    op.create_index(
        "ix_version_profiles_version_created",
        "version_profiles",
        ["dataset_version_id", "created_at"],
    )

    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", ("ingest", "profile", "purge")))

    # ---- Row-Level Security --------------------------------------------------------------
    for table in NEW_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY owner_isolation ON {table} "
            f"USING (owner_id = {OWNER_EXPR}) WITH CHECK (owner_id = {OWNER_EXPR})"
        )

    # ---- Owner consistency (function from 0001) ------------------------------------------
    for table in NEW_TABLES:
        op.execute(
            f"CREATE TRIGGER {table}_dataset_owner BEFORE INSERT OR UPDATE OF owner_id, dataset_id "
            f"ON {table} FOR EACH ROW EXECUTE FUNCTION enforce_dataset_owner()"
        )
    op.execute(
        """
        CREATE FUNCTION enforce_version_in_dataset() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NOT EXISTS (
               SELECT 1 FROM dataset_versions v
                WHERE v.id = NEW.dataset_version_id
                  AND v.dataset_id = NEW.dataset_id
                  AND v.owner_id = NEW.owner_id
             ) THEN
            RAISE EXCEPTION 'version does not belong to this dataset and owner'
              USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER version_profiles_version BEFORE INSERT ON version_profiles "
        "FOR EACH ROW EXECUTE FUNCTION enforce_version_in_dataset()"
    )

    # ---- Versions are immutable (DEVELOPMENT_GUIDE.md invariant 5); only deleted_at may change -------
    op.execute(
        """
        CREATE FUNCTION dataset_versions_immutable() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF (to_jsonb(NEW) - 'deleted_at') IS DISTINCT FROM (to_jsonb(OLD) - 'deleted_at') THEN
            RAISE EXCEPTION 'dataset_versions rows are immutable' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER dataset_versions_no_update BEFORE UPDATE ON dataset_versions "
        "FOR EACH ROW EXECUTE FUNCTION dataset_versions_immutable()"
    )
    op.execute(
        """
        CREATE FUNCTION versions_parent_same_dataset() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.parent_version_id IS NOT NULL AND NOT EXISTS (
               SELECT 1 FROM dataset_versions v
                WHERE v.id = NEW.parent_version_id AND v.dataset_id = NEW.dataset_id) THEN
            RAISE EXCEPTION 'parent version belongs to another dataset' USING ERRCODE = '42501';
          END IF;
          IF NEW.restored_from_version_id IS NOT NULL AND NOT EXISTS (
               SELECT 1 FROM dataset_versions v
                WHERE v.id = NEW.restored_from_version_id AND v.dataset_id = NEW.dataset_id) THEN
            RAISE EXCEPTION 'restored version belongs to another dataset' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER dataset_versions_same_dataset BEFORE INSERT ON dataset_versions "
        "FOR EACH ROW EXECUTE FUNCTION versions_parent_same_dataset()"
    )

    # ---- datasets.head_version_id must point into the same dataset ------------------------
    op.execute(
        """
        CREATE FUNCTION datasets_head_in_dataset() RETURNS trigger
        LANGUAGE plpgsql AS $$
        BEGIN
          IF NEW.head_version_id IS NOT NULL AND NOT EXISTS (
               SELECT 1 FROM dataset_versions v
                WHERE v.id = NEW.head_version_id AND v.dataset_id = NEW.id) THEN
            RAISE EXCEPTION 'head version belongs to another dataset' USING ERRCODE = '42501';
          END IF;
          RETURN NEW;
        END $$
        """
    )
    op.execute(
        "CREATE TRIGGER datasets_head_version BEFORE INSERT OR UPDATE OF head_version_id "
        "ON datasets FOR EACH ROW EXECUTE FUNCTION datasets_head_in_dataset()"
    )

    # ---- Grants: insert-only tables get no UPDATE ------------------------------------------
    op.execute("GRANT SELECT, INSERT, DELETE ON dataset_versions TO dm_app")
    op.execute("GRANT UPDATE (deleted_at) ON dataset_versions TO dm_app")
    op.execute("GRANT SELECT, INSERT, DELETE ON artifacts TO dm_app")
    op.execute("GRANT SELECT, INSERT, DELETE ON version_profiles TO dm_app")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS datasets_head_version ON datasets")
    op.execute("DROP FUNCTION IF EXISTS datasets_head_in_dataset()")
    op.drop_table("version_profiles")
    op.drop_table("artifacts")
    op.drop_constraint(
        "fk_datasets_head_version_id_dataset_versions", "datasets", type_="foreignkey"
    )
    op.drop_column("datasets", "head_version_id")
    op.drop_table("dataset_versions")
    op.execute("DROP FUNCTION IF EXISTS versions_parent_same_dataset()")
    op.execute("DROP FUNCTION IF EXISTS dataset_versions_immutable()")
    op.execute("DROP FUNCTION IF EXISTS enforce_version_in_dataset()")
    op.execute("DELETE FROM jobs WHERE type = 'profile'")
    op.drop_constraint("ck_jobs_type", "jobs", type_="check")
    op.create_check_constraint("ck_jobs_type", "jobs", _in("type", ("ingest", "purge")))

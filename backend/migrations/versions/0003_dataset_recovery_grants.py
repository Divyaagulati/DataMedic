"""Stale-job recovery: the maintenance role may read datasets/original_files and move a
dataset out of 'validating' (status and updated_at only).

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-09
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("GRANT SELECT ON datasets, original_files TO dm_maintenance")
    op.execute("GRANT UPDATE (status, updated_at) ON datasets TO dm_maintenance")


def downgrade() -> None:
    op.execute("REVOKE UPDATE (status, updated_at) ON datasets FROM dm_maintenance")
    op.execute("REVOKE SELECT ON datasets, original_files FROM dm_maintenance")

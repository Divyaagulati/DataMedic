"""Phase 6/7 MVP: Biomedical Intelligence Lite summaries and exports.

- jobs.type gains biomedical_summary and export;
- analysis_runs gain kind 'biomedical' (a descriptive cohort summary; no findings rows);
- artifacts gain kind 'export' (a platform-written export file, owner only, served only through
  the authorised download endpoint).

Expand-only: existing rows are untouched; only CHECK constraints are widened.

Revision ID: 0008
Revises: 0007
Create Date: 2026-11-26
"""

from alembic import op

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None

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
    "purge",
)
JOB_TYPES_NEW = (*JOB_TYPES_OLD[:-1], "biomedical_summary", "export", "purge")
KINDS_OLD = (
    "preview",
    "fold_assignment",
    "impact_preview",
    "version_comparison",
    "oof_predictions",
)
KINDS_NEW = (*KINDS_OLD, "export")
ANALYSIS_OLD = ("diagnostics", "reliability_audit")
ANALYSIS_NEW = (*ANALYSIS_OLD, "biomedical")


def _in(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(repr(v) for v in values)})"


def _swap(name: str, table: str, column: str, values: tuple[str, ...]) -> None:
    op.drop_constraint(name, table, type_="check")
    op.create_check_constraint(name, table, _in(column, values))


def upgrade() -> None:
    _swap("ck_jobs_type", "jobs", "type", JOB_TYPES_NEW)
    _swap("ck_artifacts_kind", "artifacts", "kind", KINDS_NEW)
    _swap("ck_analysis_runs_kind", "analysis_runs", "kind", ANALYSIS_NEW)


def downgrade() -> None:
    op.execute("DELETE FROM analysis_runs WHERE kind = 'biomedical'")
    op.execute("DELETE FROM artifacts WHERE kind = 'export'")
    op.execute("DELETE FROM jobs WHERE type IN ('biomedical_summary', 'export')")
    _swap("ck_analysis_runs_kind", "analysis_runs", "kind", ANALYSIS_OLD)
    _swap("ck_artifacts_kind", "artifacts", "kind", KINDS_OLD)
    _swap("ck_jobs_type", "jobs", "type", JOB_TYPES_OLD)

"""RLS, grants and triggers of the Part E tables, tested directly in PostgreSQL."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError, ProgrammingError

from datamedic.context import AppContext
from datamedic.db.session import bind_user, session_scope
from tests.conftest import ClientFactory, signup_and_login
from tests.integration.test_pipelines_api import (
    STEPS,
    _approve_all,
    _commit,
    _pipeline,
    _preview,
    _setup,
)

pytestmark = pytest.mark.integration

TABLES = ("pipelines", "pipeline_steps", "approvals", "transformation_runs", "version_events")


@pytest.fixture
def committed(make_client: ClientFactory, ctx: AppContext):  # type: ignore[no-untyped-def]
    a = signup_and_login(make_client(), ctx)
    b = signup_and_login(make_client(), ctx)
    ds_id, _ = _setup(a)
    p = _pipeline(a, ds_id, STEPS[:4])
    _preview(a, p["id"])
    _approve_all(a, p["id"])
    assert _commit(a, p["id"], p["spec_hash"]).json()["job"]["status"] == "succeeded"
    r = a.client.post(
        f"/api/v1/datasets/{ds_id}/revert",
        headers=a.headers(),
        json={"version_id": p["source_version_id"]},
    )
    assert r.json()["job"]["status"] == "succeeded"
    return a, b, ds_id, p


def _count(db, table: str) -> int:  # type: ignore[no-untyped-def]
    return int(db.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())  # noqa: S608


def test_rls_hides_part_e_rows(ctx: AppContext, committed) -> None:  # type: ignore[no-untyped-def]
    a, b, *_ = committed
    with session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        assert all(_count(db, t) > 0 for t in TABLES)
    with session_scope(ctx.sessions) as db:
        assert all(_count(db, t) == 0 for t in TABLES)  # fails closed
    with session_scope(ctx.sessions) as db:
        bind_user(db, b.user_id)
        assert all(_count(db, t) == 0 for t in TABLES)
        assert db.execute(text("UPDATE pipelines SET name = 'x'")).rowcount == 0  # type: ignore[attr-defined]
        assert db.execute(text("DELETE FROM pipeline_steps")).rowcount == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE approvals SET approved_hash = 'forged'",
        "UPDATE transformation_runs SET spec_hash = 'forged'",
        "UPDATE version_events SET event = 'created'",
        "UPDATE dataset_versions SET model_pipeline_spec = '[]'::jsonb",
        "UPDATE dataset_versions SET transformation_run_id = NULL",
    ],
)
def test_history_is_insert_only(ctx: AppContext, committed, statement: str) -> None:  # type: ignore[no-untyped-def]
    a, *_ = committed
    with pytest.raises((ProgrammingError, DBAPIError)), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(text(statement))


def test_committed_pipeline_steps_are_frozen(ctx: AppContext, committed) -> None:  # type: ignore[no-untyped-def]
    a, _, _, p = committed
    for statement in (
        "UPDATE pipeline_steps SET params = '{}'::jsonb WHERE pipeline_id = :p",
        "DELETE FROM pipeline_steps WHERE pipeline_id = :p",
    ):
        with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
            bind_user(db, a.user_id)
            db.execute(text(statement), {"p": p["id"]})


def test_a_run_must_match_its_result_version(ctx: AppContext, committed) -> None:  # type: ignore[no-untyped-def]
    """A transformation run inserted for a version it did not create is refused at commit
    (deferred constraint trigger)."""
    a, _, _, p = committed
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(
            text(
                "INSERT INTO transformation_runs (id, owner_id, dataset_id, pipeline_id, job_id, "
                "source_version_id, result_version_id, spec_snapshot, spec_hash, approval_ids, "
                "step_reports, model_pipeline_spec, roles_report, seeds, engine_version, "
                "library_versions, executed_by) SELECT :id, owner_id, dataset_id, pipeline_id, "
                "(SELECT id FROM jobs WHERE type = 'revert' LIMIT 1), source_version_id, "
                "(SELECT id FROM dataset_versions WHERE kind = 'revert' LIMIT 1), spec_snapshot, "
                "spec_hash, approval_ids, step_reports, model_pipeline_spec, roles_report, seeds, "
                "engine_version, library_versions, executed_by FROM transformation_runs LIMIT 1"
            ),
            {"id": uuid.uuid4()},
        )


def test_one_mutation_job_per_dataset(ctx: AppContext, committed) -> None:  # type: ignore[no-untyped-def]
    a, _, ds_id, _ = committed
    insert = text(
        "INSERT INTO jobs (owner_id, type, queue, dataset_id, status, time_limit_seconds) "
        "VALUES (:o, :t, 'execute', :d, 'queued', 60)"
    )
    with pytest.raises(IntegrityError), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(insert, {"o": a.user_id, "t": "commit", "d": ds_id})
        db.execute(insert, {"o": a.user_id, "t": "revert", "d": ds_id})

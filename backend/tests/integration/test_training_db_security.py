"""RLS, grants, constraints and triggers of the Phase 5 tables, tested directly in PostgreSQL."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError

from datamedic.context import AppContext
from datamedic.db.session import bind_user, session_scope
from tests.conftest import ClientFactory, signup_and_login
from tests.integration.test_pipelines_api import _setup
from tests.integration.test_training_api import FAST, _train

pytestmark = pytest.mark.integration

TABLES = ("training_runs", "model_evaluations")


@pytest.fixture
def trained(make_client: ClientFactory, ctx: AppContext):  # type: ignore[no-untyped-def]
    a = signup_and_login(make_client(), ctx)
    b = signup_and_login(make_client(), ctx)
    ds_id, vid = _setup(a)
    r = _train(a, vid, FAST)
    assert r.json()["job"]["status"] == "succeeded", r.text
    return a, b, ds_id, r.json()["training_run"]["id"]


def _count(db, table: str, where: str = "") -> int:  # type: ignore[no-untyped-def]
    return int(db.execute(text(f"SELECT count(*) FROM {table} {where}")).scalar_one())  # noqa: S608


def test_rls_hides_phase5_rows(ctx: AppContext, trained) -> None:  # type: ignore[no-untyped-def]
    a, b, *_ = trained
    audit = "WHERE kind = 'reliability_audit'"
    with session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        assert all(_count(db, t) > 0 for t in TABLES) and _count(db, "analysis_runs", audit) == 1
    with session_scope(ctx.sessions) as db:
        assert all(_count(db, t) == 0 for t in TABLES)  # fails closed without a user
    with session_scope(ctx.sessions) as db:
        bind_user(db, b.user_id)
        assert all(_count(db, t) == 0 for t in TABLES) and _count(db, "analysis_runs", audit) == 0
        assert db.execute(text("DELETE FROM model_evaluations")).rowcount == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE training_runs SET summary = '{}'::jsonb",  # completed: written once
        "UPDATE training_runs SET config = '{}'::jsonb",  # configuration: no grant
        "UPDATE training_runs SET evaluation_mode = 'unbiased_direct'",
        "UPDATE model_evaluations SET metrics = '{}'::jsonb",  # insert-only
        "UPDATE analysis_runs SET kind = 'diagnostics' WHERE kind = 'reliability_audit'",
    ],
)
def test_results_are_immutable(ctx: AppContext, trained, statement: str) -> None:  # type: ignore[no-untyped-def]
    a, *_ = trained
    with pytest.raises((ProgrammingError, DBAPIError)), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(text(statement))


def test_constraints(ctx: AppContext, trained) -> None:  # type: ignore[no-untyped-def]
    a, _, _, rid = trained
    copy = (
        "INSERT INTO training_runs (id, owner_id, dataset_id, dataset_version_id, "
        "evaluation_version_id, job_id, name, task_type, target_column, evaluation_mode, "
        "eligibility, split_config, models, tuning, preprocess_steps, spec_hash, config, "
        "config_hash, source_content_hash, seed, engine_version) SELECT :id, owner_id, "
        "dataset_id, dataset_version_id, evaluation_version_id, (SELECT id FROM jobs WHERE "
        "type = 'reliability_audit' LIMIT 1), name, task_type, target_column, {mode}, "
        "eligibility, split_config, models, tuning, preprocess_steps, spec_hash, config, "
        "config_hash, source_content_hash, seed, engine_version FROM training_runs LIMIT 1"
    )
    # A replay run must evaluate another version than the selected one.
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(text(copy.format(mode="'unbiased_replay'")), {"id": uuid.uuid4()})
    # An audit run always belongs to a training run.
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(
            text("UPDATE analysis_runs SET training_run_id = NULL WHERE kind = 'reliability_audit'")
        )
    # A comparable evaluation must be complete.
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(
            text(
                "INSERT INTO model_evaluations (id, owner_id, dataset_id, training_run_id, "
                "model_key, status, comparable, metrics, details, fold_results, selection, "
                "importance, failures) VALUES (:id, :o, (SELECT dataset_id FROM training_runs "
                "WHERE id = :r), :r, 'hgb', 'partial', true, '{}', '{}', '[]', '{}', '{}', '[]')"
            ),
            {"id": uuid.uuid4(), "o": a.user_id, "r": rid},
        )


def test_other_users_run_cannot_be_referenced(
    ctx: AppContext, trained, make_client: ClientFactory
) -> None:  # type: ignore[no-untyped-def]
    a, b, _, rid = trained
    ds_b, _ = _setup(b)
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, b.user_id)
        db.execute(
            text(
                "INSERT INTO model_evaluations (id, owner_id, dataset_id, training_run_id, "
                "model_key, status, comparable, comparability_reason, metrics, details, "
                "fold_results, selection, importance, failures) VALUES (:id, :o, :d, :r, 'hgb', "
                "'failed', false, 'x', '{}', '{}', '[]', '{}', '{}', '[]')"
            ),
            {"id": uuid.uuid4(), "o": b.user_id, "d": ds_b, "r": rid},
        )

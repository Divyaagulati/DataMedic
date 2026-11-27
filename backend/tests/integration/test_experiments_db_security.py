"""Row-Level Security, grants and consistency triggers of the Part D tables, tested directly
in PostgreSQL."""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError

from datamedic.context import AppContext
from datamedic.db.session import bind_user, session_scope
from tests.conftest import ClientFactory, signup_and_login
from tests.integration.test_experiments_api import (
    _diagnose,
    _experiment_body,
    _post,
    _recommend,
    _setup,
)

pytestmark = pytest.mark.integration

TABLES = (
    "recommendation_sets",
    "recommendations",
    "recommendation_findings",
    "split_definitions",
    "experiments",
    "experiment_candidates",
    "experiment_results",
)


@pytest.fixture
def two_users(make_client: ClientFactory, ctx: AppContext):  # type: ignore[no-untyped-def]
    a = signup_and_login(make_client(), ctx)
    b = signup_and_login(make_client(), ctx)
    ds_id, vid = _setup(a)
    _diagnose(a, vid)
    rec = _recommend(a, vid)
    exp = _post(a, vid, _experiment_body(rec)).json()["experiment"]["id"]
    _, b_vid = _setup(b)
    return a, b, ds_id, vid, exp, b_vid


def _count(db, table: str) -> int:  # type: ignore[no-untyped-def]
    return int(db.execute(text(f"SELECT count(*) FROM {table}")).scalar_one())  # noqa: S608


def test_rls_hides_and_protects_part_d_rows(ctx: AppContext, two_users) -> None:  # type: ignore[no-untyped-def]
    a, b, *_ = two_users
    with session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        assert all(_count(db, t) > 0 for t in TABLES)
    with session_scope(ctx.sessions) as db:
        assert all(_count(db, t) == 0 for t in TABLES)  # no user bound: fail closed
    with session_scope(ctx.sessions) as db:
        bind_user(db, b.user_id)
        assert all(_count(db, t) == 0 for t in TABLES)
        assert db.execute(text("UPDATE recommendations SET status = 'dismissed'")).rowcount == 0  # type: ignore[attr-defined]
        assert db.execute(text("DELETE FROM experiments")).rowcount == 0  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE experiment_results SET status = 'failed'",
        "UPDATE experiment_candidates SET steps = '[]'::jsonb",
        "UPDATE split_definitions SET seed = 99",
        "UPDATE recommendation_sets SET content = '{}'::jsonb",
        "UPDATE recommendations SET op = 'select.drop_columns'",
        "UPDATE experiments SET config = '{}'::jsonb",
    ],
)
def test_app_role_cannot_rewrite_part_d_history(ctx: AppContext, two_users, statement: str) -> None:  # type: ignore[no-untyped-def]
    a, *_ = two_users
    with pytest.raises((ProgrammingError, DBAPIError)), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(text(statement))


def test_a_completed_experiment_cannot_change(ctx: AppContext, two_users) -> None:  # type: ignore[no-untyped-def]
    a, _, _, _, exp, _ = two_users
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(text("UPDATE experiments SET summary = '{}'::jsonb WHERE id = :id"), {"id": exp})


def test_references_must_stay_inside_the_dataset(ctx: AppContext, two_users) -> None:  # type: ignore[no-untyped-def]
    """A's experiment cannot point at another dataset's split, and a recommendation cannot
    cite a finding of another dataset (triggers), even for the same owner."""
    a, _, ds_id, vid, exp, _ = two_users
    _, other_vid = _setup(a)
    _diagnose(a, other_vid)
    other_rec = _recommend(a, other_vid)
    other_exp = _post(a, other_vid, _experiment_body(other_rec)).json()["experiment"]["id"]
    with session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        other_split = db.execute(
            text("SELECT split_definition_id FROM experiments WHERE id = :e"), {"e": other_exp}
        ).scalar_one()
        rec_id = db.execute(
            text("SELECT id FROM recommendations WHERE set_id = :s LIMIT 1"),
            {"s": _recommend(a, vid)["id"]},
        ).scalar_one()
        other_finding = db.execute(
            text(
                "SELECT f.id FROM findings f JOIN analysis_runs r ON r.id = f.analysis_run_id "
                "WHERE r.dataset_version_id = :v LIMIT 1"
            ),
            {"v": other_vid},
        ).scalar_one()
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(
            text(
                "INSERT INTO experiments (id, owner_id, dataset_id, dataset_version_id, job_id, name, "
                "task_type, models, seed, config, config_hash, source_content_hash, "
                "split_definition_id, engine_version) SELECT gen_random_uuid(), owner_id, dataset_id,"
                " dataset_version_id, (SELECT id FROM jobs WHERE type='ingest' AND dataset_id = "
                ":d LIMIT 1), 'x', task_type, models, seed, config, config_hash, "
                "source_content_hash, :split, engine_version FROM experiments WHERE id = :e"
            ),
            {"d": ds_id, "split": other_split, "e": exp},
        )
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(
            text(
                "INSERT INTO recommendation_findings (recommendation_id, finding_id, owner_id, "
                "dataset_id) VALUES (:r, :f, :o, :d)"
            ),
            {"r": rec_id, "f": other_finding, "o": a.user_id, "d": ds_id},
        )


def test_other_users_cannot_insert_into_part_d_tables(ctx: AppContext, two_users) -> None:  # type: ignore[no-untyped-def]
    a, b, ds_id, vid, _, _ = two_users
    with pytest.raises((ProgrammingError, DBAPIError)), session_scope(ctx.sessions) as db:
        bind_user(db, b.user_id)
        db.execute(
            text(
                "INSERT INTO split_definitions (id, owner_id, dataset_id, dataset_version_id, "
                "config_hash, method, n_splits, seed, duplicate_columns, fold_assignment_artifact_id,"
                " assignment_sha256, cohort, fold_stats, warnings, engine_version, library_versions)"
                " VALUES (:id, :o, :d, :v, '\\x00', 'kfold', 2, 0, '[]', :id, '\\x00', '{}', '[]', "
                "'[]', 'x', '{}')"
            ),
            {"id": uuid.uuid4(), "o": a.user_id, "d": ds_id, "v": vid},
        )

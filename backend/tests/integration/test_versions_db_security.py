"""Database-level protection of versions, artifacts and profiles: RLS, immutability,
least-privilege grants and consistency triggers."""

import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, ProgrammingError

from datamedic.context import AppContext
from datamedic.db.session import bind_user, session_scope
from tests.conftest import ClientFactory, DbUrls, create_and_upload, signup_and_login

pytestmark = pytest.mark.integration

NEW_TABLES = ("dataset_versions", "artifacts", "version_profiles")


@pytest.fixture
def parsed(make_client: ClientFactory, ctx: AppContext):  # type: ignore[no-untyped-def]
    a = signup_and_login(make_client(), ctx)
    b = signup_and_login(make_client(), ctx)
    first = create_and_upload(a)
    second = create_and_upload(a)
    ds1 = uuid.UUID(first["dataset"]["id"])  # type: ignore[index]
    ds2 = uuid.UUID(second["dataset"]["id"])  # type: ignore[index]
    v1 = uuid.UUID(a.client.get(f"/api/v1/datasets/{ds1}").json()["head_version"]["id"])
    v2 = uuid.UUID(a.client.get(f"/api/v1/datasets/{ds2}").json()["head_version"]["id"])
    return a, b, ds1, v1, ds2, v2


def test_other_user_sees_and_changes_nothing(ctx: AppContext, parsed) -> None:  # type: ignore[no-untyped-def]
    a, b, ds1, *_ = parsed
    with session_scope(ctx.sessions) as db:
        bind_user(db, b.user_id)
        for table in NEW_TABLES:
            count = db.execute(
                text(f"SELECT count(*) FROM {table} WHERE owner_id = :a"),  # noqa: S608
                {"a": a.user_id},
            ).scalar_one()
            assert count == 0, table
            deleted = db.execute(
                text(f"DELETE FROM {table} WHERE dataset_id = :d"),  # noqa: S608
                {"d": ds1},
            ).rowcount  # type: ignore[attr-defined]
            assert deleted == 0, table
    with session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        for table in NEW_TABLES:
            count = db.execute(
                text(f"SELECT count(*) FROM {table} WHERE dataset_id = :d"),  # noqa: S608
                {"d": ds1},
            ).scalar_one()
            assert count == 1, table


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE dataset_versions SET n_rows = 0 WHERE id = :v",
        "UPDATE dataset_versions SET object_key = 'x' WHERE id = :v",
        "UPDATE artifacts SET object_key = 'x' WHERE owner_ref_id = :v",
        "UPDATE version_profiles SET summary = '{}' WHERE dataset_version_id = :v",
    ],
)
def test_app_role_cannot_rewrite_history(ctx: AppContext, parsed, statement: str) -> None:  # type: ignore[no-untyped-def]
    a, _, _, v1, *_ = parsed
    with pytest.raises(ProgrammingError, match="permission denied"):
        with session_scope(ctx.sessions) as db:
            bind_user(db, a.user_id)
            db.execute(text(statement), {"v": v1})


def test_versions_are_immutable_even_for_the_schema_owner(
    ctx: AppContext, database: DbUrls, parsed
) -> None:  # type: ignore[no-untyped-def]
    a, _, _, v1, *_ = parsed
    engine = create_engine(database.migrator)
    try:
        with pytest.raises(DBAPIError, match="immutable"):
            with engine.begin() as conn:
                conn.execute(
                    text("SELECT set_config('app.user_id', :u, true)"), {"u": str(a.user_id)}
                )
                conn.execute(
                    text("UPDATE dataset_versions SET n_rows = 1 WHERE id = :v"), {"v": v1}
                )
    finally:
        engine.dispose()
    # Soft deletion (deleted_at) is the one permitted change.
    with session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        db.execute(text("UPDATE dataset_versions SET deleted_at = now() WHERE id = :v"), {"v": v1})


def test_head_version_must_belong_to_the_dataset(ctx: AppContext, parsed) -> None:  # type: ignore[no-untyped-def]
    a, _, ds1, _, _, v2 = parsed
    with pytest.raises(DBAPIError, match="head version belongs to another dataset"):
        with session_scope(ctx.sessions) as db:
            bind_user(db, a.user_id)
            db.execute(
                text("UPDATE datasets SET head_version_id = :v WHERE id = :d"),
                {"v": v2, "d": ds1},
            )


def test_profile_must_reference_a_version_of_its_dataset(ctx: AppContext, parsed) -> None:  # type: ignore[no-untyped-def]
    a, _, ds1, _, _, v2 = parsed
    with pytest.raises(DBAPIError, match="version does not belong"):
        with session_scope(ctx.sessions) as db:
            bind_user(db, a.user_id)
            job_id = db.execute(
                text("SELECT job_id FROM version_profiles WHERE dataset_id = :d"), {"d": ds1}
            ).scalar_one()
            db.execute(
                text(
                    "INSERT INTO version_profiles (id, owner_id, dataset_id, dataset_version_id, "
                    "job_id, profiler, profiler_version, summary, columns, correlations, "
                    "sampling, limitations, engine_version, library_versions) VALUES "
                    "(gen_random_uuid(), :a, :d, :v, :j, 'x', '1', '{}', '[]', '{}', '{}', "
                    "'[]', '0', '{}')"
                ),
                {"a": a.user_id, "d": ds1, "v": v2, "j": job_id},
            )

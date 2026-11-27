"""Row-Level Security, grants, triggers and claim_job(), tested directly in PostgreSQL."""

import uuid

import psycopg
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError, ProgrammingError

from datamedic.context import AppContext
from datamedic.db.models import Base
from datamedic.db.session import bind_user, session_scope
from tests.conftest import ClientFactory, DbUrls, create_and_upload, signup_and_login

pytestmark = pytest.mark.integration


@pytest.fixture
def two_users_with_data(make_client: ClientFactory, ctx: AppContext):  # type: ignore[no-untyped-def]
    a = signup_and_login(make_client(), ctx)
    b = signup_and_login(make_client(), ctx)
    body = create_and_upload(a)
    return a, b, uuid.UUID(body["dataset"]["id"])  # type: ignore[index]


def test_rls_fails_closed_without_user(ctx: AppContext, two_users_with_data) -> None:  # type: ignore[no-untyped-def]
    with session_scope(ctx.sessions) as db:  # no bind_user
        for table in (
            "datasets",
            "original_files",
            "jobs",
            "dataset_versions",
            "artifacts",
            "version_profiles",
        ):
            assert db.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0  # noqa: S608


def test_rls_hides_other_users_rows(ctx: AppContext, two_users_with_data) -> None:  # type: ignore[no-untyped-def]
    a, b, ds_id = two_users_with_data
    with session_scope(ctx.sessions) as db:
        bind_user(db, b.user_id)
        assert (
            db.execute(
                text("SELECT count(*) FROM datasets WHERE id = :id"), {"id": ds_id}
            ).scalar_one()
            == 0
        )
        assert (
            db.execute(
                text("SELECT count(*) FROM jobs WHERE owner_id = :o"), {"o": a.user_id}
            ).scalar_one()
            == 0
        )
        # Even a direct UPDATE/DELETE of A's rows affects nothing.
        assert (
            db.execute(
                text("UPDATE datasets SET name = 'pwned' WHERE id = :id"), {"id": ds_id}
            ).rowcount
            == 0
        )  # type: ignore[attr-defined]
        assert (
            db.execute(text("DELETE FROM jobs WHERE owner_id = :o"), {"o": a.user_id}).rowcount == 0
        )  # type: ignore[attr-defined]
    with session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        assert (
            db.execute(text("SELECT name FROM datasets WHERE id = :id"), {"id": ds_id}).scalar_one()
            == "Cohort"
        )


def test_rls_blocks_inserting_rows_for_another_user(ctx: AppContext, two_users_with_data) -> None:  # type: ignore[no-untyped-def]
    a, b, _ = two_users_with_data
    with pytest.raises(ProgrammingError, match="row-level security"):
        with session_scope(ctx.sessions) as db:
            bind_user(db, b.user_id)
            db.execute(
                text(
                    "INSERT INTO datasets (id, owner_id, name, declared_filename, "
                    "declared_extension, declared_size_bytes) "
                    "VALUES (gen_random_uuid(), :a, 'x', 'x.csv', 'csv', 1)"
                ),
                {"a": a.user_id},
            )


def test_child_owner_must_match_dataset_owner(ctx: AppContext, two_users_with_data) -> None:  # type: ignore[no-untyped-def]
    _, b, ds_id = two_users_with_data
    # B tries to attach a job to A's dataset while acting as B.
    with pytest.raises(DBAPIError, match="owner_id does not match"):
        with session_scope(ctx.sessions) as db:
            bind_user(db, b.user_id)
            db.execute(
                text(
                    "INSERT INTO jobs (id, owner_id, type, queue, dataset_id, time_limit_seconds) "
                    "VALUES (gen_random_uuid(), :b, 'ingest', 'ingest', :ds, 60)"
                ),
                {"b": b.user_id, "ds": ds_id},
            )


def test_original_files_are_immutable(
    ctx: AppContext, database: DbUrls, two_users_with_data
) -> None:  # type: ignore[no-untyped-def]
    a, _, ds_id = two_users_with_data
    # App role: no UPDATE privilege on the hash/key columns at all.
    with pytest.raises(ProgrammingError, match="permission denied"):
        with session_scope(ctx.sessions) as db:
            bind_user(db, a.user_id)
            db.execute(
                text("UPDATE original_files SET sha256 = '\\x00' WHERE dataset_id = :d"),
                {"d": ds_id},
            )
    # Even the schema owner is stopped by the trigger.
    engine = create_engine(database.migrator)
    try:
        with pytest.raises(DBAPIError, match="immutable"):
            with engine.begin() as conn:
                conn.execute(
                    text("SELECT set_config('app.user_id', :u, true)"), {"u": str(a.user_id)}
                )
                conn.execute(
                    text("UPDATE original_files SET object_key = 'x' WHERE dataset_id = :d"),
                    {"d": ds_id},
                )
    finally:
        engine.dispose()


def test_app_role_has_no_bypass(database: DbUrls) -> None:
    with psycopg.connect(database.app.replace("+psycopg", "")) as conn:
        row = conn.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
    assert row == (False, False)


def test_claim_job_is_atomic(ctx: AppContext, two_users_with_data) -> None:  # type: ignore[no-untyped-def]
    a, _, ds_id = two_users_with_data
    with session_scope(ctx.sessions) as db:
        bind_user(db, a.user_id)
        job_id = db.execute(
            text(
                "INSERT INTO jobs (id, owner_id, type, queue, dataset_id, time_limit_seconds) "
                "VALUES (gen_random_uuid(), :a, 'ingest', 'ingest', :ds, 60) RETURNING id"
            ),
            {"a": a.user_id, "ds": ds_id},
        ).scalar_one()
    with session_scope(ctx.sessions) as db:  # unbound session, as a worker starts
        first = db.execute(
            text("SELECT owner_id, job_type FROM claim_job(:id, 'w1')"), {"id": job_id}
        ).all()
        second = db.execute(
            text("SELECT owner_id, job_type FROM claim_job(:id, 'w2')"), {"id": job_id}
        ).all()
    assert first == [(a.user_id, "ingest")]
    assert second == []


def test_models_match_migrations(database: DbUrls) -> None:
    engine = create_engine(database.migrator)
    try:
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    finally:
        engine.dispose()
    assert diff == []

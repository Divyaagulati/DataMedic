"""Stale-job recovery and safe retries (Phase 2 hardening).

The reaper fails jobs whose heartbeat is too old; recover_stuck_datasets() then moves
datasets out of 'validating' when nothing processes them any more. Active work is never
touched, and no sequence of retries can create a second version 0.
"""

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

from datamedic.context import AppContext, build_context
from datamedic.db.session import bind_user, session_scope
from datamedic.jobs.publisher import CeleryJobPublisher
from datamedic.workers.celery_app import create_celery
from datamedic.workers.maintenance import reap_stale, recover_stuck_datasets
from datamedic.workers.runner import run_job
from tests.conftest import (
    CSV_BYTES,
    ClientFactory,
    DbUrls,
    LoggedInUser,
    create_and_upload,
    make_settings,
    signup_and_login,
)

pytestmark = pytest.mark.integration

FAILED_ERROR = '{"code": "INTERNAL", "message": "x", "retryable": true}'


@pytest.fixture
def broker_ctx(database: DbUrls, tmp_path: Path) -> Iterator[AppContext]:
    """Jobs stay queued until the test runs them, like a real broker + worker."""
    context = build_context(
        make_settings(database, tmp_path, jobs_eager=False, redis_url="memory://")
    )
    context.publisher = CeleryJobPublisher(create_celery(context.settings))
    yield context
    context.dispose()


def _maint(ctx: AppContext, sql: str, **params: object) -> None:
    with session_scope(ctx.maintenance_sessions) as db:
        db.execute(text(sql), params)


def _app(ctx: AppContext, user: LoggedInUser, sql: str, **params: object) -> None:
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        db.execute(text(sql), params)


def _status(user: LoggedInUser, ds_id: str) -> str:
    return str(user.client.get(f"/api/v1/datasets/{ds_id}").json()["status"])


def _versions(ctx: AppContext, user: LoggedInUser, ds_id: str) -> int:
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        return int(
            db.execute(
                text("SELECT count(*) FROM dataset_versions WHERE dataset_id = :d"), {"d": ds_id}
            ).scalar_one()
        )


def _upload(user: LoggedInUser) -> tuple[str, str]:
    body = create_and_upload(user, CSV_BYTES, "cohort.csv")
    return body["dataset"]["id"], body["job"]["id"]  # type: ignore[index]


def _reap(ctx: AppContext) -> dict[str, int]:
    reap_stale(ctx)
    return recover_stuck_datasets(ctx)


def test_stale_ingest_is_recovered_then_retried_once(
    make_client: ClientFactory, broker_ctx: AppContext
) -> None:
    user = signup_and_login(make_client(broker_ctx), broker_ctx)
    ds_id, job_id = _upload(user)
    long_ago = datetime.now(UTC) - timedelta(hours=2)
    # A worker claimed the job and then died: running, heartbeat two hours old.
    _maint(
        broker_ctx,
        "UPDATE jobs SET status = 'running', started_at = :t, heartbeat_at = :t WHERE id = :j",
        t=long_ago,
        j=job_id,
    )
    # The reaper is global (all users), so assert on this dataset, not on total counts.
    assert _reap(broker_ctx)["parse_failed"] >= 1
    job = user.client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "failed" and job["error"]["code"] == "JOB_STALE"
    assert _status(user, ds_id) == "parse_failed"

    r = user.client.post(f"/api/v1/datasets/{ds_id}/parse", headers=user.headers())
    assert r.status_code == 202
    run_job(broker_ctx, r.json()["job"]["id"])
    assert _status(user, ds_id) == "ready"
    assert _versions(broker_ctx, user, ds_id) == 1
    _reap(broker_ctx)  # a later reaper pass leaves the recovered, re-parsed dataset alone
    assert _status(user, ds_id) == "ready" and _versions(broker_ctx, user, ds_id) == 1


def test_active_ingest_is_never_touched(make_client: ClientFactory, broker_ctx: AppContext) -> None:
    user = signup_and_login(make_client(broker_ctx), broker_ctx)
    queued_ds, _ = _upload(user)  # queued, never claimed
    running_ds, running_job = _upload(user)
    _maint(
        broker_ctx,
        "UPDATE jobs SET status = 'running', started_at = now(), heartbeat_at = now() "
        "WHERE id = :j",
        j=running_job,
    )
    # Even when the dataset row itself looks old, an active job protects it.
    _app(
        broker_ctx,
        user,
        "UPDATE datasets SET updated_at = now() - interval '3 hours' WHERE id IN (:a, :b)",
        a=queued_ds,
        b=running_ds,
    )
    _reap(broker_ctx)
    assert _status(user, queued_ds) == "validating"
    assert _status(user, running_ds) == "validating"
    assert user.client.get(f"/api/v1/jobs/{running_job}").json()["status"] == "running"


def test_cancelled_ingest_is_recovered(make_client: ClientFactory, broker_ctx: AppContext) -> None:
    user = signup_and_login(make_client(broker_ctx), broker_ctx)
    ds_id, job_id = _upload(user)
    r = user.client.post(f"/api/v1/jobs/{job_id}/cancel", headers=user.headers())
    assert r.status_code == 202
    assert _status(user, ds_id) == "validating"  # cancelling alone leaves it stuck
    _reap(broker_ctx)
    assert _status(user, ds_id) == "parse_failed"


def test_failed_ingest_left_validating_is_recovered(
    make_client: ClientFactory, broker_ctx: AppContext
) -> None:
    user = signup_and_login(make_client(broker_ctx), broker_ctx)
    ds_id, job_id = _upload(user)
    # e.g. the failure hook itself could not run (database briefly unavailable)
    _maint(
        broker_ctx,
        "UPDATE jobs SET status = 'failed', finished_at = now(), error = CAST(:e AS jsonb) "
        "WHERE id = :j",
        e=FAILED_ERROR,
        j=job_id,
    )
    _reap(broker_ctx)
    assert _status(user, ds_id) == "parse_failed"


def test_interrupted_upload_is_reopened_after_grace(
    make_client: ClientFactory, broker_ctx: AppContext
) -> None:
    user = signup_and_login(make_client(broker_ctx), broker_ctx)
    fresh, old = (
        user.client.post(
            "/api/v1/datasets",
            headers=user.headers(),
            json={"name": n, "filename": "a.csv", "size_bytes": len(CSV_BYTES)},
        ).json()["dataset"]["id"]
        for n in ("fresh", "old")
    )
    # The API process died mid-stream: 'validating' but no stored original.
    _app(
        broker_ctx,
        user,
        "UPDATE datasets SET status = 'validating' WHERE id IN (:a, :b)",
        a=fresh,
        b=old,
    )
    _app(
        broker_ctx,
        user,
        "UPDATE datasets SET updated_at = now() - interval '2 hours' WHERE id = :b",
        b=old,
    )
    assert _reap(broker_ctx)["upload_pending"] >= 1
    assert _status(user, old) == "upload_pending"
    assert _status(user, fresh) == "validating"  # may still be streaming
    r = user.client.put(
        f"/api/v1/datasets/{old}/content", headers=user.headers(), content=CSV_BYTES
    )
    assert r.status_code == 202  # the owner can upload again


def test_late_finished_dataset_is_marked_ready(
    make_client: ClientFactory, broker_ctx: AppContext
) -> None:
    user = signup_and_login(make_client(broker_ctx), broker_ctx)
    ds_id, job_id = _upload(user)
    run_job(broker_ctx, job_id)
    _app(broker_ctx, user, "UPDATE datasets SET status = 'validating' WHERE id = :d", d=ds_id)
    assert _reap(broker_ctx)["ready"] >= 1
    assert _status(user, ds_id) == "ready"


def test_duplicate_ingest_never_creates_a_second_version(
    make_client: ClientFactory, broker_ctx: AppContext
) -> None:
    from datamedic.jobs.service import create_job

    user = signup_and_login(make_client(broker_ctx), broker_ctx)
    ds_id, first = _upload(user)
    with session_scope(broker_ctx.sessions) as db:
        bind_user(db, user.user_id)
        second = create_job(
            db, broker_ctx.settings, user.user_id, "ingest", uuid.UUID(ds_id), enforce_limits=False
        ).id
    run_job(broker_ctx, first)
    run_job(broker_ctx, str(second))
    assert _versions(broker_ctx, user, ds_id) == 1
    job = user.client.get(f"/api/v1/jobs/{second}").json()
    assert job["status"] == "succeeded" and job["result"]["skipped"] == "already parsed"


def test_memory_limit_fails_cleanly(
    make_client: ClientFactory, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datamedic.datasets import jobs as dataset_jobs

    def too_big(*args: object, **kwargs: object) -> None:
        raise MemoryError

    monkeypatch.setattr(dataset_jobs, "_parse_into_version", too_big)
    user = signup_and_login(make_client(), ctx)
    ds_id, job_id = _upload(user)
    job = user.client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "failed" and job["error"]["code"] == "LIMIT_EXCEEDED"
    assert job["error"]["retryable"] is False
    assert _status(user, ds_id) == "parse_failed"  # failure hook restored a recoverable state

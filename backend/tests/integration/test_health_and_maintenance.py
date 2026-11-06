import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import text

from datamedic.context import AppContext
from datamedic.db.session import bind_user, session_scope
from datamedic.workers.maintenance import reap_stale, requeue_stuck
from tests.conftest import ClientFactory, signup_and_login

pytestmark = pytest.mark.integration


def test_health_endpoints(make_client: ClientFactory) -> None:
    client = make_client()
    assert client.get("/healthz").json() == {"status": "ok"}
    r = client.get("/readyz")
    assert r.status_code == 200 and r.json()["checks"] == {"database": True, "storage": True}
    assert r.headers["cache-control"] == "no-store"
    assert "x-request-id" in r.headers


class RecordingPublisher:
    def __init__(self) -> None:
        self.published: list[tuple[uuid.UUID, str]] = []

    def publish(self, job_id: uuid.UUID, job_type: str) -> None:
        self.published.append((job_id, job_type))


def _insert_job(ctx: AppContext, owner: uuid.UUID, **cols: object) -> uuid.UUID:
    job_id = uuid.uuid4()
    with session_scope(ctx.sessions) as db:
        bind_user(db, owner)
        db.execute(
            text(
                "INSERT INTO jobs (id, owner_id, type, queue, status, time_limit_seconds, queued_at, "
                "last_published_at, heartbeat_at) VALUES (:id, :o, 'purge', 'maintenance', :status, "
                ":limit, COALESCE(:queued_at, now()), :published, :heartbeat)"
            ),
            {
                "id": job_id,
                "o": owner,
                "status": cols.get("status", "queued"),
                "limit": cols.get("limit", 60),
                "queued_at": cols.get("queued_at"),
                "published": cols.get("published"),
                "heartbeat": cols.get("heartbeat"),
            },
        )
    return job_id


def _status(ctx: AppContext, job_id: uuid.UUID) -> str:
    with session_scope(ctx.maintenance_sessions) as db:
        return str(
            db.execute(text("SELECT status FROM jobs WHERE id = :id"), {"id": job_id}).scalar_one()
        )


def test_requeue_republishes_only_stuck_jobs(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    old = datetime.now(UTC) - timedelta(hours=1)
    stuck = _insert_job(ctx, user.user_id, queued_at=old, published=None)
    fresh = _insert_job(ctx, user.user_id, queued_at=datetime.now(UTC), published=datetime.now(UTC))
    recorder = RecordingPublisher()
    ctx.publisher = recorder
    requeue_stuck(ctx)
    ids = [jid for jid, _ in recorder.published]
    assert stuck in ids and fresh not in ids


def test_reaper_fails_stale_running_jobs(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    long_ago = datetime.now(UTC) - timedelta(hours=2)
    stale = _insert_job(ctx, user.user_id, status="running", heartbeat=long_ago, queued_at=long_ago)
    alive = _insert_job(
        ctx, user.user_id, status="running", heartbeat=datetime.now(UTC), queued_at=long_ago
    )
    reap_stale(ctx)
    assert _status(ctx, stale) == "failed"
    assert _status(ctx, alive) == "running"

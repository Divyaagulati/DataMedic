"""Broker messages must contain only the job id (PRIVACY_AND_SECURITY §2)."""

import uuid
from typing import Any

from celery.signals import before_task_publish
from pydantic import SecretStr

from datamedic.config import Settings
from datamedic.jobs.publisher import CeleryJobPublisher
from datamedic.workers.celery_app import create_celery


def test_published_message_contains_only_job_id() -> None:
    settings = Settings(
        _env_file=None,
        redis_url="memory://",  # type: ignore[call-arg]
        database_url=SecretStr("postgresql+psycopg://x/y"),
        maintenance_database_url=SecretStr("postgresql+psycopg://x/y"),
    )
    app = create_celery(settings)
    captured: list[dict[str, Any]] = []

    def capture(
        sender: str | None = None,
        body: Any = None,
        headers: Any = None,
        routing_key: str | None = None,
        **_: Any,
    ) -> None:
        captured.append(
            {"sender": sender, "body": body, "headers": headers, "routing_key": routing_key}
        )

    before_task_publish.connect(capture)
    try:
        job_id = uuid.uuid4()
        CeleryJobPublisher(app).publish(job_id, "ingest")
    finally:
        before_task_publish.disconnect(capture)

    assert len(captured) == 1
    msg = captured[0]
    args, kwargs, _embed = msg["body"]
    assert list(args) == []
    assert kwargs == {"job_id": str(job_id)}
    assert msg["sender"] == "datamedic.jobs.ingest"
    assert msg["routing_key"] == "ingest"
    assert app.conf.task_serializer == "json" and app.conf.accept_content == ["json"]

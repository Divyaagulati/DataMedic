"""Shared fixtures.

Unit tests need nothing external. Integration tests (marker ``integration``) need a PostgreSQL
superuser URL in TEST_ADMIN_DATABASE_URL, e.g.
    postgresql://postgres:postgres@127.0.0.1:5432/postgres
A fresh database (with the dm_* roles) is created per test session and dropped afterwards.
"""

import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from pydantic import SecretStr
from sqlalchemy.engine import make_url

from datamedic.api.app import create_app
from datamedic.config import Settings
from datamedic.context import AppContext, build_context
from datamedic.db.bootstrap import bootstrap
from datamedic.email import MemoryEmailSender

BACKEND_DIR = Path(__file__).resolve().parents[1]
ADMIN_URL = os.environ.get("TEST_ADMIN_DATABASE_URL")
PASSWORD = "Correct-Horse-Battery-9"


@dataclass(frozen=True)
class DbUrls:
    name: str
    migrator: str
    app: str
    maintenance: str


def _role_url(admin: str, role: str, password: str, db: str) -> str:
    url = make_url(admin).set(
        drivername="postgresql+psycopg", username=role, password=password, database=db
    )
    return url.render_as_string(hide_password=False)


@pytest.fixture(scope="session")
def database() -> Iterator[DbUrls]:
    if not ADMIN_URL:
        pytest.skip("TEST_ADMIN_DATABASE_URL not set; skipping integration tests")
    name = f"datamedic_test_{uuid.uuid4().hex[:10]}"
    passwords = {
        "dm_migrator": "test-migrator",
        "dm_app": "test-app",
        "dm_maintenance": "test-maintenance",
    }
    bootstrap(ADMIN_URL, name, passwords)
    urls = DbUrls(
        name=name,
        migrator=_role_url(ADMIN_URL, "dm_migrator", passwords["dm_migrator"], name),
        app=_role_url(ADMIN_URL, "dm_app", passwords["dm_app"], name),
        maintenance=_role_url(ADMIN_URL, "dm_maintenance", passwords["dm_maintenance"], name),
    )
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.attributes["url"] = urls.migrator
    command.upgrade(cfg, "head")
    yield urls
    admin = make_url(ADMIN_URL).set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(admin, autocommit=True) as conn:
        conn.execute(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            "WHERE datname = %s AND pid <> pg_backend_pid()",
            (name,),
        )
        conn.execute(f'DROP DATABASE IF EXISTS "{name}"')


def make_settings(database: DbUrls, tmp: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "test",
        "database_url": SecretStr(database.app),
        "maintenance_database_url": SecretStr(database.maintenance),
        "storage_root": tmp / "storage",
        "email_backend": "memory",
        "jobs_eager": True,
        "cookie_secure": False,
        "web_origin": "http://localhost:3000",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type, call-arg]


@pytest.fixture
def ctx(database: DbUrls, tmp_path: Path) -> Iterator[AppContext]:
    context = build_context(make_settings(database, tmp_path))
    yield context
    context.dispose()


@pytest.fixture
def make_client(ctx: AppContext) -> Iterator["ClientFactory"]:
    clients: list[TestClient] = []

    def factory(context: AppContext | None = None) -> TestClient:
        client = TestClient(create_app(ctx=context or ctx))
        clients.append(client)
        return client

    yield factory
    for c in clients:
        c.close()


class ClientFactory:  # typing helper
    def __call__(self, context: AppContext | None = None) -> TestClient: ...


@dataclass
class LoggedInUser:
    client: TestClient
    email: str
    user_id: uuid.UUID
    csrf: str

    def headers(self, **extra: str) -> dict[str, str]:
        return {"X-CSRF-Token": self.csrf, **extra}


def outbox(context: AppContext) -> MemoryEmailSender:
    assert isinstance(context.email, MemoryEmailSender)
    return context.email


def last_token(context: AppContext, to: str) -> str:
    msgs = [m for m in outbox(context).outbox if m.to == to]
    assert msgs, f"no email to {to}"
    return msgs[-1].body.split("token=")[1].split()[0]


def signup_and_login(
    client: TestClient, context: AppContext, *, verify: bool = True, email: str | None = None
) -> LoggedInUser:
    email = email or f"user-{uuid.uuid4().hex[:8]}@example.org"
    r = client.post("/api/v1/auth/signup", json={"email": email, "password": PASSWORD})
    assert r.status_code == 202, r.text
    if verify:
        r = client.post("/api/v1/auth/verify-email", json={"token": last_token(context, email)})
        assert r.status_code == 204, r.text
    r = client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    assert r.status_code == 200, r.text
    body = r.json()
    return LoggedInUser(client, email, uuid.UUID(body["user"]["id"]), body["csrf_token"])


CSV_BYTES = b"patient_id,age,outcome\n1,54,0\n2,61,1\n"


def create_and_upload(
    user: LoggedInUser, content: bytes = CSV_BYTES, filename: str = "cohort.csv"
) -> dict[str, object]:
    r = user.client.post(
        "/api/v1/datasets",
        headers=user.headers(),
        json={"name": "Cohort", "filename": filename, "size_bytes": len(content)},
    )
    assert r.status_code == 201, r.text
    ds_id = r.json()["dataset"]["id"]
    r = user.client.put(
        f"/api/v1/datasets/{ds_id}/content",
        content=content,
        headers=user.headers(**{"Content-Type": "application/octet-stream"}),
    )
    assert r.status_code == 202, r.text
    return r.json()  # type: ignore[no-any-return]

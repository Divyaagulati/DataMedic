"""User isolation matrix generated from the live route table.

Every /api/v1 route is exercised as (a) an anonymous client and (b) user B targeting user A's
resources. A new route with an unknown path parameter makes this test fail until the matrix
is extended, so no route can silently skip the isolation check.
"""

import re
import uuid

import pytest
from fastapi.testclient import TestClient

from datamedic.api.app import create_app
from datamedic.context import AppContext
from tests.conftest import (
    CSV_BYTES,
    ClientFactory,
    LoggedInUser,
    create_and_upload,
    signup_and_login,
)

pytestmark = pytest.mark.integration

PUBLIC_ROUTES = {
    ("POST", "/api/v1/auth/signup"),
    ("POST", "/api/v1/auth/verify-email"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/password-reset/request"),
    ("POST", "/api/v1/auth/password-reset/confirm"),
    ("GET", "/api/v1/meta/version"),
}
KNOWN_PARAMS = {"dataset_id", "job_id", "session_id"}

JSON_BODIES = {
    ("POST", "/api/v1/datasets"): {"name": "x", "filename": "x.csv", "size_bytes": 3},
}


HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _routes(ctx: AppContext) -> list[tuple[str, str, set[str]]]:
    """Every operation in the public OpenAPI schema under /api/v1."""
    schema = create_app(ctx=ctx).openapi()
    out = []
    for path, ops in schema["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        params = set(re.findall(r"{(\w+)}", path))
        for method in sorted(m for m in ops if m in HTTP_METHODS):
            out.append((method.upper(), path, params))
    return out


def _call(client: TestClient, method: str, path: str, headers: dict[str, str]):  # type: ignore[no-untyped-def]
    if path.endswith("/content"):
        return client.request(method, path, headers=headers, content=CSV_BYTES)
    body = JSON_BODIES.get((method, path))
    return client.request(method, path, headers=headers, json=body)


@pytest.fixture
def victim_and_attacker(
    make_client: ClientFactory, ctx: AppContext
) -> tuple[LoggedInUser, LoggedInUser, dict[str, str]]:
    a = signup_and_login(make_client(), ctx)
    b = signup_and_login(make_client(), ctx)
    # A's resources: a dataset that has not received content yet (so PUT content is a
    # meaningful target), an uploaded dataset with a job, and A's session.
    pending = a.client.post(
        "/api/v1/datasets",
        headers=a.headers(),
        json={"name": "pending", "filename": "p.csv", "size_bytes": len(CSV_BYTES)},
    ).json()
    uploaded = create_and_upload(a)
    session_id = a.client.get("/api/v1/auth/sessions").json()["items"][0]["id"]
    ids = {
        "dataset_id": pending["dataset"]["id"],
        "uploaded_dataset_id": uploaded["dataset"]["id"],  # type: ignore[index]
        "job_id": uploaded["job"]["id"],  # type: ignore[index]
        "session_id": session_id,
    }
    return a, b, ids


def test_every_route_is_covered(ctx: AppContext) -> None:
    for method, path, params in _routes(ctx):
        unknown = params - KNOWN_PARAMS
        assert not unknown, f"{method} {path}: extend the isolation matrix for {unknown}"


def test_anonymous_gets_401_on_every_protected_route(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    client = make_client()
    checked = 0
    for method, path, params in _routes(ctx):
        if (method, path) in PUBLIC_ROUTES:
            continue
        concrete = path
        for p in params:
            concrete = concrete.replace("{" + p + "}", str(uuid.uuid4()))
        r = _call(client, method, concrete, {})
        assert r.status_code == 401, f"{method} {path} -> {r.status_code}"
        checked += 1
    assert checked >= 10


def test_user_b_cannot_touch_user_a_resources(ctx: AppContext, victim_and_attacker) -> None:  # type: ignore[no-untyped-def]
    a, b, ids = victim_and_attacker
    checked = 0
    for method, path, params in _routes(ctx):
        if not params:
            continue
        concrete = path
        for p in params:
            concrete = concrete.replace("{" + p + "}", ids[p])
        r = _call(b.client, method, concrete, b.headers())
        assert r.status_code == 404, f"B {method} {path} -> {r.status_code}: {r.text}"
        assert r.json()["code"] == "NOT_FOUND"
        checked += 1
    assert checked >= 6

    # The uploaded dataset and job are equally invisible.
    assert b.client.get(f"/api/v1/datasets/{ids['uploaded_dataset_id']}").status_code == 404
    assert (
        b.client.delete(
            f"/api/v1/datasets/{ids['uploaded_dataset_id']}", headers=b.headers()
        ).status_code
        == 404
    )

    # Lists only ever show the caller's own resources, whatever the query string says.
    assert b.client.get("/api/v1/datasets").json()["items"] == []
    assert b.client.get(f"/api/v1/jobs?owner_id={a.user_id}").json()["items"] == []
    b_sessions = b.client.get("/api/v1/auth/sessions").json()["items"]
    assert ids["session_id"] not in {s["id"] for s in b_sessions}

    # A's data is intact after all of B's attempts.
    a_ds = a.client.get(f"/api/v1/datasets/{ids['dataset_id']}")
    assert a_ds.status_code == 200 and a_ds.json()["status"] == "upload_pending"
    assert a.client.get("/api/v1/auth/me").status_code == 200
    assert a.client.get(f"/api/v1/jobs/{ids['job_id']}").json()["status"] == "succeeded"

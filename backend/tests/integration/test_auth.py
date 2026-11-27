import pytest
from fastapi.testclient import TestClient

from datamedic.context import AppContext
from tests.conftest import PASSWORD, ClientFactory, last_token, outbox, signup_and_login

pytestmark = pytest.mark.integration


def test_signup_verify_login_me_logout(make_client: ClientFactory, ctx: AppContext) -> None:
    client = make_client()
    user = signup_and_login(client, ctx)
    me = client.get("/api/v1/auth/me")
    assert me.status_code == 200
    assert me.json()["user"]["email"] == user.email
    assert me.json()["user"]["email_verified"] is True

    assert client.post("/api/v1/auth/logout", headers=user.headers()).status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401


def test_session_cookie_flags(make_client: ClientFactory, ctx: AppContext) -> None:
    client = make_client()
    email = "cookie-check@example.org"
    client.post("/api/v1/auth/signup", json={"email": email, "password": PASSWORD})
    r = client.post("/api/v1/auth/login", json={"email": email, "password": PASSWORD})
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "path=/" in cookie
    # The raw session token never appears in the JSON body.
    assert client.cookies.get("dm_session") not in r.text


def test_signup_does_not_reveal_existing_accounts(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    client = make_client()
    email = "dupe@example.org"
    first = client.post("/api/v1/auth/signup", json={"email": email, "password": PASSWORD})
    sent = len([m for m in outbox(ctx).outbox if m.to == email])
    second = client.post("/api/v1/auth/signup", json={"email": email, "password": PASSWORD})
    assert first.status_code == second.status_code == 202
    assert first.json() == second.json()
    assert len([m for m in outbox(ctx).outbox if m.to == email]) == sent


def test_resend_verification_is_rate_limited_and_delivers_a_working_link(
    make_client: ClientFactory, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datetime import timedelta

    from datamedic.auth import service

    client = make_client()
    user = signup_and_login(client, ctx, verify=False)
    url = "/api/v1/auth/verify-email/resend"
    # Right after sign-up the cooldown applies (protects the mailbox and the SMTP quota).
    r = client.post(url, headers=user.headers())
    assert r.status_code == 429 and r.json()["code"] == "RATE_LIMITED"
    monkeypatch.setattr(service, "RESEND_COOLDOWN", timedelta(0))
    before = len([m for m in outbox(ctx).outbox if m.to == user.email])
    assert client.post(url, headers=user.headers()).status_code == 202
    assert len([m for m in outbox(ctx).outbox if m.to == user.email]) == before + 1
    r = client.post("/api/v1/auth/verify-email", json={"token": last_token(ctx, user.email)})
    assert r.status_code == 204
    assert client.get("/api/v1/auth/me").json()["user"]["email_verified"] is True
    # Once verified, resending sends nothing.
    assert client.post(url, headers=user.headers()).status_code == 202
    assert len([m for m in outbox(ctx).outbox if m.to == user.email]) == before + 1


def test_weak_password_rejected(make_client: ClientFactory) -> None:
    r = make_client().post(
        "/api/v1/auth/signup", json={"email": "weak@example.org", "password": "short"}
    )
    assert r.status_code == 422
    assert r.headers["content-type"].startswith("application/problem+json")
    assert r.json()["code"] == "VALIDATION_FAILED"


def test_wrong_password_is_generic_and_progressively_delayed(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    client = make_client()
    user = signup_and_login(client, ctx)
    for _ in range(ctx.settings.login_free_attempts):
        r = client.post(
            "/api/v1/auth/login", json={"email": user.email, "password": "nope-nope-nope"}
        )
        assert r.status_code == 401 and r.json()["code"] == "AUTH_INVALID"
    # Now even the correct password is delayed (no hard lockout, but brute force is slowed).
    r = client.post("/api/v1/auth/login", json={"email": user.email, "password": PASSWORD})
    assert r.status_code == 429
    assert r.json()["code"] == "RATE_LIMITED"
    assert int(r.headers["retry-after"]) >= 1


def test_unknown_email_login_is_generic(make_client: ClientFactory) -> None:
    r = make_client().post(
        "/api/v1/auth/login", json={"email": "nobody@example.org", "password": PASSWORD}
    )
    assert r.status_code == 401 and r.json()["code"] == "AUTH_INVALID"


def test_csrf_token_required_for_unsafe_methods(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    client = make_client()
    signup_and_login(client, ctx)
    r = client.post("/api/v1/auth/logout")
    assert r.status_code == 403 and r.json()["code"] == "CSRF_FAILED"
    r = client.post("/api/v1/auth/logout", headers={"X-CSRF-Token": "forged"})
    assert r.status_code == 403


def test_foreign_origin_rejected(make_client: ClientFactory, ctx: AppContext) -> None:
    client = make_client()
    user = signup_and_login(client, ctx)
    r = client.post("/api/v1/auth/logout", headers=user.headers(Origin="https://evil.example"))
    assert r.status_code == 403 and r.json()["code"] == "CSRF_FAILED"
    r = client.post("/api/v1/auth/logout", headers=user.headers(Origin="http://localhost:3000"))
    assert r.status_code == 204


def test_verification_token_single_use(make_client: ClientFactory, ctx: AppContext) -> None:
    client = make_client()
    email = "single-use@example.org"
    client.post("/api/v1/auth/signup", json={"email": email, "password": PASSWORD})
    token = outbox(ctx).outbox[-1].body.split("token=")[1].split()[0]
    assert client.post("/api/v1/auth/verify-email", json={"token": token}).status_code == 204
    assert client.post("/api/v1/auth/verify-email", json={"token": token}).status_code == 401


def test_sessions_list_and_revoke(make_client: ClientFactory, ctx: AppContext) -> None:
    first = make_client()
    user = signup_and_login(first, ctx)
    second = make_client()
    r = second.post("/api/v1/auth/login", json={"email": user.email, "password": PASSWORD})
    assert r.status_code == 200
    sessions = first.get("/api/v1/auth/sessions").json()["items"]
    assert len(sessions) == 2
    other = next(s for s in sessions if not s["current"])
    assert (
        first.delete(f"/api/v1/auth/sessions/{other['id']}", headers=user.headers()).status_code
        == 204
    )
    assert second.get("/api/v1/auth/me").status_code == 401
    assert first.get("/api/v1/auth/me").status_code == 200


def test_password_reset_revokes_sessions(make_client: ClientFactory, ctx: AppContext) -> None:
    client = make_client()
    user = signup_and_login(client, ctx)
    anon: TestClient = make_client()
    assert (
        anon.post("/api/v1/auth/password-reset/request", json={"email": user.email}).status_code
        == 202
    )
    token = outbox(ctx).outbox[-1].body.split("token=")[1].split()[0]
    new_password = "A-brand-new-password-42"
    assert (
        anon.post(
            "/api/v1/auth/password-reset/confirm",
            json={"token": token, "new_password": new_password},
        ).status_code
        == 204
    )
    assert client.get("/api/v1/auth/me").status_code == 401
    r = anon.post("/api/v1/auth/login", json={"email": user.email, "password": new_password})
    assert r.status_code == 200


def test_password_reset_for_unknown_email_looks_identical(make_client: ClientFactory) -> None:
    r = make_client().post(
        "/api/v1/auth/password-reset/request", json={"email": "ghost@example.org"}
    )
    assert r.status_code == 202

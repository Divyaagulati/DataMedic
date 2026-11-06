import json
import logging

import pytest
from pydantic import SecretStr, ValidationError

from datamedic.config import Settings
from datamedic.logging_setup import JsonFormatter, ScrubFilter
from datamedic.security.passwords import hash_password, password_problems, verify_password
from datamedic.security.tokens import new_token, token_digest


def test_password_hash_is_argon2id_and_verifies() -> None:
    h = hash_password("a-long-enough-password")
    assert h.startswith("$argon2id$")
    assert verify_password(h, "a-long-enough-password")
    assert not verify_password(h, "wrong-password-123")
    assert not verify_password(None, "anything-at-all")


def test_password_policy_is_length_based() -> None:
    assert password_problems("short")
    assert password_problems("x" * 12) == []


def test_tokens_are_random_and_digested() -> None:
    a, b = new_token(), new_token()
    assert a != b and len(a) >= 43
    assert len(token_digest(a)) == 32 and token_digest(a) != token_digest(b)


def _settings(**kw: object) -> Settings:
    base: dict[str, object] = {
        "database_url": SecretStr("postgresql+psycopg://x/y"),
        "maintenance_database_url": SecretStr("postgresql+psycopg://x/y"),
    }
    base.update(kw)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type, call-arg]


def test_production_guards() -> None:
    with pytest.raises(ValidationError):
        _settings(environment="production", cookie_secure=False)
    with pytest.raises(ValidationError):
        _settings(environment="production", jobs_eager=True)
    with pytest.raises(ValidationError):
        _settings(environment="production", email_backend="console")
    assert _settings(environment="production").session_cookie_name == "__Host-dm_session"
    assert _settings(cookie_secure=False).session_cookie_name == "dm_session"


def test_secrets_not_in_repr() -> None:
    s = _settings(database_url=SecretStr("postgresql+psycopg://u:hunter2@h/db"))
    assert "hunter2" not in repr(s)


def test_log_scrubber_redacts_dataset_content() -> None:
    record = logging.LogRecord("t", logging.INFO, __file__, 1, "processed", None, None)
    record.values = ["patient-123", "HIV positive"]
    record.columns = ["name", "diagnosis"]
    record.job_id = "abc"
    ScrubFilter().filter(record)
    out = json.loads(JsonFormatter().format(record))
    assert out["values"] == "[redacted]" and out["columns"] == "[redacted]"
    assert out["job_id"] == "abc"
    assert "patient-123" not in json.dumps(out)

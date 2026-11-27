"""The parse sandbox subprocess: success, expected failures, crashes and timeouts."""

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from pydantic import SecretStr

from datamedic.config import Settings
from datamedic.datasets import parsing
from datamedic.datasets.parsing import run_parse_sandbox


def settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "database_url": SecretStr("postgresql+psycopg://x/y"),
        "maintenance_database_url": SecretStr("postgresql+psycopg://x/y"),
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type, call-arg]


def run(tmp_path: Path, content: bytes, fmt: str = "csv", ext: str = "csv", **kw: object):  # type: ignore[no-untyped-def]
    source = tmp_path / "original"
    source.write_bytes(content)
    workdir = tmp_path / "work"
    workdir.mkdir()
    return run_parse_sandbox(settings(**kw), source, workdir, fmt, ext)


def test_success_writes_parquet_and_result(tmp_path: Path) -> None:
    outcome = run(tmp_path, b"a,b\n1,x\n2,y\n")
    assert outcome.ok and outcome.parquet_path is not None
    assert outcome.result is not None
    assert outcome.result["n_rows"] == 2 and outcome.result["n_cols"] == 2
    assert len(bytes.fromhex(outcome.result["content_hash"])) == 32
    frame = pd.read_parquet(outcome.parquet_path)
    assert list(frame.columns) == ["_dm_row_id", "a", "b"]
    assert "resource_limits_enforced" in outcome.result["report"]


def test_expected_failure_is_reported(tmp_path: Path) -> None:
    outcome = run(tmp_path, b"a,b\n1,2\n3,4,5\n")
    assert not outcome.ok and outcome.code == "PARSE_FAILED"
    assert outcome.message is not None and "Line 3" in outcome.message


def test_limits_are_passed_to_the_child(tmp_path: Path) -> None:
    outcome = run(tmp_path, b"a,b,c\n1,2,3\n", max_columns=2)
    assert not outcome.ok and outcome.code == "LIMIT_EXCEEDED"


def test_timeout_is_a_limit_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)

    monkeypatch.setattr(parsing.subprocess, "run", fake_run)
    outcome = run(tmp_path, b"a\n1\n", parse_timeout_seconds=1)
    assert (
        not outcome.ok and outcome.code == "LIMIT_EXCEEDED" and "1 seconds" in str(outcome.message)
    )


def test_crash_is_generic_and_hides_details(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=1)

    monkeypatch.setattr(parsing.subprocess, "run", fake_run)
    outcome = run(tmp_path, b"a\n1\n")
    assert not outcome.ok and outcome.code == "PARSE_FAILED"
    assert outcome.message == "The file could not be parsed."


def test_killed_child_is_a_limit_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args=[], returncode=-9)

    monkeypatch.setattr(parsing.subprocess, "run", fake_run)
    outcome = run(tmp_path, b"a\n1\n")
    assert not outcome.ok and outcome.code == "LIMIT_EXCEEDED"


def test_child_gets_a_minimal_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        seen.update(kwargs, cmd=cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=1)

    monkeypatch.setenv("DATAMEDIC_DATABASE_URL", "postgresql://secret@db/x")
    monkeypatch.setattr(parsing.subprocess, "run", fake_run)
    run(tmp_path, b"a\n1\n")
    env = seen["env"]
    assert isinstance(env, dict) and not any(k.startswith("DATAMEDIC") for k in env)
    assert env["ARROW_DEFAULT_MEMORY_POOL"] == "system"
    assert env["OPENBLAS_NUM_THREADS"] == "1"  # see parsing._child_env
    cmd = seen["cmd"]
    assert isinstance(cmd, list) and cmd[1] == "-I"  # isolated mode
    assert json.loads(cmd[-1])["max_columns"] == 2000
    assert seen["stdout"] == subprocess.DEVNULL and seen["stderr"] == subprocess.DEVNULL

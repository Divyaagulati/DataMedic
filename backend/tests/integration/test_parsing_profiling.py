"""End to end through the API: upload -> sandboxed parse -> version 0 -> preview -> profile.

Covers every supported format, malformed/empty/unsupported files, original immutability,
re-profiling, re-parsing and the version/preview/profile endpoints. Synthetic data only.
"""

import hashlib
import io
import json
import uuid
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select, text

from datamedic.context import AppContext, build_context
from datamedic.db.models import OriginalFile
from datamedic.db.session import bind_user, session_scope
from datamedic.jobs.publisher import CeleryJobPublisher
from datamedic.workers.celery_app import create_celery
from datamedic.workers.runner import run_job
from tests.conftest import (
    ClientFactory,
    DbUrls,
    LoggedInUser,
    create_and_upload,
    make_settings,
    signup_and_login,
)

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

COHORT = pd.DataFrame(
    {
        "patient_id": [f"P{i:03d}" for i in range(30)],
        "age": [40 + (i * 7) % 35 for i in range(30)],
        "sex": ["F" if i % 3 else "M" for i in range(30)],
        "creatinine": [round(0.6 + (i % 9) * 0.15, 2) if i % 10 else None for i in range(30)],
        "outcome": [1 if i % 4 == 0 else 0 for i in range(30)],
    }
)


def _xlsx(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()


def _parquet(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


FORMATS = {
    "cohort.csv": lambda: COHORT.to_csv(index=False).encode(),
    "cohort.tsv": lambda: COHORT.to_csv(index=False, sep="\t").encode(),
    "cohort.txt": lambda: COHORT.to_csv(index=False, sep=";").encode(),
    "cohort.xlsx": lambda: _xlsx(COHORT),
    "cohort.parquet": lambda: _parquet(COHORT),
    "cohort.json": lambda: COHORT.to_json(orient="records").encode(),
    "cohort.jsonl": lambda: COHORT.to_json(orient="records", lines=True).encode(),
}


def _dataset(user: LoggedInUser, ds_id: str) -> dict:  # type: ignore[type-arg]
    r = user.client.get(f"/api/v1/datasets/{ds_id}")
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def _job(user: LoggedInUser, job_id: str) -> dict:  # type: ignore[type-arg]
    return user.client.get(f"/api/v1/jobs/{job_id}").json()  # type: ignore[no-any-return]


@pytest.mark.parametrize("filename", list(FORMATS))
def test_every_format_parses_and_profiles(
    make_client: ClientFactory, ctx: AppContext, filename: str
) -> None:
    user = signup_and_login(make_client(), ctx)
    content = FORMATS[filename]()
    body = create_and_upload(user, content, filename)
    ds = _dataset(user, body["dataset"]["id"])  # type: ignore[index]
    assert ds["status"] == "ready", _job(user, body["job"]["id"])  # type: ignore[index]
    head = ds["head_version"]
    assert (head["version_number"], head["kind"]) == (0, "ingest")
    assert (head["n_rows"], head["n_cols"]) == (30, 5)

    version = user.client.get(f"/api/v1/versions/{head['id']}").json()
    assert version["is_head"] and version["parent_version_id"] is None
    types = {c["name"]: c["inferred_type"] for c in version["columns"]}
    assert types["age"] == "numeric" and types["outcome"] == "numeric"
    assert types["sex"] == "categorical" and types["patient_id"] == "identifier"
    assert len(version["content_hash"]) == 64 and len(version["file_sha256"]) == 64
    assert version["library_versions"]["pandas"].startswith("2.3")

    preview = user.client.get(f"/api/v1/versions/{head['id']}/preview")
    assert preview.status_code == 200
    assert preview.headers["cache-control"] == "no-store"
    pv = preview.json()
    assert len(pv["rows"]) == 20 and pv["truncated"] and pv["n_rows_total"] == 30
    assert [c["name"] for c in pv["columns"]] == list(COHORT.columns)  # no _dm_row_id
    assert pv["rows"][0][0] == "P000"

    state = user.client.get(f"/api/v1/versions/{head['id']}/profile").json()
    assert state["job"]["status"] == "succeeded"
    profile = state["profile"]
    assert profile["summary"]["n_rows"] == 30
    assert profile["summary"]["missing_cells"] == 3
    creat = next(c for c in profile["columns"] if c["name"] == "creatinine")
    assert creat["n_missing"] == 3 and creat["missing_pct"] == 10.0
    assert creat["numeric"]["median"] is not None and creat["histogram"]["kind"] == "numeric"
    outcome = next(c for c in profile["columns"] if c["name"] == "outcome")
    counts = {v["value"]: v["count"] for v in outcome["frequencies"]["values"]}
    assert counts == {0: 22, 1: 8} and outcome["frequencies"]["complete"]
    assert profile["correlations"]["status"] == "ok"
    assert profile["limitations"]


def test_xls_fixture(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, (FIXTURES / "cohort.xls").read_bytes(), "cohort.xls")
    ds = _dataset(user, body["dataset"]["id"])  # type: ignore[index]
    assert ds["status"] == "ready" and ds["detected_format"] == "xls"
    version = user.client.get(f"/api/v1/versions/{ds['head_version']['id']}").json()
    assert version["parse_options"]["sheet"] == "cohort"
    assert version["parse_report"]["sheets"] == ["cohort", "notes"]


def test_original_is_unchanged_after_parsing(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    content = FORMATS["cohort.csv"]()
    body = create_and_upload(user, content, "cohort.csv")
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    assert _dataset(user, ds_id)["status"] == "ready"
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        original = db.scalar(select(OriginalFile).where(OriginalFile.dataset_id == ds_id))
        assert original is not None
        assert original.sha256 == hashlib.sha256(content).digest()
        with ctx.storage.open_reader(original.object_key) as fh:
            assert fh.read() == content


@pytest.mark.parametrize(
    "filename,content,status,code",
    [
        ("bad.csv", b"a,b\n1,2\n3,4,5\n", "parse_failed", "PARSE_FAILED"),
        ("header.csv", b"a,b,c\n", "parse_failed", "PARSE_FAILED"),
        ("blank.csv", b"\n\n\n", "parse_failed", "PARSE_FAILED"),
        ("bad.json", b"[1, 2, 3]", "parse_failed", "PARSE_FAILED"),
        ("trunc.json", b'[{"a": 1}, {"a":', "parse_failed", "PARSE_FAILED"),
        ("bad.parquet", b"PAR1" + b"\x00" * 32 + b"PAR1", "parse_failed", "PARSE_FAILED"),
        ("fake.xlsx", b"a,b\n1,2\n", "rejected", "FORMAT_MISMATCH"),
        ("fake.parquet", b"a,b\n1,2\n", "rejected", "FORMAT_MISMATCH"),
    ],
)
def test_bad_files_fail_safely(
    make_client: ClientFactory,
    ctx: AppContext,
    filename: str,
    content: bytes,
    status: str,
    code: str,
) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, content, filename)
    ds = _dataset(user, body["dataset"]["id"])  # type: ignore[index]
    assert ds["status"] == status and ds["head_version"] is None
    job = _job(user, body["job"]["id"])  # type: ignore[index]
    assert job["status"] == "failed" and job["error"]["code"] == code
    assert job["error"]["message"]
    # No profile job is chained for a failed ingest.
    jobs = user.client.get("/api/v1/jobs", params={"dataset_id": ds["id"]}).json()["items"]
    assert [j["type"] for j in jobs] == ["ingest"]  # nothing chained after a failed ingest


def test_column_limit_rejects(make_client: ClientFactory, database: DbUrls, tmp_path: Path) -> None:
    context = build_context(make_settings(database, tmp_path, max_columns=3))
    try:
        user = signup_and_login(make_client(context), context)
        body = create_and_upload(user, b"a,b,c,d\n1,2,3,4\n", "wide.csv")
        ds = _dataset(user, body["dataset"]["id"])  # type: ignore[index]
        assert ds["status"] == "rejected"
        job = _job(user, body["job"]["id"])  # type: ignore[index]
        assert job["error"]["code"] == "LIMIT_EXCEEDED"
    finally:
        context.dispose()


def test_versions_list_and_profile_rerun(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, FORMATS["cohort.csv"](), "cohort.csv")
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    versions = user.client.get(f"/api/v1/datasets/{ds_id}/versions").json()["items"]
    assert [v["version_number"] for v in versions] == [0]
    vid = versions[0]["id"]
    first = user.client.get(f"/api/v1/versions/{vid}/profile").json()["profile"]

    r = user.client.post(f"/api/v1/versions/{vid}/profile", headers=user.headers())
    assert r.status_code == 202
    assert _job(user, r.json()["job"]["id"])["status"] == "succeeded"
    second = user.client.get(f"/api/v1/versions/{vid}/profile").json()["profile"]
    assert second["id"] != first["id"]  # a new, stored result; the old one is kept
    assert second["summary"] == first["summary"]  # deterministic
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        count = db.execute(
            text("SELECT count(*) FROM version_profiles WHERE dataset_version_id = :v"),
            {"v": vid},
        ).scalar_one()
    assert count == 2


def test_profile_conflict_while_active(
    make_client: ClientFactory, database: DbUrls, tmp_path: Path
) -> None:
    context = build_context(
        make_settings(database, tmp_path, jobs_eager=False, redis_url="memory://")
    )
    context.publisher = CeleryJobPublisher(create_celery(context.settings))
    try:
        user = signup_and_login(make_client(context), context)
        body = create_and_upload(user, FORMATS["cohort.csv"](), "cohort.csv")
        run_job(context, body["job"]["id"])  # type: ignore[index]  (worker runs the ingest)
        ds = _dataset(user, body["dataset"]["id"])  # type: ignore[index]
        vid = ds["head_version"]["id"]
        state = user.client.get(f"/api/v1/versions/{vid}/profile").json()
        assert state["profile"] is None and state["job"]["status"] == "queued"
        r = user.client.post(f"/api/v1/versions/{vid}/profile", headers=user.headers())
        assert r.status_code == 409 and r.json()["code"] == "CONFLICT"
        run_job(context, state["job"]["id"])
        state = user.client.get(f"/api/v1/versions/{vid}/profile").json()
        assert state["profile"] is not None and state["job"]["status"] == "succeeded"
    finally:
        context.dispose()


def test_parse_can_be_queued_for_unparsed_upload(
    make_client: ClientFactory, database: DbUrls, tmp_path: Path
) -> None:
    """Datasets uploaded before parsing existed stopped at 'validated'."""
    context = build_context(
        make_settings(database, tmp_path, jobs_eager=False, redis_url="memory://")
    )
    context.publisher = CeleryJobPublisher(create_celery(context.settings))
    try:
        user = signup_and_login(make_client(context), context)
        body = create_and_upload(user, FORMATS["cohort.csv"](), "cohort.csv")
        ds_id = body["dataset"]["id"]  # type: ignore[index]
        # Not yet parseable: the upload is still being processed.
        r = user.client.post(f"/api/v1/datasets/{ds_id}/parse", headers=user.headers())
        assert r.status_code == 409
        user.client.post(f"/api/v1/jobs/{body['job']['id']}/cancel", headers=user.headers())  # type: ignore[index]
        with session_scope(context.sessions) as db:  # simulate a Phase 1 'validated' dataset
            bind_user(db, user.user_id)
            db.execute(text("UPDATE datasets SET status = 'validated' WHERE id = :d"), {"d": ds_id})
        r = user.client.post(f"/api/v1/datasets/{ds_id}/parse", headers=user.headers())
        assert r.status_code == 202 and r.json()["dataset"]["status"] == "validating"
        run_job(context, r.json()["job"]["id"])
        assert _dataset(user, ds_id)["status"] == "ready"
        # Once a version exists, parsing cannot be queued again.
        r = user.client.post(f"/api/v1/datasets/{ds_id}/parse", headers=user.headers())
        assert r.status_code == 409
    finally:
        context.dispose()


def test_parse_failed_can_be_retried(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, b"a,b\n1,2\n3,4,5\n", "bad.csv")
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    assert _dataset(user, ds_id)["status"] == "parse_failed"
    r = user.client.post(f"/api/v1/datasets/{ds_id}/parse", headers=user.headers())
    assert r.status_code == 202
    assert _dataset(user, ds_id)["status"] == "parse_failed"  # same bytes, same result


def test_ingest_crash_marks_dataset_parse_failed(
    make_client: ClientFactory, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from datamedic.datasets import jobs as dataset_jobs

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("worker bug")

    monkeypatch.setattr(dataset_jobs, "_parse_into_version", boom)
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, FORMATS["cohort.csv"](), "cohort.csv")
    assert _dataset(user, body["dataset"]["id"])["status"] == "parse_failed"  # type: ignore[index]
    job = _job(user, body["job"]["id"])  # type: ignore[index]
    assert job["status"] == "failed" and job["error"]["code"] == "INTERNAL"


def test_unknown_version_is_404(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    for path in ("", "/preview", "/profile"):
        r = user.client.get(f"/api/v1/versions/{uuid.uuid4()}{path}")
        assert r.status_code == 404 and r.json()["code"] == "NOT_FOUND"


def test_profile_job_message_carries_only_the_job_id(
    make_client: ClientFactory, database: DbUrls, tmp_path: Path
) -> None:
    from typing import Any

    from celery.signals import before_task_publish

    context = build_context(
        make_settings(database, tmp_path, jobs_eager=False, redis_url="memory://")
    )
    context.publisher = CeleryJobPublisher(create_celery(context.settings))
    captured: list[Any] = []

    def capture(body: Any = None, **_: Any) -> None:
        captured.append(body)

    before_task_publish.connect(capture, weak=False)
    try:
        user = signup_and_login(make_client(context), context)
        body = create_and_upload(user, FORMATS["cohort.csv"](), "cohort.csv")
        run_job(context, body["job"]["id"])  # type: ignore[index]  (publishes the profile job)
    finally:
        before_task_publish.disconnect(capture)
        context.dispose()
    assert len(captured) == 3  # ingest + chained profile + chained diagnostics
    for message in captured:
        args, kwargs, _ = message
        assert args == [] or args == ()
        assert set(kwargs) == {"job_id"}
        assert "creatinine" not in json.dumps(message, default=str)


def test_delete_removes_versions_artifacts_profiles_and_objects(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    from datamedic.storage import dataset_prefix

    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, FORMATS["cohort.csv"](), "cohort.csv")
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    vid = _dataset(user, ds_id)["head_version"]["id"]
    user.client.post(f"/api/v1/versions/{vid}/profile", headers=user.headers())  # 2nd profile
    r = user.client.delete(f"/api/v1/datasets/{ds_id}", headers=user.headers())
    assert _job(user, r.json()["job"]["id"])["status"] == "succeeded"
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        for table in ("dataset_versions", "artifacts", "version_profiles", "original_files"):
            count = db.execute(
                text(f"SELECT count(*) FROM {table} WHERE dataset_id = :d"),  # noqa: S608
                {"d": ds_id},
            ).scalar_one()
            assert count == 0, table
    for area in ("raw", "versions", "artifacts"):
        assert ctx.storage.list_prefix(dataset_prefix(area, user.user_id, uuid.UUID(ds_id))) == []
    assert user.client.get(f"/api/v1/versions/{vid}").status_code == 404
    assert user.client.get(f"/api/v1/versions/{vid}/preview").status_code == 404


def test_parse_temp_files_are_removed(make_client: ClientFactory, ctx: AppContext) -> None:
    import tempfile

    def leftovers() -> set[str]:
        return {p.name for p in Path(tempfile.gettempdir()).glob("dm-parse-*")}

    before = leftovers()
    user = signup_and_login(make_client(), ctx)
    create_and_upload(user, FORMATS["cohort.csv"](), "cohort.csv")  # success path
    create_and_upload(user, b"a,b\n1,2\n3,4,5\n", "bad.csv")  # failure path
    assert leftovers() == before


def test_new_write_endpoints_require_csrf(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, FORMATS["cohort.csv"](), "cohort.csv")
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    vid = _dataset(user, ds_id)["head_version"]["id"]
    for path in (f"/api/v1/versions/{vid}/profile", f"/api/v1/datasets/{ds_id}/parse"):
        r = user.client.post(path)  # session cookie but no X-CSRF-Token
        assert r.status_code == 403 and r.json()["code"] == "CSRF_FAILED", path
        r = user.client.post(path, headers={"X-CSRF-Token": "forged"})
        assert r.status_code == 403, path


def test_hostile_column_names_and_values_are_returned_as_data(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    hostile = '<img src=x onerror=alert(1)>,=cmd|"/c calc"!A1,"a,b"\n<b>x</b>,=1+1,ok\n'
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, hostile.encode(), "hostile.csv")
    ds = _dataset(user, body["dataset"]["id"])  # type: ignore[index]
    assert ds["status"] == "ready"
    preview = user.client.get(f"/api/v1/versions/{ds['head_version']['id']}/preview").json()
    assert [c["name"] for c in preview["columns"]] == [
        "<img src=x onerror=alert(1)>",
        '=cmd|"/c calc"!A1',
        "a,b",
    ]
    assert preview["rows"][0] == ["<b>x</b>", "=1+1", "ok"]  # stored verbatim, never executed


def test_larger_file_end_to_end(make_client: ClientFactory, ctx: AppContext) -> None:
    import numpy as np

    rng = np.random.default_rng(1)
    n = 60_000
    frame = pd.DataFrame(
        {
            "id": np.arange(n),
            "x": rng.normal(size=n).round(4),
            "y": rng.integers(0, 2, size=n),
            "site": rng.choice(["A", "B", "C"], size=n),
        }
    )
    content = frame.to_csv(index=False).encode()
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, content, "big.csv")
    ds = _dataset(user, body["dataset"]["id"])  # type: ignore[index]
    assert ds["status"] == "ready" and ds["head_version"]["n_rows"] == n
    profile = user.client.get(f"/api/v1/versions/{ds['head_version']['id']}/profile").json()
    y = next(c for c in profile["profile"]["columns"] if c["name"] == "y")
    assert sum(v["count"] for v in y["frequencies"]["values"]) == n
    assert next(c for c in profile["profile"]["columns"] if c["name"] == "id")["inferred_type"] == (
        "identifier"
    )

import io
import uuid
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from datamedic.context import AppContext, build_context
from datamedic.db.models import Job, OriginalFile
from datamedic.db.session import bind_user, session_scope
from datamedic.jobs.publisher import CeleryJobPublisher
from datamedic.storage import dataset_prefix
from datamedic.workers.celery_app import create_celery
from datamedic.workers.runner import run_job
from tests.conftest import (
    CSV_BYTES,
    ClientFactory,
    DbUrls,
    create_and_upload,
    make_settings,
    signup_and_login,
)

pytestmark = pytest.mark.integration


def test_upload_requires_verified_email(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx, verify=False)
    r = user.client.post(
        "/api/v1/datasets",
        headers=user.headers(),
        json={"name": "x", "filename": "x.csv", "size_bytes": 10},
    )
    assert r.status_code == 403 and r.json()["code"] == "FORBIDDEN"


def test_csv_upload_is_stored_and_parsed(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user)
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    # Eager job execution in tests: the ingest job (and the chained profile job) already ran.
    ds = user.client.get(f"/api/v1/datasets/{ds_id}").json()
    assert ds["status"] == "ready" and ds["detected_format"] == "csv"
    assert ds["head_version"]["version_number"] == 0
    assert (ds["head_version"]["n_rows"], ds["head_version"]["n_cols"]) == (2, 3)
    job = user.client.get(f"/api/v1/jobs/{body['job']['id']}").json()  # type: ignore[index]
    assert job["status"] == "succeeded" and job["poll_after_ms"] == 0
    # The original bytes are stored unchanged.
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        original = db.scalar(select(OriginalFile).where(OriginalFile.dataset_id == ds_id))
        assert original is not None
        with ctx.storage.open_reader(original.object_key) as fh:
            assert fh.read() == CSV_BYTES


def test_macro_workbook_is_rejected(make_client: ClientFactory, ctx: AppContext) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("[Content_Types].xml", b"<Types/>")
        zf.writestr("xl/workbook.xml", b"<workbook/>")
        zf.writestr("xl/vbaProject.bin", b"\x00" * 64)
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, buf.getvalue(), "report.xlsx")
    ds = user.client.get(f"/api/v1/datasets/{body['dataset']['id']}").json()  # type: ignore[index]
    assert ds["status"] == "rejected"
    job = user.client.get(f"/api/v1/jobs/{body['job']['id']}").json()  # type: ignore[index]
    assert job["status"] == "failed" and job["error"]["code"] == "MALICIOUS_FILE"


def test_extension_content_mismatch_is_rejected(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, b"MZ\x90\x00" + b"\x00" * 64, "data.csv")
    job = user.client.get(f"/api/v1/jobs/{body['job']['id']}").json()  # type: ignore[index]
    assert job["error"]["code"] == "FORMAT_MISMATCH"


def test_upload_validation_errors(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    c, h = user.client, user.headers()
    r = c.post(
        "/api/v1/datasets", headers=h, json={"name": "x", "filename": "virus.exe", "size_bytes": 10}
    )
    assert r.status_code == 415 and r.json()["code"] == "UNSUPPORTED_FORMAT"
    r = c.post(
        "/api/v1/datasets",
        headers=h,
        json={"name": "x", "filename": "big.csv", "size_bytes": ctx.settings.max_upload_bytes + 1},
    )
    assert r.status_code == 413 and r.json()["code"] == "UPLOAD_TOO_LARGE"

    ds_id = c.post(
        "/api/v1/datasets", headers=h, json={"name": "x", "filename": "a.csv", "size_bytes": 10}
    ).json()["dataset"]["id"]
    # Content-Length must match the declared size.
    r = c.put(f"/api/v1/datasets/{ds_id}/content", headers=h, content=b"a,b\n1,2\n")
    assert r.status_code == 422 and r.json()["code"] == "VALIDATION_FAILED"
    # Content-Length is required (a streamed body without it is refused).
    r = c.put(f"/api/v1/datasets/{ds_id}/content", headers=h, content=iter([b"a,b\n", b"1,22\n"]))
    assert r.status_code == 411 and r.json()["code"] == "LENGTH_REQUIRED"
    # The failed attempts left the dataset uploadable.
    r = c.put(f"/api/v1/datasets/{ds_id}/content", headers=h, content=b"a,b\n1,222\n")
    assert r.status_code == 202
    # A second upload is refused: originals are immutable.
    r = c.put(f"/api/v1/datasets/{ds_id}/content", headers=h, content=b"a,b\n1,333\n")
    assert r.status_code == 409


def test_client_cannot_choose_owner(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    r = user.client.post(
        "/api/v1/datasets",
        headers=user.headers(),
        json={"name": "x", "filename": "a.csv", "size_bytes": 3, "owner_id": str(uuid.uuid4())},
    )
    assert r.status_code == 422


def test_dataset_quota(make_client: ClientFactory, database: DbUrls, tmp_path: Path) -> None:
    context = build_context(make_settings(database, tmp_path, max_datasets_per_user=2))
    try:
        user = signup_and_login(make_client(context), context)
        for _ in range(2):
            assert (
                user.client.post(
                    "/api/v1/datasets",
                    headers=user.headers(),
                    json={"name": "x", "filename": "a.csv", "size_bytes": 3},
                ).status_code
                == 201
            )
        r = user.client.post(
            "/api/v1/datasets",
            headers=user.headers(),
            json={"name": "x", "filename": "a.csv", "size_bytes": 3},
        )
        assert r.status_code == 409 and r.json()["code"] == "QUOTA_EXCEEDED"
    finally:
        context.dispose()


def test_delete_purges_storage_and_rows(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user)
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    r = user.client.delete(f"/api/v1/datasets/{ds_id}", headers=user.headers())
    assert r.status_code == 202
    purge = user.client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()
    assert purge["status"] == "succeeded"
    # raw original + version-0 Parquet + preview artifact
    assert purge["result"] == {"objects_deleted": 3, "verified_empty": True}
    assert user.client.get(f"/api/v1/datasets/{ds_id}").status_code == 404
    for area in ("raw", "versions", "artifacts"):
        assert ctx.storage.list_prefix(dataset_prefix(area, user.user_id, uuid.UUID(ds_id))) == []


def test_job_cancel_and_listing(
    make_client: ClientFactory, database: DbUrls, tmp_path: Path
) -> None:
    # Broker-backed publisher (in-memory transport): the job stays queued until a worker runs.
    context = build_context(
        make_settings(database, tmp_path, jobs_eager=False, redis_url="memory://")
    )
    context.publisher = CeleryJobPublisher(create_celery(context.settings))
    try:
        user = signup_and_login(make_client(context), context)
        body = create_and_upload(user)
        job_id = body["job"]["id"]  # type: ignore[index]
        assert body["job"]["status"] == "queued"  # type: ignore[index]
        listed = user.client.get("/api/v1/jobs").json()["items"]
        assert [j["id"] for j in listed] == [job_id]
        r = user.client.post(f"/api/v1/jobs/{job_id}/cancel", headers=user.headers())
        assert r.status_code == 202 and r.json()["status"] == "cancelled"
        # A worker receiving the (stale) message does nothing: claim_job refuses it.
        run_job(context, job_id)
        assert user.client.get(f"/api/v1/jobs/{job_id}").json()["status"] == "cancelled"
        r = user.client.post(f"/api/v1/jobs/{job_id}/cancel", headers=user.headers())
        assert r.status_code == 409
    finally:
        context.dispose()


def test_worker_processes_broker_job(
    make_client: ClientFactory, database: DbUrls, tmp_path: Path
) -> None:
    context = build_context(
        make_settings(database, tmp_path, jobs_eager=False, redis_url="memory://")
    )
    context.publisher = CeleryJobPublisher(create_celery(context.settings))
    try:
        user = signup_and_login(make_client(context), context)
        body = create_and_upload(user)
        job_id = body["job"]["id"]  # type: ignore[index]
        run_job(context, job_id)  # what the Celery task does with {"job_id": ...}
        run_job(context, job_id)  # duplicate delivery is a no-op
        with session_scope(context.sessions) as db:
            bind_user(db, user.user_id)
            job = db.get(Job, uuid.UUID(job_id))
            assert job is not None and job.status == "succeeded" and job.attempt == 1
    finally:
        context.dispose()


def test_job_admission_limit(make_client: ClientFactory, database: DbUrls, tmp_path: Path) -> None:
    context = build_context(
        make_settings(
            database,
            tmp_path,
            jobs_eager=False,
            redis_url="memory://",
            max_queued_jobs=1,
            max_running_jobs=0,
        )
    )
    context.publisher = CeleryJobPublisher(create_celery(context.settings))
    try:
        user = signup_and_login(make_client(context), context)
        create_and_upload(user)  # 1 active job now
        r = user.client.post(
            "/api/v1/datasets",
            headers=user.headers(),
            json={"name": "y", "filename": "b.csv", "size_bytes": len(CSV_BYTES)},
        )
        ds_id = r.json()["dataset"]["id"]
        r = user.client.put(
            f"/api/v1/datasets/{ds_id}/content", headers=user.headers(), content=CSV_BYTES
        )
        assert r.status_code == 429 and r.json()["code"] == "JOB_LIMIT_EXCEEDED"
        # Deletion is never blocked by the limit.
        assert (
            user.client.delete(f"/api/v1/datasets/{ds_id}", headers=user.headers()).status_code
            == 202
        )
    finally:
        context.dispose()


def test_unauthenticated_upload_is_refused(make_client: ClientFactory) -> None:
    client: TestClient = make_client()
    r = client.put(f"/api/v1/datasets/{uuid.uuid4()}/content", content=b"abc")
    assert r.status_code == 401

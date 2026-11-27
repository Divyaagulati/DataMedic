"""MVP completion (Phase 6/7): biomedical workspace, Data Passport, Copilot and exports through
the API and the worker. Synthetic data only."""

from typing import Any

import pytest

from datamedic.context import AppContext
from tests.conftest import ClientFactory, LoggedInUser, create_and_upload, signup_and_login
from tests.unit.test_mvp_reports import cohort

pytestmark = pytest.mark.integration


def _csv() -> bytes:
    df = cohort().drop(columns=["_dm_row_id", "sex"])
    df["code"] = ["k1", "k2", "k3"] * (len(df) // 3)
    df["note"] = ["=1+1", "ok", "-x"] * (len(df) // 3)  # formula-injection probe
    return df.to_csv(index=False).encode()


@pytest.fixture
def bio(make_client: ClientFactory, ctx: AppContext) -> tuple[LoggedInUser, str, str]:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, _csv(), "bio.csv")
    ds_id = str(body["dataset"]["id"])  # type: ignore[index]
    r = user.client.patch(
        f"/api/v1/datasets/{ds_id}", headers=user.headers(), json={"domain": "biomedical"}
    )
    assert r.status_code == 200, r.text
    vid = user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"]
    roles = [
        {"column": "y", "role": "target", "task_type": "binary"},
        {"column": "patient", "role": "bio.patient_id"},
        {"column": "batch", "role": "bio.batch"},
        {"column": "code", "role": "sensitive"},
    ]
    r = user.client.put(
        f"/api/v1/versions/{vid}/roles", headers=user.headers(), json={"roles": roles}
    )
    assert r.status_code == 200, r.text
    return user, ds_id, vid


def test_biomedical_workspace(bio) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = bio
    empty = user.client.get(f"/api/v1/versions/{vid}/biomedical").json()
    assert empty["summary"] is None
    r = user.client.post(f"/api/v1/versions/{vid}/biomedical", headers=user.headers())
    assert r.status_code == 202 and r.json()["job"]["status"] == "succeeded", r.text
    w = user.client.get(f"/api/v1/versions/{vid}/biomedical").json()
    s = w["summary"]
    assert s["cohort"]["patients"]["unique"] == 120 and s["comparisons"]["unit"] == "patient"
    assert "code" in s["roles"]["sensitive"] and "Research use only" in s["disclaimer"]
    assert {c["kind"] for c in s["confounding"]} >= {"batch_outcome_association"}
    assert w["stale"] is False and w["model_reliability"][0]["status"] == "not_assessed"
    # Diagnostics are unaffected by biomedical summaries (kind filter).
    latest = user.client.get(f"/api/v1/versions/{vid}/analyses/latest").json()["run"]
    assert latest is None or latest["id"] != w["summary_created_at"]


def test_passport_and_copilot(bio) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = bio
    r = user.client.get(f"/api/v1/versions/{vid}/passport")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store"
    p = r.json()
    # bio.patient_id is a sensitive role itself.
    assert p["version"]["content_hash"] and p["protected_columns"] == ["code", "patient"]
    assert p["models"]["status"] == "not_available" and "score" not in p
    assert p["history"]["versions"][0]["version_number"] == 0
    d = user.client.get(f"/api/v1/versions/{vid}/passport?download=true")
    assert d.headers["content-disposition"].startswith('attachment; filename="Cohort_v0_passport')
    a = user.client.post(
        f"/api/v1/versions/{vid}/copilot",
        headers=user.headers(),
        json={"question": "What should I do next?"},
    ).json()
    assert a["topic"] == "next" and a["assistant"]["kind"] == "built-in rule-based assistant"
    assert any(ref["href"].startswith(f"/datasets/{ds_id}") for ref in a["references"])
    bad = user.client.post(
        f"/api/v1/versions/{vid}/copilot", headers=user.headers(), json={"question": ""}
    )
    assert bad.status_code == 422


def _export(user: LoggedInUser, vid: str, **body: Any) -> Any:
    return user.client.post(f"/api/v1/versions/{vid}/exports", headers=user.headers(), json=body)


def test_csv_export_rules(bio, make_client: ClientFactory, ctx: AppContext) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = bio
    before = user.client.get(f"/api/v1/versions/{vid}").json()
    assert _export(user, vid).status_code == 422  # acknowledgement required
    assert _export(user, vid, acknowledged=True, include_sensitive=True).status_code == 422
    r = _export(user, vid, acknowledged=True)
    assert r.status_code == 202 and r.json()["excluded_sensitive_columns"] == ["code", "patient"]
    job = r.json()["job"]
    assert job["status"] == "succeeded", job
    dl = user.client.get(f"/api/v1/exports/{job['id']}/download")
    assert dl.status_code == 200 and dl.headers["cache-control"] == "no-store"
    assert dl.headers["content-disposition"] == 'attachment; filename="Cohort_v0.csv"'
    text = dl.text
    header = text.splitlines()[0].split(",")
    assert not {"code", "patient", "_dm_row_id"} & set(header)
    assert "'=1+1" in text and "\n=1+1" not in text
    full = _export(user, vid, acknowledged=True, include_sensitive=True, acknowledge_sensitive=True)
    full_text = user.client.get(f"/api/v1/exports/{full.json()['job']['id']}/download").text
    assert "code" in full_text.splitlines()[0]
    # Another user gets 404; the source version is unchanged.
    other = signup_and_login(make_client(), ctx)
    assert other.client.get(f"/api/v1/exports/{job['id']}/download").status_code == 404
    after = user.client.get(f"/api/v1/versions/{vid}").json()
    assert after["content_hash"] == before["content_hash"]
    assert len(user.client.get(f"/api/v1/datasets/{ds_id}/versions").json()["items"]) == 1

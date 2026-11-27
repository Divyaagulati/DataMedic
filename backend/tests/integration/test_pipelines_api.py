"""Part E end to end through the API and the worker: pipelines, impact previews, hash-bound
approvals, commits, reverts, comparisons, generated code, crash safety, concurrency, quotas
and idempotency. Synthetic data only."""

import hashlib
import io
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from datamedic.context import AppContext
from datamedic.db.models import (
    ColumnRole,
    Dataset,
    DatasetVersion,
    Job,
    OriginalFile,
    TransformationRun,
)
from datamedic.db.session import bind_user, session_scope
from datamedic.engine.hashing import content_hash
from datamedic.storage import dataset_prefix
from datamedic.workers.maintenance import reconcile_orphans
from datamedic.workers.runner import run_job
from tests.conftest import ClientFactory, LoggedInUser, create_and_upload, signup_and_login
from tests.integration.test_experiments_api import Recorder

pytestmark = pytest.mark.integration


def cohort_csv(n: int = 240, seed: int = 3) -> bytes:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": rng.normal(10, 2, n).round(3),
            "b": rng.normal(size=n).round(3),
            "lab": np.exp(rng.normal(1, 0.7, n)).round(3),
            "code": rng.choice(["k1", "k2", "k3"], n),
            "y": (rng.random(n) < 0.3).astype(int),
        }
    )
    df.loc[rng.random(n) < 0.25, "b"] = None
    df.loc[rng.random(n) < 0.1, "code"] = None
    df.loc[[3, 7], "a"] = [80.0, -40.0]
    return df.to_csv(index=False).encode()


STEPS = [
    {"op": "missing.indicator", "columns": ["b"]},
    {"op": "impute.median", "columns": ["b"]},
    {"op": "outliers.cap", "columns": ["a"], "params": {"method": "iqr", "iqr_multiplier": 3.0}},
    {"op": "transform.log", "columns": ["lab"]},
    {"op": "scale.standard", "columns": ["a", "b"]},
    {"op": "imbalance.class_weight"},
]


def _setup(user: LoggedInUser) -> tuple[str, str]:
    body = create_and_upload(user, cohort_csv(), "cohort.csv")
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    vid = user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"]
    r = user.client.put(
        f"/api/v1/versions/{vid}/roles",
        headers=user.headers(),
        json={
            "roles": [
                {"column": "y", "role": "target", "task_type": "binary"},
                {"column": "code", "role": "sensitive"},
            ]
        },
    )
    assert r.status_code == 200, r.text
    return ds_id, vid


def _pipeline(user: LoggedInUser, ds_id: str, steps: list[dict] = STEPS) -> dict:  # type: ignore[type-arg]
    r = user.client.post(
        f"/api/v1/datasets/{ds_id}/pipelines",
        headers=user.headers(),
        json={"name": "Clean", "steps": steps},
    )
    assert r.status_code == 201, r.text
    return r.json()  # type: ignore[no-any-return]


def _get(user: LoggedInUser, pid: str) -> dict:  # type: ignore[type-arg]
    return user.client.get(f"/api/v1/pipelines/{pid}").json()  # type: ignore[no-any-return]


def _preview(user: LoggedInUser, pid: str) -> dict:  # type: ignore[type-arg]
    r = user.client.post(f"/api/v1/pipelines/{pid}/impact-preview", headers=user.headers())
    assert r.status_code == 202, r.text
    assert r.json()["job"]["status"] == "succeeded", r.json()["job"]
    return user.client.get(f"/api/v1/pipelines/{pid}/impact-preview").json()  # type: ignore[no-any-return]


def _approve_all(user: LoggedInUser, pid: str) -> dict:  # type: ignore[type-arg]
    p = _get(user, pid)
    for s in p["steps"]:
        if s["risk"] == "high":
            r = user.client.post(
                f"/api/v1/pipelines/{pid}/steps/{s['id']}/approve",
                headers=user.headers(),
                json={
                    "step_hash": s["step_hash"],
                    "spec_hash": p["spec_hash"],
                    "acknowledge_destructive": s["destructive"],
                },
            )
            assert r.status_code == 201, r.text
    if any(s["risk"] == "medium" for s in p["steps"]):
        r = user.client.post(
            f"/api/v1/pipelines/{pid}/confirm",
            headers=user.headers(),
            json={"spec_hash": p["spec_hash"]},
        )
        assert r.status_code == 201, r.text
    return _get(user, pid)


def _commit(user: LoggedInUser, pid: str, spec_hash: str, key: str | None = None) -> Any:
    return user.client.post(
        f"/api/v1/pipelines/{pid}/commit",
        json={"spec_hash": spec_hash},
        headers=user.headers(**{"Idempotency-Key": key or str(uuid.uuid4())}),
    )


def _versions(user: LoggedInUser, ds_id: str) -> list[dict]:  # type: ignore[type-arg]
    return user.client.get(f"/api/v1/datasets/{ds_id}/versions").json()["items"]  # type: ignore[no-any-return]


def _objects(ctx: AppContext, user: LoggedInUser, ds_id: str) -> dict[str, str]:
    out = {}
    for area in ("raw", "versions"):
        for key in ctx.storage.list_prefix(dataset_prefix(area, user.user_id, uuid.UUID(ds_id))):
            with ctx.storage.open_reader(key) as fh:
                out[key] = hashlib.sha256(fh.read()).hexdigest()
    return out


def _frame(ctx: AppContext, user: LoggedInUser, vid: str) -> pd.DataFrame:
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        v = db.get(DatasetVersion, uuid.UUID(vid))
        assert v is not None
        key = v.object_key
    with ctx.storage.open_reader(key) as fh:
        return pd.read_parquet(io.BytesIO(fh.read()))


@pytest.fixture
def ready(make_client: ClientFactory, ctx: AppContext) -> tuple[LoggedInUser, str, str]:
    user = signup_and_login(make_client(), ctx)
    ds_id, vid = _setup(user)
    return user, ds_id, vid


# ---- Approvals and commit ---------------------------------------------------------------------


def test_full_commit_flow(ready, ctx: AppContext, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, v0 = ready
    before = _objects(ctx, user, ds_id)
    p = _pipeline(user, ds_id)
    assert p["state"] == "validated" and p["validation"]["model_steps"] == 2
    stages = [(s["op"], s["stage"]) for s in p["steps"]]
    assert stages[-2:] == [("scale.standard", "model"), ("imbalance.class_weight", "model")]
    assert stages[1] == ("impute.median", "data")
    r = _commit(user, p["id"], p["spec_hash"])
    assert r.status_code == 409 and r.json()["code"] == "APPROVAL_REQUIRED"
    preview = _preview(user, p["id"])
    assert preview["current"] and preview["summary"]["spec_hash"] == p["spec_hash"]
    steps = {s["op"]: s for s in preview["summary"]["steps"]}
    assert steps["impute.median"]["rows_affected"] > 0
    assert steps["outliers.cap"]["columns"][0]["observed_overwritten"] >= 2
    assert [m["op"] for m in preview["summary"]["model_steps"]] == [
        "scale.standard",
        "imbalance.class_weight",
    ]
    r = _commit(user, p["id"], p["spec_hash"])
    assert r.status_code == 409 and r.json()["code"] == "APPROVAL_REQUIRED"
    cap = next(s for s in _get(user, p["id"])["steps"] if s["op"] == "outliers.cap")
    r = user.client.post(
        f"/api/v1/pipelines/{p['id']}/steps/{cap['id']}/approve",
        headers=user.headers(),
        json={"step_hash": cap["step_hash"], "spec_hash": p["spec_hash"]},
    )
    assert r.status_code == 422 and r.json()["errors"][0]["code"] == "ACKNOWLEDGEMENT_REQUIRED"
    approved = _approve_all(user, p["id"])
    assert approved["readiness"]["ready"], approved["readiness"]
    statements = [a["statement"] for a in approved["approvals"]]
    assert any("Destructive" in s for s in statements)
    key = str(uuid.uuid4())
    r = _commit(user, p["id"], p["spec_hash"], key)
    assert r.status_code == 202, r.text
    assert r.json()["job"]["status"] == "succeeded", r.json()["job"]
    again = _commit(user, p["id"], p["spec_hash"], key)
    assert again.status_code == 200 and again.json()["job"]["id"] == r.json()["job"]["id"]

    versions = _versions(user, ds_id)
    assert [(v["version_number"], v["kind"]) for v in versions] == [(0, "ingest"), (1, "transform")]
    v1 = versions[1]
    assert v1["parent_version_id"] == v0 and v1["fitted_on_full_data"] and v1["has_model_spec"]
    assert user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"] == v1["id"]
    run = user.client.get(f"/api/v1/transformation-runs/{v1['transformation_run_id']}").json()
    assert run["source_version_id"] == v0 and run["result_version_id"] == v1["id"]
    assert [m["op"] for m in run["model_pipeline_spec"]] == [
        "scale.standard",
        "imbalance.class_weight",
    ]
    assert len(run["approvals"]) == 2 and len(run["step_reports"]) == 4
    assert {(r["column"], r["role"]) for r in run["roles_report"]["kept"]} == {
        ("y", "target"),
        ("code", "sensitive"),
    }
    detail = user.client.get(f"/api/v1/versions/{v1['id']}").json()
    assert detail["model_pipeline_spec"][0]["op"] == "scale.standard"

    old, new = _frame(ctx, user, v0), _frame(ctx, user, v1["id"])
    assert new["_dm_row_id"].tolist() == old["_dm_row_id"].tolist()  # no added/removed rows
    assert "_dm_sample_weight" not in new.columns  # model-stage step not materialised
    assert new["b"].notna().all() and "b__missing" in new.columns
    np.testing.assert_allclose(new["a"].std() < old["a"].std(), True)
    assert (new["a"] - old["a"]).abs().max() > 0  # capped
    # Original and version 0 are byte-identical; nothing was overwritten.
    after = _objects(ctx, user, ds_id)
    assert all(after[k] == h for k, h in before.items())
    assert len(after) == len(before) + 1
    # Follow-up profiling and diagnostics ran for the new version; roles were copied.
    jobs = user.client.get(f"/api/v1/jobs?dataset_id={ds_id}&limit=100").json()["items"]
    follow = [j for j in jobs if j["type"] in ("profile", "analysis")]
    assert len(follow) == 4 and all(j["status"] == "succeeded" for j in follow)  # v0 + v1
    assert user.client.get(f"/api/v1/versions/{v1['id']}/profile").json()["profile"]
    latest = user.client.get(f"/api/v1/versions/{v1['id']}/analyses/latest").json()
    assert latest["run"] is not None and latest["run"]["stale"] is False
    # The committed pipeline is frozen.
    r = user.client.delete(f"/api/v1/pipelines/{p['id']}/steps/{cap['id']}", headers=user.headers())
    assert r.status_code == 409

    # Generated pipeline.py reproduces version 1 from the original file exactly.
    code = user.client.get(f"/api/v1/versions/{v1['id']}/code")
    assert code.status_code == 200 and code.headers["cache-control"] == "no-store"
    assert "pickle" not in code.text and str(ctx.settings.storage_root) not in code.text
    digest = _run_code(ctx, user, ds_id, code.text, tmp_path)
    assert digest == _content_hash(ctx, user, v1["id"])


def _content_hash(ctx: AppContext, user: LoggedInUser, vid: str) -> str:
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        v = db.get(DatasetVersion, uuid.UUID(vid))
        assert v is not None
        return v.content_hash.hex()


def _run_code(ctx: AppContext, user: LoggedInUser, ds_id: str, code: str, tmp: Path) -> str:
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        original = db.scalar(
            select(OriginalFile).where(OriginalFile.dataset_id == uuid.UUID(ds_id))
        )
        assert original is not None
        key = original.object_key
    src = tmp / "original.csv"
    with ctx.storage.open_reader(key) as fh:
        src.write_bytes(fh.read())
    (tmp / "pipeline.py").write_text(code, encoding="utf-8")
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-I", "pipeline.py", "original.csv", "out.parquet"],
        cwd=tmp,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert content_hash(pd.read_parquet(tmp / "out.parquet")).hex() == proc.stdout.strip()
    return proc.stdout.strip()


@pytest.mark.parametrize("change", ["edit", "delete", "reorder", "stage"])
def test_any_change_after_approval_makes_it_stale(ready, change: str) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, _ = ready
    p = _pipeline(user, ds_id, STEPS[:4])
    _preview(user, p["id"])
    approved = _approve_all(user, p["id"])
    assert approved["readiness"]["ready"]
    steps = approved["steps"]
    if change == "edit":
        r = user.client.patch(
            f"/api/v1/pipelines/{p['id']}/steps/{steps[2]['id']}",
            headers=user.headers(),
            json={"params": {"method": "iqr", "iqr_multiplier": 2.5}},
        )
    elif change == "delete":
        r = user.client.delete(
            f"/api/v1/pipelines/{p['id']}/steps/{steps[3]['id']}", headers=user.headers()
        )
    elif change == "reorder":
        r = user.client.patch(
            f"/api/v1/pipelines/{p['id']}/steps/{steps[3]['id']}",
            headers=user.headers(),
            json={"position": 2},
        )
    else:
        r = user.client.patch(
            f"/api/v1/pipelines/{p['id']}/steps/{steps[3]['id']}",
            headers=user.headers(),
            json={"stage": "model"},
        )
    assert r.status_code == 200, r.text
    changed = r.json()
    assert changed["spec_hash"] != approved["spec_hash"]
    assert not changed["readiness"]["ready"] and not changed["readiness"]["preview_current"]
    assert all(not a["valid"] for a in changed["approvals"])  # history kept, not applicable
    assert len(changed["approvals"]) == len(approved["approvals"])
    for spec_hash in (approved["spec_hash"], changed["spec_hash"]):
        r = _commit(user, p["id"], spec_hash)
        assert r.status_code == 409 and r.json()["code"] in ("APPROVAL_STALE", "APPROVAL_REQUIRED")
    assert len(_versions(user, ds_id)) == 1


def test_forged_hashes_and_unpreviewed_approvals_are_refused(ready) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, _ = ready
    p = _pipeline(user, ds_id, STEPS[:4])
    cap = p["steps"][2]
    body = {
        "step_hash": cap["step_hash"],
        "spec_hash": p["spec_hash"],
        "acknowledge_destructive": True,
    }
    r = user.client.post(
        f"/api/v1/pipelines/{p['id']}/steps/{cap['id']}/approve", headers=user.headers(), json=body
    )
    assert r.status_code == 409 and r.json()["code"] == "APPROVAL_STALE"  # no preview yet
    _preview(user, p["id"])
    for forged in ({**body, "step_hash": "0" * 64}, {**body, "spec_hash": "f" * 64}):
        r = user.client.post(
            f"/api/v1/pipelines/{p['id']}/steps/{cap['id']}/approve",
            headers=user.headers(),
            json=forged,
        )
        assert r.status_code == 409 and r.json()["code"] == "APPROVAL_STALE"
    low = p["steps"][0]
    r = user.client.post(
        f"/api/v1/pipelines/{p['id']}/steps/{low['id']}/approve",
        headers=user.headers(),
        json={**body, "step_hash": low["step_hash"]},
    )
    assert r.status_code == 422
    _approve_all(user, p["id"])
    r = _commit(user, p["id"], "a" * 64)
    assert r.status_code == 409 and r.json()["code"] == "APPROVAL_STALE"
    r = user.client.post(
        f"/api/v1/pipelines/{p['id']}/commit",
        headers=user.headers(),
        json={"spec_hash": p["spec_hash"]},
    )  # no Idempotency-Key
    assert r.status_code == 422


def test_worker_revalidates_changes_made_after_queueing(
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, _ = ready
    p = _pipeline(user, ds_id, STEPS[:4])
    _preview(user, p["id"])
    _approve_all(user, p["id"])
    monkeypatch.setattr(ctx, "publisher", Recorder())
    r = _commit(user, p["id"], p["spec_hash"])
    assert r.status_code == 202 and r.json()["job"]["status"] == "queued"
    # A second mutation for the same dataset is refused while this one is queued.
    other = user.client.post(
        f"/api/v1/datasets/{ds_id}/revert",
        headers=user.headers(),
        json={"version_id": p["source_version_id"]},
    )
    assert other.status_code == 409
    step = _get(user, p["id"])["steps"][3]
    user.client.patch(
        f"/api/v1/pipelines/{p['id']}/steps/{step['id']}",
        headers=user.headers(),
        json={"columns": ["a"]},
    )
    run_job(ctx, r.json()["job"]["id"])
    job = user.client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()
    assert job["status"] == "failed" and job["error"]["code"] in (
        "APPROVAL_STALE",
        "APPROVAL_REQUIRED",
    )
    assert len(_versions(user, ds_id)) == 1


def test_source_must_still_be_head(ready) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, _ = ready
    first, second = _pipeline(user, ds_id, STEPS[:2]), _pipeline(user, ds_id, STEPS[3:4])
    for p in (first, second):
        _preview(user, p["id"])
        _approve_all(user, p["id"])
    assert _commit(user, first["id"], first["spec_hash"]).json()["job"]["status"] == "succeeded"
    r = _commit(user, second["id"], second["spec_hash"])
    assert r.status_code == 409 and r.json()["code"] == "CONFLICT"
    assert not _get(user, second["id"])["source_is_head"]
    assert len(_versions(user, ds_id)) == 2


def test_version_quota(ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, v0 = ready
    monkeypatch.setattr(ctx.settings, "max_versions_per_dataset", 2)
    p = _pipeline(user, ds_id, STEPS[:2])
    _preview(user, p["id"])
    _approve_all(user, p["id"])
    assert _commit(user, p["id"], p["spec_hash"]).json()["job"]["status"] == "succeeded"
    r = user.client.post(
        f"/api/v1/datasets/{ds_id}/revert", headers=user.headers(), json={"version_id": v0}
    )
    assert r.status_code == 409 and r.json()["code"] == "QUOTA_EXCEEDED"
    assert len(_versions(user, ds_id)) == 2


def test_version_quota_refuses_commit_in_api(  # type: ignore[no-untyped-def]
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    user, ds_id, _ = ready
    monkeypatch.setattr(ctx.settings, "max_versions_per_dataset", 1)
    p = _pipeline(user, ds_id, STEPS[:2])
    _preview(user, p["id"])
    _approve_all(user, p["id"])
    r = _commit(user, p["id"], p["spec_hash"])
    assert r.status_code == 409 and r.json()["code"] == "QUOTA_EXCEEDED"
    assert len(_versions(user, ds_id)) == 1
    assert _get(user, p["id"])["state"] != "committed"


def test_worker_enforces_quota_reached_after_queueing(  # type: ignore[no-untyped-def]
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker re-checks the cap under the dataset lock, for commit and for revert, so a
    cap reached (or lowered) after a mutation was queued still creates no version."""
    user, ds_id, v0 = ready
    p = _pipeline(user, ds_id, STEPS[:2])
    _preview(user, p["id"])
    _approve_all(user, p["id"])
    monkeypatch.setattr(ctx, "publisher", Recorder())
    r = _commit(user, p["id"], p["spec_hash"])
    assert r.status_code == 202 and r.json()["job"]["status"] == "queued"
    monkeypatch.setattr(ctx.settings, "max_versions_per_dataset", 1)
    run_job(ctx, r.json()["job"]["id"])
    job = user.client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()
    assert job["status"] == "failed" and job["error"]["code"] == "QUOTA_EXCEEDED"
    assert len(_versions(user, ds_id)) == 1

    monkeypatch.setattr(ctx.settings, "max_versions_per_dataset", 20)
    q = _pipeline(user, ds_id, STEPS[:2])
    pv = user.client.post(f"/api/v1/pipelines/{q['id']}/impact-preview", headers=user.headers())
    run_job(ctx, pv.json()["job"]["id"])
    q = _approve_all(user, q["id"])
    c = _commit(user, q["id"], q["spec_hash"])
    run_job(ctx, c.json()["job"]["id"])
    assert len(_versions(user, ds_id)) == 2
    r = user.client.post(
        f"/api/v1/datasets/{ds_id}/revert",
        headers=user.headers(**{"Idempotency-Key": str(uuid.uuid4())}),
        json={"version_id": v0},
    )
    assert r.status_code == 202 and r.json()["job"]["status"] == "queued", r.text
    monkeypatch.setattr(ctx.settings, "max_versions_per_dataset", 2)
    run_job(ctx, r.json()["job"]["id"])
    job = user.client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()
    assert job["status"] == "failed" and job["error"]["code"] == "QUOTA_EXCEEDED"
    assert len(_versions(user, ds_id)) == 2


# ---- Revert, comparison, code -----------------------------------------------------------------


def test_revert_comparison_and_code(ready, ctx: AppContext, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, v0 = ready
    p = _pipeline(user, ds_id, STEPS[:4])
    _preview(user, p["id"])
    _approve_all(user, p["id"])
    _commit(user, p["id"], p["spec_hash"])
    v1 = _versions(user, ds_id)[1]["id"]
    r = user.client.post(f"/api/v1/versions/{v0}/compare/{v1}", headers=user.headers())
    assert r.status_code == 202 and r.json()["job"]["status"] == "succeeded"
    cmp = user.client.get(f"/api/v1/version-comparisons/{r.json()['job']['id']}").json()["result"]
    assert cmp["columns_added"] == ["b__missing"] and cmp["rows_affected"] > 0
    cols = {c["column"]: c for c in cmp["columns"]}
    assert cols["b"]["filled"] > 0 and cols["lab"]["test"]["status"] == "assessed"
    assert [v["kind"] for v in cmp["lineage"]] == ["ingest", "transform"]
    assert "rows" not in cmp and "values" not in str(cmp.keys())

    r = user.client.post(
        f"/api/v1/datasets/{ds_id}/revert", headers=user.headers(), json={"version_id": v0}
    )
    assert r.status_code == 202 and r.json()["job"]["status"] == "succeeded", r.text
    versions = _versions(user, ds_id)
    assert [(v["version_number"], v["kind"]) for v in versions] == [
        (0, "ingest"),
        (1, "transform"),
        (2, "revert"),
    ]
    v2 = versions[2]
    assert v2["restored_from_version_id"] == v0 and v2["parent_version_id"] == v1
    assert _content_hash(ctx, user, v2["id"]) == _content_hash(ctx, user, v0)
    pd.testing.assert_frame_equal(_frame(ctx, user, v2["id"]), _frame(ctx, user, v0))
    events = user.client.get(f"/api/v1/datasets/{ds_id}/version-events").json()["items"]
    assert [e["event"] for e in events] == ["created", "reverted"]
    r = user.client.post(
        f"/api/v1/datasets/{ds_id}/revert", headers=user.headers(), json={"version_id": v2["id"]}
    )
    assert r.status_code == 409  # already the current version
    # Code for the reverted version replays version 0's chain, not the undone commit.
    code = user.client.get(f"/api/v1/versions/{v2['id']}/code").text
    assert _run_code(ctx, user, ds_id, code, tmp_path) == _content_hash(ctx, user, v0)
    sub = tmp_path / "v1"
    sub.mkdir()
    code1 = user.client.get(f"/api/v1/versions/{v1}/code").text
    assert _run_code(ctx, user, ds_id, code1, sub) == _content_hash(ctx, user, v1)


# ---- Crash safety, retries, preview bounds, lab bridge ----------------------------------------


def test_crash_after_object_write_leaves_no_version_and_orphan_is_reconciled(
    ready,
    ctx: AppContext,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    user, ds_id, _ = ready
    p = _pipeline(user, ds_id, STEPS[:2])
    _preview(user, p["id"])
    _approve_all(user, p["id"])
    before = _objects(ctx, user, ds_id)
    import datamedic.pipelines.jobs as jobs

    def boom(*a: Any, **k: Any) -> None:
        raise RuntimeError("crash zqxjw")  # not hex: cannot occur in ids or times

    monkeypatch.setattr(jobs, "copy_roles", boom)
    r = _commit(user, p["id"], p["spec_hash"])
    job = r.json()["job"]
    assert job["status"] == "failed" and job["error"]["code"] == "INTERNAL"
    assert "zqxjw" not in str(job)
    monkeypatch.undo()
    assert len(_versions(user, ds_id)) == 1
    head = user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"]
    assert head == p["source_version_id"]
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        assert db.scalar(select(func.count()).select_from(TransformationRun)) == 0
    after = _objects(ctx, user, ds_id)
    orphans = set(after) - set(before)
    assert len(orphans) == 1  # the version object written before the crash
    assert all(after[k] == h for k, h in before.items())
    monkeypatch.setattr(ctx.settings, "orphan_grace_seconds", 10**9)
    assert reconcile_orphans(ctx) == 0  # too young: never deleted early
    monkeypatch.setattr(ctx.settings, "orphan_grace_seconds", -1)
    reconcile_orphans(ctx)
    final = _objects(ctx, user, ds_id)
    assert set(final) == set(before) and all(final[k] == h for k, h in before.items())
    # The pipeline can still be committed after the failure.
    assert _commit(user, p["id"], p["spec_hash"]).json()["job"]["status"] == "succeeded"


def test_retried_commit_never_creates_a_second_version(ready, ctx: AppContext) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, _ = ready
    p = _pipeline(user, ds_id, STEPS[:2])
    _preview(user, p["id"])
    _approve_all(user, p["id"])
    job = _commit(user, p["id"], p["spec_hash"]).json()["job"]
    with session_scope(ctx.maintenance_sessions) as db:
        db.execute(
            Job.__table__.update().where(Job.id == uuid.UUID(job["id"])).values(status="queued")
        )
    run_job(ctx, job["id"])
    assert len(_versions(user, ds_id)) == 2
    again = user.client.get(f"/api/v1/jobs/{job['id']}").json()
    assert again["status"] == "succeeded"


def test_preview_rows_are_bounded_and_redacted(
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, _ = ready
    monkeypatch.setattr(ctx.settings, "impact_preview_max_rows", 5)
    p = _pipeline(
        user,
        ds_id,
        [{"op": "impute.mode", "columns": ["code"]}, {"op": "impute.median", "columns": ["b"]}],
    )
    summary = _preview(user, p["id"])["summary"]
    mode = summary["steps"][0]
    assert mode["rows_affected"] > 5 and mode["rows_listed"] == 5 and mode["rows_truncated"]
    assert mode["columns"][0]["before"] == {"redacted": True}
    rows = user.client.get(f"/api/v1/pipelines/{p['id']}/impact-preview/rows?step=0").json()
    assert (
        rows["rows_truncated"] and len(rows["rows"]) == 5 and rows["redacted_columns"] == ["code"]
    )
    assert all(r["after"]["code"] == "[redacted]" for r in rows["rows"])
    ids = [r["row_id"] for r in rows["rows"]]
    assert ids == sorted(ids)
    b = user.client.get(f"/api/v1/pipelines/{p['id']}/impact-preview/rows?step=1").json()
    assert b["redacted_columns"] == [] and b["rows"][0]["before"]["b"] is None
    assert isinstance(b["rows"][0]["after"]["b"], float)


def test_lab_candidate_becomes_an_editable_pipeline(ready, ctx: AppContext) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready
    r = user.client.post(
        f"/api/v1/versions/{vid}/experiments",
        headers=user.headers(),
        json={
            "name": "lab",
            "models": ["dummy"],
            "split": {"n_splits": 2},
            "candidates": [
                {
                    "label": "core",
                    "steps": [
                        {"op": "impute.median", "columns": ["b"]},
                        {"op": "scale.robust", "columns": ["a", "b"]},
                        {"op": "transform.yeo_johnson", "columns": ["lab"]},
                        {"op": "imbalance.class_weight"},
                    ],
                }
            ],
        },
    )
    assert r.status_code == 202, r.text
    exp = user.client.get(f"/api/v1/experiments/{r.json()['experiment']['id']}").json()
    cand = exp["candidates"][1]
    p = user.client.post(
        f"/api/v1/experiment-candidates/{cand['id']}/to-pipeline", headers=user.headers()
    )
    assert p.status_code == 201, p.text
    body = p.json()
    assert [(s["op"], s["stage"]) for s in body["steps"]] == [
        ("impute.median", "data"),
        ("scale.robust", "model"),
        ("transform.yeo_johnson", "model"),  # after a model step: kept in model order
        ("imbalance.class_weight", "model"),
    ]
    assert body["from_experiment_candidate_id"] == cand["id"] and body["state"] == "validated"
    assert not body["readiness"]["ready"] and body["approvals"] == []
    assert user.client.get(f"/api/v1/experiments/{exp['id']}").json() == exp  # unchanged
    # A stage chosen explicitly afterwards is kept as chosen, never reassigned.
    first = body["steps"][0]
    r = user.client.patch(
        f"/api/v1/pipelines/{body['id']}/steps/{first['id']}",
        headers=user.headers(),
        json={"stage": "model"},
    )
    assert r.status_code == 200, r.text
    assert [s["stage"] for s in _get(user, body["id"])["steps"]] == ["model"] * 4
    explicit = _pipeline(
        user,
        ds_id,
        [
            {"op": "impute.median", "columns": ["b"], "stage": "model"},
            {"op": "transform.log", "columns": ["lab"]},
        ],
    )
    assert [s["stage"] for s in explicit["steps"]] == ["model", "data"]
    assert explicit["state"] == "draft"  # data after model: STAGE_ORDER, shown, not moved
    base = exp["candidates"][0]
    r = user.client.post(
        f"/api/v1/experiment-candidates/{base['id']}/to-pipeline", headers=user.headers()
    )
    assert r.status_code == 422
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        assert (
            db.scalar(
                select(func.count())
                .select_from(DatasetVersion)
                .where(DatasetVersion.dataset_id == uuid.UUID(ds_id))
            )
            == 1
        )
        roles = db.scalars(
            select(ColumnRole.role).where(ColumnRole.dataset_id == uuid.UUID(ds_id))
        ).all()
        assert sorted(roles) == ["sensitive", "target"]
        assert db.get(Dataset, uuid.UUID(ds_id)) is not None


def test_concurrent_mutations_cannot_both_be_queued(
    ready,
    ctx: AppContext,
    make_client: ClientFactory,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    """Two commits on the same head issued at the same time: the dataset row lock and the
    one-mutation-per-dataset index let exactly one through."""
    import threading

    user, ds_id, v0 = ready
    first, second = _pipeline(user, ds_id, STEPS[:2]), _pipeline(user, ds_id, STEPS[3:4])
    for p in (first, second):
        _preview(user, p["id"])
        _approve_all(user, p["id"])
    monkeypatch.setattr(ctx, "publisher", Recorder())
    other = make_client()
    other.cookies = user.client.cookies
    clients = [user.client, other]
    results: list[int] = []

    def go(i: int, p: dict) -> None:  # type: ignore[type-arg]
        r = clients[i].post(
            f"/api/v1/pipelines/{p['id']}/commit",
            json={"spec_hash": p["spec_hash"]},
            headers=user.headers(**{"Idempotency-Key": str(uuid.uuid4())}),
        )
        results.append(r.status_code)

    threads = [threading.Thread(target=go, args=(i, p)) for i, p in enumerate((first, second))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [202, 409]
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        assert db.scalar(select(func.count()).select_from(Job).where(Job.type == "commit")) == 1

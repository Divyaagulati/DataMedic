"""Recommendations and the Experiment Lab end to end through the API and the worker:
generation and reuse, dismissal, experiments on frozen folds, reproducible re-runs,
idempotency, validation and limits, failure isolation, cancellation, and proof that no
dataset version or stored version object is ever written. Synthetic data only."""

import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select, text, update

from datamedic.context import AppContext
from datamedic.db.models import DatasetVersion, Experiment, ExperimentResult, Job
from datamedic.db.session import bind_user, session_scope
from datamedic.storage import dataset_prefix
from datamedic.workers.runner import run_job
from tests.conftest import ClientFactory, LoggedInUser, create_and_upload, signup_and_login

pytestmark = pytest.mark.integration


class Recorder:
    """A publisher that only records: jobs stay queued until run by hand."""

    def __init__(self) -> None:
        self.published: list[uuid.UUID] = []

    def publish(self, job_id: uuid.UUID, job_type: str) -> None:
        self.published.append(job_id)


def cohort_csv(n: int = 240, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.12).astype(int)  # below the 20% imbalance threshold
    df = pd.DataFrame(
        {
            "a": (rng.normal(size=n) + y).round(4),
            "b": rng.normal(size=n).round(4),
            "lab": np.exp(rng.normal(1, 0.8, n)).round(3),
            "site": rng.choice(["north", "south", "east"], n),
            "y": y,
        }
    )
    df["b2"] = (df["b"] * 0.9 + rng.normal(0, 0.2, n)).round(4)
    df.loc[rng.random(n) < 0.2, "b"] = None
    return df.to_csv(index=False).encode()


def _setup(user: LoggedInUser) -> tuple[str, str]:
    body = create_and_upload(user, cohort_csv(), "cohort.csv")
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    vid = user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"]
    r = user.client.put(
        f"/api/v1/versions/{vid}/roles",
        headers=user.headers(),
        json={"roles": [{"column": "y", "role": "target", "task_type": "binary"}]},
    )
    assert r.status_code == 200, r.text
    return ds_id, vid


def _diagnose(user: LoggedInUser, vid: str) -> None:
    r = user.client.post(f"/api/v1/versions/{vid}/analyses", headers=user.headers())
    assert r.status_code == 202, r.text


def _recommend(user: LoggedInUser, vid: str) -> dict[str, Any]:
    r = user.client.post(f"/api/v1/versions/{vid}/recommendations", headers=user.headers())
    assert r.status_code in (200, 201), r.text
    return r.json()  # type: ignore[no-any-return]


def _experiment_body(rec: dict[str, Any], **extra: Any) -> dict[str, Any]:
    cands = [c for c in rec["candidates"] if c["steps"]][:2]
    assert cands
    return {
        "name": "Lab run",
        "models": ["dummy", "linear"],
        "split": {"n_splits": 3, "seed": 1},
        "candidates": [{"label": c["label"], "steps": c["steps"]} for c in cands],
        "recommendation_set_id": rec["id"],
        **extra,
    }


def _post(user: LoggedInUser, vid: str, body: dict[str, Any], **headers: str) -> Any:
    return user.client.post(
        f"/api/v1/versions/{vid}/experiments", headers=user.headers(**headers), json=body
    )


def _versions(ctx: AppContext, owner: uuid.UUID) -> int:
    with session_scope(ctx.sessions) as db:
        bind_user(db, owner)
        return int(db.scalar(select(func.count()).select_from(DatasetVersion)) or 0)


@pytest.fixture
def ready(make_client: ClientFactory, ctx: AppContext) -> tuple[LoggedInUser, str, str]:
    user = signup_and_login(make_client(), ctx)
    ds_id, vid = _setup(user)
    _diagnose(user, vid)
    return user, ds_id, vid


# ---- Recommendations --------------------------------------------------------------------------


def test_recommendations_need_current_diagnostics(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    user = signup_and_login(make_client(), ctx)
    _, vid = _setup(user)  # roles changed after the automatic diagnostics run
    r = user.client.post(f"/api/v1/versions/{vid}/recommendations", headers=user.headers())
    assert r.status_code == 409 and "out of date" in r.json()["detail"]
    assert user.client.get(f"/api/v1/versions/{vid}/recommendations").json()["set"] is None


def test_recommendations_are_generated_once_per_input_and_can_be_dismissed(ready) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    r = user.client.post(f"/api/v1/versions/{vid}/recommendations", headers=user.headers())
    assert r.status_code == 201, r.text
    first = r.json()
    again = user.client.post(f"/api/v1/versions/{vid}/recommendations", headers=user.headers())
    assert again.status_code == 200 and again.json()["id"] == first["id"]
    assert [x["key"] for x in again.json()["recommendations"]] == [
        x["key"] for x in first["recommendations"]
    ]
    ops = {x["op"] for x in first["recommendations"]}
    assert {"impute.median", "imbalance.class_weight"} <= ops
    for item in first["recommendations"]:
        d = item["detail"]
        assert d["reason"] and d["evidence"] and d["ranking"]["factors"] and d["limitations"]
        assert item["stage"] in ("data", "model", "either")
    assert first["split_plan"]["method"] == "stratified"
    assert first["candidates"][0]["key"] == "baseline"
    target = first["recommendations"][0]
    r = user.client.patch(
        f"/api/v1/recommendations/{target['id']}",
        headers=user.headers(),
        json={"status": "dismissed"},
    )
    assert r.status_code == 200 and r.json()["status"] == "dismissed"
    bad = user.client.patch(
        f"/api/v1/recommendations/{target['id']}",
        headers=user.headers(),
        json={"status": "applied"},
    )
    assert bad.status_code == 422
    latest = user.client.get(f"/api/v1/versions/{vid}/recommendations").json()["set"]
    assert latest["stale"] is False
    assert {x["id"]: x["status"] for x in latest["recommendations"]}[target["id"]] == "dismissed"
    # Changing roles makes the set stale (its inputs are no longer current).
    user.client.put(f"/api/v1/versions/{vid}/roles", headers=user.headers(), json={"roles": []})
    assert user.client.get(f"/api/v1/versions/{vid}/recommendations").json()["set"]["stale"]


# ---- Experiments ------------------------------------------------------------------------------


def test_experiment_runs_on_frozen_folds_and_writes_no_version(ready, ctx: AppContext) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready
    rec = _recommend(user, vid)
    before_versions = _versions(ctx, user.user_id)
    before_objects = ctx.storage.list_prefix(
        dataset_prefix("versions", user.user_id, uuid.UUID(ds_id))
    )
    r = _post(user, vid, _experiment_body(rec))
    assert r.status_code == 202, r.text
    eid = r.json()["experiment"]["id"]
    detail = user.client.get(f"/api/v1/experiments/{eid}").json()
    assert detail["job"]["status"] == "succeeded", detail["job"]
    assert detail["summary"] and detail["completed_at"]
    split = detail["split"]
    assert split["method"] == "stratified" and split["n_splits"] == 3
    assert len(split["assignment_sha256"]) == 64
    cands = detail["candidates"]
    assert cands[0]["is_baseline"] and cands[0]["steps"] == []
    test_rows = None
    for c in cands:
        res = c["result"]
        assert res["status"] in ("completed", "partial")
        sizes = [f["quality"]["test_rows"] for f in res["folds"]]
        test_rows = test_rows or sizes
        assert sizes == test_rows  # identical held-out rows for every candidate
        assert res["metrics"]["linear"]["roc_auc"]["n_folds"] == 3
        assert c["max_risk"] in ("low", "medium", "high")
    assert detail["summary"]["comparisons"]
    assert detail["config"]["roles"]["roles"] == {"target": ["y"]}
    # Nothing was written as a dataset version, and no version object was stored.
    assert _versions(ctx, user.user_id) == before_versions
    assert (
        ctx.storage.list_prefix(dataset_prefix("versions", user.user_id, uuid.UUID(ds_id)))
        == before_objects
    )
    head = user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"]
    assert head == vid
    listed = user.client.get(f"/api/v1/versions/{vid}/experiments").json()["items"]
    assert [e["id"] for e in listed] == [eid]


def test_rerun_reuses_the_split_and_reproduces_results(ready) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    rec = _recommend(user, vid)
    eid = _post(user, vid, _experiment_body(rec)).json()["experiment"]["id"]
    r = user.client.post(f"/api/v1/experiments/{eid}/rerun", headers=user.headers())
    assert r.status_code == 202, r.text
    first = user.client.get(f"/api/v1/experiments/{eid}").json()
    second = user.client.get(f"/api/v1/experiments/{r.json()['experiment']['id']}").json()
    assert second["rerun_of_id"] == eid and second["job"]["status"] == "succeeded"
    assert second["split"]["id"] == first["split"]["id"]  # the stored folds were reused

    def metrics(d: dict[str, Any]) -> list[Any]:
        out = []
        for c in d["candidates"]:
            out.append(
                {
                    m: {k: v["values"] for k, v in scores.items() if k != "fit_ms"}
                    for m, scores in c["result"]["metrics"].items()
                }
            )
        return out

    assert metrics(first) == metrics(second)
    # A second experiment with the same split configuration reuses the same split too.
    other = _post(user, vid, _experiment_body(rec, name="Other")).json()["experiment"]["id"]
    assert (
        user.client.get(f"/api/v1/experiments/{other}").json()["split"]["id"]
        == first["split"]["id"]
    )


def test_idempotency_key_returns_the_original_experiment(ready) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    rec = _recommend(user, vid)
    key = str(uuid.uuid4())
    a = _post(user, vid, _experiment_body(rec), **{"Idempotency-Key": key})
    b = _post(user, vid, _experiment_body(rec), **{"Idempotency-Key": key})
    assert a.status_code == 202 and b.status_code == 200
    assert a.json()["experiment"]["id"] == b.json()["experiment"]["id"]
    jobs = user.client.get("/api/v1/jobs?type=experiment").json()["items"]
    assert len(jobs) == 1


@pytest.mark.parametrize(
    "change,status,code",
    [
        ({"candidates": [{"label": "x", "steps": [{"op": "nope.op"}]}]}, 422, "UNKNOWN_OP"),
        (
            {"candidates": [{"label": "x", "steps": [{"op": "impute.median", "columns": ["y"]}]}]},
            422,
            "COLUMN_PROTECTED",
        ),
        ({"candidates": [{"label": "x", "steps": []}]}, 422, "EMPTY_CANDIDATE"),
        (
            {
                "candidates": [
                    {
                        "label": "Baseline: no preprocessing",
                        "steps": [{"op": "scale.standard", "columns": ["a"]}],
                    }
                ]
            },
            422,
            "DUPLICATE_LABEL",
        ),
        ({"models": ["kmeans"]}, 422, "MODEL_INVALID"),
        ({"split": {"method": "group"}}, 422, "SPLIT_ROLE_MISSING"),
        (
            {
                "candidates": [
                    {"label": f"c{i}", "steps": [{"op": "scale.standard", "columns": ["a"]}]}
                    for i in range(7)
                ]
            },
            409,
            "QUOTA_EXCEEDED",
        ),
    ],
)
def test_invalid_experiments_are_refused(ready, change: dict, status: int, code: str) -> None:  # type: ignore[no-untyped-def, type-arg]
    user, _, vid = ready
    body = {
        "name": "x",
        "candidates": [{"label": "s", "steps": [{"op": "scale.standard", "columns": ["a"]}]}],
        **change,
    }
    r = _post(user, vid, body)
    assert r.status_code == status, r.text
    problem = r.json()
    codes = {e["code"] for e in problem.get("errors") or []} | {problem["code"]}
    assert code in codes, problem
    assert user.client.get(f"/api/v1/versions/{vid}/experiments").json()["items"] == []


def test_clustering_needs_no_target_but_refuses_target_steps(ready) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    body = {
        "name": "clusters",
        "task": "clustering",
        "n_clusters": 3,
        "candidates": [
            {
                "label": "scaled",
                "steps": [
                    {"op": "impute.median", "columns": ["b"]},
                    {"op": "scale.robust", "columns": ["a", "b", "lab"]},
                ],
            }
        ],
    }
    r = _post(user, vid, body)
    assert r.status_code == 202, r.text
    d = user.client.get(f"/api/v1/experiments/{r.json()['experiment']['id']}").json()
    assert d["task_type"] == "clustering" and d["job"]["status"] == "succeeded"
    assert d["candidates"][1]["result"]["metrics"]["kmeans"]["silhouette"]["n_folds"] == 5
    body["candidates"][0]["steps"].append({"op": "imbalance.class_weight"})
    r = _post(user, vid, body)
    assert r.status_code == 422 and "TARGET_REQUIRED" in {e["code"] for e in r.json()["errors"]}


def test_failing_candidate_and_crashing_job_corrupt_nothing(
    ready,
    ctx: AppContext,
    monkeypatch: pytest.MonkeyPatch,  # type: ignore[no-untyped-def]
) -> None:
    user, _, vid = ready
    rec = _recommend(user, vid)
    body = _experiment_body(rec)
    body["candidates"].append(
        {"label": "log of negatives", "steps": [{"op": "transform.log", "columns": ["a"]}]}
    )
    ok = _post(user, vid, body).json()["experiment"]["id"]
    d = user.client.get(f"/api/v1/experiments/{ok}").json()
    assert d["job"]["status"] == "succeeded"
    bad = d["candidates"][-1]["result"]
    assert bad["status"] == "failed" and not bad["comparable"]
    assert all(c["result"]["comparable"] for c in d["candidates"][:-1])

    import datamedic.engine.preprocess.experiment as exp_pkg

    def crash(*a: Any, **k: Any) -> None:
        raise RuntimeError("boom zqxjv")  # not hex: cannot occur in ids or hashes

    monkeypatch.setattr(exp_pkg, "run_experiment", crash)
    crashed = _post(user, vid, _experiment_body(rec, name="crash")).json()["experiment"]["id"]
    c = user.client.get(f"/api/v1/experiments/{crashed}").json()
    assert c["job"]["status"] == "failed" and c["job"]["error"]["code"] == "INTERNAL"
    assert "zqxjv" not in str(c)
    assert c["summary"] is None and all(x["result"] is None for x in c["candidates"])
    monkeypatch.undo()
    # The earlier experiment is untouched by the crash.
    assert user.client.get(f"/api/v1/experiments/{ok}").json() == d


def test_cancellation_queued_and_running(
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    rec = _recommend(user, vid)
    recorder = Recorder()
    monkeypatch.setattr(ctx, "publisher", recorder)
    queued = _post(user, vid, _experiment_body(rec)).json()
    job_id = queued["job"]["id"]
    r = user.client.post(f"/api/v1/jobs/{job_id}/cancel", headers=user.headers())
    assert r.status_code == 202
    run_job(ctx, job_id)  # a late delivery of the message is a no-op
    d = user.client.get(f"/api/v1/experiments/{queued['experiment']['id']}").json()
    assert d["job"]["status"] == "cancelled" and d["summary"] is None

    running = _post(user, vid, _experiment_body(rec, name="running")).json()
    rid = running["job"]["id"]
    from datamedic.engine.preprocess.experiment import runner

    original = runner._run_fold

    def cancel_during(*a: Any, **k: Any) -> Any:
        with session_scope(ctx.sessions) as db:
            bind_user(db, user.user_id)
            db.execute(
                update(Job).where(Job.id == uuid.UUID(rid)).values(status="cancel_requested")
            )
        return original(*a, **k)

    monkeypatch.setattr(runner, "_run_fold", cancel_during)
    run_job(ctx, rid)
    d = user.client.get(f"/api/v1/experiments/{running['experiment']['id']}").json()
    assert d["job"]["status"] == "cancelled" and d["job"]["error"]["code"] == "JOB_CANCELLED"
    assert d["summary"] is None
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        assert (
            db.scalar(
                select(func.count())
                .select_from(ExperimentResult)
                .where(ExperimentResult.experiment_id == uuid.UUID(running["experiment"]["id"]))
            )
            == 0
        )


def test_a_retried_job_never_runs_twice(ready, ctx: AppContext) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    rec = _recommend(user, vid)
    created = _post(user, vid, _experiment_body(rec)).json()
    eid = uuid.UUID(created["experiment"]["id"])
    run_job(ctx, created["job"]["id"])  # duplicate delivery after success: not claimable
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        assert (
            db.scalar(
                select(func.count())
                .select_from(ExperimentResult)
                .where(ExperimentResult.experiment_id == eid)
            )
            == 3
        )
        exp = db.get(Experiment, eid)
        assert exp is not None and exp.completed_at is not None
    # Even a forced re-queue of a completed experiment writes nothing new.
    with session_scope(ctx.maintenance_sessions) as db:
        db.execute(
            text("UPDATE jobs SET status = 'queued' WHERE id = :id"), {"id": created["job"]["id"]}
        )
    run_job(ctx, created["job"]["id"])
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        assert (
            db.scalar(
                select(func.count())
                .select_from(ExperimentResult)
                .where(ExperimentResult.experiment_id == eid)
            )
            == 3
        )
        job = db.get(Job, uuid.UUID(created["job"]["id"]))
        assert job is not None and job.result["skipped"] == "completed"

"""Phase 5 end to end through the API and the worker: training plans, runs, nested selection,
replay of full-data-fitted versions (including through a revert), blocked and exploratory
evaluation, the reliability audit, reruns, idempotency, cancellation, limits, purge and the
Experiment Lab warning. Synthetic data only."""

import uuid
from typing import Any

import pytest
from sqlalchemy import func, select

from datamedic.context import AppContext
from datamedic.db.models import (
    AnalysisRun,
    Artifact,
    DatasetVersion,
    FindingRow,
    ModelEvaluation,
    TrainingRun,
)
from datamedic.db.session import bind_user, session_scope
from datamedic.workers.runner import run_job
from tests.conftest import ClientFactory, LoggedInUser, signup_and_login
from tests.integration.test_experiments_api import Recorder
from tests.integration.test_pipelines_api import (
    STEPS,
    _approve_all,
    _commit,
    _pipeline,
    _preview,
    _setup,
    _versions,
)

pytestmark = pytest.mark.integration

FAST = {"models": ["dummy", "linear"], "tuning": {"mode": "defaults"}, "split": {"n_splits": 3}}


def _key() -> dict[str, str]:
    return {"Idempotency-Key": str(uuid.uuid4())}


def _plan(user: LoggedInUser, vid: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    r = user.client.post(
        f"/api/v1/versions/{vid}/training-plan", headers=user.headers(), json=body or FAST
    )
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def _train(
    user: LoggedInUser,
    vid: str,
    body: dict[str, Any] | None = None,
    key: dict[str, str] | None = None,
) -> Any:
    return user.client.post(
        f"/api/v1/versions/{vid}/training-runs",
        headers=user.headers(**(key or _key())),
        json=body or FAST,
    )


def _run(user: LoggedInUser, rid: str) -> dict[str, Any]:
    return user.client.get(f"/api/v1/training-runs/{rid}").json()  # type: ignore[no-any-return]


def _audit(user: LoggedInUser, rid: str) -> dict[str, Any]:
    return user.client.get(f"/api/v1/training-runs/{rid}/audit").json()  # type: ignore[no-any-return]


def _roles(user: LoggedInUser, vid: str, roles: list[dict[str, Any]]) -> None:
    r = user.client.put(
        f"/api/v1/versions/{vid}/roles", headers=user.headers(), json={"roles": roles}
    )
    assert r.status_code == 200, r.text


@pytest.fixture
def ready(make_client: ClientFactory, ctx: AppContext) -> tuple[LoggedInUser, str, str]:
    user = signup_and_login(make_client(), ctx)
    ds_id, vid = _setup(user)  # target y (binary), code sensitive
    return user, ds_id, vid


def _commit_v1(user: LoggedInUser, ds_id: str) -> str:
    p = _pipeline(user, ds_id, STEPS[:2])  # missing indicator + median imputation (fitted)
    _preview(user, p["id"])
    _approve_all(user, p["id"])
    assert _commit(user, p["id"], p["spec_hash"]).json()["job"]["status"] == "succeeded"
    v1 = _versions(user, ds_id)[-1]
    assert v1["fitted_on_full_data"]
    return str(v1["id"])


def test_catalogue(ready) -> None:  # type: ignore[no-untyped-def]
    user, *_ = ready
    cat = user.client.get("/api/v1/ml/models").json()
    assert [m["key"] for m in cat["models"]] == ["dummy", "linear", "random_forest", "hgb"]
    assert len(cat["audit_checks"]) == 20
    assert all(len(m["grids"]["classification"]) <= 4 for m in cat["models"])


def test_binary_run_end_to_end(ready, ctx: AppContext) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready
    plan = _plan(user, vid)
    assert plan["valid"] and plan["evaluation_mode"] == "unbiased_direct"
    assert plan["task"] == "binary" and "y" not in plan["features"]
    assert plan["split"]["method"] == "stratified" and plan["estimated_fits"] == 6
    key = _key()
    r = _train(user, vid, {**FAST, "name": "first"}, key)
    assert r.status_code == 202 and r.json()["job"]["status"] == "succeeded", r.text
    rid = r.json()["training_run"]["id"]
    again = _train(user, vid, {**FAST, "name": "first"}, key)  # idempotent
    assert again.status_code == 200 and again.json()["training_run"]["id"] == rid

    run = _run(user, rid)
    assert run["evaluation_mode"] == "unbiased_direct" and run["completed_at"]
    assert [e["model"] for e in run["evaluations"]] == ["dummy", "linear"]
    lin = run["evaluations"][1]
    assert lin["status"] == "completed" and lin["comparable"]
    assert lin["metrics"]["folds"]["roc_auc"]["n_folds"] == 3
    assert run["summary"]["coverage"]["complete"] and run["split"]["method"] == "stratified"
    assert run["oof_artifact_sha256"] and run["headline"]["model"] == "linear"
    assert "rows" not in str(lin["details"].get("curves", {}).keys())

    audit = _audit(user, rid)
    assert audit["run"] is not None and len(audit["findings"]) == 20
    assert audit["run"]["summary"]["evaluation_status"]["status"] in (
        "no_issues_detected_among_assessed",
        "concerns",
    )
    assert not audit["stale"]
    # Training and auditing never write versions; the diagnostics view ignores audit runs.
    assert len(_versions(user, ds_id)) == 1
    latest = user.client.get(f"/api/v1/versions/{vid}/analyses/latest").json()["run"]
    assert latest is None or latest["id"] != audit["run"]["id"]
    listed = user.client.get(f"/api/v1/datasets/{ds_id}/training-runs").json()["items"]
    assert listed[0]["id"] == rid and listed[0]["audit"]["status"]

    # A rerun reproduces the configuration, the frozen folds and the results.
    rr = user.client.post(f"/api/v1/training-runs/{rid}/rerun", headers=user.headers(**_key()))
    assert rr.status_code == 202, rr.text
    run2 = _run(user, rr.json()["training_run"]["id"])
    assert run2["rerun_of_id"] == rid and run2["split"]["id"] == run["split"]["id"]
    assert run2["evaluations"][1]["metrics"]["folds"] == lin["metrics"]["folds"]
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        assert db.scalar(select(func.count()).select_from(TrainingRun)) == 2
        art = db.scalar(select(Artifact).where(Artifact.kind == "oof_predictions"))
        assert art is not None and art.owner_type == "training_run"


def test_regression_nested_run(ready) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    _roles(user, vid, [{"column": "lab", "role": "target", "task_type": "regression"}])
    body = {"models": ["dummy", "linear"], "tuning": {"mode": "nested"}, "split": {"n_splits": 3}}
    plan = _plan(user, vid, body)
    assert plan["task"] == "regression" and plan["split"]["method"] == "kfold"
    r = _train(user, vid, body)
    assert r.json()["job"]["status"] == "succeeded", r.text
    run = _run(user, r.json()["training_run"]["id"])
    assert run["summary"]["tuning"]["used"] == "nested"
    sel = run["evaluations"][1]["selection"]
    assert sum(f["folds"] for f in sel["frequency"]) == 3
    assert run["evaluations"][1]["details"]["residuals"]["available"]


def test_task_and_target_validation(ready) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    # A task type the data contradicts is refused already when roles are saved; training
    # re-checks the same rule (API plan and worker) for roles stored before that check.
    r = user.client.put(
        f"/api/v1/versions/{vid}/roles",
        headers=user.headers(),
        json={"roles": [{"column": "lab", "role": "target", "task_type": "binary"}]},
    )
    assert r.status_code == 422 and r.json()["errors"][0]["code"] == "TASK_TYPE_MISMATCH"
    _roles(user, vid, [])
    r = _train(user, vid)
    assert r.status_code == 422 and r.json()["errors"][0]["code"] == "TARGET_REQUIRED"
    assert _plan(user, vid)["errors"][0]["code"] == "TARGET_REQUIRED"


def test_fitted_version_is_evaluated_by_replay_including_through_revert(ready) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, v0 = ready
    v1 = _commit_v1(user, ds_id)
    plan = _plan(user, v1)
    assert plan["evaluation_mode"] == "unbiased_replay" and plan["evaluation_version_id"] == v0
    assert [s["op"] for s in plan["eligibility"]["replay"]["steps"]] == [
        "missing.indicator",
        "impute.median",
    ]
    r = _train(user, v1)
    assert r.json()["job"]["status"] == "succeeded", r.text
    run = _run(user, r.json()["training_run"]["id"])
    assert run["evaluation_mode"] == "unbiased_replay" and run["evaluation_version_id"] == v0
    assert run["replay"]["equivalence"]["status"] == "exact"
    a9 = next(
        f
        for f in _audit(user, run["id"])["findings"]
        if f["check_id"] == "leakage.pipeline.fit_on_full_data"
    )
    assert a9["status"] == "not_detected"

    # v2 = revert to v0 (direct); v3 = revert to v1 (replay through the revert lineage).
    for target in (v0, v1):
        rv = user.client.post(
            f"/api/v1/datasets/{ds_id}/revert", headers=user.headers(), json={"version_id": target}
        )
        assert rv.json()["job"]["status"] == "succeeded", rv.text
    versions = _versions(user, ds_id)
    assert _plan(user, versions[2]["id"])["evaluation_mode"] == "unbiased_direct"
    p3 = _plan(user, versions[3]["id"])
    assert p3["evaluation_mode"] == "unbiased_replay" and p3["evaluation_version_id"] == v0


def test_unsafe_replay_is_blocked_and_exploratory_is_permanently_labelled(ready) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, _ = ready
    v1 = _commit_v1(user, ds_id)
    # The imputed column becomes the target: replay would alter test labels.
    _roles(user, v1, [{"column": "b", "role": "target", "task_type": "regression"}])
    plan = _plan(user, v1)
    assert plan["evaluation_mode"] == "blocked" and not plan["valid"]
    assert plan["eligibility"]["reasons"][0]["code"] == "TARGET_TRANSFORMED_IN_LINEAGE"
    r = _train(user, v1)
    assert r.status_code == 409 and r.json()["code"] == "EVALUATION_BLOCKED"
    r = _train(user, v1, {**FAST, "allow_exploratory": True})
    assert r.status_code == 202 and r.json()["job"]["status"] == "succeeded", r.text
    rid = r.json()["training_run"]["id"]
    assert r.json()["training_run"]["evaluation_mode"] == "exploratory"
    audit = _audit(user, rid)
    assert audit["run"]["summary"]["evaluation_status"]["status"] in ("exploratory", "blocked")
    a9 = next(f for f in audit["findings"] if f["check_id"] == "leakage.pipeline.fit_on_full_data")
    assert a9["status"] == "detected" and a9["severity"] == "critical"
    rr = user.client.post(f"/api/v1/training-runs/{rid}/rerun", headers=user.headers(**_key()))
    assert rr.json()["training_run"]["evaluation_mode"] == "exploratory"
    listed = user.client.get(f"/api/v1/datasets/{ds_id}/training-runs").json()["items"]
    assert {i["evaluation_mode"] for i in listed} == {"exploratory"}


def test_lab_warns_on_full_data_fitted_source(ready) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, _ = ready
    v1 = _commit_v1(user, ds_id)
    r = user.client.post(
        f"/api/v1/versions/{v1}/experiments",
        headers=user.headers(),
        json={
            "name": "lab",
            "models": ["dummy"],
            "split": {"n_splits": 2},
            "candidates": [{"label": "s", "steps": [{"op": "scale.standard", "columns": ["a"]}]}],
        },
    )
    assert r.status_code == 202, r.text
    exp = user.client.get(f"/api/v1/experiments/{r.json()['experiment']['id']}").json()
    assert "FITTED_ON_FULL_DATA_SOURCE" in {w["code"] for w in exp["summary"]["warnings"]}


def test_cancellation_publishes_no_results(
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    monkeypatch.setattr(ctx, "publisher", Recorder())
    r = _train(user, vid)
    job_id, rid = r.json()["job"]["id"], r.json()["training_run"]["id"]
    import datamedic.training.jobs as tj

    monkeypatch.setattr(tj._Progress, "cancelled", lambda self: True)
    run_job(ctx, job_id)
    job = user.client.get(f"/api/v1/jobs/{job_id}").json()
    assert job["status"] == "cancelled"
    run = _run(user, rid)
    assert run["completed_at"] is None and run["evaluations"] == [] and run["summary"] is None
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        assert db.scalar(select(func.count()).select_from(ModelEvaluation)) == 0


def test_limits_and_fit_budget_fallback(
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    monkeypatch.setattr(ctx.settings, "training_max_fits", 10)
    plan = _plan(user, vid, {**FAST, "tuning": {"mode": "nested"}})
    assert plan["tuning"]["planned"] == "defaults" and plan["tuning"]["fallback_reason"]
    monkeypatch.setattr(ctx.settings, "training_max_fits", 2)
    r = _train(user, vid)
    assert r.status_code == 409 and r.json()["code"] == "QUOTA_EXCEEDED"
    monkeypatch.setattr(ctx.settings, "training_max_fits", 400)
    monkeypatch.setattr(ctx.settings, "training_max_runs_per_dataset", 0)
    assert _train(user, vid).json()["code"] == "QUOTA_EXCEEDED"


def test_worker_rechecks_memory_budget(
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    user, _, vid = ready
    monkeypatch.setattr(ctx.settings, "training_memory_budget_mb", 0)
    r = _train(user, vid)
    job = r.json()["job"]
    assert job["status"] == "failed" and job["error"]["code"] == "LIMIT_EXCEEDED"


def test_audit_rerun_and_purge(ready, ctx: AppContext) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready
    rid = _train(user, vid).json()["training_run"]["id"]
    r = user.client.post(f"/api/v1/training-runs/{rid}/audit", headers=user.headers())
    assert r.status_code == 202 and r.json()["job"]["status"] == "succeeded"
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        audits = db.scalar(
            select(func.count())
            .select_from(AnalysisRun)
            .where(AnalysisRun.kind == "reliability_audit")
        )
        assert audits == 2
    assert user.client.delete(f"/api/v1/datasets/{ds_id}", headers=user.headers()).status_code in (
        200,
        202,
        204,
    )
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        purge = db.scalar(select(func.count()).select_from(TrainingRun))
        versions = db.scalar(select(func.count()).select_from(DatasetVersion))
        findings = db.scalar(select(func.count()).select_from(FindingRow))
    # The eager publisher runs the purge immediately.
    assert purge == 0 and versions == 0 and findings == 0

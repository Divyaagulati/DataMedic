"""Phase 5 closure: sensitive-column protection across role changes, served headlines,
identifier handling and exploratory labelling through the API and the worker.

Sensitivity is enforced when results are served (training.privacy): a column sensitive at
training time, sensitive now on any version of the dataset, or derived from such a column by
committed steps is never shown as error slices. Stored rows are never rewritten. Synthetic
data only."""

from typing import Any

import pytest
from sqlalchemy import select

from datamedic.context import AppContext
from datamedic.db.models import ModelEvaluation
from datamedic.db.session import bind_user, session_scope
from datamedic.workers.runner import run_job
from tests.conftest import ClientFactory, LoggedInUser, signup_and_login
from tests.integration.test_experiments_api import Recorder
from tests.integration.test_pipelines_api import _setup
from tests.integration.test_training_api import (
    FAST,
    _commit_v1,
    _key,
    _plan,
    _roles,
    _run,
    _train,
)

pytestmark = pytest.mark.integration


@pytest.fixture
def ready(make_client: ClientFactory, ctx: AppContext) -> tuple[LoggedInUser, str, str]:
    user = signup_and_login(make_client(), ctx)
    ds_id, vid = _setup(user)  # target y (binary), code sensitive
    return user, ds_id, vid


TARGET = {"column": "y", "role": "target", "task_type": "binary"}
SENSITIVE_CODE = {"column": "code", "role": "sensitive"}


def _sliced(run: dict[str, Any], model: str = "linear") -> set[str]:
    e = next(x for x in run["evaluations"] if x["model"] == model)
    return {s["column"] for s in e["details"]["error_slices"]}


def _listed(user: LoggedInUser, ds_id: str) -> dict[str, dict[str, Any]]:
    items = user.client.get(f"/api/v1/datasets/{ds_id}/training-runs").json()["items"]
    return {i["id"]: i for i in items}


def _stored_slices(ctx: AppContext, user: LoggedInUser, rid: str) -> set[str]:
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        e = db.scalar(
            select(ModelEvaluation).where(
                ModelEvaluation.training_run_id == rid, ModelEvaluation.model_key == "linear"
            )
        )
        assert e is not None
        return {s["column"] for s in e.details["error_slices"]}


def test_sensitivity_added_after_training_is_enforced_on_historical_results(
    ready, ctx: AppContext
) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready
    _roles(user, vid, [TARGET])  # code not sensitive at training time
    rid = _train(user, vid, {**FAST, "name": "before"}).json()["training_run"]["id"]
    run = _run(user, rid)
    assert "code" in _sliced(run)  # sliced while not sensitive (fixture has 3 large categories)
    assert run["privacy"]["protected_columns"] == [] and not run["privacy"]["withheld_error_slices"]

    _roles(user, vid, [TARGET, SENSITIVE_CODE])  # becomes sensitive after training
    later = _run(user, rid)
    assert "code" not in _sliced(later)
    assert later["privacy"]["withheld_error_slices"] == ["code"]
    assert later["privacy"]["protected_columns"] == [
        {"column": "code", "reasons": ["sensitive_now"]}
    ]
    # Provenance and stored results are not rewritten; only the served copy is filtered.
    assert later["config"]["sensitive_columns"] == [] and "code" not in str(
        later["config"]["roles"]["roles"].get("sensitive", [])
    )
    assert "code" in _stored_slices(ctx, user, rid)
    assert later["evaluations"][1]["metrics"] == run["evaluations"][1]["metrics"]

    # A re-run keeps the training-time roles snapshot but applies current sensitivity.
    rr = user.client.post(f"/api/v1/training-runs/{rid}/rerun", headers=user.headers(**_key()))
    assert rr.status_code == 202 and rr.json()["job"]["status"] == "succeeded", rr.text
    rerun = _run(user, rr.json()["training_run"]["id"])
    assert rerun["config"]["roles"] == run["config"]["roles"]
    assert rerun["config"]["sensitive_columns"] == ["code"]
    assert "code" not in _stored_slices(ctx, user, rerun["id"])
    assert rerun["evaluations"][1]["metrics"]["folds"] == run["evaluations"][1]["metrics"]["folds"]


def test_sensitive_at_training_stays_protected_after_the_role_is_removed(
    ready, ctx: AppContext
) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready  # the fixture declares code sensitive before training
    rid = _train(user, vid).json()["training_run"]["id"]
    assert _run(user, rid)["config"]["sensitive_columns"] == ["code"]
    assert "code" not in _stored_slices(ctx, user, rid)  # never sliced by the engine
    _roles(user, vid, [TARGET])  # sensitivity removed later
    run = _run(user, rid)
    assert {"column": "code", "reasons": ["sensitive_at_training"]} in run["privacy"][
        "protected_columns"
    ]
    assert "code" not in _sliced(run)


def test_sensitivity_on_another_version_and_derived_columns(ready) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, v0 = ready
    v1 = _commit_v1(user, ds_id)  # missing indicator b__missing derived from b, then imputation
    _roles(user, v1, [TARGET, {"column": "b", "role": "sensitive"}])
    plan = _plan(user, v1)
    # b is sensitive on v1, code still on v0 (any version counts), b__missing derives from b.
    assert plan["sensitive_columns"] == ["b", "b__missing", "code"]
    rid = _train(user, v1).json()["training_run"]["id"]
    run = _run(user, rid)
    assert run["evaluation_mode"] == "unbiased_replay"
    reasons = {p["column"]: p["reasons"] for p in run["privacy"]["protected_columns"]}
    assert "sensitive_at_training" in reasons["b__missing"]
    assert {"b", "code"} <= set(reasons) and not ({"b", "code"} & _sliced(run))

    # Derived from a column that now has a non-feature role: disclosed before training.
    _roles(user, v1, [TARGET, {"column": "b", "role": "exclude"}])
    warned = _plan(user, v1)
    w = next(x for x in warned["warnings"] if x["code"] == "DERIVED_FROM_PROTECTED")
    assert "b__missing" in w["message"] and "b" not in warned["features"]


def test_target_values_withheld_when_target_becomes_sensitive(ready) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready
    rid = _train(user, vid).json()["training_run"]["id"]
    assert _run(user, rid)["summary"]["classes"] == ["0", "1"]
    _roles(user, vid, [TARGET, SENSITIVE_CODE, {"column": "y", "role": "sensitive"}])
    run = _run(user, rid)
    assert run["summary"]["classes"] == ["class 1", "class 2"]
    assert run["summary"]["positive_class"] == "class 2"
    assert run["privacy"]["target_values_withheld"] is True


def test_list_headline_states_and_direction(ready, ctx: AppContext) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready
    rid = _train(user, vid).json()["training_run"]["id"]
    h = _listed(user, ds_id)[rid]["headline"]
    assert h["state"] == "available" and h["model"] == "linear" and h["metric"] == "roc_auc"
    assert h["higher_is_better"] is True and h["mean"] is not None and not h["exploratory"]
    assert h["baseline_mean"] == 0.5  # dummy: constant scores tie, ROC-AUC is exactly 0.5
    assert "not a declared winner" in h["note"]
    only_dummy = _train(user, vid, {**FAST, "models": ["dummy"]}).json()["training_run"]["id"]
    d = _listed(user, ds_id)[only_dummy]["headline"]
    assert d["state"] == "unavailable" and d["mean"] is None and "dummy" in d["reason"]

    # Regression: R² (higher is better) on the imputed version via an explicit role change.
    v1 = _commit_v1(user, ds_id)
    _roles(user, v1, [{"column": "a", "role": "target", "task_type": "regression"}])
    reg = _train(user, v1).json()["training_run"]["id"]
    r = _listed(user, ds_id)[reg]["headline"]
    assert r["metric"] == "r2" and r["higher_is_better"] is True and r["state"] == "available"


def test_list_headline_pending_and_cancelled(
    ready, ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid = ready
    monkeypatch.setattr(ctx, "publisher", Recorder())  # queued, not executed
    r = _train(user, vid)
    job_id, rid = r.json()["job"]["id"], r.json()["training_run"]["id"]
    h = _listed(user, ds_id)[rid]["headline"]
    assert h["state"] == "pending" and h["mean"] is None and h["reason"]
    import datamedic.training.jobs as tj

    monkeypatch.setattr(tj._Progress, "cancelled", lambda self: True)
    run_job(ctx, job_id)
    h = _listed(user, ds_id)[rid]["headline"]
    assert h["state"] == "cancelled" and h["mean"] is None and "no results" in h["reason"]


def test_exploratory_headline_is_qualified_and_roles_are_not_reinterpreted(ready) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, v0 = ready
    v1 = _commit_v1(user, ds_id)
    # The smoke-test configuration: only a regression target on the transformed column, so no
    # group and no sensitive role exist at training time.
    regression_only = [{"column": "b", "role": "target", "task_type": "regression"}]
    _roles(user, v1, regression_only)
    # code is still sensitive on v0: since the closure fix, that alone protects it at training.
    protected = _plan(user, v1, {**FAST, "allow_exploratory": True})
    assert protected["sensitive_columns"] == ["code"]
    _roles(user, v0, [TARGET])  # no sensitive role anywhere, as in the historical run
    r = _train(user, v1, {**FAST, "allow_exploratory": True})
    rid = r.json()["training_run"]["id"]
    run = _run(user, rid)
    assert run["config"]["sensitive_columns"] == []
    assert "code" in _sliced(run)  # sliced: not sensitive anywhere at training time
    h = _listed(user, ds_id)[rid]["headline"]
    assert h["exploratory"] and "not a leakage-controlled" in h["exploratory_note"]
    assert run["split"]["method"] == "kfold" and run["split_config"]["group"] is None
    assert "code" in run["config"]["features"]
    # Roles restored afterwards (as the smoke test did): code is sensitive now, so its slices
    # are withheld from the historical run, while its stored configuration is unchanged.
    _roles(user, v1, [TARGET, SENSITIVE_CODE])
    later = _run(user, rid)
    assert "code" not in _sliced(later) and later["split"]["method"] == "kfold"
    assert later["config"] == run["config"]
    audit = user.client.get(f"/api/v1/training-runs/{rid}/audit").json()
    assert audit["stale"] is True  # roles changed since training: shown, not reinterpreted
    a9 = next(f for f in audit["findings"] if f["check_id"] == "leakage.pipeline.fit_on_full_data")
    assert a9["status"] == "detected" and a9["severity"] == "critical"
    again = user.client.post(f"/api/v1/training-runs/{rid}/audit", headers=user.headers())
    assert again.json()["job"]["status"] == "succeeded"
    rerun_audit = user.client.get(f"/api/v1/training-runs/{rid}/audit").json()
    assert rerun_audit["run"]["summary"]["evaluation_status"]["status"] in (
        "exploratory",
        "blocked",
    )
    a9 = next(
        f for f in rerun_audit["findings"] if f["check_id"] == "leakage.pipeline.fit_on_full_data"
    )
    assert a9["status"] == "detected" and a9["severity"] == "critical"

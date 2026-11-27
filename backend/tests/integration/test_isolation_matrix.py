"""User isolation matrix generated from the live route table.

Every /api/v1 route is exercised as (a) an anonymous client and (b) user B targeting user A's
resources. A new route with an unknown path parameter makes this test fail until the matrix
is extended, so no route can silently skip the isolation check.
"""

import re
import uuid

import pytest
from fastapi.testclient import TestClient

from datamedic.api.app import create_app
from datamedic.context import AppContext
from tests.conftest import (
    CSV_BYTES,
    ClientFactory,
    LoggedInUser,
    create_and_upload,
    signup_and_login,
)
from tests.integration.test_pipelines_api import cohort_csv

pytestmark = pytest.mark.integration

PUBLIC_ROUTES = {
    ("POST", "/api/v1/auth/signup"),
    ("POST", "/api/v1/auth/verify-email"),
    ("POST", "/api/v1/auth/login"),
    ("POST", "/api/v1/auth/password-reset/request"),
    ("POST", "/api/v1/auth/password-reset/confirm"),
    ("GET", "/api/v1/meta/version"),
}
KNOWN_PARAMS = {
    "analysis_id",
    "candidate_id",
    "dashboard_id",
    "dataset_id",
    "experiment_id",
    "job_id",
    "other_version_id",
    "pipeline_id",
    "recommendation_id",
    "session_id",
    "step_id",
    "training_run_id",
    "transformation_run_id",
    "version_id",
    "widget_id",
}

JSON_BODIES = {
    ("POST", "/api/v1/datasets"): {"name": "x", "filename": "x.csv", "size_bytes": 3},
    ("PATCH", "/api/v1/datasets/{dataset_id}"): {"domain": "biomedical"},
    ("PUT", "/api/v1/versions/{version_id}/roles"): {"roles": []},
    ("PATCH", "/api/v1/recommendations/{recommendation_id}"): {"status": "dismissed"},
    ("POST", "/api/v1/versions/{version_id}/experiments"): {
        "name": "x",
        "candidates": [{"label": "s", "steps": [{"op": "scale.standard", "columns": ["age"]}]}],
    },
    ("POST", "/api/v1/datasets/{dataset_id}/pipelines"): {"name": "x", "steps": []},
    ("POST", "/api/v1/pipelines/{pipeline_id}/steps"): {"op": "impute.median", "columns": ["age"]},
    ("PATCH", "/api/v1/pipelines/{pipeline_id}/steps/{step_id}"): {"params": {}},
    ("POST", "/api/v1/pipelines/{pipeline_id}/steps/{step_id}/approve"): {
        "step_hash": "x",
        "spec_hash": "x",
        "acknowledge_destructive": True,
    },
    ("POST", "/api/v1/pipelines/{pipeline_id}/confirm"): {"spec_hash": "x"},
    ("POST", "/api/v1/pipelines/{pipeline_id}/commit"): {"spec_hash": "x"},
    ("POST", "/api/v1/versions/{version_id}/training-plan"): {"models": ["dummy"]},
    ("POST", "/api/v1/versions/{version_id}/training-runs"): {"models": ["dummy"]},
    ("POST", "/api/v1/versions/{version_id}/copilot"): {"question": "What next?"},
    ("POST", "/api/v1/versions/{version_id}/exports"): {"acknowledged": True},
    ("POST", "/api/v1/versions/{version_id}/analytics/query"): {
        "spec": {"kind": "aggregate", "measures": [{"agg": "count"}]}
    },
    ("POST", "/api/v1/versions/{version_id}/analytics/export"): {
        "spec": {"kind": "aggregate", "measures": [{"agg": "count"}]}
    },
    ("POST", "/api/v1/datasets/{dataset_id}/dashboards"): {
        "version_id": "00000000-0000-4000-8000-000000000000",
        "name": "x",
    },
    ("PATCH", "/api/v1/dashboards/{dashboard_id}"): {"name": "x"},
    ("POST", "/api/v1/dashboards/{dashboard_id}/duplicate"): {},
    ("POST", "/api/v1/dashboards/{dashboard_id}/switch-version"): {
        "version_id": "00000000-0000-4000-8000-000000000000",
        "confirm": True,
    },
    ("POST", "/api/v1/dashboards/{dashboard_id}/widgets"): {
        "kind": "kpi",
        "title": "x",
        "spec": {"query": {"kind": "aggregate", "measures": [{"agg": "count"}]}},
    },
    ("PATCH", "/api/v1/dashboards/{dashboard_id}/widgets/{widget_id}"): {"title": "x"},
    ("POST", "/api/v1/dashboards/{dashboard_id}/reorder"): {"widget_ids": []},
    ("POST", "/api/v1/datasets/{dataset_id}/revert"): {
        "version_id": "00000000-0000-4000-8000-000000000000"
    },
}


HTTP_METHODS = {"get", "post", "put", "patch", "delete"}


def _routes(ctx: AppContext) -> list[tuple[str, str, set[str]]]:
    """Every operation in the public OpenAPI schema under /api/v1."""
    schema = create_app(ctx=ctx).openapi()
    out = []
    for path, ops in schema["paths"].items():
        if not path.startswith("/api/v1"):
            continue
        params = set(re.findall(r"{(\w+)}", path))
        for method in sorted(m for m in ops if m in HTTP_METHODS):
            out.append((method.upper(), path, params))
    return out


def _call(client: TestClient, method: str, path: str, headers: dict[str, str]):  # type: ignore[no-untyped-def]
    if path.endswith("/content"):
        return client.request(method, path, headers=headers, content=CSV_BYTES)
    body = JSON_BODIES.get((method, _template(path)))
    return client.request(method, path, headers=headers, json=body)


def _template(path: str) -> str:
    """Map a concrete path back to its route template (for the body lookup)."""
    parts = path.split("/")
    for _, template in JSON_BODIES:
        t_parts = template.split("/")
        if len(t_parts) == len(parts) and all(
            a == b or b.startswith("{") for a, b in zip(parts, t_parts, strict=True)
        ):
            return template
    return path


@pytest.fixture
def victim_and_attacker(
    make_client: ClientFactory, ctx: AppContext
) -> tuple[LoggedInUser, LoggedInUser, dict[str, str]]:
    a = signup_and_login(make_client(), ctx)
    b = signup_and_login(make_client(), ctx)
    # A's resources: a dataset that has not received content yet (so PUT content is a
    # meaningful target), an uploaded dataset with a job, and A's session.
    pending = a.client.post(
        "/api/v1/datasets",
        headers=a.headers(),
        json={"name": "pending", "filename": "p.csv", "size_bytes": len(CSV_BYTES)},
    ).json()
    uploaded = create_and_upload(a)
    session_id = a.client.get("/api/v1/auth/sessions").json()["items"][0]["id"]
    uploaded_id = uploaded["dataset"]["id"]  # type: ignore[index]
    head = a.client.get(f"/api/v1/datasets/{uploaded_id}").json()["head_version"]
    # Recommendations and an experiment for A (target declared, diagnostics re-run).
    vid = head["id"]
    a.client.put(
        f"/api/v1/versions/{vid}/roles",
        headers=a.headers(),
        json={"roles": [{"column": "outcome", "role": "target", "task_type": "binary"}]},
    )
    a.client.post(f"/api/v1/versions/{vid}/analyses", headers=a.headers())
    rec = a.client.post(f"/api/v1/versions/{vid}/recommendations", headers=a.headers())
    assert rec.status_code == 201, rec.text
    experiment = a.client.post(
        f"/api/v1/versions/{vid}/experiments",
        headers=a.headers(),
        json={
            "name": "a",
            "models": ["kmeans"],
            "task": "clustering",
            "n_clusters": 2,
            "split": {"n_splits": 2},
            "candidates": [{"label": "s", "steps": [{"op": "scale.standard", "columns": ["age"]}]}],
        },
    )
    assert experiment.status_code == 202, experiment.text
    recommendation_id = (rec.json()["recommendations"] or [{"id": str(uuid.uuid4())}])[0]["id"]
    candidate_id = a.client.get(
        f"/api/v1/experiments/{experiment.json()['experiment']['id']}"
    ).json()["candidates"][1]["id"]
    # Part E: a committed pipeline (with its transformation run) and an open draft pipeline.
    committed = a.client.post(
        f"/api/v1/datasets/{uploaded_id}/pipelines",
        headers=a.headers(),
        json={"name": "c", "steps": [{"op": "impute.median", "columns": ["age"]}]},
    ).json()
    a.client.post(f"/api/v1/pipelines/{committed['id']}/impact-preview", headers=a.headers())
    a.client.post(
        f"/api/v1/pipelines/{committed['id']}/confirm",
        headers=a.headers(),
        json={"spec_hash": committed["spec_hash"]},
    )
    commit = a.client.post(
        f"/api/v1/pipelines/{committed['id']}/commit",
        headers=a.headers(**{"Idempotency-Key": str(uuid.uuid4())}),
        json={"spec_hash": committed["spec_hash"]},
    )
    assert commit.status_code == 202 and commit.json()["job"]["status"] == "succeeded", commit.text
    draft = a.client.post(
        f"/api/v1/datasets/{uploaded_id}/pipelines",
        headers=a.headers(),
        json={"name": "d", "steps": [{"op": "impute.median", "columns": ["age"]}]},
    ).json()
    # Phase 5: a completed training run (with its reliability audit) on a larger dataset.
    trained = create_and_upload(a, cohort_csv(120), "train.csv")
    tvid = a.client.get(f"/api/v1/datasets/{trained['dataset']['id']}").json()["head_version"]["id"]  # type: ignore[index]
    a.client.put(
        f"/api/v1/versions/{tvid}/roles",
        headers=a.headers(),
        json={"roles": [{"column": "y", "role": "target", "task_type": "binary"}]},
    )
    run = a.client.post(
        f"/api/v1/versions/{tvid}/training-runs",
        headers=a.headers(**{"Idempotency-Key": str(uuid.uuid4())}),
        json={"models": ["dummy"], "tuning": {"mode": "defaults"}, "split": {"n_splits": 2}},
    )
    assert run.status_code == 202 and run.json()["job"]["status"] == "succeeded", run.text
    # Phase 8: a dashboard with one widget on the uploaded dataset's head version.
    dash = a.client.post(
        f"/api/v1/datasets/{trained['dataset']['id']}/dashboards",  # type: ignore[index]
        headers=a.headers(),
        json={"version_id": tvid, "name": "d"},
    )
    assert dash.status_code == 201, dash.text
    widget = a.client.post(
        f"/api/v1/dashboards/{dash.json()['id']}/widgets",
        headers=a.headers(),
        json={
            "kind": "kpi",
            "title": "rows",
            "spec": {"query": {"kind": "aggregate", "measures": [{"agg": "count"}]}},
        },
    )
    assert widget.status_code == 201, widget.text
    ids = {
        "dashboard_id": dash.json()["id"],
        "widget_id": widget.json()["id"],
        "training_run_id": run.json()["training_run"]["id"],
        "candidate_id": candidate_id,
        "pipeline_id": draft["id"],
        "step_id": draft["steps"][0]["id"],
        "transformation_run_id": commit.json()["transformation_run_id"],
        "other_version_id": a.client.get(f"/api/v1/datasets/{uploaded_id}").json()["head_version"][
            "id"
        ],
        "experiment_id": experiment.json()["experiment"]["id"],
        "recommendation_id": recommendation_id,
        "dataset_id": pending["dataset"]["id"],
        "uploaded_dataset_id": uploaded_id,
        "job_id": uploaded["job"]["id"],  # type: ignore[index]
        "session_id": session_id,
        "version_id": head["id"],
        "analysis_id": a.client.get(f"/api/v1/versions/{head['id']}/analyses/latest").json()["run"][
            "id"
        ],
    }
    return a, b, ids


def test_every_route_is_covered(ctx: AppContext) -> None:
    for method, path, params in _routes(ctx):
        unknown = params - KNOWN_PARAMS
        assert not unknown, f"{method} {path}: extend the isolation matrix for {unknown}"


def test_anonymous_gets_401_on_every_protected_route(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    client = make_client()
    checked = 0
    for method, path, params in _routes(ctx):
        if (method, path) in PUBLIC_ROUTES:
            continue
        concrete = path
        for p in params:
            concrete = concrete.replace("{" + p + "}", str(uuid.uuid4()))
        r = _call(client, method, concrete, {})
        assert r.status_code == 401, f"{method} {path} -> {r.status_code}"
        checked += 1
    assert checked >= 10


def test_user_b_cannot_touch_user_a_resources(ctx: AppContext, victim_and_attacker) -> None:  # type: ignore[no-untyped-def]
    a, b, ids = victim_and_attacker
    checked = 0
    for method, path, params in _routes(ctx):
        if not params:
            continue
        concrete = path
        for p in params:
            concrete = concrete.replace("{" + p + "}", ids[p])
        r = _call(b.client, method, concrete, b.headers())
        assert r.status_code == 404, f"B {method} {path} -> {r.status_code}: {r.text}"
        assert r.json()["code"] == "NOT_FOUND"
        checked += 1
    assert checked >= 6

    # The uploaded dataset and job are equally invisible.
    assert b.client.get(f"/api/v1/datasets/{ids['uploaded_dataset_id']}").status_code == 404
    for sub in ("versions", "parse"):
        method = "GET" if sub == "versions" else "POST"
        r = b.client.request(
            method, f"/api/v1/datasets/{ids['uploaded_dataset_id']}/{sub}", headers=b.headers()
        )
        assert r.status_code == 404
    assert (
        b.client.delete(
            f"/api/v1/datasets/{ids['uploaded_dataset_id']}", headers=b.headers()
        ).status_code
        == 404
    )

    # Lists only ever show the caller's own resources, whatever the query string says.
    assert b.client.get("/api/v1/datasets").json()["items"] == []
    assert b.client.get(f"/api/v1/jobs?owner_id={a.user_id}").json()["items"] == []
    b_sessions = b.client.get("/api/v1/auth/sessions").json()["items"]
    assert ids["session_id"] not in {s["id"] for s in b_sessions}

    # A's data is intact after all of B's attempts.
    a_ds = a.client.get(f"/api/v1/datasets/{ids['dataset_id']}")
    assert a_ds.status_code == 200 and a_ds.json()["status"] == "upload_pending"
    assert a.client.get("/api/v1/auth/me").status_code == 200
    assert a.client.get(f"/api/v1/jobs/{ids['job_id']}").json()["status"] == "succeeded"
    # B's profile request did not create a job for A's version.
    profile = a.client.get(f"/api/v1/versions/{ids['version_id']}/profile").json()
    assert profile["profile"] is not None and profile["job"]["status"] == "succeeded"
    types = [j["type"] for j in a.client.get("/api/v1/jobs").json()["items"]]
    # Ingest's profile and analysis, the re-run analysis, the commit and its follow-ups.
    assert types.count("profile") == 3 and types.count("analysis") == 4
    assert types.count("experiment") == 1  # B created no experiment for A
    assert types.count("commit") == 1 and types.count("revert") == 0
    assert types.count("train") == 1 and types.count("reliability_audit") == 1
    # B could not change A's draft pipeline or create versions.
    a_draft = a.client.get(f"/api/v1/pipelines/{ids['pipeline_id']}").json()
    assert [s["op"] for s in a_draft["steps"]] == ["impute.median"] and not a_draft["approvals"]
    a_versions = a.client.get(f"/api/v1/datasets/{ids['uploaded_dataset_id']}/versions").json()
    assert [v["kind"] for v in a_versions["items"]] == ["ingest", "transform"]
    # B's PATCH did not change A's domain; B's PUT did not touch A's roles.
    assert a.client.get(f"/api/v1/datasets/{ids['dataset_id']}").json()["domain"] == "generic"
    roles = a.client.get(f"/api/v1/versions/{ids['version_id']}/roles").json()["roles"]
    assert [(r["column"], r["role"]) for r in roles] == [("outcome", "target")]
    # B's PATCH did not dismiss A's recommendation.
    rec_state = a.client.get(f"/api/v1/versions/{ids['version_id']}/recommendations").json()
    assert all(r["status"] == "open" for r in rec_state["set"]["recommendations"])

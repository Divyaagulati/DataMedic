"""Diagnostics end to end through the API: chained run, roles, domains, staleness, re-runs,
database constraints and cleanup. Synthetic data only."""

import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError

from datamedic.context import AppContext, build_context
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


def cohort_csv(n: int = 300) -> bytes:
    rng = np.random.default_rng(0)
    patients = np.repeat([f"PT{i:03d}" for i in range(n // 3)], 3)
    frame = pd.DataFrame(
        {
            "patient": patients,
            "site": rng.choice(["A", "B"], n),
            "hr": rng.normal(75, 10, n).round(1),
            "crp": rng.lognormal(1, 0.5, n).round(2),
            "dx": np.repeat(rng.choice([0, 1], n // 3), 3),
        }
    )
    frame.loc[0, "hr"] = 999.0
    frame.loc[:59, "crp"] = None
    return frame.to_csv(index=False).encode()


def _setup(user: LoggedInUser) -> tuple[str, str]:
    body = create_and_upload(user, cohort_csv(), "cohort.csv")
    ds_id = body["dataset"]["id"]  # type: ignore[index]
    vid = user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"]
    return ds_id, vid


def _latest(user: LoggedInUser, vid: str) -> dict:  # type: ignore[type-arg]
    r = user.client.get(f"/api/v1/versions/{vid}/analyses/latest")
    assert r.status_code == 200, r.text
    return r.json()  # type: ignore[no-any-return]


def _run(user: LoggedInUser, vid: str) -> dict:  # type: ignore[type-arg]
    r = user.client.post(f"/api/v1/versions/{vid}/analyses", headers=user.headers())
    assert r.status_code == 202, r.text
    assert user.client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()["status"] == "succeeded"
    return _latest(user, vid)


def test_diagnostics_run_automatically_after_upload(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    user = signup_and_login(make_client(), ctx)
    _, vid = _setup(user)
    state = _latest(user, vid)
    assert state["job"]["status"] == "succeeded"
    run = state["run"]
    assert run["domain"] == "generic" and run["stale"] is False
    assert run["summary"]["check_errors"] == 0
    by_check = {f["check_id"]: f for f in run["findings"]}
    missing = [f for f in run["findings"] if f["check_id"] == "universal.missingness.column_rate"]
    assert missing[0]["status"] == "detected" and missing[0]["affected_rows"] == 60
    assert by_check["universal.target.imbalance"]["status"] == "not_assessed"  # no target yet
    for f in run["findings"]:
        assert f["limitations"] and f["explanation"]
        if f["status"] == "detected":
            assert f["evidence"] and f["recommendation"]
    assert user.client.get(f"/api/v1/analyses/{run['id']}").json()["id"] == run["id"]


def test_domain_roles_staleness_and_rerun(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    ds_id, vid = _setup(user)
    first = _latest(user, vid)["run"]

    r = user.client.patch(
        f"/api/v1/datasets/{ds_id}", headers=user.headers(), json={"domain": "biomedical"}
    )
    assert r.status_code == 200 and r.json()["domain"] == "biomedical"
    assert r.json()["domain_changed_at"] is not None
    assert _latest(user, vid)["run"]["stale"] is True  # the domain changed after the run

    roles = user.client.get(f"/api/v1/versions/{vid}/roles").json()
    assert {r["key"] for r in roles["available_roles"]} >= {"target", "bio.patient_id", "bio.batch"}
    r = user.client.put(
        f"/api/v1/versions/{vid}/roles",
        headers=user.headers(),
        json={
            "roles": [
                {"column": "dx", "role": "target", "task_type": "binary"},
                {"column": "patient", "role": "bio.patient_id"},
                {"column": "site", "role": "bio.batch"},
                {"column": "hr", "role": "bio.heart_rate"},
            ]
        },
    )
    assert r.status_code == 200, r.text
    assert len(r.json()["roles"]) == 4

    second = _run(user, vid)["run"]
    assert (
        second["id"] != first["id"]
        and second["stale"] is False
        and second["domain"] == "biomedical"
    )
    found = {f["check_id"]: f for f in second["findings"]}
    assert found["bio.leakage.patient_level"]["status"] == "detected"
    assert found["bio.clinical.plausibility"]["status"] == "detected"
    assert found["bio.clinical.plausibility"]["affected_rows"] == 1
    assert found["universal.target.imbalance"]["status"] in ("detected", "not_detected")
    # The earlier run is kept and can still be read.
    assert user.client.get(f"/api/v1/analyses/{first['id']}").json()["stale"] is True


@pytest.mark.parametrize(
    "roles,code",
    [
        ([{"column": "nope", "role": "target", "task_type": "binary"}], "COLUMN_NOT_FOUND"),
        ([{"column": "dx", "role": "target"}], "TASK_TYPE_REQUIRED"),
        ([{"column": "dx", "role": "target", "task_type": "multiclass"}], "TASK_TYPE_MISMATCH"),
        ([{"column": "site", "role": "target", "task_type": "regression"}], "TASK_TYPE_MISMATCH"),
        ([{"column": "hr", "role": "id", "task_type": "binary"}], "TASK_TYPE_NOT_ALLOWED"),
        ([{"column": "patient", "role": "bio.patient_id"}], "ROLE_NOT_AVAILABLE"),
        ([{"column": "site", "role": "weight"}], "ROLE_TYPE_MISMATCH"),
        ([{"column": "hr", "role": "id"}, {"column": "hr", "role": "id"}], "DUPLICATE_ROLE"),
        (
            [
                {"column": "dx", "role": "target", "task_type": "binary"},
                {"column": "hr", "role": "target", "task_type": "regression"},
            ],
            "TOO_MANY",
        ),
    ],
)
def test_role_validation(
    make_client: ClientFactory, ctx: AppContext, roles: list, code: str
) -> None:  # type: ignore[type-arg]
    user = signup_and_login(make_client(), ctx)
    _, vid = _setup(user)  # domain is generic: bio.* roles are not available
    r = user.client.put(
        f"/api/v1/versions/{vid}/roles", headers=user.headers(), json={"roles": roles}
    )
    assert r.status_code == 422 and r.json()["code"] == "VALIDATION_FAILED"
    assert code in {e["code"] for e in r.json()["errors"]}
    assert user.client.get(f"/api/v1/versions/{vid}/roles").json()["roles"] == []  # unchanged


def test_unknown_domain_and_extra_fields_rejected(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    user = signup_and_login(make_client(), ctx)
    ds_id, _ = _setup(user)
    r = user.client.patch(
        f"/api/v1/datasets/{ds_id}", headers=user.headers(), json={"domain": "astro"}
    )
    assert r.status_code == 422 and r.json()["errors"][0]["code"] == "UNKNOWN_DOMAIN"
    r = user.client.patch(
        f"/api/v1/datasets/{ds_id}", headers=user.headers(), json={"owner_id": str(uuid.uuid4())}
    )
    assert r.status_code == 422


def test_catalogue_endpoints(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    domains = user.client.get("/api/v1/domains").json()["items"]
    assert [d["key"] for d in domains] == [
        "generic",
        "biomedical",
        "financial",
        "business",
        "industrial",
    ]
    checks = user.client.get("/api/v1/checks", params={"domain": "industrial"}).json()["items"]
    ids = {c["id"] for c in checks}
    assert "ind.sensors.drift" in ids and "universal.outliers.multivariate" in ids
    assert not any(i.startswith("bio.") for i in ids)
    assert all(c["limitations"] for c in checks)


def test_analysis_conflict_while_active(
    make_client: ClientFactory, database: DbUrls, tmp_path: Path
) -> None:
    context = build_context(
        make_settings(database, tmp_path, jobs_eager=False, redis_url="memory://")
    )
    context.publisher = CeleryJobPublisher(create_celery(context.settings))
    try:
        user = signup_and_login(make_client(context), context)
        body = create_and_upload(user, cohort_csv(), "cohort.csv")
        run_job(context, body["job"]["id"])  # type: ignore[index]
        vid = user.client.get(f"/api/v1/datasets/{body['dataset']['id']}").json()["head_version"][
            "id"
        ]  # type: ignore[index]
        state = _latest(user, vid)
        assert state["run"] is None and state["job"]["status"] == "queued"
        r = user.client.post(f"/api/v1/versions/{vid}/analyses", headers=user.headers())
        assert r.status_code == 409
        run_job(context, state["job"]["id"])
        assert _latest(user, vid)["run"] is not None
    finally:
        context.dispose()


def test_findings_constraints_and_immutability(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    _, vid = _setup(user)
    run_id = _latest(user, vid)["run"]["id"]
    with pytest.raises(ProgrammingError, match="permission denied"):
        with session_scope(ctx.sessions) as db:
            bind_user(db, user.user_id)
            db.execute(
                text("UPDATE findings SET severity = 'info' WHERE analysis_run_id = :r"),
                {"r": run_id},
            )
    with pytest.raises(DBAPIError, match="ck_findings_not_assessed_reason"):
        with session_scope(ctx.sessions) as db:
            bind_user(db, user.user_id)
            db.execute(
                text(
                    "INSERT INTO findings (id, owner_id, dataset_id, analysis_run_id, position, "
                    "check_id, check_version, domain, category, status, severity, title, "
                    "explanation, evidence, columns, limitations) SELECT gen_random_uuid(), "
                    "owner_id, dataset_id, id, 999, 'x', '1', 'universal', 'x', 'not_assessed', "
                    "'info', 't', 'e', '{}', '[]', '[\"l\"]' FROM analysis_runs WHERE id = :r"
                ),
                {"r": run_id},
            )


def test_delete_removes_diagnostics(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    ds_id, vid = _setup(user)
    user.client.put(
        f"/api/v1/versions/{vid}/roles",
        headers=user.headers(),
        json={"roles": [{"column": "dx", "role": "target", "task_type": "binary"}]},
    )
    r = user.client.delete(f"/api/v1/datasets/{ds_id}", headers=user.headers())
    assert user.client.get(f"/api/v1/jobs/{r.json()['job']['id']}").json()["status"] == "succeeded"
    with session_scope(ctx.sessions) as db:
        bind_user(db, user.user_id)
        for table in ("findings", "analysis_runs", "column_roles"):
            assert (
                db.execute(
                    text(f"SELECT count(*) FROM {table} WHERE dataset_id = :d"),  # noqa: S608
                    {"d": ds_id},
                ).scalar_one()
                == 0
            ), table


def test_rls_hides_other_users_diagnostics(make_client: ClientFactory, ctx: AppContext) -> None:
    a = signup_and_login(make_client(), ctx)
    b = signup_and_login(make_client(), ctx)
    _, vid = _setup(a)
    run_id = _latest(a, vid)["run"]["id"]
    assert b.client.get(f"/api/v1/analyses/{run_id}").status_code == 404
    assert b.client.get(f"/api/v1/versions/{vid}/analyses/latest").status_code == 404
    with session_scope(ctx.sessions) as db:
        bind_user(db, b.user_id)
        for table in ("findings", "analysis_runs", "column_roles"):
            assert db.execute(text(f"SELECT count(*) FROM {table}")).scalar_one() == 0  # noqa: S608

"""Phase 8 Analytics Studio through the API: queries, exports, version comparison, insights,
dashboards (CRUD, explicit version binding), natural-language Copilot, privacy, isolation and
RLS. Synthetic data only; expected values are computed independently with pandas."""

import uuid
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from datamedic.context import AppContext
from datamedic.db.session import bind_user, session_scope
from tests.conftest import ClientFactory, LoggedInUser, create_and_upload, signup_and_login
from tests.integration.test_mvp_api import _csv as bio_csv

pytestmark = pytest.mark.integration


def business_frame(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(4)
    return pd.DataFrame(
        {
            "region": rng.choice(["North", "South", "East"], n),
            "product": rng.choice(["A", "B"], n),
            "amount": np.round(rng.gamma(2, 40, n), 2),
            "units": rng.integers(1, 8, n),
            "date": pd.date_range("2025-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
            "note": ["=SUM(A1)", "ok", "x"] * (n // 3),
        }
    )


@pytest.fixture
def biz(make_client: ClientFactory, ctx: AppContext) -> tuple[LoggedInUser, str, str, pd.DataFrame]:
    user = signup_and_login(make_client(), ctx)
    df = business_frame()
    body = create_and_upload(user, df.to_csv(index=False).encode(), "sales.csv")
    ds_id = str(body["dataset"]["id"])  # type: ignore[index]
    vid = user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"]
    return user, ds_id, vid, df


def _q(user: LoggedInUser, vid: str, spec: dict[str, Any]) -> Any:
    return user.client.post(
        f"/api/v1/versions/{vid}/analytics/query", headers=user.headers(), json={"spec": spec}
    )


def test_columns_query_and_export(biz) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid, df = biz
    cat = user.client.get(f"/api/v1/versions/{vid}/analytics/columns").json()
    types = {c["name"]: c["inferred_type"] for c in cat["columns"]}
    assert types["amount"] == "numeric" and types["date"] == "datetime" and not cat["privacy_mode"]
    spec = {
        "kind": "aggregate",
        "group_by": [{"column": "region"}],
        "measures": [{"agg": "mean", "column": "amount"}],
        "filters": [{"column": "units", "op": "gte", "value": 3}],
        "sort": {"by": "measure", "index": 0, "dir": "desc"},
    }
    r = _q(user, vid, spec)
    assert r.status_code == 200 and r.headers["cache-control"] == "no-store", r.text
    expected = df[df["units"] >= 3].groupby("region")["amount"].mean()
    rows = r.json()["result"]["rows"]
    assert rows[0]["groups"] == [expected.idxmax()] and rows[0]["values"][0] == pytest.approx(
        expected.max(), abs=1e-6
    )
    assert (
        _q(
            user, vid, {"kind": "aggregate", "measures": [{"agg": "sum", "column": "region"}]}
        ).status_code
        == 422
    )
    assert _q(user, vid, {"kind": "sql", "query": "SELECT 1"}).status_code == 422
    csv = user.client.post(
        f"/api/v1/versions/{vid}/analytics/export",
        headers=user.headers(),
        json={
            "spec": {
                "kind": "aggregate",
                "group_by": [{"column": "note"}],
                "measures": [{"agg": "count"}],
            }
        },
    )
    assert csv.status_code == 200 and csv.headers["content-disposition"].startswith(
        'attachment; filename="Cohort_v0_aggregate.csv"'
    )
    assert "'=SUM(A1)" in csv.text and "\n=SUM" not in csv.text
    assert (
        user.client.post(
            f"/api/v1/versions/{vid}/analytics/export",
            headers=user.headers(),
            json={"spec": {"kind": "scatter", "columns": ["amount", "units"]}},
        ).status_code
        == 422
    )


def test_dashboards_crud_and_version_binding(biz) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid, _ = biz
    h = user.headers()
    d = user.client.post(
        f"/api/v1/datasets/{ds_id}/dashboards",
        headers=h,
        json={
            "version_id": vid,
            "name": "Sales",
            "description": "Q1",
            "filters": [{"column": "region", "op": "ne", "value": "East"}],
        },
    )
    assert d.status_code == 201, d.text
    did = d.json()["id"]
    chart = {
        "kind": "chart",
        "title": "Avg by region",
        "spec": {
            "query": {
                "kind": "aggregate",
                "group_by": [{"column": "region"}],
                "measures": [{"agg": "mean", "column": "amount"}],
            },
            "display": {"chart_type": "bar", "x_label": "Region"},
        },
    }
    kpi = {
        "kind": "kpi",
        "title": "Rows",
        "size": "small",
        "spec": {"query": {"kind": "aggregate", "measures": [{"agg": "count"}]}, "display": {}},
    }
    w1 = user.client.post(f"/api/v1/dashboards/{did}/widgets", headers=h, json=chart).json()
    w2 = user.client.post(f"/api/v1/dashboards/{did}/widgets", headers=h, json=kpi).json()
    bad = user.client.post(
        f"/api/v1/dashboards/{did}/widgets",
        headers=h,
        json={**chart, "spec": {**chart["spec"], "display": {"chart_type": "histogram"}}},
    )
    assert bad.status_code == 422
    dup = user.client.post(
        f"/api/v1/dashboards/{did}/widgets/{w1['id']}/duplicate", headers=h
    ).json()
    user.client.patch(
        f"/api/v1/dashboards/{did}/widgets/{w2['id']}",
        headers=h,
        json={"title": "Row count", "size": "large"},
    )
    r = user.client.post(
        f"/api/v1/dashboards/{did}/reorder",
        headers=h,
        json={"widget_ids": [w2["id"], dup["id"], w1["id"]]},
    )
    assert [w["id"] for w in r.json()["widgets"]] == [w2["id"], dup["id"], w1["id"]]
    assert (
        user.client.delete(f"/api/v1/dashboards/{did}/widgets/{dup['id']}", headers=h).status_code
        == 204
    )
    full = user.client.get(f"/api/v1/dashboards/{did}").json()
    assert full["version_id"] == vid and [w["title"] for w in full["widgets"]] == [
        "Row count",
        "Avg by region",
    ]
    listed = user.client.get(f"/api/v1/datasets/{ds_id}/dashboards").json()["items"]
    assert listed[0]["widget_count"] == 2
    exp = user.client.get(f"/api/v1/dashboards/{did}/export")
    assert (
        exp.status_code == 200
        and exp.json()["format"] == "datamedic.dashboard/1"
        and "rows" not in exp.text
    )
    copy = user.client.post(f"/api/v1/dashboards/{did}/duplicate", headers=h, json={}).json()
    assert copy["dashboard"]["widget_count"] == 2 and copy["skipped_widgets"] == []
    # Switching versions is explicit only; another dataset's version is refused.
    assert (
        user.client.post(
            f"/api/v1/dashboards/{did}/switch-version", headers=h, json={"version_id": vid}
        ).status_code
        == 422
    )
    compat = user.client.get(
        f"/api/v1/dashboards/{did}/compatibility", params={"version_id": vid}
    ).json()
    assert compat["compatible"]
    assert (
        user.client.delete(f"/api/v1/dashboards/{copy['dashboard']['id']}", headers=h).status_code
        == 204
    )


def test_dashboard_on_new_version_explains_incompatible_widgets(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    from tests.integration.test_pipelines_api import _setup
    from tests.integration.test_training_api import _commit_v1

    user = signup_and_login(make_client(), ctx)
    ds_id, v0 = _setup(user)
    h = user.headers()
    did = user.client.post(
        f"/api/v1/datasets/{ds_id}/dashboards", headers=h, json={"version_id": v0, "name": "D"}
    ).json()["id"]
    hist = {
        "kind": "chart",
        "title": "a hist",
        "spec": {
            "query": {"kind": "histogram", "columns": ["a"]},
            "display": {"chart_type": "histogram"},
        },
    }
    b = {
        "kind": "chart",
        "title": "b hist",
        "spec": {
            "query": {"kind": "histogram", "columns": ["b"]},
            "display": {"chart_type": "histogram"},
        },
    }
    user.client.post(f"/api/v1/dashboards/{did}/widgets", headers=h, json=hist)
    user.client.post(f"/api/v1/dashboards/{did}/widgets", headers=h, json=b)
    v1 = _commit_v1(user, ds_id)
    compat = user.client.get(
        f"/api/v1/dashboards/{did}/compatibility", params={"version_id": v1}
    ).json()
    assert compat["compatible"] is True  # a and b exist in v1
    # Mark b sensitive on v1: its widget becomes incompatible there.
    user.client.put(
        f"/api/v1/versions/{v1}/roles",
        headers=h,
        json={
            "roles": [
                {"column": "y", "role": "target", "task_type": "binary"},
                {"column": "b", "role": "sensitive"},
            ]
        },
    )
    compat = user.client.get(
        f"/api/v1/dashboards/{did}/compatibility", params={"version_id": v1}
    ).json()
    assert not compat["compatible"] and any(
        "sensitive" in p for w in compat["widgets"] for p in w["problems"]
    )
    refused = user.client.post(
        f"/api/v1/dashboards/{did}/switch-version",
        headers=h,
        json={"version_id": v1, "confirm": True},
    )
    assert refused.status_code == 422
    assert (
        user.client.get(f"/api/v1/dashboards/{did}").json()["version_id"] == v0
    )  # never switched silently
    dup = user.client.post(
        f"/api/v1/dashboards/{did}/duplicate", headers=h, json={"version_id": v1}
    ).json()
    assert dup["dashboard"]["version_id"] == v1 and dup["skipped_widgets"] == ["b hist"]


def test_copilot_analytics_compare_and_insights(biz) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid, df = biz

    def ask(q: str) -> Any:
        return user.client.post(
            f"/api/v1/versions/{vid}/copilot", headers=user.headers(), json={"question": q}
        ).json()

    a = ask("Which region has the highest average amount?")
    expected = df.groupby("region")["amount"].mean()
    assert a["analysis"]["intent"] == "top_group" and a["analysis"]["spec"]["kind"] == "aggregate"
    assert (
        f"{expected.idxmax()} has the highest mean(amount): {round(expected.max(), 6)}"
        in " ".join(a["answer"])
    )
    assert ask("Show average salary by region")["analysis"]["status"] == "unknown_column"
    assert ask("Can I trust the model results?")["topic"] == "reliability"  # report-level fallback
    m = ask("Which numerical column has the most missing values?")
    assert m["analysis"]["intent"] == "most_missing"
    r = user.client.post(f"/api/v1/versions/{vid}/insights", headers=user.headers())
    assert r.status_code == 202 and r.json()["job"]["status"] == "succeeded", r.text
    ins = user.client.get(f"/api/v1/versions/{vid}/insights").json()
    assert ins["summary"]["insights"] and all(
        i["basis"] in ("descriptive", "association", "suspicion", "configuration")
        for i in ins["summary"]["insights"]
    )
    p = user.client.get(f"/api/v1/versions/{vid}/passport").json()
    assert p["analytics"]["insights"]["status"] == "available"


def test_privacy_mode_and_isolation(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    body = create_and_upload(user, bio_csv(), "bio.csv")
    ds_id = str(body["dataset"]["id"])  # type: ignore[index]
    user.client.patch(
        f"/api/v1/datasets/{ds_id}", headers=user.headers(), json={"domain": "biomedical"}
    )
    vid = user.client.get(f"/api/v1/datasets/{ds_id}").json()["head_version"]["id"]
    user.client.put(
        f"/api/v1/versions/{vid}/roles",
        headers=user.headers(),
        json={
            "roles": [
                {"column": "y", "role": "target", "task_type": "binary"},
                {"column": "code", "role": "sensitive"},
            ]
        },
    )
    cat = user.client.get(f"/api/v1/versions/{vid}/analytics/columns").json()
    assert (
        cat["privacy_mode"] and next(c for c in cat["columns"] if c["name"] == "code")["protected"]
    )
    blocked = _q(
        user,
        vid,
        {"kind": "aggregate", "group_by": [{"column": "code"}], "measures": [{"agg": "count"}]},
    )
    assert blocked.status_code == 422 and "sensitive" in blocked.text
    scatter = _q(user, vid, {"kind": "scatter", "columns": ["marker", "noise"]}).json()["result"]
    assert scatter["mode"] == "binned" and "points" not in scatter
    small = _q(
        user,
        vid,
        {
            "kind": "aggregate",
            "filters": [{"column": "patient", "op": "lt", "value": 2}],
            "measures": [{"agg": "mean", "column": "marker"}],
        },
    ).json()["result"]
    assert small["suppressed"]
    ask = user.client.post(
        f"/api/v1/versions/{vid}/copilot",
        headers=user.headers(),
        json={"question": "count by code"},
    ).json()
    assert "sensitive" in " ".join(ask["answer"])
    other = signup_and_login(make_client(), ctx)
    assert _q(other, vid, {"kind": "aggregate", "measures": [{"agg": "count"}]}).status_code == 404
    did = user.client.post(
        f"/api/v1/datasets/{ds_id}/dashboards",
        headers=user.headers(),
        json={"version_id": vid, "name": "P"},
    ).json()["id"]
    assert other.client.get(f"/api/v1/dashboards/{did}").status_code == 404
    assert (
        other.client.delete(f"/api/v1/dashboards/{did}", headers=other.headers()).status_code == 404
    )
    before = user.client.get(f"/api/v1/versions/{vid}").json()["content_hash"]
    _q(
        user,
        vid,
        {
            "kind": "aggregate",
            "group_by": [{"column": "batch"}],
            "measures": [{"agg": "mean", "column": "marker"}],
        },
    )
    assert user.client.get(f"/api/v1/versions/{vid}").json()["content_hash"] == before
    with session_scope(ctx.sessions) as db:
        bind_user(db, other.user_id)
        assert db.execute(text("SELECT count(*) FROM dashboards")).scalar_one() == 0
        assert db.execute(text("SELECT count(*) FROM dashboard_widgets")).scalar_one() == 0
    with session_scope(ctx.sessions) as db:
        assert db.execute(text("SELECT count(*) FROM dashboards")).scalar_one() == 0  # fails closed
    with pytest.raises(DBAPIError), session_scope(ctx.sessions) as db:
        bind_user(db, other.user_id)
        db.execute(
            text(
                "INSERT INTO dashboards (id, owner_id, dataset_id, dataset_version_id, name) VALUES "
                "(:id, :o, :d, :v, 'x')"
            ),
            {"id": uuid.uuid4(), "o": other.user_id, "d": ds_id, "v": vid},
        )


def test_version_compare(make_client: ClientFactory, ctx: AppContext) -> None:
    from tests.integration.test_pipelines_api import _setup
    from tests.integration.test_training_api import _commit_v1

    user = signup_and_login(make_client(), ctx)
    ds_id, v0 = _setup(user)
    v1 = _commit_v1(user, ds_id)  # missing indicator + median imputation on b
    r = user.client.get(f"/api/v1/versions/{v0}/compare/{v1}")
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["columns_added"] == ["b__missing"] and c["rows"]["in_both"] == 240
    b = next(x for x in c["columns"] if x["column"] == "b")
    assert b["missing_pct"][0] > 0 and b["missing_pct"][1] == 0 and len(b["edges"]) == 21
    # The fixture declares a sensitive column, so privacy mode suppresses bins under 10 rows.
    assert c["privacy_mode"] and all(x is None or x >= 10 or x == 0 for x in b["histogram"][1])
    assert 0 < sum(x or 0 for x in b["histogram"][1]) <= 240
    effects = c["transformation_effects"]
    assert effects and any(col.get("cells_filled") for e in effects for col in e["columns"])


def test_saved_chart_configuration_round_trips_exactly(biz) -> None:  # type: ignore[no-untyped-def]
    user, ds_id, vid, df = biz
    h = user.headers()
    did = user.client.post(
        f"/api/v1/datasets/{ds_id}/dashboards", headers=h, json={"version_id": vid, "name": "Sum"}
    ).json()["id"]
    spec = {
        "query": {
            "kind": "aggregate",
            "group_by": [{"column": "region"}],
            "measures": [{"agg": "sum", "column": "amount"}],
            "filters": [
                {"column": "units", "op": "gte", "value": 2},
                {"column": "product", "op": "in", "values": ["A", "B"]},
            ],
            "sort": {"by": "measure", "index": 0, "dir": "asc"},
            "limit": 20,
        },
        "display": {
            "chart_type": "hbar",
            "title": "Total amount by region",
            "x_label": "Region",
            "y_label": "Amount (sum)",
        },
    }
    w = user.client.post(
        f"/api/v1/dashboards/{did}/widgets",
        headers=h,
        json={"kind": "chart", "title": "Total amount by region", "size": "large", "spec": spec},
    )
    assert w.status_code == 201 and w.json()["spec"] == spec
    saved = user.client.get(f"/api/v1/dashboards/{did}").json()
    assert saved["version_id"] == vid and saved["widgets"][0]["spec"] == spec
    assert saved["widgets"][0]["size"] == "large"
    exported = user.client.get(f"/api/v1/dashboards/{did}/export").json()
    assert exported["version_id"] == vid and exported["dataset_id"] == ds_id
    assert exported["widgets"][0]["spec"] == spec and exported["widgets"][0]["kind"] == "chart"
    # The saved query computes the same numbers as the configured chart.
    r = user.client.post(
        f"/api/v1/versions/{vid}/analytics/query",
        headers=h,
        json={"spec": saved["widgets"][0]["spec"]["query"]},
    ).json()["result"]
    sub = df[(df["units"] >= 2) & df["product"].isin(["A", "B"])]
    expected = sub.groupby("region")["amount"].sum().sort_values()
    assert [row["groups"][0] for row in r["rows"]] == list(expected.index)
    assert [row["values"][0] for row in r["rows"]] == pytest.approx(list(expected.values), abs=1e-6)

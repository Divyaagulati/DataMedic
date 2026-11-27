"""The preprocessing op catalogue endpoint (Part A)."""

import pytest

from datamedic.context import AppContext
from datamedic.engine.preprocess.catalog import all_ops
from tests.conftest import ClientFactory, signup_and_login

pytestmark = pytest.mark.integration


def test_catalogue_lists_every_op_with_server_classification(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    user = signup_and_login(make_client(), ctx)
    r = user.client.get("/api/v1/ops")
    assert r.status_code == 200, r.text
    items = {item["op"]: item for item in r.json()["items"]}
    assert set(items) == {s.op for s in all_ops()}
    knn = items["impute.knn"]
    assert knn["risk"] == "medium" and knn["kind"] == "fitted" and knn["stage"] == "either"
    assert knn["destructive"] is False and knn["column_types"] == ["numeric"]
    assert {p["name"] for p in knn["params"]} == {"n_neighbors", "weights", "predictors"}
    assert items["missing.indicator"]["adds_columns"] is True
    assert items["impute.constant"]["params"][0]["required"] is True
    assert all(item["downsides"] and item["limitations"] for item in items.values())


def test_catalogue_family_filter(make_client: ClientFactory, ctx: AppContext) -> None:
    user = signup_and_login(make_client(), ctx)
    missing = [s for s in all_ops() if s.family == "missing"]
    assert len(user.client.get("/api/v1/ops?family=missing").json()["items"]) == len(missing)
    assert user.client.get("/api/v1/ops?family=nope").json()["items"] == []


def test_catalogue_requires_authentication(make_client: ClientFactory) -> None:
    r = make_client().get("/api/v1/ops")
    assert r.status_code == 401
    assert r.json()["code"] == "AUTH_REQUIRED"


def test_catalogue_serialises_part_b_classification(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    user = signup_and_login(make_client(), ctx)
    items = {i["op"]: i for i in user.client.get("/api/v1/ops").json()["items"]}
    assert items["outliers.cap"]["destructive"] is True and items["outliers.cap"]["risk"] == "high"
    assert items["encode.onehot"]["stage"] == "model" and items["scale.robust"]["stage"] == "model"
    order = next(p for p in items["encode.ordinal"]["params"] if p["name"] == "order")
    assert order["type"] == "text_list" and order["required"] is True
    families = {
        f: len(user.client.get(f"/api/v1/ops?family={f}").json()["items"])
        for f in ("missing", "outliers", "encoding", "scaling")
    }
    assert families == {"missing": 7, "outliers": 4, "encoding": 3, "scaling": 4}


def test_catalogue_serialises_part_c_classification(
    make_client: ClientFactory, ctx: AppContext
) -> None:
    user = signup_and_login(make_client(), ctx)
    items = {i["op"]: i for i in user.client.get("/api/v1/ops").json()["items"]}
    smote = items["imbalance.smote"]
    assert smote["kind"] == "training_only" and smote["stage"] == "model"
    assert smote["uses_columns"] is False and smote["needs_target"] is True
    assert items["select.mutual_information"]["stage"] == "model"
    assert items["select.correlation"]["destructive"] is False
    assert items["transform.quantile"]["destructive"] is True
    assert items["transform.log"]["uses_columns"] is True
    # Resolved parameters (target, task, role_kept, passthrough) are never request params.
    for item in items.values():
        names = {p["name"] for p in item["params"]}
        assert not names & {"target", "task", "role_kept", "passthrough", "discrete"}
    families = {
        f: len(user.client.get(f"/api/v1/ops?family={f}").json()["items"])
        for f in ("transform", "selection", "imbalance")
    }
    assert families == {"transform": 4, "selection": 5, "imbalance": 4}

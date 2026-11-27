"""Recommendation engine (Phase 4 Part D): evidence, honesty (not_assessed / not
recommended), domain policies, conflicts with trade-offs, transparent ranking, candidate
pipelines and reproducibility. Synthetic data only; inputs come from the real profiler and
diagnostics."""

import json
from typing import Any

import numpy as np
import pandas as pd
import pytest

from datamedic.engine.preprocess.recommend import recommend
from datamedic.engine.preprocess.recommend.facts import Facts
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import frame
from tests.unit.recommend_helpers import clinical_frame, make_input, recs_for

TARGET = {"target": ["y"]}
REQUIRED = (
    "id",
    "op",
    "params",
    "columns",
    "reason",
    "evidence",
    "expected_benefit",
    "risks",
    "information_loss",
    "risk",
    "domain_compatibility",
    "task_compatibility",
    "required_decisions",
    "stage",
    "evidence_strength",
    "limitations",
    "ranking",
    "not_a_guarantee",
)


@pytest.fixture(scope="module")
def clinical() -> dict[str, Any]:
    return recs_for(clinical_frame(), TARGET, task="binary")


def ops(out: dict[str, Any]) -> list[str]:
    return [r["op"] for r in out["recommendations"]]


def by_op(out: dict[str, Any], op: str) -> dict[str, Any]:
    return next(r for r in out["recommendations"] if r["op"] == op)


def test_every_recommendation_carries_the_required_explanation(clinical: dict) -> None:  # type: ignore[type-arg]
    assert clinical["recommendations"]
    for rec in clinical["recommendations"]:
        row_op = rec["kind"] == "training_only"  # acts on whole training rows, no columns
        for key in REQUIRED:
            optional = key == "params" or (key == "columns" and row_op)
            assert rec[key] not in (None, "", []) or optional, (rec["op"], key)
        assert rec["applies_to"] and (not row_op or "target" in rec["applies_to"])
        assert rec["stage"] in ("data", "model", "either")
        assert rec["evidence_strength"]["level"] in ("strong", "moderate", "weak")
        assert rec["domain_compatibility"]["status"] in ("compatible", "caution")
        assert all("source" in e and "metric" in e for e in rec["evidence"])
        assert "not a guaranteed improvement" in rec["not_a_guarantee"]
    json.dumps(clinical, allow_nan=False)


def test_recommendations_follow_measured_evidence(clinical: dict) -> None:  # type: ignore[type-arg]
    assert by_op(clinical, "impute.median")["columns"] == ["b"]  # the only incomplete column
    assert by_op(clinical, "transform.log")["columns"] == ["lab"]  # positive, right-skewed
    assert "imbalance.class_weight" in ops(clinical)  # 8% minority detected by diagnostics
    assert by_op(clinical, "select.correlation")["columns"] == ["a", "b"]  # r ~ 0.95
    assert "mean" not in " ".join(ops(clinical))  # mean imputation is never suggested


def test_recommendations_do_not_depend_on_column_names() -> None:
    """Renaming every column leaves the recommended operations and evidence unchanged."""
    df = clinical_frame()
    names = {"a": "glucose_mg_dl", "b": "patient_age", "lab": "id_like_name", "noise": "zz"}
    renamed = df.rename(columns=names)
    original = recs_for(df, TARGET, task="binary")
    other = recs_for(renamed, TARGET, task="binary")

    def shape(out: dict[str, Any], mapping: dict[str, str]) -> list[tuple[str, tuple[str, ...]]]:
        return sorted(
            (r["op"], tuple(sorted(mapping.get(c, c) for c in r["columns"])))
            for r in out["recommendations"]
        )

    assert shape(original, names) == shape(other, {})


def test_clean_columns_get_nothing_whatever_their_name() -> None:
    rng = np.random.default_rng(3)
    df = frame(glucose=rng.normal(5, 1, 300), age=rng.normal(50, 5, 300), y=rng.integers(0, 2, 300))
    out = recs_for(df, TARGET, task="binary")
    assert out["recommendations"] == []
    assert [c["key"] for c in out["candidates"]] == ["baseline"]


def test_missing_metadata_is_reported_as_not_assessed() -> None:
    df = clinical_frame()
    no_target = recs_for(df, {})
    topics = {n["rule_id"] for n in no_target["not_assessed"]}
    assert {"target.imbalance", "selection.wide"} <= topics
    assert not any(op.startswith("imbalance.") for op in ops(no_target))
    no_profile = recommend(make_input(df, TARGET, task="binary", profile=False))
    assert "distribution.skew" in {n["rule_id"] for n in no_profile["not_assessed"]}
    assert not any(op.startswith("transform.") for op in ops(no_profile))
    no_diag = recommend(make_input(df, TARGET, task="binary", diagnostics=False))
    rules = {n["rule_id"] for n in no_diag["not_assessed"]}
    assert {"outliers.extreme", "diagnostics"} <= rules
    assert not any(op.startswith(("outliers.", "imbalance.")) for op in ops(no_diag))


def test_regression_target_does_not_get_imbalance_recommendations() -> None:
    df = clinical_frame()
    df["y"] = df["a"] * 2.0
    out = recs_for(df, TARGET, task="regression")
    assert not any(op.startswith("imbalance.") for op in ops(out))
    assert any(n["rule_id"] == "target.imbalance" for n in out["not_assessed"])


def test_no_recommendation_removes_rows_or_columns(clinical: dict) -> None:  # type: ignore[type-arg]
    for rec in clinical["recommendations"]:
        assert not rec["op"].startswith("rows.") and rec["op"] != "select.drop_columns"
        assert rec["params"].get("action", "report") == "report"
    for cand in clinical["candidates"]:
        for step in cand["steps"]:
            assert step["params"].get("action", "report") == "report"


def test_conflicts_explain_trade_offs_and_include_doing_nothing(clinical: dict) -> None:  # type: ignore[type-arg]
    groups = {c["group"]: c for c in clinical["conflicts"]}
    outliers = groups["outliers.treatment"]
    members = {o["recommendation_id"] for o in outliers["options"]}
    assert {
        by_op(clinical, op)["id"] for op in ("outliers.flag", "outliers.cap", "scale.robust")
    } <= members
    assert "destructive" in outliers["trade_off"]
    for key in ("impute.numeric", "missing.indicator", "imbalance", "redundancy", "transform.log"):
        conflict = groups[key]
        assert conflict["trade_off"] and "no universal answer" in conflict["how_to_decide"]
        assert conflict["options"][-1]["recommendation_id"] is None  # "Do nothing"
        assert conflict["options"][-1]["consequence"]
    # Quantile versus preserving the distribution: quantile is a ranked alternative, never
    # the top choice, and "do nothing" keeps the original units.
    transform = groups["transform.log"]
    assert by_op(clinical, "transform.quantile")["id"] in {
        o["recommendation_id"] for o in transform["options"]
    }
    assert transform["options"][0]["recommendation_id"] == by_op(clinical, "transform.log")["id"]


def test_class_weights_rank_above_resampling_and_alternatives_are_linked(clinical: dict) -> None:  # type: ignore[type-arg]
    weight = by_op(clinical, "imbalance.class_weight")
    over = by_op(clinical, "imbalance.random_oversample")
    assert weight["rank"] < over["rank"]
    assert over["id"] in weight["alternatives"] and weight["id"] in over["alternatives"]
    smote = next((r for r in clinical["recommendations"] if r["op"] == "imbalance.smote"), None)
    if smote is not None:
        assert smote["rank"] > weight["rank"]
    undersample = [
        n for n in clinical["not_recommended"] if n["op"] == "imbalance.random_undersample"
    ]
    assert undersample and "discard" in undersample[0]["reason"]


def test_ranking_is_transparent_and_ordered(clinical: dict) -> None:  # type: ignore[type-arg]
    recs = clinical["recommendations"]
    assert [r["rank"] for r in recs] == list(range(1, len(recs) + 1))
    scores = [r["ranking"]["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)
    for r in recs:
        factors = r["ranking"]["factors"]
        assert r["ranking"]["score"] == sum(f["points"] for f in factors)
        assert {"evidence", "task", "domain", "risk", "information_loss"} <= {
            f["factor"] for f in factors
        }
        assert all(f["detail"] for f in factors)
    cap = by_op(clinical, "outliers.cap")
    flag = by_op(clinical, "outliers.flag")
    assert cap["rank"] > flag["rank"]  # destructive and high information loss ranks lower


def test_recommendations_are_reproducible() -> None:
    a = recs_for(clinical_frame(), TARGET, task="binary")
    b = recs_for(clinical_frame(), TARGET, task="binary")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_suggested_candidates_are_valid_and_avoid_destructive_core(clinical: dict) -> None:  # type: ignore[type-arg]
    ctx = Facts(make_input(clinical_frame(), TARGET, task="binary")).context
    cands = {c["key"]: c for c in clinical["candidates"]}
    assert cands["baseline"]["steps"] == []
    assert "core" in cands and len(cands) <= 5
    destructive = {r["id"] for r in clinical["recommendations"] if r["destructive"]}
    assert not set(cands["core"]["recommendation_ids"]) & destructive
    for cand in clinical["candidates"]:
        steps = [Step(s["op"], tuple(s["columns"]), s["params"]) for s in cand["steps"]]
        assert validate_steps(steps, ctx).valid, cand["key"]
        kinds = [s["op"].startswith("imbalance.") for s in cand["steps"]]
        assert sum(kinds) <= 1 and (not any(kinds) or kinds[-1])


def test_smote_is_validated_after_imputation_and_refused_for_entities() -> None:
    df = clinical_frame()
    plain = recs_for(df, TARGET, task="binary")
    assert "imbalance.smote" in ops(plain)  # valid once the median imputation runs first
    df["pid"] = [f"p{i // 4}" for i in range(len(df))]
    grouped = recs_for(df, {"target": ["y"], "group": ["pid"]}, task="binary")
    assert "imbalance.smote" not in ops(grouped)
    reasons = [n["reason"] for n in grouped["not_recommended"] if n["op"] == "imbalance.smote"]
    assert reasons and "entities" in reasons[0]


def test_biomedical_policy_preserves_clinical_values_and_interpretability() -> None:
    df = clinical_frame()
    roles = {"target": ["y"], "bio.heart_rate": ["noise"], "bio.bmi": ["a"]}
    out = recs_for(df, roles, task="binary", domain="biomedical")
    cap = next((r for r in out["recommendations"] if r["op"] == "outliers.cap"), None)
    assert cap is None or not {"noise", "a"} & set(cap["columns"])
    blocked = {c for n in out["not_recommended"] if n["op"] == "outliers.cap" for c in n["columns"]}
    assert {"noise", "a"} & blocked
    assert "transform.quantile" not in ops(out)
    assert any(
        n["op"] == "transform.quantile" and "interpretation" in n["reason"]
        for n in out["not_recommended"]
    )
    corr = next(r for r in out["recommendations"] if r["op"] == "select.correlation")
    assert corr["domain_compatibility"]["status"] == "caution"
    assert "biology" in corr["domain_compatibility"]["detail"]
    log = by_op(out, "transform.log")
    assert "laboratory" in log["domain_compatibility"]["detail"]


def test_mostly_empty_and_identifier_like_columns_are_advised_not_treated() -> None:
    rng = np.random.default_rng(5)
    n = 300
    df = frame(
        x=rng.normal(size=n),
        sparse=rng.normal(size=n),
        code=pd.Series([f"c{i}" for i in range(n)], dtype=object),
        city=pd.Series(rng.choice([f"t{i}" for i in range(40)], n), dtype=object),
        y=rng.integers(0, 2, n),
    )
    df.loc[df.index[: int(n * 0.8)], "sparse"] = np.nan
    out = recs_for(df, TARGET, task="binary", types={"code": "categorical", "city": "categorical"})
    codes = {a["code"] for a in out["advisories"]}
    assert {"MOSTLY_EMPTY_COLUMNS", "IDENTIFIER_LIKE_COLUMN"} <= codes
    assert not any(
        "sparse" in r["columns"] and r["op"].startswith("impute.") for r in out["recommendations"]
    )
    assert not any("code" in r["columns"] for r in out["recommendations"])
    city = {r["op"] for r in out["recommendations"] if r["columns"] == ["city"]}
    assert city == {"encode.frequency", "encode.onehot"}  # 40 levels: a dimensionality choice


def test_split_plan_follows_roles() -> None:
    df = clinical_frame()
    df["pid"] = [f"p{i // 4}" for i in range(len(df))]
    df["when"] = pd.date_range("2020-01-01", periods=len(df), freq="D")
    plain = recs_for(df.drop(columns=["pid", "when"]), TARGET, task="binary")["split_plan"]
    assert plain["method"] == "stratified" and plain["n_splits"] == 5
    assert "y" in plain["duplicate_columns"]
    grouped = recs_for(
        df.drop(columns=["when"]),
        {"target": ["y"], "bio.patient_id": ["pid"]},
        task="binary",
        domain="biomedical",
    )["split_plan"]
    assert grouped["method"] == "stratified_group" and grouped["group"] == "pid"
    timed = recs_for(df, {"target": ["y"], "time": ["when"], "group": ["pid"]}, task="binary")[
        "split_plan"
    ]
    assert timed["method"] == "time" and timed["time"] == "when"
    assert "GROUPS_NOT_ENFORCED_WITH_TIME" in {w["code"] for w in timed["warnings"]}

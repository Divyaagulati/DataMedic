"""Phase 5 evaluation eligibility and the Reliability Auditor, on synthetic data with
deliberately introduced leakage (and clean controls)."""

from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.ml import audit_spec
from datamedic.engine.ml.audit import AuditContext, run_audit
from datamedic.engine.ml.eligibility import (
    BLOCKED,
    DIRECT,
    REPLAY,
    Lineage,
    LineageRun,
    assess,
)
from datamedic.engine.ml.runner import run_training
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.split import make_splits
from datamedic.engine.preprocess.table_schema import schema_of
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.test_ml_engine import ROLE_SPECS, ctx_for, frame, setup

# ---- Eligibility --------------------------------------------------------------------------------


def _lineage(
    df: pd.DataFrame, steps: list[Step], roles: dict[str, Any] | None = None, **kw: Any
) -> Lineage:
    roles = roles or {"roles": {"target": ["y"]}, "task_type": "binary"}
    result = validate_steps(steps, ctx_for(df, "binary", roles["roles"]))
    run = LineageRun(
        run_id="r1",
        result_version_id="v1",
        source_version_id="v0",
        source_schema=schema_of(df, {}),
        source_n_rows=len(df),
        data_steps=[s.as_dict() for s in steps],
        data_step_hashes=kw.pop("hashes", [v.step_hash for v in result.steps]),
        commit_roles=roles,
        seed=0,
    )
    return Lineage("v1", True, "v0", schema_of(df, {}), len(df), kw.pop("n_rows", len(df)), [run])


CURRENT = {"roles": {"target": ["y"]}, "task_type": "binary"}


def test_unfitted_version_is_direct() -> None:
    df = frame("binary")
    e = assess(Lineage("v0", False, "v0", schema_of(df, {}), len(df), len(df)), CURRENT, ROLE_SPECS)
    assert e.mode == DIRECT and e.evaluation_version_id == "v0"


def test_fitted_version_replays_from_ancestor() -> None:
    df = frame("binary")
    e = assess(_lineage(df, [Step("impute.median", ("b",), {})]), CURRENT, ROLE_SPECS)
    assert e.mode == REPLAY and e.evaluation_version_id == "v0"
    assert [s["op"] for s in e.replay["steps"]] == ["impute.median"]  # type: ignore[index]


def test_replay_is_blocked_when_not_faithful_or_not_safe() -> None:
    df = frame("binary")
    steps = [Step("impute.median", ("b",), {})]
    cases = {
        "REPLAY_HASH_MISMATCH": (_lineage(df, steps, hashes=["0" * 64]), CURRENT),
        "REPLAY_ROWS_DIFFER": (_lineage(df, steps, n_rows=len(df) - 1), CURRENT),
        # The imputed column is now the target: replaying would change test labels.
        "TARGET_TRANSFORMED_IN_LINEAGE": (
            _lineage(df, steps),
            {"roles": {"target": ["b"]}, "task_type": "regression"},
        ),
        "ROLE_ON_DERIVED_COLUMN": (
            _lineage(df, steps),
            {"roles": {"target": ["y"], "group": ["b__missing"]}, "task_type": "binary"},
        ),
        "LINEAGE_INCOMPLETE": (Lineage("v1", True, None, problem="missing"), CURRENT),
    }
    for code, (lineage, roles) in cases.items():
        e = assess(lineage, roles, ROLE_SPECS)
        assert e.mode == BLOCKED and e.reasons[0]["code"] == code, code


# ---- Auditor ------------------------------------------------------------------------------------


def _audit(
    df: pd.DataFrame,
    task: str = "binary",
    *,
    roles: dict[str, list[str]] | None = None,
    mode: str = "unbiased_direct",
    method: str | None = None,
    group: str | None = None,
    time: str | None = None,
    splits: Any = None,
    datetime_columns: list[str] | None = None,
    diagnostics: str = "current",
    models: tuple[str, ...] = ("dummy", "linear", "hgb"),
) -> dict[str, Any]:
    roles = roles or {"target": ["y"]}
    s, pre, spec = setup(
        df, task, roles=roles, method=method, group=group, time=time, models=models
    )
    splits = splits or s
    out = run_training(df, splits, pre, spec)
    protected = {c for cols in roles.values() for c in cols}
    features = [c for c in df.columns if c != ROW_ID and c not in protected]
    ctx = AuditContext(
        task=task,
        target="y",
        data=df,
        features=features,
        splits=splits,
        summary=out.summary,
        models=out.models,
        evidence=out.evidence,
        mode=mode,
        eligibility={},
        roles=roles,
        split_config={"method": splits.method},
        preprocess_steps=[],
        lineage_steps=[],
        sensitive=frozenset(),
        datetime_columns=datetime_columns or [],
        diagnostics={"state": diagnostics},
    )
    findings, summary = run_audit(ctx)
    return {"by": {f.check_id: f for f in findings}, "summary": summary, "findings": findings}


def test_every_check_runs_with_required_fields() -> None:
    res = _audit(frame("binary"))
    assert [f.check_id for f in res["findings"]] == list(audit_spec.AUDIT_IDS)
    assert len(res["findings"]) == 20
    for f in res["findings"]:
        assert (
            f.limitations
            and f.evidence.get("basis") in audit_spec.BASES
            or f.status == "not_assessed"
        )
        if f.status == "detected":
            assert f.recommendation and f.evidence and f.severity
        if f.status == "not_assessed":
            assert f.not_assessed_reason


def test_clean_control_has_no_high_or_critical_detection() -> None:
    res = _audit(frame("binary", n=300))
    bad = [
        f.check_id
        for f in res["findings"]
        if f.status == "detected" and f.severity in ("high", "critical")
    ]
    assert bad == []
    status = res["summary"]["evaluation_status"]
    assert status["status"] == "no_issues_detected_among_assessed"
    assert status["not_assessed"]  # e.g. groups and time: listed, never counted as passed
    assert res["by"]["leakage.groups.cross_fold"].status == "not_assessed"
    assert res["by"]["leakage.temporal.ordering"].status == "not_assessed"
    assert res["by"]["leakage.temporal.post_outcome"].status == "not_assessed"


def test_target_copy_is_a_suspicion_not_a_confirmed_leak() -> None:
    df = frame("binary", n=240)
    df["leak"] = df["y"] * 3.0 + 1.0
    res = _audit(df)
    f = res["by"]["leakage.target.single_feature"]
    assert f.status == "detected" and f.severity == "critical" and "leak" in f.columns
    assert f.evidence["basis"] == "statistical" and f.title.startswith("Possible")
    assert res["by"]["leakage.target.derived"].status == "detected"
    sus = res["by"]["reliability.suspicious_performance"]
    assert (
        sus.status == "detected"
        and "leakage.target.single_feature" in sus.evidence["related_detections"]
    )
    assert res["summary"]["evaluation_status"]["status"] == "concerns"


def test_identifier_sorted_by_outcome() -> None:
    df = frame("binary", n=240).sort_values("y", kind="stable").reset_index(drop=True)
    df[ROW_ID] = np.arange(len(df))
    df["record_no"] = np.arange(len(df)) * 7 + 1000
    f = _audit(df)["by"]["leakage.identifier"]
    assert f.status == "detected" and "record_no" in f.columns


def test_duplicates_across_folds_with_a_naive_split() -> None:
    base = frame("binary", n=150)
    df = pd.concat([base, base.iloc[:60]], ignore_index=True)
    df[ROW_ID] = np.arange(len(df))
    naive = make_splits(df, method="stratified", n_splits=4, seed=0, target="y")  # no grouping
    f = _audit(df, splits=naive)["by"]["leakage.duplicates.cross_fold"]
    assert f.status == "detected" and f.evidence["basis"] == "confirmed"
    grouped = _audit(df)["by"]["leakage.duplicates.cross_fold"]  # duplicates kept together
    assert grouped.status == "not_detected"


def test_group_role_ignored_is_a_confirmed_critical_violation() -> None:
    df = frame("binary", n=240)
    df["patient"] = np.repeat(np.arange(80), 3)
    roles = {"target": ["y"], "group": ["patient"]}
    res = _audit(df, roles=roles, method="stratified")
    f = res["by"]["leakage.groups.cross_fold"]
    assert (
        f.status == "detected" and f.severity == "critical" and f.evidence["basis"] == "confirmed"
    )
    assert res["summary"]["evaluation_status"]["status"] == "blocked"
    ok = _audit(df, roles=roles, method="stratified_group", group="patient")
    assert ok["by"]["leakage.groups.cross_fold"].status == "not_detected"


def test_undeclared_repeated_entities() -> None:
    df = frame("binary", n=240)
    df["pid"] = np.repeat(np.arange(60), 4)
    res = _audit(df, roles={"target": ["y"], "id": ["pid"]})
    assert res["by"]["leakage.groups.undeclared"].status == "detected"
    meta = res["by"]["reliability.metadata"]
    assert meta.status == "detected" and meta.evidence["items"][0]["item"] == "entity_role"


def test_time_role_ignored_and_time_split_verified() -> None:
    df = frame("regression", n=240)
    df["t"] = np.arange(240)
    roles = {"target": ["y"], "time": ["t"]}
    bad = _audit(df, "regression", roles=roles, method="kfold")["by"]["leakage.temporal.ordering"]
    assert (
        bad.status == "detected"
        and bad.severity == "critical"
        and bad.evidence["basis"] == "configuration"
    )
    good = _audit(df, "regression", roles=roles, method="time", time="t")["by"][
        "leakage.temporal.ordering"
    ]
    assert good.status == "not_detected"
    no_role = _audit(frame("binary"), datetime_columns=["when"])["by"]["leakage.temporal.ordering"]
    assert no_role.status == "not_assessed" and "Date or time" in no_role.not_assessed_reason


def test_exploratory_run_is_labelled_and_never_unbiased() -> None:
    res = _audit(frame("binary"), mode="exploratory")
    f = res["by"]["leakage.pipeline.fit_on_full_data"]
    assert (
        f.status == "detected" and f.severity == "critical" and f.evidence["basis"] == "confirmed"
    )
    assert res["summary"]["evaluation_status"]["status"] == "exploratory"


def test_imbalance_sample_size_and_missing_diagnostics() -> None:
    df = frame("binary", n=240)
    df.loc[:, "y"] = 0
    df.loc[df.index[:12], "y"] = 1
    res = _audit(df, diagnostics="missing", models=("dummy", "linear"))
    assert res["by"]["reliability.class_imbalance"].severity == "medium"
    assert res["by"]["reliability.sample_size"].status == "detected"
    assert any(
        i["item"] == "diagnostics" for i in res["by"]["reliability.metadata"].evidence["items"]
    )


def test_evaluation_status_rules() -> None:
    def f(check: str, status: str, severity: str, basis: str) -> dict[str, object]:
        return {"check_id": check, "status": status, "severity": severity, "basis": basis}

    assert audit_spec.evaluation_status([f("a", "not_assessed", "info", "confirmed")], False)[
        "status"
    ] == ("no_issues_detected_among_assessed")
    full = f("leakage.pipeline.fit_on_full_data", "detected", "critical", "confirmed")
    assert audit_spec.evaluation_status([full], True)["status"] == "exploratory"
    other = f("leakage.groups.cross_fold", "detected", "critical", "confirmed")
    assert audit_spec.evaluation_status([full, other], True)["status"] == "blocked"
    assert (
        audit_spec.evaluation_status([f("x", "detected", "high", "statistical")], False)["status"]
        == "concerns"
    )
    # A statistical critical suspicion never blocks on its own.
    assert audit_spec.evaluation_status([f("x", "detected", "critical", "statistical")], False)[
        "status"
    ] == ("concerns")

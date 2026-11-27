"""Sensitive data and error hygiene for Part C ops.

Planted secret values must never appear in reports (learned parameters, effects, warnings),
error messages or error details. Learned values of sensitive columns are redacted; so are
distribution statistics, correlations involving a sensitive column, and class labels, counts
and weights of a sensitive target. Decisions (kept/removed) stay visible: they are not values.
"""

import json

import numpy as np
import pandas as pd
import pytest

from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.report import REDACTED
from datamedic.engine.preprocess.spec import Step
from tests.unit.preprocess_helpers import frame, validated

SECRET = 987654.321
SECRET_LABEL = "DX-SECRET-LABEL"


def secret_frame(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    lab = rng.lognormal(3, 1, n)
    lab[0] = SECRET
    y = (rng.random(n) < 0.3).astype("int64")
    return frame(
        lab=lab,
        other=lab * 0.9 + rng.normal(0, 1, n),
        noise=rng.normal(0, 1, n),
        diag=pd.Series(np.where(y == 1, SECRET_LABEL, "plain"), dtype=object),
        y=y,
    )


ROLES = {"target": ["y"], "sensitive": ["lab", "diag"]}

STEPS = [
    Step("transform.log", ("lab",)),
    Step("transform.log1p", ("lab",)),
    Step("transform.yeo_johnson", ("lab",)),
    Step("transform.quantile", ("lab",)),
    Step("select.variance", ("lab", "noise")),
    Step("select.correlation", ("lab", "other", "noise"), {"threshold": 0.5}),
    Step("select.mutual_information", ("lab", "diag", "noise")),
    Step("select.univariate", ("lab", "noise"), {"test": "anova_f"}),
    Step("select.univariate", ("diag",), {"test": "chi2"}),
]


def dump(obj: object) -> str:
    return json.dumps(obj, default=str)


def contains_secret(text: str) -> bool:
    return any(s in text for s in ("987654", SECRET_LABEL, f"{np.log(SECRET):.6f}"[:8]))


@pytest.mark.parametrize("step", STEPS, ids=lambda s: f"{s.op}:{s.params.get('test', '')}")
def test_reports_of_sensitive_columns_contain_no_values(step: Step) -> None:
    df = secret_frame()
    result = validated(df, [step], ROLES, task="binary")
    out = PreprocessExecutor(result).fit(df).transform(df)
    report = out.reports[0].as_dict()
    text = dump(report)
    assert not contains_secret(text)
    learned = report["learned"]
    for column in ("lab", "diag"):
        if column in learned.get("columns", {}):
            assert learned["columns"][column] == REDACTED
        if column in report["effects"].get("distribution", {}):
            assert report["effects"]["distribution"][column] == REDACTED
    for pair in learned.get("pairs", []):
        if {"lab", "diag"} & set(pair["columns"]):
            assert "correlation" not in pair and pair["redacted"] is True
    if "decisions" in learned:  # decisions are not values and remain visible for lineage
        assert set(learned["decisions"]) == set(step.columns)


def test_non_sensitive_columns_keep_their_statistics() -> None:
    df = secret_frame()
    result = validated(
        df, [Step("select.correlation", ("other", "noise", "lab"), {"threshold": 0.5})], ROLES
    )
    learned = PreprocessExecutor(result).fit(df).transform(df).reports[0].learned
    assert "max_abs_correlation" in learned["columns"]["noise"]


@pytest.mark.parametrize(
    "op",
    ["imbalance.class_weight", "imbalance.random_oversample", "imbalance.random_undersample"],
)
def test_sensitive_target_labels_counts_and_weights_are_redacted(op: str) -> None:
    df = secret_frame()
    roles = {"target": ["diag"], "sensitive": ["diag"]}
    result = validated(df, [Step(op, ())], roles, task="binary")
    training = PreprocessExecutor(result).fit_resample(df)
    report = training.reports[0].as_dict()
    assert report["learned"]["columns"]["diag"] == REDACTED
    assert not contains_secret(dump(report))
    # Row counts are not values and stay visible.
    assert report["effects"]["rows_before"] == len(df)


def test_domain_error_messages_never_quote_values() -> None:
    df = secret_frame()
    df.loc[df.index[3], "lab"] = -SECRET
    for step in (Step("transform.log", ("lab",)), Step("transform.log1p", ("lab",))):
        with pytest.raises(EngineError) as exc:
            PreprocessExecutor(validated(df, [step], ROLES)).fit(df)
        assert exc.value.code == "VALUE_OUT_OF_DOMAIN"
        assert not contains_secret(exc.value.message + dump(exc.value.details))


def test_resampling_errors_never_quote_labels() -> None:
    df = secret_frame()
    df["diag"] = SECRET_LABEL
    df.loc[df.index[:3], "diag"] = "plain"  # 3 minority rows
    df["lab"] = df["lab"].astype(float)
    data = df[["_dm_row_id", "lab", "noise", "diag"]]
    result = validated(
        data, [Step("imbalance.smote", (), {"k_neighbors": 5})], {"target": ["diag"]}, task="binary"
    )
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(result).fit(data)
    assert exc.value.code == "SMOTE_TOO_FEW_SAMPLES"
    assert not contains_secret(exc.value.message + dump(exc.value.details))


def test_unexpected_library_failures_carry_only_the_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from datamedic.engine.preprocess import runtime as rt

    def boom(*args: object, **kwargs: object) -> None:
        raise ValueError(f"bad value {SECRET}")

    monkeypatch.setattr(rt, "fit_yeo_johnson", boom)
    df = secret_frame()
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(validated(df, [Step("transform.yeo_johnson", ("lab",))], ROLES)).fit(df)
    assert exc.value.code == "OP_FAILED" and exc.value.details["error_type"] == "ValueError"
    assert not contains_secret(exc.value.message + dump(exc.value.details))


def test_lineage_carries_column_names_and_decisions_only() -> None:
    df = secret_frame()
    steps = [Step("select.variance", ("lab", "noise"), {"action": "drop", "threshold": 1e12})]
    out = PreprocessExecutor(validated(df, steps, ROLES)).fit(df).transform(df)
    assert not contains_secret(dump(out.lineage))
    events = {e["column"]: e["events"] for e in out.lineage}
    assert events["lab"][0]["decision"] == "removed"


def test_control_without_the_sensitive_role_values_are_reported() -> None:
    """Proves the redaction above is real: the same reports carry values when not sensitive."""
    df = secret_frame()
    roles = {"target": ["y"]}
    out = (
        PreprocessExecutor(validated(df, [Step("transform.yeo_johnson", ("lab",))], roles))
        .fit(df)
        .transform(df)
    )
    assert "lambda" in out.reports[0].learned["columns"]["lab"]
    assert out.reports[0].effects["distribution"]["lab"]["before"]["skewness"] is not None
    roles = {"target": ["diag"]}
    result = validated(df, [Step("imbalance.class_weight", ())], roles, task="binary")
    learned = PreprocessExecutor(result).fit_resample(df).reports[0].learned
    assert SECRET_LABEL in learned["columns"]["diag"]["classes"]

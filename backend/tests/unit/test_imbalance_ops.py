"""Class imbalance (Part C): class weights, random over/undersampling and SMOTE.

Resampling only ever happens in ``fit_resample`` on the rows the step was fitted on;
``transform`` (test rows, previews, commits) never changes rows. Covers binary and multiclass
targets, label preservation, class counts, row identity of added rows, pass-through columns,
k-neighbour validation, refused combinations, determinism and limits."""

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.catalog import SAMPLE_WEIGHT, Limits
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import context, frame, validated

ROLES = {"target": ["y"], "id": ["pid"]}
OPS = [
    "imbalance.class_weight",
    "imbalance.random_oversample",
    "imbalance.random_undersample",
    "imbalance.smote",
]

ACK = {"acknowledge_dependent_rows": True}  # roles declare ids/groups (see audit)


def data(
    n: int = 120, seed: int = 0, labels: tuple = (0, 1), shares: tuple = (0.8, 0.2)
) -> pd.DataFrame:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    y = rng.choice(np.array(labels, dtype=object), size=n, p=shares)
    y[: len(labels)] = list(labels)  # every class present
    return frame(
        pid=[f"p{i}" for i in range(n)],
        a=rng.normal(0, 1, n),
        b=np.arange(n, dtype="int64"),
        y=pd.Series(list(y)).astype(type(labels[0]) if not isinstance(labels[0], str) else object),
    )


def executor(
    df: pd.DataFrame,
    steps: list[Step],
    roles: dict | None = None,
    task: str = "binary",
    seed: int = 0,
) -> PreprocessExecutor:  # type: ignore[type-arg]
    return PreprocessExecutor(validated(df, steps, roles or ROLES, task=task), seed=seed)


def counts(df: pd.DataFrame) -> dict:  # type: ignore[type-arg]
    return df["y"].value_counts().to_dict()


# ---- Training table versus other rows ----------------------------------------------------------


@pytest.mark.parametrize("op", OPS)
def test_transform_never_changes_rows_and_fit_resample_changes_only_training(op: str) -> None:
    df = data()
    train, test = df.iloc[:90], df.iloc[90:]
    ex = executor(df, [Step(op, (), ACK if op == "imbalance.smote" else {})])
    training = ex.fit_resample(train)
    test_out = ex.transform(test)
    pd.testing.assert_frame_equal(test_out.frame, test)  # untouched
    pd.testing.assert_frame_equal(ex.transform(train).frame, train)
    assert counts(test_out.frame) == counts(test)
    assert training.frame[ROW_ID].is_unique
    original = training.frame[training.frame[ROW_ID] >= 0]
    assert set(original[ROW_ID]) <= set(train[ROW_ID])


@pytest.mark.parametrize("op", OPS)
def test_fit_transform_never_resamples_so_nothing_resampled_can_be_committed(op: str) -> None:
    df = data()
    out = executor(df, [Step(op, (), ACK if op == "imbalance.smote" else {})]).fit_transform(df)
    pd.testing.assert_frame_equal(out.frame, df)
    assert SAMPLE_WEIGHT not in out.frame.columns


@pytest.mark.parametrize("op", OPS[1:])
def test_resampling_other_rows_than_the_fitting_rows_is_refused(op: str) -> None:
    df = data()
    ex = executor(df, [Step(op, (), ACK if op == "imbalance.smote" else {})])
    ex.fit(df.iloc[:90])
    transformer = ex.transformers()[-1]
    with pytest.raises(EngineError) as exc:
        transformer.resample(df.iloc[90:])
    assert exc.value.code == "RESAMPLE_NOT_FIT_ROWS"


# ---- Each method ----------------------------------------------------------------------------------


def test_class_weights_are_balanced_and_only_in_the_training_table() -> None:
    df = data()
    training = executor(df, [Step("imbalance.class_weight", ())]).fit_resample(df)
    n0, n1 = (df["y"] == 0).sum(), (df["y"] == 1).sum()
    w = training.frame[SAMPLE_WEIGHT]
    assert w[df["y"] == 0].unique().tolist() == pytest.approx([len(df) / (2 * n0)])
    assert w[df["y"] == 1].unique().tolist() == pytest.approx([len(df) / (2 * n1)])
    assert w.sum() == pytest.approx(len(df))  # balanced weights keep the total
    assert len(training.frame) == len(df)
    learned = training.reports[0].learned["columns"]["y"]
    assert learned["classes"] == [0, 1] and learned["counts_after"] == [n0, n1]


@pytest.mark.parametrize(
    ("labels", "shares", "task"),
    [((0, 1), (0.8, 0.2), "binary"), (("a", "b", "c"), (0.6, 0.3, 0.1), "multiclass")],
)
def test_random_oversampling_balances_and_preserves_labels(
    labels: tuple, shares: tuple, task: str
) -> None:  # type: ignore[type-arg]
    df = data(n=150, labels=labels, shares=shares)
    training = executor(df, [Step("imbalance.random_oversample", ())], task=task).fit_resample(df)
    out = training.frame
    largest = df["y"].value_counts().max()
    assert set(counts(out).values()) == {largest}
    assert (
        out.iloc[: len(df)]
        .drop(columns=[])
        .reset_index(drop=True)
        .equals(df.reset_index(drop=True))
    )
    added = out.iloc[len(df) :]
    assert (added[ROW_ID] < 0).all() and added[ROW_ID].tolist() == list(
        range(-1, -len(added) - 1, -1)
    )
    # Duplicates are exact copies of real rows (including their identifier).
    merged = added.drop(columns=[ROW_ID]).merge(
        df.drop(columns=[ROW_ID]), how="left", indicator=True
    )
    assert (merged["_merge"] == "both").all()
    report = training.reports[0]
    assert report.rows_in == len(df) and report.rows_out == len(out)
    assert report.effects["rows_added"] == len(added) and report.effects["rows_removed"] == 0
    assert report.learned["columns"]["y"]["counts_after"] == [largest] * len(labels)


def test_minority_strategy_raises_only_the_smallest_class() -> None:
    df = data(n=200, labels=("a", "b", "c"), shares=(0.6, 0.3, 0.1))
    before = counts(df)
    out = (
        executor(
            df,
            [Step("imbalance.random_oversample", (), {"sampling_strategy": "minority"})],
            task="multiclass",
        )
        .fit_resample(df)
        .frame
    )
    after = counts(out)
    assert after["a"] == before["a"] and after["b"] == before["b"] and after["c"] == before["a"]


def test_random_undersampling_keeps_real_rows_in_order_and_reports_discards() -> None:
    df = data(n=150)
    training = executor(df, [Step("imbalance.random_undersample", ())]).fit_resample(df)
    out = training.frame
    smallest = df["y"].value_counts().min()
    assert set(counts(out).values()) == {smallest}
    assert (out[ROW_ID] >= 0).all() and out[ROW_ID].is_monotonic_increasing
    kept = df.set_index(ROW_ID).loc[out[ROW_ID]]
    assert kept["y"].tolist() == out["y"].tolist()
    assert training.reports[0].effects["rows_removed"] == len(df) - len(out)
    assert "ROWS_DISCARDED" in [w["code"] for w in training.reports[0].warnings]


@pytest.mark.parametrize(
    ("labels", "shares", "task"),
    [((0, 1), (0.8, 0.2), "binary"), (("a", "b", "c"), (0.6, 0.3, 0.1), "multiclass")],
)
def test_smote_synthesises_numeric_rows_without_identifiers(
    labels: tuple, shares: tuple, task: str
) -> None:  # type: ignore[type-arg]
    df = data(n=200, labels=labels, shares=shares)
    training = executor(
        df, [Step("imbalance.smote", (), {**ACK, "k_neighbors": 3})], task=task
    ).fit_resample(df)
    out = training.frame
    assert set(counts(out).values()) == {df["y"].value_counts().max()}
    synthetic = out[out[ROW_ID] < 0]
    assert len(synthetic) == len(out) - len(df)
    assert synthetic["pid"].isna().all()  # identifiers are never synthesised
    assert synthetic["y"].isin(labels).all()  # labels preserved, never interpolated
    assert out["b"].dtype == "float64"  # integer features become decimal (documented)
    # Originals are unchanged apart from the documented integer -> float change.
    originals = out[out[ROW_ID] >= 0].reset_index(drop=True)
    assert originals["a"].equals(df["a"].reset_index(drop=True))
    assert originals["pid"].tolist() == df["pid"].tolist()
    # Synthetic points lie on segments between same-class rows: inside the class's range.
    for label in labels:
        real = df[df["y"] == label]["a"]
        fake = synthetic[synthetic["y"] == label]["a"]
        assert fake.between(real.min(), real.max()).all()
    assert "SYNTHETIC_ROWS" in [w["code"] for w in training.reports[0].warnings]


def test_resampling_is_reproducible_with_a_seed() -> None:
    df = data(n=150)
    for op in OPS[1:]:
        a = (
            executor(df, [Step(op, (), ACK if op == "imbalance.smote" else {})], seed=4)
            .fit_resample(df)
            .frame
        )
        b = (
            executor(df, [Step(op, (), ACK if op == "imbalance.smote" else {})], seed=4)
            .fit_resample(df)
            .frame
        )
        c = (
            executor(df, [Step(op, (), ACK if op == "imbalance.smote" else {})], seed=5)
            .fit_resample(df)
            .frame
        )
        pd.testing.assert_frame_equal(a, b)
        assert not a.equals(c)


# ---- Refused configurations ------------------------------------------------------------------------


def test_smote_validates_k_neighbors_against_minority_counts_before_resampling() -> None:
    df = data(n=40, shares=(0.9, 0.1))
    df.loc[df.index[2:], "y"] = 0
    df.loc[df.index[:4], "y"] = 1  # 4 minority rows
    ex = executor(df, [Step("imbalance.smote", (), {**ACK, "k_neighbors": 4})])
    with pytest.raises(EngineError) as exc:
        ex.fit(df)
    assert exc.value.code == "SMOTE_TOO_FEW_SAMPLES"
    executor(df, [Step("imbalance.smote", (), {**ACK, "k_neighbors": 3})]).fit_resample(df)


def test_smote_refuses_non_numeric_or_incomplete_features() -> None:
    df = data()
    df["cat"] = "x"
    assert _codes(df, [Step("imbalance.smote", (), ACK)]) == ["FEATURE_NOT_NUMERIC"]
    df = data()
    df.loc[df.index[3], "a"] = np.nan
    assert _codes(df, [Step("imbalance.smote", (), ACK)]) == ["FEATURE_MISSING_VALUES"]
    df = data()
    df["flag"] = df["a"] > 0
    assert _codes(df, [Step("imbalance.smote", (), ACK)]) == ["FEATURE_NOT_NUMERIC"]
    steps = [Step("encode.onehot", ("pid2",)), Step("imbalance.smote", (), ACK)]
    df = data()
    df["pid2"] = pd.Series(["u", "v"] * 60, dtype=object)
    assert "FEATURE_NOT_NUMERIC" in _codes(df, steps)


def test_smote_runtime_refuses_missing_features_after_earlier_steps() -> None:
    df = data()
    df.loc[df.index[5], "a"] = np.nan
    # Validation knows the gap here; at runtime the same check protects un-validated callers.
    from datamedic.engine.preprocess import runtime as rt

    with pytest.raises(rt.PreprocessError) as exc:
        rt.resample_smote(df, "y", ["pid"], "not_majority", 3, 0, 100, 10_000, 1_000_000)
    assert exc.value.code == "FEATURE_MISSING_VALUES"


def _codes(
    df: pd.DataFrame, steps: list[Step], roles: dict | None = None, task: str | None = "binary"
) -> list[str]:  # type: ignore[type-arg]
    result = validate_steps(steps, context(df, roles if roles is not None else ROLES, task=task))
    return [e.code for e in result.errors]


def test_imbalance_ops_need_a_classification_target() -> None:
    df = data()
    for op in OPS:
        assert _codes(
            df, [Step(op, (), ACK if op == "imbalance.smote" else {})], roles={"id": ["pid"]}
        ) == ["TARGET_REQUIRED"]
        assert _codes(
            df, [Step(op, (), ACK if op == "imbalance.smote" else {})], task="regression"
        ) == ["TASK_UNSUPPORTED"]
        assert _codes(df, [Step(op, ("a",), ACK if op == "imbalance.smote" else {})]) == [
            "COLUMNS_NOT_APPLICABLE"
        ]


def test_training_only_step_must_be_last_and_unique() -> None:
    df = data()
    assert _codes(
        df, [Step("imbalance.random_oversample", ()), Step("scale.standard", ("a",))]
    ) == ["TRAINING_ONLY_NOT_LAST"]
    assert _codes(df, [Step("imbalance.class_weight", ()), Step("imbalance.smote", (), ACK)]) == [
        "TOO_MANY_TRAINING_ONLY"
    ]


def test_class_weight_is_refused_when_a_weight_column_exists() -> None:
    df = data()
    df["w"] = 1.0
    roles = {**ROLES, "weight": ["w"]}
    assert _codes(df, [Step("imbalance.class_weight", ())], roles) == ["WEIGHT_ROLE_PRESENT"]


def test_missing_and_single_class_targets_fail_clearly() -> None:
    df = data()
    df["y"] = df["y"].astype("float64")
    df.loc[df.index[7], "y"] = np.nan
    with pytest.raises(EngineError) as exc:
        executor(df, [Step("imbalance.random_oversample", ())]).fit(df)
    assert exc.value.code == "TARGET_MISSING"
    df = data()
    df["y"] = 0
    with pytest.raises(EngineError) as exc:
        executor(df, [Step("imbalance.random_oversample", ())]).fit(df)
    assert exc.value.code == "TARGET_SINGLE_CLASS"


def test_grouped_or_temporal_data_warns_for_smote() -> None:
    df = data()
    df["visit"] = np.arange(len(df)) // 3
    result = validate_steps(
        [Step("imbalance.smote", (), ACK)],
        context(df, {**ROLES, "group": ["visit"]}, task="binary"),
    )
    assert "GROUPED_OR_TEMPORAL_DATA" in [w.code for w in result.warnings]


def test_resampling_limits_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    df = data()
    monkeypatch.setattr(Limits, "RESAMPLE_MAX_OUTPUT_ROWS", 100)
    with pytest.raises(EngineError) as exc:
        executor(df, [Step("imbalance.random_oversample", ())]).fit_resample(df)
    assert exc.value.code == "OP_LIMIT_EXCEEDED"
    monkeypatch.setattr(Limits, "SMOTE_MAX_FIT_ROWS", 50)
    from datamedic.engine.preprocess.ops import imbalance

    monkeypatch.setattr(imbalance.SmoteOp, "fit_cap", 50)
    with pytest.raises(EngineError) as exc:
        executor(df, [Step("imbalance.smote", (), ACK)]).fit(df)
    assert exc.value.code == "OP_LIMIT_EXCEEDED"


def test_resampling_after_preprocessing_uses_transformed_training_rows() -> None:
    df = data()
    df.loc[df.index[4], "a"] = np.nan
    steps = [
        Step("impute.median", ("a",)),
        Step("scale.standard", ("a", "b")),
        Step("imbalance.smote", (), ACK),
    ]
    ex = executor(df, steps)
    training = ex.fit_resample(df.iloc[:90])
    test = ex.transform(df.iloc[90:])
    assert len(test.frame) == 30 and test.frame["a"].notna().all()
    assert training.frame["a"].notna().all() and len(training.frame) > 90
    assert [r.op for r in training.reports] == [s.op for s in steps]

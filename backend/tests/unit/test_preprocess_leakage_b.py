"""Leakage proof for Part B ops: thresholds, scaling parameters, category mappings,
frequencies and Isolation Forest models are learned from training rows only and applied
unchanged to other rows. Same three proofs as Part A (spy, perturbation, negative control)."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from tests.unit.preprocess_helpers import MIXED_ROLES, mixed_frame, validated
from tests.unit.test_preprocess_leakage import Spy, split

FITTED_B = [
    Step("outliers.flag", ("x1",), {"method": "iqr"}),
    Step("outliers.flag", ("x1",), {"method": "zscore"}),
    Step("outliers.flag", ("x1",), {"method": "mad"}),
    Step("outliers.cap", ("x2",), {"method": "percentile"}),
    Step("outliers.isolation_forest_flag", ("x1", "x2")),
    Step("encode.onehot", ("city",)),
    Step("encode.frequency", ("city",)),
    Step("scale.standard", ("x1",)),
    Step("scale.minmax", ("x1",)),
    Step("scale.robust", ("x1",)),
]


def ident(step: Step) -> str:
    return f"{step.op}:{step.params.get('method', '')}"


def data() -> pd.DataFrame:
    df = mixed_frame(n=120, seed=8)
    df["city"] = df["city"].fillna("north")
    return df


def perturb(df: pd.DataFrame, test: np.ndarray, *, leak_values: bool = True) -> pd.DataFrame:
    out = df.copy()
    rows = out[ROW_ID].isin(test)
    out.loc[rows, ["x1", "x2"]] = 1e6 if leak_values else np.nan
    out.loc[rows, "city"] = "zzz-test-only"
    return out


def test_spy_every_part_b_step_fits_on_training_rows_only() -> None:
    df = data()
    train, _ = split(df)
    steps = [
        Step("outliers.flag", ("x1",)),
        Step("outliers.cap", ("x2",)),
        Step("outliers.isolation_forest_flag", ("x1", "x2")),
        Step("scale.robust", ("x1",)),
        Step("encode.frequency", ("city",)),
        Step("encode.onehot", ("flag",)),
    ]
    executor = PreprocessExecutor(validated(df, steps, MIXED_ROLES, {"flag": "boolean"}))
    executor.pipeline.steps = [(name, Spy(t)) for name, t in executor.pipeline.steps]
    Spy.seen = []
    executor.fit(df[df[ROW_ID].isin(train)])
    executor.transform(df)
    assert len(Spy.seen) == len(steps)
    assert all(seen == set(train.tolist()) for seen in Spy.seen)


def _learned_and_train_rows(
    df: pd.DataFrame, step: Step, train: np.ndarray, *, fit_on_all: bool = False
) -> tuple[Any, pd.DataFrame]:
    executor = PreprocessExecutor(validated(df, [step], MIXED_ROLES), seed=0)
    executor.fit(df if fit_on_all else df[df[ROW_ID].isin(train)])
    out = executor.transform(df)
    return out.reports[0].learned, out.frame[out.frame[ROW_ID].isin(train)]


@pytest.mark.parametrize("step", FITTED_B, ids=ident)
def test_perturbing_test_rows_changes_nothing_learned(step: Step) -> None:
    df = data()
    train, test = split(df)
    learned_a, train_a = _learned_and_train_rows(df, step, train)
    learned_b, train_b = _learned_and_train_rows(perturb(df, test), step, train)
    assert learned_a == learned_b
    pd.testing.assert_frame_equal(train_a, train_b)


@pytest.mark.parametrize("step", FITTED_B, ids=ident)
def test_perturbation_detector_catches_a_deliberate_leak(step: Step) -> None:
    df = data()
    train, test = split(df)
    learned_a, _ = _learned_and_train_rows(df, step, train, fit_on_all=True)
    learned_b, _ = _learned_and_train_rows(perturb(df, test), step, train, fit_on_all=True)
    assert learned_a != learned_b


def test_isolation_forest_scores_test_rows_with_the_training_model() -> None:
    df = data()
    train, test = split(df)
    step = Step("outliers.isolation_forest_flag", ("x1", "x2"), {"contamination": 0.1})
    executor = PreprocessExecutor(validated(df, [step], MIXED_ROLES), seed=4)
    executor.fit(df[df[ROW_ID].isin(train)])
    full = executor.transform(df).frame.set_index(ROW_ID)["isolation_forest_outlier"]
    only_test = executor.transform(df[df[ROW_ID].isin(test)]).frame.set_index(ROW_ID)
    # Scoring a row does not depend on which other rows are transformed with it.
    assert only_test["isolation_forest_outlier"].equals(full.loc[only_test.index])


def test_caps_and_scales_from_training_rows_apply_unchanged() -> None:
    df = data()
    train, test = split(df)
    for step in (Step("outliers.cap", ("x2",)), Step("scale.minmax", ("x2",))):
        executor = PreprocessExecutor(validated(df, [step], MIXED_ROLES))
        executor.fit(df[df[ROW_ID].isin(train)])
        bigger = perturb(df, test)
        out = executor.transform(bigger).frame.set_index(ROW_ID)
        if step.op == "outliers.cap":
            upper = executor.transform(df).reports[0].learned["columns"]["x2"]["upper"]
            assert (out.loc[test, "x2"].dropna() == upper).all()  # capped at the training bound
        else:
            assert (out.loc[test, "x2"].dropna() > 1).all()  # not re-fitted to [0, 1]


def test_target_never_influences_part_b_ops() -> None:
    df = data()
    flipped = df.assign(outcome=1 - df["outcome"])
    for step in FITTED_B:
        a = PreprocessExecutor(validated(df, [step], MIXED_ROLES)).fit_transform(df).frame
        b = PreprocessExecutor(validated(flipped, [step], MIXED_ROLES)).fit_transform(flipped)
        pd.testing.assert_frame_equal(a.drop(columns="outcome"), b.frame.drop(columns="outcome"))

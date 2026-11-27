"""Proof that preprocessing learns only from the fitting (training) rows.

1. Spy: every step's ``fit`` receives only training-row ids, including later steps.
2. Perturbation: changing test rows (values and missingness) changes neither learned
   parameters nor the transformed training rows.
3. Test rows are filled with statistics of the training rows, never their own.
4. The target never influences imputation (it is never a predictor).
5. Fitting on all rows gives different results than in-fold fitting (the tests above would
   catch a leak), and the executor equals scikit-learn's ``Pipeline.transform``.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.preprocess.base import DMTransformer
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from tests.unit.preprocess_helpers import MIXED_ROLES, frame, mixed_frame, validated

FITTED_STEPS = [
    Step("missing.indicator", ("x1",)),
    Step("impute.mean", ("x1",)),
    Step("impute.median", ("age",)),
    Step("impute.mode", ("city", "flag")),
    Step("impute.knn", ("x2",), {"n_neighbors": 3}),
]
ALL_STEPS = [
    *FITTED_STEPS,
    Step("missing.indicator", ("city",)),
    Step("impute.constant", ("x1__missing",), {"fill_value": False}),
]


def split(df: pd.DataFrame, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    ids = df[ROW_ID].to_numpy()
    rng = np.random.default_rng(seed)
    test = np.sort(rng.choice(ids, size=len(ids) // 4, replace=False))
    train = np.setdiff1d(ids, test)
    return train, test


class Spy(DMTransformer):
    """Wraps a transformer and records the row ids seen by ``fit``."""

    op = "spy"
    seen: list[set[int]] = []

    def __init__(self, inner: Any = None) -> None:
        self.inner = inner

    def fit(self, X: pd.DataFrame, y: Any = None) -> "Spy":  # noqa: N803
        Spy.seen.append(set(X[ROW_ID].tolist()))
        self.inner.fit(X)
        self.n_fit_rows_ = len(X)
        self.warnings_ = self.inner.warnings_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out: pd.DataFrame = self.inner.transform(X)
        self.transform_warnings_ = self.inner.transform_warnings_
        return out

    def learned_params(self) -> dict[str, Any]:
        params: dict[str, Any] = self.inner.learned_params()
        return params


def test_spy_every_step_fits_on_training_rows_only() -> None:
    df = mixed_frame(n=80)
    train, _ = split(df)
    executor = PreprocessExecutor(validated(df, FITTED_STEPS, MIXED_ROLES))
    executor.pipeline.steps = [(name, Spy(t)) for name, t in executor.pipeline.steps]
    Spy.seen = []
    executor.fit(df[df[ROW_ID].isin(train)])
    executor.transform(df)  # transforming everything must not trigger any fit
    assert len(Spy.seen) == len(FITTED_STEPS)
    for seen in Spy.seen:
        assert seen == set(train.tolist())


@pytest.mark.parametrize(
    "step", FITTED_STEPS + [Step("impute.iterative", ("x2",))], ids=lambda s: s.op
)
def test_perturbing_test_rows_changes_nothing_learned(step: Step) -> None:
    df = mixed_frame(n=80)
    train, test = split(df)
    perturbed = df.copy()
    is_test = perturbed[ROW_ID].isin(test)
    perturbed.loc[is_test, ["x1", "x2"]] = 1e6  # extreme values ...
    perturbed.loc[is_test, "age"] = pd.NA  # ... and different missingness
    perturbed.loc[is_test, "city"] = "zzz-test-only"
    perturbed.loc[is_test, "flag"] = pd.NA

    results = []
    for data in (df, perturbed):
        executor = PreprocessExecutor(validated(data, [step], MIXED_ROLES), seed=0)
        executor.fit(data[data[ROW_ID].isin(train)])
        out = executor.transform(data)
        results.append((out.reports[0].learned, out.frame[out.frame[ROW_ID].isin(train)]))
    (learned_a, train_a), (learned_b, train_b) = results
    assert learned_a == learned_b
    pd.testing.assert_frame_equal(train_a, train_b)


@pytest.mark.parametrize(
    "step", FITTED_STEPS[1:] + [Step("impute.iterative", ("x2",))], ids=lambda s: s.op
)
def test_perturbation_detector_catches_a_deliberate_leak(step: Step) -> None:
    """Negative control: if fitting saw the test rows, the perturbation test would fail."""
    df = mixed_frame(n=80)
    _, test = split(df)
    perturbed = df.copy()
    is_test = perturbed[ROW_ID].isin(test)
    perturbed.loc[is_test, ["x1", "x2"]] = 1e6
    perturbed.loc[is_test, "age"] = 1000
    perturbed.loc[is_test, "city"] = "zzz-test-only"
    perturbed.loc[is_test, "flag"] = False
    learned = [
        PreprocessExecutor(validated(d, [step], MIXED_ROLES)).fit(d).transform(d).reports[0].learned
        for d in (df, perturbed)  # deliberate leak: fitted on all rows
    ]
    assert learned[0] != learned[1]


def test_test_rows_receive_training_statistics() -> None:
    # Training rows: values 1, 2, 3 (median 2). Test rows: 100, 200 and a gap.
    df = frame(a=[1.0, 2.0, 3.0, None, 100.0, 200.0, None])
    train = df[ROW_ID].isin([0, 1, 2, 3])
    executor = PreprocessExecutor(validated(df, [Step("impute.median", ("a",))]))
    executor.fit(df[train])
    out = executor.transform(df).frame
    assert out["a"].iloc[3] == 2.0 and out["a"].iloc[6] == 2.0  # never median(all) = 3
    full = PreprocessExecutor(validated(df, [Step("impute.median", ("a",))])).fit_transform(df)
    assert full.frame["a"].iloc[6] == 3.0  # proves the in-fold value differs from a leak


def test_mode_learned_only_from_training_rows() -> None:
    df = frame(c=pd.Series(["a", "a", "b", None, "b", "b", "b", None], dtype=object))
    train = df[ROW_ID] < 4
    executor = PreprocessExecutor(validated(df, [Step("impute.mode", ("c",))]))
    out = executor.fit(df[train]).transform(df).frame
    assert out["c"].iloc[7] == "a"  # all rows would say "b"


@pytest.mark.parametrize("op", ["impute.knn", "impute.iterative"])
def test_target_never_influences_imputation(op: str) -> None:
    df = mixed_frame(n=60)
    flipped = df.copy()
    flipped["outcome"] = 1 - flipped["outcome"]
    outputs = []
    for data in (df, flipped):
        result = validated(data, [Step(op, ("x1", "x2"))], MIXED_ROLES)
        assert "outcome" not in result.steps[0].step.params["predictors"]
        outputs.append(PreprocessExecutor(result).fit_transform(data).frame[["x1", "x2"]])
    pd.testing.assert_frame_equal(outputs[0], outputs[1])


def test_executor_matches_sklearn_pipeline_transform() -> None:
    df = mixed_frame(n=60)
    train, _ = split(df)
    executor = PreprocessExecutor(validated(df, FITTED_STEPS, MIXED_ROLES))
    executor.fit(df[df[ROW_ID].isin(train)])
    pd.testing.assert_frame_equal(executor.transform(df).frame, executor.pipeline.transform(df))


def test_test_rows_are_never_removed_or_reordered() -> None:
    df = mixed_frame(n=60)
    train, _ = split(df)
    executor = PreprocessExecutor(validated(df, ALL_STEPS, MIXED_ROLES))
    out = executor.fit(df[df[ROW_ID].isin(train)]).transform(df).frame
    assert out[ROW_ID].tolist() == df[ROW_ID].tolist()

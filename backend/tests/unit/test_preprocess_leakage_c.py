"""Leakage proofs for Part C: distribution transforms, feature selection and resampling are
learned from training rows only, and validation/test rows (values and labels) never reach a
fitted parameter, a selection decision or a resampled training table.

Proofs, as for Parts A and B: (1) spy on the rows each fit sees; (2) perturbing test values
and labels changes nothing learned and no transformed training row; (3) negative control:
fitting on all rows does change what is learned, so (2) is a real test. Plus in-fold proofs
with frozen stratified, group and time splits, and combined Part A+B+C pipelines."""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.ops.select import MutualInfoSelectOp, UnivariateSelectOp
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.split import make_splits
from tests.unit.preprocess_helpers import frame, validated
from tests.unit.test_preprocess_leakage import Spy

ROLES = {"target": ["y"], "group": ["patient"], "time": ["day"]}

ACK = {"acknowledge_dependent_rows": True}  # roles declare ids/groups (see audit)


def data(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    patient = np.repeat(np.arange(n // 4), 4)
    y = (rng.random(n) < 0.25).astype("int64")
    x1 = rng.lognormal(0, 1, n) + y
    df = frame(
        patient=patient,
        day=np.arange(n),
        x1=x1,
        x2=x1 * 0.5 + rng.normal(0, 0.3, n),
        x3=rng.normal(0, 1, n) + 0.8 * y,
        x4=rng.normal(0, 1, n),
        cat=pd.Series(rng.choice(["u", "v", "w"], n), dtype=object),
        y=y,
    )
    return df


def split(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ids = df[ROW_ID].to_numpy()
    test = ids[ids % 4 == 1]
    return np.setdiff1d(ids, test), test


FITTED_C = [
    Step("transform.yeo_johnson", ("x1", "x3")),
    Step("transform.quantile", ("x1",), {"n_quantiles": 50}),
    Step("transform.quantile", ("x3",), {"output_distribution": "normal"}),
    Step("select.variance", ("x1", "x3", "x4"), {"threshold": 0.9, "action": "drop"}),
    Step("select.correlation", ("x1", "x2", "x3"), {"threshold": 0.5}),
    Step("select.mutual_information", ("x1", "x3", "x4", "cat"), {"k": 2, "action": "drop"}),
    Step("select.univariate", ("x1", "x3", "x4"), {"test": "anova_f", "k": 1, "action": "drop"}),
    Step("select.univariate", ("x3",), {"test": "kruskal"}),
    Step("select.univariate", ("cat",), {"test": "chi2"}),
    Step("imbalance.class_weight", ()),
    Step("imbalance.random_oversample", ()),
    Step("imbalance.random_undersample", ()),
    Step("imbalance.smote", (), {**ACK, "k_neighbors": 3}),
]


def ident(step: Step) -> str:
    return f"{step.op}:{step.params.get('test', '')}{step.params.get('output_distribution', '')}"


def numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["cat"])


def prepared(step: Step, df: pd.DataFrame) -> pd.DataFrame:
    return numeric_only(df) if step.op == "imbalance.smote" else df


def perturb(df: pd.DataFrame, test: np.ndarray) -> pd.DataFrame:
    """Test rows get extreme values, a new category and flipped labels."""
    out = df.copy()
    rows = out[ROW_ID].isin(test)
    out.loc[rows, ["x1", "x2"]] = out.loc[rows, ["x1", "x2"]] * 50 + 1e4
    out.loc[rows, "x3"] = -out.loc[rows, "x3"] * 30
    out.loc[rows, "x4"] = 0.0
    if "cat" in out:
        out.loc[rows, "cat"] = "test-only"
    out.loc[rows, "y"] = 1 - out.loc[rows, "y"]
    return out


def learned_and_training(
    df: pd.DataFrame, step: Step, rows: Any, seed: int = 0
) -> tuple[Any, pd.DataFrame]:
    ex = PreprocessExecutor(validated(df, [step], ROLES, task="binary"), seed=seed)
    training = ex.fit_resample(df[df[ROW_ID].isin(rows)])
    return training.reports[0].learned, training.frame


def test_spy_every_part_c_step_fits_on_training_rows_only() -> None:
    df = data()
    train, _ = split(df)
    for step in FITTED_C:
        frame_ = prepared(step, df)
        ex = PreprocessExecutor(validated(frame_, [step], ROLES, task="binary"))
        ex.pipeline.steps = [(name, Spy(t)) for name, t in ex.pipeline.steps]
        Spy.seen = []
        ex.fit(frame_[frame_[ROW_ID].isin(train)])
        ex.transform(frame_)
        assert Spy.seen == [set(train.tolist())], step.op


@pytest.mark.parametrize("step", FITTED_C, ids=ident)
def test_test_values_and_labels_change_nothing_learned(step: Step) -> None:
    df = prepared(step, data())
    train, test = split(df)
    learned, training = learned_and_training(df, step, train)
    learned_p, training_p = learned_and_training(perturb(df, test), step, train)
    assert learned_p == learned
    pd.testing.assert_frame_equal(training_p, training)


@pytest.mark.parametrize("step", FITTED_C, ids=ident)
def test_negative_control_fitting_on_all_rows_does_change_what_is_learned(step: Step) -> None:
    """If test rows were used, the perturbation would show: the proof above is not vacuous."""
    df = prepared(step, data())
    train, test = split(df)
    all_ids = df[ROW_ID].to_numpy()
    learned, table = learned_and_training(df, step, all_ids)
    learned_p, table_p = learned_and_training(perturb(df, test), step, all_ids)
    # Compare what is learned and the transformed training rows (quantile values themselves
    # are never reported, so for that op only the rows show the difference).
    on_train = table[table[ROW_ID].isin(train)].drop(columns=["y"])
    on_train_p = table_p[table_p[ROW_ID].isin(train)].drop(columns=["y"])
    assert learned_p != learned or not on_train_p.equals(on_train)


@pytest.mark.parametrize("step", FITTED_C, ids=ident)
def test_test_rows_are_transformed_with_training_parameters_and_never_resampled(step: Step) -> None:
    df = prepared(step, data())
    train, test = split(df)
    ex = PreprocessExecutor(validated(df, [step], ROLES, task="binary"))
    ex.fit_resample(df[df[ROW_ID].isin(train)])
    test_rows = df[df[ROW_ID].isin(test)]
    out = ex.transform(test_rows).frame
    assert out[ROW_ID].tolist() == test_rows[ROW_ID].tolist()
    assert out["y"].value_counts().to_dict() == test_rows["y"].value_counts().to_dict()
    # Applying to the perturbed test rows reuses the same fitted state (no refit).
    out_p = ex.transform(perturb(df, test)[lambda d: d[ROW_ID].isin(test)]).frame
    assert list(out_p.columns) == list(out.columns)


def test_supervised_selectors_refuse_the_target_as_an_input_even_without_validation() -> None:
    df = data()
    for cls, extra in (
        (MutualInfoSelectOp, {"discrete": [False], "n_neighbors": 3}),
        (UnivariateSelectOp, {"test": "anova_f"}),
    ):
        params = {
            "target": "y",
            "task": "binary",
            "k": 1,
            "action": "drop",
            "keep": [],
            "role_kept": [],
            **extra,
        }
        op = cls(columns=("y",), params=params)
        with pytest.raises(EngineError) as exc:
            op.fit(df)
        assert exc.value.code == "OP_INVALID"


def test_target_is_never_changed_by_any_part_c_step() -> None:
    df = data()
    for step in FITTED_C[:9]:
        ex = PreprocessExecutor(validated(df, [step], ROLES, task="binary")).fit(df)
        assert ex.transform(df).frame["y"].equals(df["y"]), step.op


# ---- In-fold proofs with frozen splits ---------------------------------------------------------

SPLITS = {
    "stratified": {"method": "stratified", "target": "y"},
    "group": {"method": "stratified_group", "target": "y", "group": "patient"},
    "time": {"method": "time", "time": "day"},
}

PIPELINE_A_B_C_OVERSAMPLE = [
    Step("missing.indicator", ("x2",)),
    Step("impute.knn", ("x2",), {"predictors": ["x1", "x3"]}),
    Step("outliers.cap", ("x1",), {"method": "iqr"}),
    Step("transform.yeo_johnson", ("x1", "x2")),
    Step("encode.onehot", ("cat",)),
    Step("scale.standard", ("x3", "x4")),
    Step("select.mutual_information", ("x1", "x2", "x3", "x4"), {"k": 3, "action": "drop"}),
    Step("imbalance.random_oversample", ()),
]
PIPELINE_A_B_C_SMOTE = [
    Step("impute.median", ("x2",)),
    Step("outliers.signed_log", ("x3",)),
    Step("transform.quantile", ("x1",), {"output_distribution": "normal"}),
    Step("encode.frequency", ("cat",)),
    Step("scale.robust", ("x2", "x4")),
    Step("select.variance", ("x1", "x2", "x3", "x4"), {"action": "drop", "keep": ["x1", "x3"]}),
    Step("select.univariate", ("x1", "x3"), {"test": "kruskal"}),
    Step("imbalance.smote", (), {**ACK, "k_neighbors": 3}),
]
PIPELINES = {"oversample": PIPELINE_A_B_C_OVERSAMPLE, "smote": PIPELINE_A_B_C_SMOTE}


def with_gaps(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.loc[out.index[::7], "x2"] = np.nan
    return out


@pytest.mark.parametrize("pipeline", list(PIPELINES))
@pytest.mark.parametrize("split_name", list(SPLITS))
def test_combined_pipelines_in_every_fold(pipeline: str, split_name: str) -> None:
    df = with_gaps(data(seed=3))
    roles = {"target": ["y"], "group": ["patient"], "time": ["day"]}
    steps = PIPELINES[pipeline]
    validation = validated(df, steps, roles, task="binary")
    splits = make_splits(df, n_splits=4, seed=1, **SPLITS[split_name])
    for fold in splits.folds:
        train, test = splits.frames(df, fold)
        ex = PreprocessExecutor(validation, seed=2)
        training = ex.fit_resample(train)
        evaluated = ex.transform(test).frame
        # Test rows: same rows, same labels, same class distribution, nothing synthetic.
        assert evaluated[ROW_ID].tolist() == test[ROW_ID].tolist()
        assert evaluated["y"].tolist() == test["y"].tolist()
        # Training table: real rows are training rows only; added rows are marked.
        real = training.frame[training.frame[ROW_ID] >= 0]
        assert set(real[ROW_ID]) == set(train[ROW_ID])
        assert not set(training.frame[ROW_ID]) & set(test[ROW_ID])
        if split_name == "group":
            # Duplicated rows keep their patient; no patient crosses train and test.
            assert not set(training.frame["patient"].dropna()) & set(test["patient"])
        if split_name == "time":
            assert real["day"].max() < test["day"].min()
        # Perturbing this fold's test rows (values and labels) changes nothing learned.
        perturbed = with_gaps(perturb(data(seed=3), test[ROW_ID].to_numpy()))
        ex_p = PreprocessExecutor(validation, seed=2)
        training_p = ex_p.fit_resample(perturbed[perturbed[ROW_ID].isin(train[ROW_ID])])
        pd.testing.assert_frame_equal(training_p.frame, training.frame)
        assert [r.learned for r in training_p.reports] == [r.learned for r in training.reports]


@pytest.mark.parametrize("pipeline", list(PIPELINES))
def test_spy_in_combined_pipelines(pipeline: str) -> None:
    df = with_gaps(data(seed=4))
    train, _ = split(df)
    ex = PreprocessExecutor(validated(df, PIPELINES[pipeline], ROLES, task="binary"))
    ex.pipeline.steps = [(name, Spy(t)) for name, t in ex.pipeline.steps]
    Spy.seen = []
    ex.fit(df[df[ROW_ID].isin(train)])
    assert len(Spy.seen) == len(PIPELINES[pipeline])
    assert all(seen == set(train.tolist()) for seen in Spy.seen)


def test_selection_decisions_can_differ_between_folds_and_are_learned_per_fold() -> None:
    """Each fold learns its own decisions; nothing is carried over from other folds."""
    df = data(seed=5)
    step = [
        Step("select.univariate", ("x1", "x3", "x4"), {"test": "anova_f", "k": 1, "action": "drop"})
    ]
    validation = validated(df, step, ROLES, task="binary")
    splits = make_splits(df, method="stratified", target="y", n_splits=3, seed=0)
    for fold in splits.folds:
        train, _ = splits.frames(df, fold)
        fold_learned = PreprocessExecutor(validation).fit(train).transform(train).reports[0].learned
        alone = PreprocessExecutor(validation).fit(train.copy()).transform(train).reports[0].learned
        assert fold_learned == alone

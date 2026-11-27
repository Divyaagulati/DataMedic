"""Property-based tests (Hypothesis) of the op contract for every missing-value op."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, event, given, settings
from hypothesis import strategies as st

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.hashing import content_hash
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from tests.unit.preprocess_helpers import frame, validated

TYPES = {"f": "numeric", "g": "numeric", "i": "numeric", "c": "categorical", "b": "boolean"}

STEPS = {
    "impute.mean": [Step("impute.mean", ("f", "i"))],
    "impute.median": [Step("impute.median", ("f", "i"))],
    "impute.mode": [Step("impute.mode", ("f", "i", "c", "b"))],
    "impute.constant": [
        Step("impute.constant", ("f", "i"), {"fill_value": 0}),
        Step("impute.constant", ("c",), {"fill_value": "Unknown"}),
        Step("impute.constant", ("b",), {"fill_value": False}),
    ],
    "impute.knn": [Step("impute.knn", ("f", "g"), {"n_neighbors": 3})],
    "impute.iterative": [Step("impute.iterative", ("f", "g"), {"max_iter": 3})],
    "missing.indicator": [Step("missing.indicator", ("f", "i", "c", "b"))],
}
IMPUTED = {
    op: [c for step in steps for c in step.columns] if op != "missing.indicator" else []
    for op, steps in STEPS.items()
}

# Missing in roughly one cell of four, so most examples exercise the fitted path while the
# all-missing edge cases still occur.
_finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)
floats = st.one_of(st.none(), _finite, _finite, _finite)
_int = st.integers(-1000, 1000)
ints = st.one_of(st.none(), _int, _int, _int)
_cat = st.sampled_from(["a", "b", "c", "dd"])
cats = st.one_of(st.none(), _cat, _cat, _cat)
bools = st.one_of(st.none(), st.booleans(), st.booleans(), st.booleans())


@st.composite
def tables(draw: st.DrawFn) -> tuple[pd.DataFrame, list[int]]:
    n = draw(st.integers(0, 25))
    df = frame(
        f=pd.Series(draw(st.lists(floats, min_size=n, max_size=n)), dtype="float64"),
        g=pd.Series(draw(st.lists(floats, min_size=n, max_size=n)), dtype="float64"),
        i=pd.array(draw(st.lists(ints, min_size=n, max_size=n)), dtype="Int64"),
        c=pd.Series(draw(st.lists(cats, min_size=n, max_size=n)), dtype=object),
        b=pd.array(draw(st.lists(bools, min_size=n, max_size=n)), dtype="boolean"),
    )
    in_fit = draw(st.lists(st.sampled_from([True, True, True, False]), min_size=n, max_size=n))
    return df, [i for i, keep in enumerate(in_fit) if keep]


@pytest.mark.parametrize("op", list(STEPS))
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=tables())
def test_op_contract(op: str, data: tuple[pd.DataFrame, list[int]]) -> None:
    df, fit_ids = data
    original = df.copy(deep=True)
    result = validated(df, STEPS[op], types=TYPES)
    fit_df = df[df[ROW_ID].isin(fit_ids)]

    try:
        first = PreprocessExecutor(result, seed=1).fit(fit_df).transform(df)
    except EngineError as exc:
        # The only acceptable failure: a column with no observed value in the fitting rows.
        event("refused: column empty in fit rows")
        assert exc.code == "FIT_COLUMN_EMPTY"
        assert exc.details["column"] in IMPUTED[op]
        assert fit_df[exc.details["column"]].notna().sum() == 0
        pd.testing.assert_frame_equal(df, original)
        return

    event("transformed")
    out = first.frame
    pd.testing.assert_frame_equal(df, original)  # input never mutated
    assert out[ROW_ID].tolist() == df[ROW_ID].tolist()  # rows and order preserved
    for column in df.columns:
        assert column in out.columns  # nothing removed
    for column in IMPUTED[op]:
        assert out[column].isna().sum() == 0  # every gap filled
        present = df[column].notna().to_numpy()
        before = df[column].astype(object).to_numpy()[present]
        after = out[column].astype(object).to_numpy()[present]
        assert all(a == b for a, b in zip(before, after, strict=True))  # observed unchanged
    if op == "missing.indicator":
        for column in ("f", "i", "c", "b"):
            assert out[f"{column}__missing"].tolist() == df[column].isna().tolist()

    # Reproducible: same data, same seed -> identical content hash.
    second = PreprocessExecutor(result, seed=1).fit(fit_df).transform(df)
    assert content_hash(second.frame) == content_hash(out)
    assert [r.learned for r in second.reports] == [r.learned for r in first.reports]


@settings(max_examples=30, deadline=None)
@given(data=tables())
def test_rows_outside_the_fit_set_never_affect_learning(
    data: tuple[pd.DataFrame, list[int]],
) -> None:
    df, fit_ids = data
    if not fit_ids or df.loc[df[ROW_ID].isin(fit_ids), "f"].notna().sum() == 0:
        return
    result = validated(df, [Step("impute.median", ("f",))], types=TYPES)
    shuffled = df.copy()
    others = ~shuffled[ROW_ID].isin(fit_ids)
    shuffled.loc[others, "f"] = np.where(shuffled.loc[others, "f"].isna(), 123456.0, np.nan)
    learned = [
        PreprocessExecutor(result).fit(d[d[ROW_ID].isin(fit_ids)]).transform(d).reports[0].learned
        for d in (df, shuffled)
    ]
    assert learned[0] == learned[1]


# ---- Nullable dtypes and mixed missing markers (review of Part A) -----------------------------

MARKER_VALUES = (None, np.nan, pd.NA)
NUMERIC_DTYPES = ("float64", "Float64", "Int64")
TEXT_DTYPES = ("object", "string", "category")


@st.composite
def nullable_tables(draw: st.DrawFn) -> tuple[pd.DataFrame, list[int], dict[str, str]]:
    n = draw(st.integers(1, 20))
    num_dtype = draw(st.sampled_from(NUMERIC_DTYPES))
    text_dtype = draw(st.sampled_from(TEXT_DTYPES))
    gap = st.sampled_from(MARKER_VALUES)

    def cells(values: st.SearchStrategy[Any]) -> list[Any]:
        return draw(st.lists(st.one_of(gap, values, values, values), min_size=n, max_size=n))

    nums = cells(st.integers(-50, 50))
    if num_dtype == "float64":
        nums = [np.nan if v is None or v is pd.NA else float(v) for v in nums]
    texts = cells(st.sampled_from(["a", "b", "c"]))
    df = frame(
        f=pd.Series(pd.array(nums, dtype=num_dtype)),
        g=pd.Series([float(i) for i in range(n)], dtype="float64"),
        c=pd.Series(texts, dtype=object if text_dtype == "object" else text_dtype),
    )
    in_fit = draw(st.lists(st.sampled_from([True, True, True, False]), min_size=n, max_size=n))
    fit_ids = [i for i, keep in enumerate(in_fit) if keep]
    return df, fit_ids, {"num": num_dtype, "text": text_dtype}


NULLABLE_STEPS = {
    "impute.mean": ([Step("impute.mean", ("f",))], ["f"]),
    "impute.median": ([Step("impute.median", ("f",))], ["f"]),
    "impute.mode": ([Step("impute.mode", ("f", "c"))], ["f", "c"]),
    "impute.constant": (
        [
            Step("impute.constant", ("f",), {"fill_value": 0}),
            Step("impute.constant", ("c",), {"fill_value": "Unknown"}),
        ],
        ["f", "c"],
    ),
    "impute.knn": ([Step("impute.knn", ("f",), {"n_neighbors": 2})], ["f"]),
    "impute.iterative": ([Step("impute.iterative", ("f",), {"max_iter": 3})], ["f"]),
    "missing.indicator": ([Step("missing.indicator", ("f", "c"))], []),
}
KEEPS_DTYPE = {"impute.mode", "impute.constant", "missing.indicator"}


@pytest.mark.parametrize("op", list(NULLABLE_STEPS))
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=nullable_tables())
def test_nullable_dtypes_and_markers(
    op: str, data: tuple[pd.DataFrame, list[int], dict[str, str]]
) -> None:
    df, fit_ids, dtypes = data
    steps, imputed = NULLABLE_STEPS[op]
    original = df.copy(deep=True)
    result = validated(df, steps, types={"f": "numeric", "g": "numeric", "c": "categorical"})
    fit_df = df[df[ROW_ID].isin(fit_ids)]
    test_only_gaps = any(
        df.loc[~df[ROW_ID].isin(fit_ids), c].isna().any() and fit_df[c].notna().all()
        for c in imputed
    )
    try:
        out = PreprocessExecutor(result).fit(fit_df).transform(df).frame
    except EngineError as exc:
        event("refused: column empty in fit rows")
        assert exc.code == "FIT_COLUMN_EMPTY"
        assert fit_df[exc.details["column"]].notna().sum() == 0
        return
    event("transformed" + (" (gaps only in non-fitting rows)" if test_only_gaps else ""))
    pd.testing.assert_frame_equal(df, original)
    assert out[ROW_ID].tolist() == df[ROW_ID].tolist()
    for column in imputed:
        assert out[column].isna().sum() == 0
        present = df[column].notna().to_numpy()
        assert (
            df[column][present].astype(object).tolist()
            == out[column][present].astype(object).tolist()
        )
        if op in KEEPS_DTYPE:
            assert str(out[column].dtype) == str(df[column].dtype)
        else:
            assert out[column].dtype == "float64"
    if op == "missing.indicator":
        for column in ("f", "c"):
            assert out[f"{column}__missing"].tolist() == df[column].isna().tolist()


# ---- Part B: outliers, encoding and scaling --------------------------------------------------

STEPS_B = {
    "outliers.flag": [Step("outliers.flag", ("f", "i"), {"method": "iqr"})],
    "outliers.cap": [Step("outliers.cap", ("f", "i"), {"method": "percentile"})],
    "outliers.cap.mad": [Step("outliers.cap", ("f",), {"method": "mad"})],
    "outliers.signed_log": [Step("outliers.signed_log", ("f", "i"))],
    "outliers.isolation_forest_flag": [
        Step("outliers.isolation_forest_flag", ("f", "g"), {"n_estimators": 20})
    ],
    "encode.onehot": [Step("encode.onehot", ("c", "b"), {"max_categories": 3})],
    "encode.ordinal": [Step("encode.ordinal", ("c",), {"order": ["a", "b", "c", "dd"]})],
    "encode.frequency": [Step("encode.frequency", ("c",))],
    "scale.standard": [Step("scale.standard", ("f", "i"))],
    "scale.minmax": [Step("scale.minmax", ("f",), {"clip": True})],
    "scale.robust": [Step("scale.robust", ("f", "g"))],
}
REFUSALS_B = {"FIT_COLUMN_EMPTY", "FIT_TOO_FEW_VALUES"}


@st.composite
def part_b_tables(draw: st.DrawFn) -> tuple[pd.DataFrame, list[int]]:
    """Tiny tables (refusal edge cases) or 25-80 rows, so the outlier rules (8 observed
    values) and Isolation Forest (20 complete rows) mostly reach the success path."""
    n = draw(st.one_of(st.integers(0, 6), st.integers(25, 80), st.integers(25, 80)))
    df = frame(
        f=pd.Series(draw(st.lists(floats, min_size=n, max_size=n)), dtype="float64"),
        g=pd.Series(draw(st.lists(_finite, min_size=n, max_size=n)), dtype="float64"),
        i=pd.array(draw(st.lists(ints, min_size=n, max_size=n)), dtype="Int64"),
        c=pd.Series(draw(st.lists(cats, min_size=n, max_size=n)), dtype=object),
        b=pd.array(draw(st.lists(bools, min_size=n, max_size=n)), dtype="boolean"),
    )
    in_fit = draw(st.lists(st.sampled_from([True, True, True, False]), min_size=n, max_size=n))
    return df, [i for i, keep in enumerate(in_fit) if keep]


@pytest.mark.parametrize("name", list(STEPS_B))
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=part_b_tables())
def test_part_b_op_contract(name: str, data: tuple[pd.DataFrame, list[int]]) -> None:
    df, fit_ids = data
    original = df.copy(deep=True)
    steps = STEPS_B[name]
    selected = {c for s in steps for c in s.columns}
    result = validated(df, steps, types=TYPES)
    fit_df = df[df[ROW_ID].isin(fit_ids)]
    try:
        out = PreprocessExecutor(result, seed=2).fit(fit_df).transform(df).frame
    except EngineError as exc:
        event(f"refused: {exc.code}")
        assert exc.code in REFUSALS_B, exc.code
        pd.testing.assert_frame_equal(df, original)
        return
    event("transformed")
    pd.testing.assert_frame_equal(df, original)  # input never mutated
    assert out[ROW_ID].tolist() == df[ROW_ID].tolist()  # no row removed or reordered
    for column in df.columns:
        if column not in selected:
            pd.testing.assert_series_equal(out[column], df[column])  # others untouched
    op = steps[0].op
    if op.endswith("_flag") or op == "outliers.flag":
        for column in selected & set(out.columns):
            pd.testing.assert_series_equal(out[column], df[column])  # detection only
    if op == "outliers.cap":
        for column in selected:
            before, after = df[column].astype("float64"), out[column].astype("float64")
            assert after.isna().equals(before.isna())
            if before.notna().any():
                assert after.min() >= before.min() and after.max() <= before.max()
    if op == "encode.onehot":
        for column in ("c", "b"):
            block = out[[n for n in out.columns if n.startswith(column + "=")]]
            assert block.sum(axis=1).max() <= 1
    again = PreprocessExecutor(result, seed=2).fit(fit_df).transform(df).frame
    assert content_hash(again) == content_hash(out)  # reproducible


# ---- Part C: distribution transforms, feature selection, class imbalance ------------------------

ROLES_C = {"target": ["y"]}
_positive = st.floats(1e-3, 1e6, allow_nan=False, allow_infinity=False)
STEPS_C = {
    "transform.log": [Step("transform.log", ("p",), {"base": "10"})],
    "transform.log1p": [Step("transform.log1p", ("p",))],
    "transform.yeo_johnson": [Step("transform.yeo_johnson", ("f", "g"))],
    "transform.quantile": [Step("transform.quantile", ("f", "i"), {"n_quantiles": 20})],
    "select.variance": [
        Step("select.variance", ("f", "g", "b"), {"action": "drop", "threshold": 1.0})
    ],
    "select.correlation": [Step("select.correlation", ("f", "g", "i"), {"threshold": 0.5})],
    "select.mutual_information": [
        Step("select.mutual_information", ("f", "c", "b"), {"k": 1, "action": "drop"})
    ],
    "select.univariate.anova": [
        Step("select.univariate", ("f", "g"), {"test": "anova_f", "k": 1, "action": "drop"})
    ],
    "select.univariate.chi2": [Step("select.univariate", ("c", "b"), {"test": "chi2"})],
    "imbalance.class_weight": [Step("imbalance.class_weight", ())],
    "imbalance.random_oversample": [Step("imbalance.random_oversample", ())],
    "imbalance.random_undersample": [Step("imbalance.random_undersample", ())],
    "imbalance.smote": [
        Step("select.drop_columns", ("f", "i", "c", "b", "p")),
        Step("imbalance.smote", (), {"k_neighbors": 2}),
    ],
}
REFUSALS_C = {
    "FIT_COLUMN_EMPTY",
    "FIT_TOO_FEW_VALUES",
    "TARGET_SINGLE_CLASS",
    "SMOTE_TOO_FEW_SAMPLES",
}


@st.composite
def part_c_tables(draw: st.DrawFn) -> tuple[pd.DataFrame, list[int]]:
    """Tiny tables (refusals) or 25-80 rows with a binary target in every row."""
    n = draw(st.one_of(st.integers(1, 6), st.integers(25, 80), st.integers(25, 80)))
    df = frame(
        f=pd.Series(draw(st.lists(floats, min_size=n, max_size=n)), dtype="float64"),
        g=pd.Series(draw(st.lists(_finite, min_size=n, max_size=n)), dtype="float64"),
        i=pd.array(draw(st.lists(ints, min_size=n, max_size=n)), dtype="Int64"),
        c=pd.Series(draw(st.lists(cats, min_size=n, max_size=n)), dtype=object),
        b=pd.array(draw(st.lists(bools, min_size=n, max_size=n)), dtype="boolean"),
        p=pd.Series(
            draw(st.lists(st.one_of(st.none(), _positive, _positive), min_size=n, max_size=n)),
            dtype="float64",
        ),
        y=pd.Series(
            draw(st.lists(st.sampled_from([0, 0, 0, 1]), min_size=n, max_size=n)), dtype="int64"
        ),
    )
    in_fit = draw(st.lists(st.sampled_from([True, True, True, False]), min_size=n, max_size=n))
    return df, [i for i, keep in enumerate(in_fit) if keep]


@pytest.mark.parametrize("name", list(STEPS_C))
@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=part_c_tables())
def test_part_c_op_contract(name: str, data: tuple[pd.DataFrame, list[int]]) -> None:
    df, fit_ids = data
    original = df.copy(deep=True)
    steps = STEPS_C[name]
    selected = {c for s in steps for c in s.columns}
    result = validated(df, steps, ROLES_C, types={**TYPES, "p": "numeric"}, task="binary")
    fit_df = df[df[ROW_ID].isin(fit_ids)]
    try:
        executor = PreprocessExecutor(result, seed=2)
        training = executor.fit_resample(fit_df).frame
        out = executor.transform(df).frame
    except EngineError as exc:
        event(f"refused: {exc.code}")
        assert exc.code in REFUSALS_C, exc.code
        pd.testing.assert_frame_equal(df, original)
        return
    event("transformed")
    pd.testing.assert_frame_equal(df, original)  # input never mutated
    # transform never resamples: every row, in order, with its label.
    assert out[ROW_ID].tolist() == df[ROW_ID].tolist()
    assert out["y"].tolist() == df["y"].tolist()
    removed = set(df.columns) - set(out.columns)
    assert removed <= selected  # only selected columns can be removed
    for column in df.columns:
        if column not in selected and column in out.columns:
            pd.testing.assert_series_equal(out[column], df[column])
    # The training table: unique ids; real rows are fitting rows; added rows are negative.
    assert training[ROW_ID].is_unique
    real = training[training[ROW_ID] >= 0]
    assert set(real[ROW_ID]) <= set(fit_df[ROW_ID])
    if not name.endswith("undersample"):
        assert set(real[ROW_ID]) == set(fit_df[ROW_ID])
    again = PreprocessExecutor(result, seed=2)
    assert content_hash(again.fit_resample(fit_df).frame) == content_hash(training)
    assert content_hash(again.transform(df).frame) == content_hash(out)

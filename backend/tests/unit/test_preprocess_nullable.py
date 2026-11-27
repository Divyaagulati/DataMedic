"""Nullable data: every op x dtype x missing marker (review of Part A, item 1).

Supported dtypes: float64, Float64, Int64, object (text or numbers), string, category,
boolean. Missing markers: np.nan, None, pd.NA (a plain float64 column cannot hold pd.NA).
Output dtypes: mean/median/KNN/iterative -> float64 (documented); mode/constant/indicator
keep the column's storage dtype.
"""

from typing import Any

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import context, frame, validated

MARKERS = {"nan": np.nan, "None": None, "NA": pd.NA}
GAPS = (2, 5)
FIT = [0, 1, 2, 3, 4, 5]  # rows 6-7 play the test rows


def numeric_column(dtype: str, marker: Any) -> pd.Series:
    values: list[Any] = [1, 2, None, 4, 5, None, 7, 8]
    values = [marker if v is None else v for v in values]
    if dtype == "object":
        return pd.Series([v if v is marker else float(v) for v in values], dtype=object)
    return pd.Series(pd.array(values, dtype=dtype))


def text_column(dtype: str, marker: Any) -> pd.Series:
    values = ["a", "b", marker, "a", "c", marker, "a", "b"]
    return pd.Series(values, dtype=object if dtype == "object" else dtype)


def bool_column(marker: Any) -> pd.Series:
    return pd.Series(pd.array([True, False, marker, True, True, marker, False, True], "boolean"))


NUMERIC_CASES = [
    (dtype, name)
    for dtype in ("float64", "Float64", "Int64", "object")
    for name in MARKERS
    if not (dtype == "float64" and name == "NA")  # numpy float64 cannot hold pd.NA
]
TEXT_CASES = [(dtype, name) for dtype in ("object", "string", "category") for name in MARKERS]


def run_one(df: pd.DataFrame, step: Step, types: dict[str, str]) -> pd.DataFrame:
    result = validated(df, [step], types=types)
    executor = PreprocessExecutor(result).fit(df[df[ROW_ID].isin(FIT)])
    return executor.transform(df).frame


def assert_filled_and_observed_kept(before: pd.Series, after: pd.Series) -> None:
    assert after.isna().sum() == 0
    observed = before.notna().to_numpy()
    assert before[observed].astype(object).tolist() == after[observed].astype(object).tolist()


@pytest.mark.parametrize(("dtype", "marker"), NUMERIC_CASES)
@pytest.mark.parametrize("op", ["impute.mean", "impute.median", "impute.knn", "impute.iterative"])
def test_numeric_estimators(op: str, dtype: str, marker: str) -> None:
    df = frame(c=numeric_column(dtype, MARKERS[marker]), p=np.arange(8, dtype=float))
    out = run_one(df, Step(op, ("c",)), {"c": "numeric", "p": "numeric"})
    assert_filled_and_observed_kept(df["c"], out["c"])
    assert out["c"].dtype == "float64"  # documented output dtype


@pytest.mark.parametrize(("dtype", "marker"), NUMERIC_CASES)
@pytest.mark.parametrize("op", ["impute.mode", "impute.constant"])
def test_numeric_mode_and_constant_keep_storage_type(op: str, dtype: str, marker: str) -> None:
    df = frame(c=numeric_column(dtype, MARKERS[marker]))
    params = {"fill_value": 0} if op == "impute.constant" else {}
    out = run_one(df, Step(op, ("c",), params), {"c": "numeric"})
    assert_filled_and_observed_kept(df["c"], out["c"])
    assert str(out["c"].dtype) == ("float64" if dtype == "float64" else dtype)
    filled = out["c"].iloc[list(GAPS)].tolist()
    assert all(isinstance(v, int | float | np.number) for v in filled)  # never text


@pytest.mark.parametrize(("dtype", "marker"), TEXT_CASES)
@pytest.mark.parametrize("op", ["impute.mode", "impute.constant"])
def test_text_columns_keep_storage_type(op: str, dtype: str, marker: str) -> None:
    df = frame(c=text_column(dtype, MARKERS[marker]))
    params = {"fill_value": "Unknown"} if op == "impute.constant" else {}
    out = run_one(df, Step(op, ("c",), params), {"c": "categorical"})
    assert_filled_and_observed_kept(df["c"], out["c"])
    assert str(out["c"].dtype) == str(df["c"].dtype)
    assert out["c"].iloc[2] == ("Unknown" if op == "impute.constant" else "a")


@pytest.mark.parametrize("marker", list(MARKERS))
@pytest.mark.parametrize("op", ["impute.mode", "impute.constant"])
def test_boolean_columns(op: str, marker: str) -> None:
    df = frame(c=bool_column(MARKERS[marker]))
    params = {"fill_value": False} if op == "impute.constant" else {}
    out = run_one(df, Step(op, ("c",), params), {"c": "boolean"})
    assert_filled_and_observed_kept(df["c"], out["c"])
    assert out["c"].dtype == "boolean"


@pytest.mark.parametrize("marker", list(MARKERS))
@pytest.mark.parametrize(
    "column",
    [
        lambda m: numeric_column("Float64", m),
        lambda m: numeric_column("Int64", m),
        lambda m: text_column("string", m),
        lambda m: text_column("category", m),
        lambda m: text_column("object", m),
        bool_column,
    ],
)
def test_indicator_marks_every_marker_in_every_dtype(marker: str, column: Any) -> None:
    df = frame(c=column(MARKERS[marker]))
    out = run_one(df, Step("missing.indicator", ("c",)), {"c": "categorical"})
    assert out["c__missing"].tolist() == [i in GAPS for i in range(8)]
    assert out["c__missing"].dtype == bool


def test_numbers_in_an_object_column_stay_numbers_under_mode() -> None:
    """Regression: the mode used to be converted to text ('1.0') in object columns."""
    df = frame(c=pd.Series([1.0, 1.0, None, 4.0], dtype=object))
    out = run_one(df, Step("impute.mode", ("c",)), {"c": "numeric"})
    assert out["c"].iloc[2] == 1.0 and not isinstance(out["c"].iloc[2], str)


def test_constant_adds_a_new_category() -> None:
    df = frame(c=pd.Series(["a", None, "b"], dtype="category"))
    out = run_one(
        df, Step("impute.constant", ("c",), {"fill_value": "Unknown"}), {"c": "categorical"}
    )
    assert out["c"].tolist() == ["a", "Unknown", "b"]
    assert "Unknown" in out["c"].cat.categories


# ---- Complete in training rows, missing only in test rows -------------------------------------


@pytest.mark.parametrize(
    ("op", "expected"),
    [
        ("impute.mean", 2.5),
        ("impute.median", 2.5),
        ("impute.mode", 1.0),
        ("impute.knn", 2.5),  # k=5 > 4 fitting rows: all donors averaged
    ],
)
def test_complete_in_training_missing_in_test(op: str, expected: float) -> None:
    df = frame(c=[1.0, 2.0, 3.0, 4.0, np.nan, np.nan], p=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = validated(df, [Step(op, ("c",))], types={"c": "numeric", "p": "numeric"})
    out = PreprocessExecutor(result).fit(df.iloc[:4]).transform(df).frame
    assert out["c"].iloc[4:].tolist() == [expected, expected]


def test_iterative_complete_in_training_missing_in_test() -> None:
    # c = p in training; the fitted model predicts 5 and 6, clipped to the training range [1, 4].
    df = frame(c=[1.0, 2.0, 3.0, 4.0, np.nan, np.nan], p=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = validated(df, [Step("impute.iterative", ("c",))])
    out = PreprocessExecutor(result).fit(df.iloc[:4]).transform(df).frame
    assert out["c"].iloc[4:].tolist() == pytest.approx([4.0, 4.0])
    free = validated(df, [Step("impute.iterative", ("c",), {"clip_to_fit_range": False})])
    out = PreprocessExecutor(free).fit(df.iloc[:4]).transform(df).frame
    assert out["c"].iloc[4:].tolist() == pytest.approx([5.0, 6.0], abs=1e-3)


def test_categorical_complete_in_training_missing_in_test() -> None:
    df = frame(k=pd.Series(["a", "a", "b", "a", None, pd.NA], dtype=object))
    result = validated(df, [Step("impute.mode", ("k",))], types={"k": "categorical"})
    out = PreprocessExecutor(result).fit(df.iloc[:4]).transform(df).frame
    assert out["k"].tolist() == ["a", "a", "b", "a", "a", "a"]


# ---- Entirely missing in training rows --------------------------------------------------------


@pytest.mark.parametrize("dtype", ["float64", "Float64", "Int64", "object", "string", "category"])
@pytest.mark.parametrize(
    "op", ["impute.mean", "impute.median", "impute.mode", "impute.knn", "impute.iterative"]
)
def test_entirely_missing_in_training_fails_clearly(op: str, dtype: str) -> None:
    text = dtype in ("string", "category")
    if op != "impute.mode" and text:
        # Not a fit-time case: validation already refuses numeric estimators on text columns.
        df = frame(c=pd.Series(["a", None], dtype=dtype))
        errors = validate_steps([Step(op, ("c",))], context(df, types={"c": "categorical"})).errors
        assert [e.code for e in errors] == ["COLUMN_TYPE_UNSUPPORTED"]
        return
    values: list[Any] = ["a", "b"] if text else [3.0, 4.0]
    col = pd.Series([None, None, None, *values], dtype=dtype if dtype != "Int64" else "Float64")
    if dtype == "Int64":
        col = pd.Series(pd.array([None, None, None, 3, 4], dtype="Int64"))
    df = frame(c=col, p=np.arange(len(col), dtype=float))
    result = validated(df, [Step(op, ("c",))], types={"c": "categorical" if text else "numeric"})
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(result).fit(df.iloc[:3])
    assert exc.value.code == "FIT_COLUMN_EMPTY" and exc.value.details["column"] == "c"


# ---- Part B: outliers, encoding and scaling on nullable data ----------------------------------

MANY = list(range(1, 21))  # enough observed values for the outlier rules


def many_numeric(dtype: str, marker: Any) -> pd.Series:
    values: list[Any] = [v if v % 7 else marker for v in MANY] + [500]
    if dtype == "object":
        return pd.Series([v if v is marker else float(v) for v in values], dtype=object)
    return pd.Series(pd.array(values, dtype=dtype))


@pytest.mark.parametrize(("dtype", "marker"), NUMERIC_CASES)
@pytest.mark.parametrize(
    "op",
    [
        "outliers.flag",
        "outliers.cap",
        "outliers.signed_log",
        "scale.standard",
        "scale.minmax",
        "scale.robust",
        "scale.none",
    ],
)
def test_part_b_numeric_ops_on_nullable_columns(op: str, dtype: str, marker: str) -> None:
    df = frame(c=many_numeric(dtype, MARKERS[marker]))
    params = {"method": "iqr"} if op.startswith("outliers.") and op != "outliers.signed_log" else {}
    result = validated(df, [Step(op, ("c",), params)], types={"c": "numeric"})
    out = PreprocessExecutor(result).fit(df).transform(df).frame
    gaps = df["c"].isna().to_numpy()
    assert out["c"].isna().to_numpy().tolist() == gaps.tolist()  # missing stays missing
    if op == "outliers.flag":
        assert out["c__outlier"].tolist()[-1] is True or bool(out["c__outlier"].iloc[-1])
        assert not out["c__outlier"][gaps].any()  # a missing value is never an outlier
    if op == "outliers.cap":
        assert str(out["c"].dtype) == dtype  # storage dtype kept (as in Part A)
        assert float(out["c"].iloc[-1]) < 500
        assert all(isinstance(v, int | float | np.number) for v in out["c"].dropna())
    if op in ("outliers.signed_log", "scale.standard", "scale.minmax", "scale.robust"):
        assert out["c"].dtype == "float64"


@pytest.mark.parametrize(("dtype", "marker"), TEXT_CASES)
@pytest.mark.parametrize("op", ["encode.onehot", "encode.ordinal", "encode.frequency"])
def test_part_b_encoders_on_nullable_text_columns(op: str, dtype: str, marker: str) -> None:
    df = frame(c=text_column(dtype, MARKERS[marker]))
    params = {"order": ["a", "b", "c"]} if op == "encode.ordinal" else {}
    result = validated(df, [Step(op, ("c",), params)], types={"c": "categorical"})
    out = PreprocessExecutor(result).fit(df).transform(df).frame
    if op == "encode.onehot":
        encoded = out[["c=a", "c=b", "c=c"]]
        assert encoded.sum(axis=1).tolist() == [0 if i in GAPS else 1 for i in range(8)]
    elif op == "encode.ordinal":
        assert out["c"].tolist() == [0, 1, pd.NA, 0, 2, pd.NA, 0, 1]
    else:
        assert out["c"].isna().to_numpy().tolist() == [i in GAPS for i in range(8)]
        assert out["c"].iloc[0] == pytest.approx(3 / 6)

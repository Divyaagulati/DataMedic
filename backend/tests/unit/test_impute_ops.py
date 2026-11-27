"""Correctness of every missing-value op against hand-calculated values, plus edge cases."""

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import context, frame, mixed_frame, run, validated


def nums(series: pd.Series) -> list[float]:
    return [float(v) for v in series]


# ---- Simple statistics ------------------------------------------------------------------------


def test_mean_and_median_use_hand_calculated_statistics() -> None:
    df = frame(a=[1.0, 2.0, None, 9.0], b=pd.array([2, None, 4, 10], dtype="Int64"))
    out, _ = run(df, [Step("impute.mean", ("a",)), Step("impute.median", ("b",))])
    assert nums(out.frame["a"]) == [1.0, 2.0, 4.0, 9.0]  # mean(1, 2, 9) = 4
    assert nums(out.frame["b"]) == [2.0, 4.0, 4.0, 10.0]  # median(2, 4, 10) = 4
    assert out.frame["b"].dtype == "float64"  # integer -> float is documented and reported
    mean_report, median_report = out.reports
    assert mean_report.learned == {"columns": {"a": {"statistic": 4.0}}}
    assert median_report.columns[0].dtype_before == "integer"
    assert median_report.columns[0].dtype_after == "float"


def test_median_is_robust_where_mean_is_not() -> None:
    df = frame(income=[20.0, 22.0, 25.0, 1000.0, None])
    mean, _ = run(df, [Step("impute.mean", ("income",))])
    median, _ = run(df, [Step("impute.median", ("income",))])
    assert mean.frame["income"].iloc[4] == pytest.approx(266.75)
    assert median.frame["income"].iloc[4] == pytest.approx(23.5)


def test_observed_values_are_never_changed() -> None:
    df = mixed_frame()
    steps = [
        Step("impute.median", ("age",)),
        Step("impute.knn", ("x1",)),
        Step("impute.iterative", ("x2",)),
        Step("impute.mode", ("city", "flag")),
    ]
    out, _ = run(df, steps, roles={"target": ["outcome"], "id": ["patient"]})
    for report in out.reports:
        for change in report.columns:
            assert change.observed_changed == 0, (report.op, change.column)
            assert change.missing_after == 0
            assert change.cells_filled == change.missing_before
    for column in ("x1", "x2"):
        observed = df[column].notna()
        assert np.array_equal(
            out.frame.loc[observed, column].to_numpy(dtype=float),
            df.loc[observed, column].to_numpy(dtype=float),
        )


# ---- Mode -------------------------------------------------------------------------------------


def test_mode_on_categorical_boolean_and_integer_columns() -> None:
    df = frame(
        city=pd.Series(["b", "a", None, "b", None], dtype=object),
        flag=pd.array([True, False, None, False, False], dtype="boolean"),
        n=pd.array([3, 3, None, 1, 2], dtype="Int64"),
    )
    out, _ = run(df, [Step("impute.mode", ("city", "flag", "n"))])
    assert list(out.frame["city"]) == ["b", "a", "b", "b", "b"]
    assert list(out.frame["flag"]) == [True, False, False, False, False]
    assert list(out.frame["n"]) == [3, 3, 3, 1, 2]
    assert out.frame["flag"].dtype == "boolean" and out.frame["n"].dtype == "Int64"
    assert out.frame["city"].isna().sum() == 0


def test_mode_handles_every_kind_of_missing_in_text_columns() -> None:
    """scikit-learn ignores None in object arrays; the op must not (see sklearn contract)."""
    df = frame(c=pd.Series(["x", None, np.nan, pd.NA, "x", "y"], dtype=object))
    out, _ = run(df, [Step("impute.mode", ("c",))])
    assert list(out.frame["c"]) == ["x", "x", "x", "x", "x", "y"]


def test_mode_tie_takes_smallest_and_warns() -> None:
    df = frame(c=pd.Series(["b", "a", "b", "a", None], dtype=object))
    out, _ = run(df, [Step("impute.mode", ("c",))])
    assert out.frame["c"].iloc[4] == "a"
    report = out.reports[0]
    assert report.learned["columns"]["c"] == {"value": "a", "tie": True}
    assert [w["code"] for w in report.warnings] == ["MODE_TIE"]


# ---- Constant ---------------------------------------------------------------------------------


def test_constant_fills_and_keeps_types() -> None:
    df = frame(
        n=pd.array([1, None], dtype="Int64"),
        x=[None, 2.5],
        c=pd.Series([None, "a"], dtype=object),
        b=pd.array([None, True], dtype="boolean"),
    )
    steps = [
        Step("impute.constant", ("n",), {"fill_value": 0}),
        Step("impute.constant", ("x",), {"fill_value": -1}),
        Step("impute.constant", ("c",), {"fill_value": "Unknown"}),
        Step("impute.constant", ("b",), {"fill_value": False}),
    ]
    out, _ = run(df, steps)
    assert list(out.frame["n"]) == [1, 0] and out.frame["n"].dtype == "Int64"
    assert nums(out.frame["x"]) == [-1.0, 2.5]
    assert list(out.frame["c"]) == ["Unknown", "a"]
    assert list(out.frame["b"]) == [False, True] and out.frame["b"].dtype == "boolean"
    assert all(r.n_fit_rows == 2 for r in out.reports)


def test_constant_decimal_on_integer_column_becomes_float() -> None:
    df = frame(n=pd.array([1, None], dtype="Int64"))
    result = validate_steps([Step("impute.constant", ("n",), {"fill_value": 0.5})], context(df))
    assert "DTYPE_CHANGE" in {w.code for w in result.warnings}
    out = PreprocessExecutor(result).fit_transform(df)
    assert nums(out.frame["n"]) == [1.0, 0.5] and out.frame["n"].dtype == "float64"


# ---- KNN --------------------------------------------------------------------------------------


def test_knn_hand_calculated_with_standardised_predictors() -> None:
    # x is missing in row 3. Its nearest neighbours on standardised p are rows 1 and 2.
    df = frame(p=[0.0, 10.0, 11.0, 10.5, 30.0], x=[1.0, 5.0, 7.0, None, 100.0])
    out, _ = run(df, [Step("impute.knn", ("x",), {"n_neighbors": 2})])
    assert out.frame["x"].iloc[3] == pytest.approx(6.0)  # mean of 5 and 7


def test_knn_scaling_prevents_large_columns_dominating() -> None:
    # Without scaling, 'big' (thousands) would decide the neighbours; with scaling, 'small'
    # (which separates the groups strongly) matters equally.
    df = frame(
        big=[1000.0, 1010.0, 1005.0, 1000.0, 1010.0, 1004.0],
        small=[0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        x=[1.0, 1.0, None, 9.0, 9.0, 9.0],
    )
    out, _ = run(df, [Step("impute.knn", ("x",), {"n_neighbors": 2})])
    assert out.frame["x"].iloc[2] == pytest.approx(1.0)


def test_knn_fewer_fit_rows_than_neighbours_warns() -> None:
    df = frame(p=[1.0, 2.0, 3.0], x=[1.0, None, 3.0])
    out, _ = run(df, [Step("impute.knn", ("x",), {"n_neighbors": 5})])
    assert out.frame["x"].iloc[1] == pytest.approx(2.0)
    assert "FEW_FIT_ROWS" in [w["code"] for w in out.reports[0].warnings]


def test_knn_row_limit_fails_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Limits, "KNN_MAX_FIT_ROWS", 3)
    df = frame(p=[1.0, 2.0, 3.0, 4.0], x=[1.0, None, 3.0, 4.0])
    with pytest.raises(EngineError) as exc:
        run(df, [Step("impute.knn", ("x",))])
    assert exc.value.code == "OP_LIMIT_EXCEEDED"


def test_empty_predictor_is_dropped_and_reported() -> None:
    df = frame(p=[np.nan, np.nan, np.nan, np.nan], q=[1.0, 2.0, 3.0, 4.0], x=[1.0, None, 3.0, 4.0])
    out, _ = run(df, [Step("impute.knn", ("x",), {"predictors": ["p", "q"]})])
    learned = out.reports[0].learned
    assert learned["dropped_predictors"] == ["p"] and learned["predictors_used"] == ["q"]
    assert "PREDICTOR_EMPTY" in [w["code"] for w in out.reports[0].warnings]


# ---- Iterative --------------------------------------------------------------------------------


def test_iterative_recovers_a_linear_relationship() -> None:
    rng = np.random.default_rng(1)
    p = rng.normal(0, 1, 200)
    x = 3 * p + 2
    df = frame(p=p, x=x.copy())
    missing = np.arange(0, 200, 10)
    df.loc[missing, "x"] = None
    out, _ = run(df, [Step("impute.iterative", ("x",))])
    assert np.allclose(out.frame.loc[missing, "x"], x[missing], atol=1e-6)
    learned = out.reports[0].learned
    assert learned["seed"] == 0 and learned["predictors_used"] == ["p"]


def test_iterative_clips_to_fit_range_by_default() -> None:
    p = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    df = frame(p=p, x=[0.0, 1.0, 2.0, 3.0, None])  # linear fit would give 10
    clipped, _ = run(df, [Step("impute.iterative", ("x",))])
    free, _ = run(df, [Step("impute.iterative", ("x",), {"clip_to_fit_range": False})])
    assert clipped.frame["x"].iloc[4] == pytest.approx(3.0)
    assert free.frame["x"].iloc[4] > 3.0


def test_iterative_non_convergence_is_reported() -> None:
    rng = np.random.default_rng(3)
    df = frame(**{f"c{i}": rng.normal(size=60) for i in range(4)})
    for i in range(4):
        df.loc[rng.random(60) < 0.4, f"c{i}"] = None
    df.loc[0, :] = df.loc[0, :].fillna(0.0)
    out, _ = run(df, [Step("impute.iterative", ("c0", "c1"), {"max_iter": 1})])
    learned = out.reports[0].learned
    assert learned["converged"] is False and learned["n_iter"] == 1
    assert "NOT_CONVERGED" in [w["code"] for w in out.reports[0].warnings]


# ---- Missing indicator ------------------------------------------------------------------------


def test_indicator_marks_all_kinds_of_missing() -> None:
    df = frame(
        c=pd.Series(["a", None, pd.NA, np.nan], dtype=object),
        when=pd.to_datetime(["2024-01-01", None, "2024-01-03", None]),
    )
    out, _ = run(df, [Step("missing.indicator", ("c", "when"))])
    assert list(out.frame["c__missing"]) == [False, True, True, True]
    assert list(out.frame["when__missing"]) == [False, True, False, True]
    assert out.frame["c__missing"].dtype == bool
    assert out.reports[0].columns_added == ["c__missing", "when__missing"]
    assert out.frame["c"].isna().sum() == 3  # the source column is untouched


def test_indicator_before_imputation_preserves_the_information() -> None:
    df = frame(a=[1.0, None, 3.0])
    out, _ = run(df, [Step("missing.indicator", ("a",)), Step("impute.median", ("a",))])
    assert list(out.frame["a__missing"]) == [False, True, False]
    assert out.frame["a"].isna().sum() == 0


# ---- Failures and tiny data -------------------------------------------------------------------


@pytest.mark.parametrize(
    "op", ["impute.mean", "impute.median", "impute.mode", "impute.knn", "impute.iterative"]
)
def test_column_empty_in_fit_rows_fails_clearly(op: str) -> None:
    df = frame(a=[None, None, 3.0, 4.0], p=[1.0, 2.0, 3.0, 4.0])
    result = validated(df, [Step(op, ("a",))])
    executor = PreprocessExecutor(result)
    with pytest.raises(EngineError) as exc:
        executor.fit(df.iloc[:2])
    assert exc.value.code == "FIT_COLUMN_EMPTY"
    assert exc.value.details == {"op": op, "column": "a"}


@pytest.mark.parametrize(
    "op", ["impute.mean", "impute.median", "impute.mode", "impute.knn", "impute.iterative"]
)
def test_empty_dataset_cannot_be_fitted(op: str) -> None:
    df = frame(a=pd.Series([], dtype="float64"), p=pd.Series([], dtype="float64"))
    with pytest.raises(EngineError) as exc:
        run(df, [Step(op, ("a",))])
    assert exc.value.code == "FIT_COLUMN_EMPTY"


def test_stateless_ops_work_on_empty_and_single_row_tables() -> None:
    empty = frame(a=pd.Series([], dtype="float64"))
    out, _ = run(
        empty,
        [Step("missing.indicator", ("a",)), Step("impute.constant", ("a",), {"fill_value": 0})],
    )
    assert len(out.frame) == 0 and list(out.frame.columns) == [ROW_ID, "a", "a__missing"]
    one = frame(a=pd.Series([None], dtype="float64"))
    out, _ = run(one, [Step("impute.constant", ("a",), {"fill_value": 1.5})])
    assert nums(out.frame["a"]) == [1.5]


@pytest.mark.parametrize("op", ["impute.mean", "impute.median", "impute.knn", "impute.iterative"])
def test_single_observed_row(op: str) -> None:
    df = frame(a=[7.0, None], p=[1.0, 2.0])
    out, _ = run(df, [Step(op, ("a",))])
    assert out.frame["a"].iloc[1] == pytest.approx(7.0)


def test_fitted_model_applies_to_rows_with_nothing_observed() -> None:
    df = frame(p=[1.0, 2.0, 3.0, None], x=[1.0, 2.0, 3.0, None])
    for op in ("impute.knn", "impute.iterative"):
        out, _ = run(df, [Step(op, ("x", "p"))], fit_rows=[0, 1, 2])
        assert out.frame[["x", "p"]].isna().sum().sum() == 0


@pytest.mark.parametrize(
    "op", ["impute.mean", "impute.median", "impute.mode", "impute.knn", "impute.iterative"]
)
def test_infinite_values_are_refused(op: str) -> None:
    df = frame(a=[1.0, np.inf, None], p=[1.0, 2.0, 3.0])
    with pytest.raises(EngineError) as exc:
        run(df, [Step(op, ("a",))])
    assert exc.value.code == "NON_FINITE_VALUES"


def test_transform_before_fit_and_missing_columns_fail() -> None:
    df = frame(a=[1.0, None])
    executor = PreprocessExecutor(validated(df, [Step("impute.mean", ("a",))]))
    with pytest.raises(EngineError):
        executor.transform(df)
    executor.fit(df)
    with pytest.raises(EngineError) as exc:
        executor.transform(df.drop(columns=["a"]))
    assert exc.value.code == "OP_INVALID"


def test_duplicate_or_missing_row_ids_are_refused() -> None:
    df = frame(a=[1.0, None])
    executor = PreprocessExecutor(validated(df, [Step("impute.mean", ("a",))]))
    with pytest.raises(EngineError):
        executor.fit(df.assign(**{ROW_ID: [0, 0]}))
    with pytest.raises(EngineError):
        executor.fit(df.drop(columns=[ROW_ID]))


# ---- Reports and redaction --------------------------------------------------------------------


def test_sensitive_learned_values_are_redacted() -> None:
    df = frame(code=pd.Series(["HIV+", "HIV+", None], dtype=object), a=[1.0, None, 3.0])
    roles = {"sensitive": ["code"]}
    out, _ = run(df, [Step("impute.mode", ("code", "a"))], roles=roles)
    learned = out.reports[0].learned["columns"]
    assert learned["code"] == {"redacted": True}
    assert learned["a"]["value"] == 1.0
    assert "HIV" not in repr(out.report_dicts())


def test_constant_fill_value_is_redacted_for_sensitive_columns() -> None:
    df = frame(code=pd.Series(["x", None], dtype=object))
    out, _ = run(
        df,
        [Step("impute.constant", ("code",), {"fill_value": "Private"})],
        roles={"sensitive": ["code"]},
    )
    assert "Private" not in repr(out.report_dicts())


def test_report_shape() -> None:
    df = frame(a=[1.0, None, 3.0])
    out, _ = run(df, [Step("impute.mean", ("a",))])
    report = out.report_dicts()[0]
    assert report["rows_in"] == report["rows_out"] == 3 and report["rows_removed"] == 0
    assert report["columns"][0]["cells_filled"] == 1
    assert report["n_fit_rows"] == 3 and report["duration_ms"] >= 0
    assert len(report["step_hash"]) == 64

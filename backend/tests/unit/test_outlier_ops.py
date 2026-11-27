"""Outlier ops (Part B): every detection rule against hand-calculated bounds, flag versus cap
(detection versus treatment), constant/skewed/small columns, capping boundaries, Isolation
Forest reproducibility, non-finite values and limits."""

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import context, frame, run, validated

BASE = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]  # Q1 2.75, Q3 6.25, IQR 3.5, median 4.5


def bounds(df: pd.DataFrame, params: dict[str, object], column: str = "x") -> list[float]:
    out, _ = run(df, [Step("outliers.flag", (column,), params)])
    learned = out.reports[0].learned["columns"][column]
    return [learned["lower"], learned["upper"]]


# ---- Detection rules ---------------------------------------------------------------------------


def test_iqr_bounds() -> None:
    df = frame(x=BASE + [100.0])
    q1, q3 = np.quantile(BASE + [100.0], [0.25, 0.75])
    expected = [q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)]
    assert bounds(df, {"method": "iqr"}) == pytest.approx(expected)
    assert bounds(df, {"method": "iqr", "iqr_multiplier": 3.0}) == pytest.approx(
        [q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)]
    )


def test_zscore_bounds_use_sample_sd() -> None:
    values = BASE + [100.0]
    df = frame(x=values)
    mean, sd = np.mean(values), np.std(values, ddof=1)
    assert bounds(df, {"method": "zscore", "z_threshold": 2.0}) == pytest.approx(
        [mean - 2 * sd, mean + 2 * sd]
    )


def test_mad_bounds() -> None:
    values = BASE + [100.0]
    df = frame(x=values)
    median = np.median(values)
    mad = 1.4826 * np.median(np.abs(np.array(values) - median))
    assert bounds(df, {"method": "mad"}) == pytest.approx([median - 3.5 * mad, median + 3.5 * mad])


def test_percentile_bounds() -> None:
    values = list(np.arange(1.0, 101.0))
    df = frame(x=values)
    assert bounds(
        df, {"method": "percentile", "lower_percentile": 5, "upper_percentile": 95}
    ) == pytest.approx(list(np.quantile(values, [0.05, 0.95])))


def test_mad_is_robust_where_zscore_is_masked() -> None:
    # Two huge values inflate the SD so much that z-score flags neither; MAD flags both.
    values = [10.0, 11.0, 9.0, 10.5, 9.5, 10.2, 9.8, 10.1, 500.0, 600.0]
    df = frame(x=values)
    z, _ = run(df, [Step("outliers.flag", ("x",), {"method": "zscore"})])
    m, _ = run(df, [Step("outliers.flag", ("x",), {"method": "mad"})])
    assert z.frame["x__outlier"].sum() == 0
    assert m.frame["x__outlier"].tolist()[-2:] == [True, True]


# ---- Detection versus treatment ----------------------------------------------------------------


def test_flag_never_changes_values() -> None:
    df = frame(x=BASE + [100.0, None])
    out, _ = run(df, [Step("outliers.flag", ("x",), {"method": "iqr"})])
    pd.testing.assert_series_equal(out.frame["x"], df["x"])
    assert out.frame["x__outlier"].tolist() == [False] * 8 + [True, False]
    assert out.frame["x__outlier"].dtype == bool
    report = out.reports[0]
    assert report.effects["columns"]["x"] == {
        "below": 0,
        "above": 1,
        "affected_rows": 1,
        "affected_pct": 10.0,
    }
    assert report.columns_added == ["x__outlier"]


def test_cap_clips_without_removing_rows() -> None:
    df = frame(x=BASE + [100.0, None, -50.0])
    out, _ = run(df, [Step("outliers.cap", ("x",), {"method": "iqr"})])
    learned = out.reports[0].learned["columns"]["x"]
    assert len(out.frame) == len(df) and out.frame[ROW_ID].tolist() == df[ROW_ID].tolist()
    assert out.frame["x"].iloc[8] == pytest.approx(learned["upper"])
    assert out.frame["x"].iloc[10] == pytest.approx(learned["lower"])
    assert pd.isna(out.frame["x"].iloc[9])  # missing is never an outlier and is not filled
    assert out.frame["x"].iloc[:8].tolist() == BASE  # values inside the bounds are untouched
    change = out.reports[0].columns[0]
    assert change.observed_changed == 2 and out.reports[0].rows_removed == 0


def test_values_exactly_on_the_bounds_are_not_outliers() -> None:
    df = frame(x=list(np.arange(1.0, 101.0)))
    params = {"method": "percentile", "lower_percentile": 10, "upper_percentile": 90}
    out, _ = run(df, [Step("outliers.flag", ("x",), params)])
    low, high = (out.reports[0].learned["columns"]["x"][k] for k in ("lower", "upper"))
    test = frame(x=[low, high, np.nextafter(low, -np.inf), np.nextafter(high, np.inf)])
    executor = PreprocessExecutor(validated(df, [Step("outliers.flag", ("x",), params)]))
    flags = executor.fit(df).transform(test).frame["x__outlier"].tolist()
    assert flags == [False, False, True, True]


def test_integer_columns_stay_integer_with_inward_bounds() -> None:
    df = frame(n=pd.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 1000], dtype="Int64"))
    params = {"method": "percentile", "lower_percentile": 5, "upper_percentile": 95}
    out, _ = run(df, [Step("outliers.cap", ("n",), params)])
    learned = out.reports[0].learned["columns"]["n"]
    assert out.frame["n"].dtype == "Int64"
    assert out.frame["n"].max() == np.floor(learned["upper"])  # rounded inward
    assert out.frame["n"].min() == np.ceil(learned["lower"])


def test_integer_bounds_fall_back_outward_when_no_whole_number_fits() -> None:
    from datamedic.engine.preprocess.runtime import integer_bounds

    assert integer_bounds([1.07, 1.93]) == (1, 2)
    assert integer_bounds([1.2, 5.8]) == (2, 5)


# ---- Constant, skewed and small columns -------------------------------------------------------


@pytest.mark.parametrize("method", ["iqr", "zscore", "mad", "percentile"])
def test_constant_column_is_left_unchanged_with_a_warning(method: str) -> None:
    df = frame(x=[5.0] * 10)
    test = frame(x=[5.0, 6.0, 100.0])
    for op in ("outliers.flag", "outliers.cap"):
        executor = PreprocessExecutor(validated(df, [Step(op, ("x",), {"method": method})]))
        out = executor.fit(df).transform(test)
        assert "ZERO_SPREAD" in [w["code"] for w in out.reports[0].warnings]
        assert out.reports[0].learned["columns"]["x"]["lower"] is None
        if op == "outliers.flag":
            assert out.frame["x__outlier"].sum() == 0
        else:
            assert out.frame["x"].tolist() == [5.0, 6.0, 100.0]


def test_mostly_identical_column_has_zero_mad() -> None:
    df = frame(x=[0.0] * 7 + [1.0, 2.0, 50.0])  # MAD = 0 (zero-inflated)
    out, _ = run(df, [Step("outliers.flag", ("x",), {"method": "mad"})])
    assert "ZERO_SPREAD" in [w["code"] for w in out.reports[0].warnings]
    assert out.frame["x__outlier"].sum() == 0  # nothing invented


def test_skewed_column_warns_for_symmetric_rules_only() -> None:
    rng = np.random.default_rng(0)
    df = frame(x=rng.lognormal(0, 1.2, 300))
    for method, expected in (
        ("iqr", True),
        ("zscore", True),
        ("mad", False),
        ("percentile", False),
    ):
        out, _ = run(df, [Step("outliers.flag", ("x",), {"method": method})])
        codes = [w["code"] for w in out.reports[0].warnings]
        assert ("SKEWED_COLUMN" in codes) is expected, method


@pytest.mark.parametrize("op", ["outliers.flag", "outliers.cap"])
def test_too_few_values_fail_clearly(op: str) -> None:
    df = frame(x=[1.0, 2.0, 3.0, None, None, None, None, None, 4.0, 5.0])  # 5 observed
    with pytest.raises(EngineError) as exc:
        run(df, [Step(op, ("x",))])
    assert exc.value.code == "FIT_TOO_FEW_VALUES" and exc.value.details["column"] == "x"


def test_percentile_on_small_samples_warns() -> None:
    out, _ = run(frame(x=BASE + [9.0, 10.0]), [Step("outliers.cap", ("x",))])
    assert "SMALL_SAMPLE_PERCENTILE" in [w["code"] for w in out.reports[0].warnings]


@pytest.mark.parametrize("op", ["outliers.flag", "outliers.cap", "outliers.signed_log"])
def test_infinite_values_are_refused(op: str) -> None:
    df = frame(x=BASE + [np.inf])
    with pytest.raises(EngineError) as exc:
        run(df, [Step(op, ("x",))])
    assert exc.value.code == "NON_FINITE_VALUES"


# ---- Signed log -------------------------------------------------------------------------------


def test_signed_log_values_and_inverse() -> None:
    values = [-1000.0, -1.0, 0.0, 0.5, 1.0, 1e6, None]
    df = frame(x=values)
    out, _ = run(df, [Step("outliers.signed_log", ("x",))])
    got = out.frame["x"].to_numpy()
    expected = np.array(
        [np.sign(v) * np.log1p(abs(v)) if v is not None else np.nan for v in values]
    )
    assert np.allclose(got, expected, equal_nan=True)
    restored = np.sign(got) * np.expm1(np.abs(got))
    assert np.allclose(restored[:-1], values[:-1], rtol=1e-12)
    assert list(np.argsort(got[:-1])) == list(np.argsort(values[:-1]))  # order preserved


# ---- Isolation Forest -------------------------------------------------------------------------


def iso_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    b = a + rng.normal(0, 0.1, n)
    b[:3] = [-5.0, 5.0, 4.0]  # break the a~b relationship
    return frame(a=a, b=b)


def test_isolation_forest_is_reproducible_with_a_seed() -> None:
    df = iso_frame()
    step = [Step("outliers.isolation_forest_flag", ("a", "b"), {"contamination": 0.02})]
    runs = [run(df, step, seed=7)[0].frame["isolation_forest_outlier"].tolist() for _ in range(3)]
    assert runs[0] == runs[1] == runs[2]
    assert all(runs[0][:3])  # the planted rows are isolated


def test_isolation_forest_contamination_controls_share() -> None:
    df = iso_frame(n=400)
    out, _ = run(
        df, [Step("outliers.isolation_forest_flag", ("a", "b"), {"contamination": 0.05})], seed=1
    )
    flagged = int(out.frame["isolation_forest_outlier"].sum())
    assert flagged == pytest.approx(20, abs=3)
    assert out.reports[0].learned["contamination"] == 0.05
    assert out.reports[0].effects["rows_flagged"] == flagged


def test_isolation_forest_rows_with_gaps_are_not_scored() -> None:
    df = iso_frame(n=60)
    df.loc[[5, 6], "a"] = np.nan
    out, _ = run(df, [Step("outliers.isolation_forest_flag", ("a", "b"))])
    assert out.frame.loc[[5, 6], "isolation_forest_outlier"].tolist() == [False, False]
    assert out.reports[0].effects["rows_not_scored"] == 2
    assert "ROWS_NOT_USED" in [w["code"] for w in out.reports[0].warnings]
    assert out.reports[0].learned["n_complete_fit_rows"] == 58


def test_isolation_forest_needs_enough_complete_rows() -> None:
    df = iso_frame(n=Limits.IF_MIN_ROWS - 1)
    with pytest.raises(EngineError) as exc:
        run(df, [Step("outliers.isolation_forest_flag", ("a", "b"))])
    assert exc.value.code == "FIT_TOO_FEW_VALUES"


def test_isolation_forest_output_name_is_configurable_and_checked() -> None:
    df = iso_frame(n=40)
    out, _ = run(
        df, [Step("outliers.isolation_forest_flag", ("a", "b"), {"output_column": "odd_row"})]
    )
    assert "odd_row" in out.frame.columns
    for bad in ("a", "_dm_x", "   ", "x\ny"):
        result = validate_steps(
            [Step("outliers.isolation_forest_flag", ("a", "b"), {"output_column": bad})],
            context(df),
        )
        assert not result.valid, bad


def test_isolation_forest_fit_row_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Limits, "IF_MAX_FIT_ROWS", 50)
    with pytest.raises(EngineError) as exc:
        run(iso_frame(n=60), [Step("outliers.isolation_forest_flag", ("a", "b"))])
    assert exc.value.code == "OP_LIMIT_EXCEEDED"


# ---- Validation rules -------------------------------------------------------------------------


def test_method_parameters_must_match_the_method() -> None:
    df = frame(x=BASE)
    ctx = context(df)
    wrong = validate_steps([Step("outliers.cap", ("x",), {"method": "iqr", "z_threshold": 2})], ctx)
    assert [e.code for e in wrong.errors] == ["PARAM_NOT_APPLICABLE"]
    ok = validate_steps([Step("outliers.cap", ("x",), {"method": "mad"})], ctx)
    assert ok.steps[0].step.params == {"method": "mad", "mad_threshold": 3.5}  # only what applies
    swapped = validate_steps(
        [
            Step(
                "outliers.cap",
                ("x",),
                {"method": "percentile", "lower_percentile": 40, "upper_percentile": 60},
            ),
            Step(
                "outliers.cap",
                ("x",),
                {"method": "percentile", "lower_percentile": 49, "upper_percentile": 49},
            ),
        ],
        ctx,
    )
    # The first step is valid; the second breaks the upper range (minimum 51) and the order.
    assert {(e.code, e.step) for e in swapped.errors} == {("PARAM_INVALID", 1)}


def test_outlier_ops_refuse_protected_and_non_numeric_columns() -> None:
    df = frame(x=BASE, y=[0.0, 1.0] * 4, c=["a", "b"] * 4)
    ctx = context(df, {"target": ["y"]}, types={"c": "categorical"})
    for op in (
        "outliers.flag",
        "outliers.cap",
        "outliers.signed_log",
        "outliers.isolation_forest_flag",
    ):
        codes = {e.code for e in validate_steps([Step(op, ("y",))], ctx).errors}
        assert "COLUMN_PROTECTED" in codes, op
        codes = {e.code for e in validate_steps([Step(op, ("c",))], ctx).errors}
        assert "COLUMN_TYPE_UNSUPPORTED" in codes, op


def test_flag_name_collisions() -> None:
    df = frame(x=BASE, **{"x__outlier": [True] * 8})
    assert "OUTPUT_COLUMN_CONFLICT" in {
        e.code for e in validate_steps([Step("outliers.flag", ("x",))], context(df)).errors
    }

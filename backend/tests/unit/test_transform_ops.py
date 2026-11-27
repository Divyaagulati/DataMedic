"""Distribution transforms (Part C): log, log1p, Yeo-Johnson and quantile.

Covers positive, zero, negative, missing, constant, extreme and non-finite values, domain
refusals, inverse accuracy, reported before/after statistics, information-loss reporting,
nullable dtypes, row identity, input immutability and limits."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from tests.unit.preprocess_helpers import frame, run, validated


def one(op: str, values: list[float] | np.ndarray, params: dict | None = None, **kw):  # type: ignore[no-untyped-def]
    df = frame(x=np.asarray(values, dtype="float64"))
    out, executor = run(df, [Step(op, ("x",), params or {})], **kw)
    return df, out, executor


# ---- log / log1p --------------------------------------------------------------------------------


@pytest.mark.parametrize(("base", "func"), [("e", np.log), ("2", np.log2), ("10", np.log10)])
def test_log_matches_numpy_for_each_base(base: str, func: object) -> None:
    values = [0.001, 1.0, 2.5, 1e6, 1e300, 5e-324]
    _, out, _ = one("transform.log", values, {"base": base})
    assert np.array_equal(out.frame["x"].to_numpy(), func(np.array(values)))  # type: ignore[operator]


@pytest.mark.parametrize("bad", [0.0, -1.0, -1e-300])
def test_log_refuses_zero_and_negative_values_at_fit(bad: float) -> None:
    with pytest.raises(EngineError) as exc:
        one("transform.log", [1.0, 2.0, bad])
    assert exc.value.code == "VALUE_OUT_OF_DOMAIN" and exc.value.details["column"] == "x"
    assert str(bad) not in exc.value.message


def test_log_refuses_zero_in_rows_it_is_applied_to_never_silent_nan() -> None:
    df = frame(x=[1.0, 2.0, 3.0, 0.0])
    executor = PreprocessExecutor(validated(df, [Step("transform.log", ("x",))]))
    executor.fit(df.iloc[:3])
    with pytest.raises(EngineError) as exc:
        executor.transform(df)
    assert exc.value.code == "VALUE_OUT_OF_DOMAIN"


def test_log1p_domain_is_greater_than_minus_one() -> None:
    _, out, _ = one("transform.log1p", [-0.999, 0.0, -0.0, 3.0, 1e308])
    assert np.array_equal(out.frame["x"].to_numpy(), np.log1p([-0.999, 0.0, -0.0, 3.0, 1e308]))
    for bad in (-1.0, -5.0):
        with pytest.raises(EngineError) as exc:
            one("transform.log1p", [1.0, bad])
        assert exc.value.code == "VALUE_OUT_OF_DOMAIN"


@pytest.mark.parametrize("op", ["transform.log", "transform.log1p"])
def test_log_family_refuses_infinity_and_keeps_missing(op: str) -> None:
    with pytest.raises(EngineError) as exc:
        one(op, [1.0, np.inf])
    assert exc.value.code == "NON_FINITE_VALUES"
    _, out, _ = one(op, [1.0, np.nan, 3.0])
    assert out.frame["x"].isna().tolist() == [False, True, False]


@pytest.mark.parametrize("op", ["transform.log", "transform.log1p"])
def test_log_family_inverse_is_accurate_to_rounding(op: str) -> None:
    rng = np.random.default_rng(1)
    values = np.concatenate([rng.lognormal(0, 3, 500), [1e-300, 1e300, 1.0]])
    _, out, _ = one(op, values)
    y = out.frame["x"].to_numpy()
    back = np.exp(y) if op == "transform.log" else np.expm1(y)
    assert np.max(np.abs(back - values) / values) < 1e-13


def test_log_on_integer_and_nullable_columns_gives_float_and_keeps_rows() -> None:
    df = frame(
        i=np.array([1, 10, 100], dtype="int64"),
        n=pd.array([1, None, 100], dtype="Int64"),
    )
    out, _ = run(df, [Step("transform.log", ("i", "n"), {"base": "10"})])
    assert out.frame["i"].tolist() == [0.0, 1.0, 2.0]
    assert out.frame["n"].isna().tolist() == [False, True, False]
    assert str(out.frame["i"].dtype) == "float64"
    assert out.frame[ROW_ID].tolist() == df[ROW_ID].tolist()
    assert list(out.frame.columns) == list(df.columns)


# ---- Yeo-Johnson --------------------------------------------------------------------------------


def test_yeo_johnson_matches_scikit_learn_on_mixed_sign_data() -> None:
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.lognormal(0, 1, 300) - 0.5, [0.0, -3.0]])
    _, out, executor = one("transform.yeo_johnson", values)
    reference = PowerTransformer(standardize=False).fit(values.reshape(-1, 1))
    lam = executor.transformers()[0].learned_params()["columns"]["x"]["lambda"]
    assert lam == pytest.approx(float(reference.lambdas_[0]), rel=1e-12)
    expected = reference.transform(values.reshape(-1, 1))[:, 0]
    assert np.allclose(out.frame["x"].to_numpy(), expected, rtol=1e-10, atol=1e-12)


def test_yeo_johnson_learns_lambda_from_fitting_rows_only() -> None:
    rng = np.random.default_rng(4)
    df = frame(x=np.concatenate([rng.lognormal(0, 1, 100), np.full(50, 1e4)]))
    _, executor = run(df, [Step("transform.yeo_johnson", ("x",))], fit_rows=df.index[:100])
    lam_train = executor.transformers()[0].state_["lambdas"][0]
    reference = PowerTransformer(standardize=False).fit(df["x"].to_numpy()[:100].reshape(-1, 1))
    assert lam_train == pytest.approx(float(reference.lambdas_[0]), rel=1e-12)


def test_yeo_johnson_constant_column_is_identity_and_reported() -> None:
    _, out, executor = one("transform.yeo_johnson", [4.0] * 10 + [np.nan])
    assert out.frame["x"].tolist()[:10] == [4.0] * 10  # exact (regression: was 3.999...)
    codes = [w["code"] for w in out.reports[0].warnings]
    assert "CONSTANT_COLUMN" in codes
    assert out.reports[0].learned["columns"]["x"]["constant"] is True


def test_yeo_johnson_inverse_is_approximate_but_close() -> None:
    rng = np.random.default_rng(5)
    values = rng.normal(0, 3, 400)
    _, out, executor = one("transform.yeo_johnson", values)
    lam = executor.transformers()[0].state_["lambdas"][0]
    model = PowerTransformer(standardize=False)
    model.fit(values.reshape(-1, 1))
    model.lambdas_ = np.array([lam])
    back = model.inverse_transform(out.frame["x"].to_numpy().reshape(-1, 1))[:, 0]
    assert np.max(np.abs(back - values)) < 1e-9


def test_yeo_johnson_overflow_fails_instead_of_writing_infinity() -> None:
    rng = np.random.default_rng(6)
    df = frame(x=np.concatenate([-rng.lognormal(0, 1, 60), [1e300]]))
    # Left-skewed fitting rows give lambda > 1; a huge value then overflows at transform.
    executor = PreprocessExecutor(validated(df, [Step("transform.yeo_johnson", ("x",))]))
    executor.fit(df.iloc[:60])
    assert executor.transformers()[0].state_["lambdas"][0] > 1
    with pytest.raises(EngineError) as exc:
        executor.transform(df)
    assert exc.value.code == "NUMERIC_OVERFLOW"


def test_yeo_johnson_needs_enough_observed_values() -> None:
    with pytest.raises(EngineError) as exc:
        one("transform.yeo_johnson", [1.0, 2.0, 3.0, np.nan] * 2)
    assert exc.value.code == "FIT_TOO_FEW_VALUES"


# ---- Quantile ------------------------------------------------------------------------------------


@pytest.mark.parametrize("dist", ["uniform", "normal"])
def test_quantile_matches_scikit_learn(dist: str) -> None:
    rng = np.random.default_rng(7)
    values = rng.lognormal(0, 2, 500)
    _, out, _ = one("transform.quantile", values, {"output_distribution": dist}, seed=9)
    reference = QuantileTransformer(
        n_quantiles=500,
        output_distribution=dist,
        subsample=Limits.QUANTILE_SUBSAMPLE,
        random_state=9,
    ).fit(values.reshape(-1, 1))
    assert np.allclose(out.frame["x"].to_numpy(), reference.transform(values.reshape(-1, 1))[:, 0])


def test_quantile_reduces_quantiles_to_observed_values_and_reports_it() -> None:
    _, out, _ = one("transform.quantile", np.arange(20, dtype="float64"))
    report = out.reports[0]
    assert report.learned["columns"]["x"]["n_quantiles"] == 20
    assert "QUANTILES_REDUCED" in [w["code"] for w in report.warnings]


def test_quantile_never_reports_learned_quantiles() -> None:
    values = np.linspace(1000.123, 2000.987, 50)
    _, out, _ = one("transform.quantile", values)
    text = repr(out.reports[0].learned)
    assert "1000.123" not in text and "2000.987" not in text
    assert set(out.reports[0].learned["columns"]["x"]) == {"n_quantiles", "n_values", "constant"}


def test_quantile_constant_column_maps_to_middle_and_ends() -> None:
    df = frame(x=[5.0] * 10 + [4.0, 6.0, np.nan])
    out, _ = run(df, [Step("transform.quantile", ("x",))], fit_rows=df.index[:10])
    y = out.frame["x"].tolist()
    assert y[:10] == [0.5] * 10 and y[10] == 0.0 and y[11] == 1.0 and np.isnan(y[12])
    out_n, _ = run(
        df,
        [Step("transform.quantile", ("x",), {"output_distribution": "normal"})],
        fit_rows=df.index[:10],
    )
    low, high = rt.normal_output_bounds()
    assert out_n.frame["x"].tolist()[:12] == [0.0] * 10 + [low, high]
    assert low == pytest.approx(-5.199, abs=1e-3)


def test_quantile_counts_values_outside_the_fitting_range() -> None:
    df = frame(x=list(np.linspace(0, 1, 30)) + [-5.0, 9.0, 0.5])
    out, _ = run(df, [Step("transform.quantile", ("x",))], fit_rows=df.index[:30])
    report = out.reports[0]
    assert report.effects["outside_fit_range_rows"] == {"x": 2}
    assert "OUTSIDE_FIT_RANGE" in [w["code"] for w in report.warnings]
    assert out.frame["x"].tolist()[30:32] == [0.0, 1.0]


def test_quantile_is_reproducible_with_a_seed_when_subsampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Limits, "QUANTILE_SUBSAMPLE", 50)
    rng = np.random.default_rng(8)
    values = rng.normal(size=400)
    a = one("transform.quantile", values, seed=1)[1].frame["x"]
    b = one("transform.quantile", values, seed=1)[1].frame["x"]
    c = one("transform.quantile", values, seed=2)[1].frame["x"]
    assert a.equals(b) and not a.equals(c)
    assert (
        one("transform.quantile", values)[1].reports[0].learned["columns"]["x"]["n_quantiles"] == 50
    )  # never more quantiles than sampled rows
    warnings = one("transform.quantile", values, seed=1)[1].reports[0].warnings
    assert "QUANTILES_SUBSAMPLED" in [w["code"] for w in warnings]


# ---- Distribution statistics and honesty -------------------------------------------------------


def test_before_after_statistics_show_tail_compression() -> None:
    rng = np.random.default_rng(9)
    _, out, _ = one("transform.log", rng.lognormal(0, 1.5, 2000))
    stats = out.reports[0].effects["distribution"]["x"]
    assert stats["before"]["skewness"] > 3 and abs(stats["after"]["skewness"]) < 0.2
    assert stats["before"]["tail_ratio"] > 5 and 0.7 < stats["after"]["tail_ratio"] < 1.4
    assert stats["distinct_values_merged"] == 0
    assert "SKEW_NOT_REDUCED" not in [w["code"] for w in out.reports[0].warnings]


def test_transform_that_does_not_reduce_skew_is_reported_not_praised() -> None:
    rng = np.random.default_rng(10)
    left = -rng.lognormal(0, 1, 500) + 100  # left-skewed and positive: log makes it worse
    _, out, _ = one("transform.log", left)
    assert "SKEW_NOT_REDUCED" in [w["code"] for w in out.reports[0].warnings]


def test_merged_values_are_reported_as_information_loss() -> None:
    # Values below the fitting range all map to the lower end: they become equal.
    df = frame(x=list(np.linspace(0, 1, 40)) + [-5.0, -6.0])
    out, _ = run(df, [Step("transform.quantile", ("x",))], fit_rows=df.index[:40])
    stats = out.reports[0].effects["distribution"]["x"]
    assert stats["distinct_values_merged"] > 0
    assert "VALUES_MERGED" in [w["code"] for w in out.reports[0].warnings]


@pytest.mark.parametrize(
    "op", ["transform.log", "transform.log1p", "transform.yeo_johnson", "transform.quantile"]
)
def test_transforms_keep_rows_index_order_and_never_mutate_input(op: str) -> None:
    rng = np.random.default_rng(11)
    df = frame(a=rng.lognormal(0, 1, 40), x=rng.lognormal(0, 1, 40), z=["k"] * 40)
    df.index = pd.Index(np.arange(100, 140))
    df.loc[df.index[3], "x"] = np.nan
    before = df.copy(deep=True)
    out, _ = run(df, [Step(op, ("x",))])
    pd.testing.assert_frame_equal(df, before)
    assert out.frame.index.equals(df.index)
    assert list(out.frame.columns) == list(df.columns)
    assert out.frame[ROW_ID].tolist() == df[ROW_ID].tolist()
    assert out.frame["x"].isna().tolist() == df["x"].isna().tolist()
    assert out.frame["a"].equals(df["a"])


def test_yeo_johnson_fit_row_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Limits, "POWER_MAX_FIT_ROWS", 10)
    with pytest.raises(EngineError) as exc:
        one("transform.yeo_johnson", np.arange(1, 30, dtype="float64"))
    assert exc.value.code == "OP_LIMIT_EXCEEDED"


def test_quantile_fit_row_limit_is_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Limits, "QUANTILE_MAX_FIT_ROWS", 10)
    with pytest.raises(EngineError) as exc:
        one("transform.quantile", np.arange(1, 30, dtype="float64"))
    assert exc.value.code == "OP_LIMIT_EXCEEDED"

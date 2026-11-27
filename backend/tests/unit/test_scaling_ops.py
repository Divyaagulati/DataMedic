"""Scaling ops (Part B): hand-calculated parameters, constant columns, values outside the
fitting range, missing values, and 'no scaling'."""

import numpy as np
import pandas as pd
import pytest

from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.spec import Step
from tests.unit.preprocess_helpers import frame, run

TRAIN = [2.0, 4.0, 6.0, 8.0]


def scaled(
    op: str,
    values: list[float | None],
    params: dict[str, object] | None = None,
    fit_rows: list[int] | None = None,
):
    df = frame(x=values)
    out, _ = run(df, [Step(op, ("x",), params or {})], fit_rows=fit_rows or [0, 1, 2, 3])
    return out


def test_standard_scaler_uses_training_mean_and_population_sd() -> None:
    out = scaled("scale.standard", TRAIN + [10.0, None])
    mean, sd = np.mean(TRAIN), np.std(TRAIN)  # scikit-learn: ddof = 0
    expected = [(v - mean) / sd for v in TRAIN + [10.0]]
    got = out.frame["x"].tolist()
    assert got[:5] == pytest.approx(expected) and np.isnan(got[5])
    assert out.reports[0].learned["columns"]["x"] == pytest.approx({"mean": mean, "scale": sd})
    assert out.reports[0].effects == {"outside_fit_range_rows": {"x": 1}}


def test_minmax_out_of_range_and_clip() -> None:
    free = scaled("scale.minmax", TRAIN + [10.0, 0.0])
    assert free.frame["x"].tolist() == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0, 4 / 3, -1 / 3])
    assert free.reports[0].effects == {"outside_fit_range_rows": {"x": 2}}
    clipped = scaled("scale.minmax", TRAIN + [10.0, 0.0], {"clip": True})
    assert clipped.frame["x"].tolist()[-2:] == [1.0, 0.0]
    assert clipped.reports[0].effects == {"outside_fit_range_rows": {"x": 2}}  # still counted


def test_robust_scaler_uses_median_and_iqr() -> None:
    out = scaled("scale.robust", TRAIN + [100.0])
    q1, q3 = np.quantile(TRAIN, [0.25, 0.75])
    median = np.median(TRAIN)
    assert out.frame["x"].tolist() == pytest.approx(
        [(v - median) / (q3 - q1) for v in TRAIN + [100.0]]
    )


@pytest.mark.parametrize("op", ["scale.standard", "scale.minmax", "scale.robust"])
def test_constant_column_is_shifted_not_divided_by_zero(op: str) -> None:
    out = scaled(op, [5.0, 5.0, 5.0, 5.0, 7.0])
    got = out.frame["x"].tolist()
    assert all(np.isfinite(got))
    assert got[:4] == [0.0, 0.0, 0.0, 0.0] and got[4] == pytest.approx(2.0)
    assert "CONSTANT_COLUMN" in [w["code"] for w in out.reports[0].warnings]


@pytest.mark.parametrize("op", ["scale.minmax", "scale.robust"])
def test_extreme_values_are_scaled_without_overflow(op: str) -> None:
    out = scaled(op, [1e-300, 2e-300, 1e300, 2e300, -1e308])
    assert np.isfinite(out.frame["x"].to_numpy()).all()


def test_standard_scaling_refuses_values_whose_variance_overflows() -> None:
    """The variance of values around 1e300 is about 1e600, beyond float range. scikit-learn
    would silently map the column to zeros; the op refuses instead."""
    with pytest.raises(EngineError) as exc:
        scaled("scale.standard", [1e-300, 2e-300, 1e300, 2e300, -1e308])
    assert exc.value.code == "NUMERIC_OVERFLOW" and exc.value.details["column"] == "x"


@pytest.mark.parametrize("op", ["scale.standard", "scale.minmax", "scale.robust"])
def test_infinite_and_empty_inputs_fail_clearly(op: str) -> None:
    with pytest.raises(EngineError) as exc:
        scaled(op, [1.0, np.inf, 2.0, 3.0])
    assert exc.value.code == "NON_FINITE_VALUES"
    with pytest.raises(EngineError) as exc:
        scaled(op, [None, None, None, None, 1.0])
    assert exc.value.code == "FIT_COLUMN_EMPTY"


def test_integer_columns_become_float() -> None:
    df = frame(n=pd.array([1, 2, None, 4], dtype="Int64"))
    out, _ = run(df, [Step("scale.standard", ("n",))])
    assert out.frame["n"].dtype == "float64" and np.isnan(out.frame["n"].iloc[2])


def test_no_scaling_changes_nothing() -> None:
    df = frame(x=[1.0, 2.0, None])
    out, _ = run(df, [Step("scale.none", ("x",))])
    pd.testing.assert_frame_equal(out.frame, df)
    assert out.reports[0].columns[0].observed_changed == 0

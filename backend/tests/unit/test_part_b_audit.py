"""Independent audit of Part B (owner request, 2026-11-18), one section per audit area:
signed log, one-hot, ordinal, Isolation Forest, capping, and Part A + B pipelines."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import MIXED_ROLES, context, frame, mixed_frame, run, validated
from tests.unit.test_preprocess_codegen import assert_equivalent, run_generated

CAT = {"c": "categorical"}


def fit_apply(
    df: pd.DataFrame,
    steps: list[Step],
    fit_rows: list[int] | None = None,
    types: dict[str, str] | None = None,
    seed: int = 0,
) -> Any:
    executor = PreprocessExecutor(validated(df, steps, types=types), seed=seed)
    executor.fit(df if fit_rows is None else df[df[ROW_ID].isin(fit_rows)])
    return executor.transform(df)


# ==== 1. Signed log =============================================================================

EXTREMES = [0.0, -0.0, 5e-324, -1e-320, 1e-10, -0.5, 1.0, 1e15, -1e300, 1.7976931348623157e308]


def test_signed_log_forward_values() -> None:
    out, _ = run(frame(x=EXTREMES + [None]), [Step("outliers.signed_log", ("x",))])
    y = out.frame["x"].to_numpy()
    assert y[0] == 0.0 and y[1] == 0.0
    assert y[2] == 5e-324 and y[3] == -1e-320  # log1p(x) == x for tiny x, sign kept
    assert y[-2] == pytest.approx(np.log(np.finfo(float).max), rel=1e-15)  # ~709.78, no overflow
    assert np.isfinite(y[:-1]).all() and np.isnan(y[-1])  # missing stays missing
    assert all(np.sign(a) == np.sign(b) for a, b in zip(y[:-1], EXTREMES, strict=True))


def test_signed_log_round_trip_is_exact_up_to_rounding() -> None:
    """Mathematically invertible; numerically approximate: relative error < 1e-13."""
    x = np.array(EXTREMES)
    y = np.sign(x) * np.log1p(np.abs(x))
    back = np.sign(y) * np.expm1(np.abs(y))
    assert np.isfinite(back).all()
    relative = np.abs(back - x) / np.where(x == 0, 1.0, np.abs(x))
    assert relative.max() < 1e-13
    assert relative[: EXTREMES.index(1.0) + 1].max() == 0.0  # small magnitudes: exact here


def test_signed_log_generated_code_matches_on_extremes(tmp_path: Path) -> None:
    df = frame(x=EXTREMES + [None])
    result = validated(df, [Step("outliers.signed_log", ("x",))])
    platform = PreprocessExecutor(result).fit_transform(df).frame
    assert_equivalent(platform, run_generated(tmp_path, result.steps, df, df))


# ==== 2. One-hot ================================================================================

TRAIN = ["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["rare"]  # 13 training rows


def onehot_df() -> pd.DataFrame:
    return frame(c=TRAIN + ["rare", "NEW", None])  # rows 13-15: rare (seen), unseen, missing


@pytest.mark.parametrize(
    ("max_categories", "policy", "expected"),
    [
        (4, "ignore", {"rare": [0, 0, 0, 1], "NEW": [0, 0, 0, 0]}),  # no other column
        (3, "ignore", {"rare": [0, 0, 0, 1], "NEW": [0, 0, 0, 1]}),  # rare+unseen -> other
    ],
)
def test_onehot_unknown_policy_ignore(
    max_categories: int, policy: str, expected: dict[str, list[int]]
) -> None:
    out = fit_apply(
        onehot_df(),
        [
            Step(
                "encode.onehot",
                ("c",),
                {"max_categories": max_categories, "handle_unknown": policy},
            )
        ],
        list(range(13)),
        CAT,
    )
    block = out.frame[[n for n in out.frame.columns if n.startswith("c=")]].astype(int)
    if max_categories == 4:
        assert list(block.columns) == ["c=a", "c=b", "c=c", "c=rare"]
    else:
        assert list(block.columns) == ["c=a", "c=b", "c=c", "c=__other__"]
    assert block.iloc[13].tolist() == expected["rare"]
    assert block.iloc[14].tolist() == expected["NEW"]
    assert block.iloc[15].tolist() == [0, 0, 0, 0]  # missing: all zeros
    assert out.reports[0].effects["unseen_category_rows"] == {"c": 1}


@pytest.mark.parametrize("max_categories", [3, 4])
def test_onehot_error_policy_raises_for_unseen_even_with_other(max_categories: int) -> None:
    step = Step(
        "encode.onehot", ("c",), {"max_categories": max_categories, "handle_unknown": "error"}
    )
    with pytest.raises(EngineError) as exc:
        fit_apply(onehot_df(), [step], list(range(13)), CAT)
    assert exc.value.code == "UNKNOWN_CATEGORY" and "NEW" not in exc.value.message
    # A rare category that was seen in training is not unknown: it goes to 'other'.
    seen_only = onehot_df().iloc[:14]
    out = fit_apply(seen_only, [step], list(range(13)), CAT)
    assert out.frame.filter(like="c=").iloc[13].sum() == 1


def test_onehot_legitimate_rows_have_exactly_one_true() -> None:
    out = fit_apply(
        onehot_df(), [Step("encode.onehot", ("c",), {"max_categories": 3})], list(range(13)), CAT
    )
    block = out.frame.filter(like="c=").astype(int)
    observed = onehot_df()["c"].notna().to_numpy()
    assert (block.sum(axis=1)[observed] == 1).all()  # every observed value: one column
    assert (block.sum(axis=1)[~observed] == 0).all()  # all-zero only for missing


@pytest.mark.parametrize(
    "labels",
    [
        ["a\nb", "c"],  # control character
        ["a", "x" * 250],  # longer than the column-name limit
        ["__other__", "a", "b"],  # clashes with the 'other' column name
        ["1", 1],  # duplicates as text
        ["   ", "a"],  # blank
    ],
)
def test_onehot_unsafe_labels_fall_back_to_position(labels: list[Any]) -> None:
    df = frame(c=pd.Series(labels * 2, dtype=object))
    out = fit_apply(df, [Step("encode.onehot", ("c",), {"max_categories": 2})], types=CAT)
    names = [n for n in out.frame.columns if n.startswith("c=")]
    assert all(n[2:].isdigit() or n == "c=__other__" for n in names), names
    assert all(not any(ord(ch) < 32 for ch in n) and len(n) <= 200 for n in names)
    assert "POSITIONAL_NAMES_FALLBACK" in [w["code"] for w in out.reports[0].warnings]
    assert out.reports[0].learned["columns"]["c"]["positional_names"] is True


def test_onehot_schema_is_fixed_at_fit() -> None:
    df = onehot_df()
    executor = PreprocessExecutor(
        validated(df, [Step("encode.onehot", ("c",), {"max_categories": 3})], types=CAT)
    )
    executor.fit(df.iloc[:13])
    schemas = {
        tuple(executor.transform(part).frame.columns)
        for part in (df, df.iloc[:1], df.iloc[14:15], df.iloc[:0], df.iloc[::-1])
    }
    assert len(schemas) == 1  # same columns in the same order for any rows, even none


def test_onehot_order_is_by_text_not_by_frequency_or_row_order() -> None:
    a = frame(c=["z", "z", "z", "m", "a"])
    b = frame(c=["a", "m", "z", "z", "z"])
    names = [
        list(fit_apply(d, [Step("encode.onehot", ("c",))], types=CAT).frame.columns) for d in (a, b)
    ]
    assert names[0] == names[1] == [ROW_ID, "c=a", "c=m", "c=z"]


def test_onehot_high_cardinality_is_bounded() -> None:
    df = frame(c=[f"v{i:03d}" for i in range(500)])
    out = fit_apply(df, [Step("encode.onehot", ("c",), {"max_categories": 100})], types=CAT)
    assert len(out.frame.filter(like="c=").columns) == 101  # 100 + __other__
    assert out.reports[0].learned["columns"]["c"]["grouped_as_other"] == 400


# ==== 3. Ordinal ================================================================================

ORDER = {"order": ["low", "mid", "high"]}


def test_ordinal_matrix() -> None:
    df = frame(c=["low", "high", None, "mid", "extreme"])
    train = [0, 1, 2]  # 'mid' is in the order but not in the training rows: fine
    out = fit_apply(
        df, [Step("encode.ordinal", ("c",), {**ORDER, "handle_unknown": "missing"})], train, CAT
    )
    assert out.frame["c"].tolist() == [0, 2, pd.NA, 1, pd.NA]  # missing stays missing
    assert out.reports[0].effects == {"unknown_to_missing_rows": 1}  # 'extreme' only
    with pytest.raises(EngineError) as exc:  # default policy: error at inference
        fit_apply(df, [Step("encode.ordinal", ("c",), ORDER)], train, CAT)
    assert exc.value.code == "UNKNOWN_CATEGORY" and "extreme" not in exc.value.message


def test_ordinal_duplicate_and_near_duplicate_orders() -> None:
    df = frame(c=["low", "Low"])
    dup = validate_steps(
        [Step("encode.ordinal", ("c",), {"order": ["low", "mid", "low"]})], context(df, types=CAT)
    )
    assert [e.code for e in dup.errors] == ["PARAM_INVALID"]
    # Case differs: distinct values. 'Low' is not 'low' and is therefore unknown.
    with pytest.raises(EngineError) as exc:
        fit_apply(df, [Step("encode.ordinal", ("c",), {"order": ["low", "high"]})], types=CAT)
    assert exc.value.code == "ORDER_INCOMPLETE"


def test_ordinal_generated_code_with_unknown_to_missing(tmp_path: Path) -> None:
    df = frame(c=["low", "high", None, "mid", "extreme"])
    result = validated(
        df, [Step("encode.ordinal", ("c",), {**ORDER, "handle_unknown": "missing"})], types=CAT
    )
    fit_df = df.iloc[:3]
    platform = PreprocessExecutor(result).fit(fit_df).transform(df).frame
    assert_equivalent(platform, run_generated(tmp_path, result.steps, fit_df, df))


# ==== 4. Isolation Forest =======================================================================


def iso(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    return frame(a=a, b=a * 2 + rng.normal(0, 0.01, n), c=rng.normal(size=n))


@pytest.mark.parametrize("value", [0.001, 0.01, 0.25, 0.5])
def test_contamination_bounds_accepted_and_share_respected(value: float) -> None:
    df = iso()
    out = fit_apply(
        df, [Step("outliers.isolation_forest_flag", ("a", "c"), {"contamination": value})]
    )
    share = out.frame["isolation_forest_outlier"].mean()
    assert share == pytest.approx(value, abs=max(0.01, value * 0.1))
    assert out.reports[0].learned["contamination"] == value


@pytest.mark.parametrize("value", [0, 0.0005, 0.51, 1, -0.1, "auto", True])
def test_contamination_out_of_bounds_is_refused(value: Any) -> None:
    errors = validate_steps(
        [Step("outliers.isolation_forest_flag", ("a", "c"), {"contamination": value})],
        context(iso()),
    ).errors
    assert [e.code for e in errors] == ["PARAM_INVALID"]


def test_default_contamination_is_one_percent_not_auto() -> None:
    result = validate_steps([Step("outliers.isolation_forest_flag", ("a", "c"))], context(iso()))
    assert result.steps[0].step.params["contamination"] == 0.01


def test_clean_data_is_not_mass_flagged() -> None:
    """scikit-learn's 'auto' threshold flags ~53% of uniform data; the op flags the share."""
    rng = np.random.default_rng(1)
    df = frame(a=rng.uniform(size=2000), b=rng.uniform(size=2000))
    out = fit_apply(df, [Step("outliers.isolation_forest_flag", ("a", "b"))])
    assert out.frame["isolation_forest_outlier"].mean() == pytest.approx(0.01, abs=0.005)


def test_constant_and_identical_columns() -> None:
    identical = frame(a=[1.0] * 40, b=[2.0] * 40)
    out = fit_apply(identical, [Step("outliers.isolation_forest_flag", ("a", "b"))])
    assert out.frame["isolation_forest_outlier"].sum() == 0
    codes = [w["code"] for w in out.reports[0].warnings]
    assert codes.count("CONSTANT_COLUMN_DROPPED") == 2 and "ZERO_SPREAD" in codes
    assert (
        out.reports[0].learned["columns_used"] == []
        and out.reports[0].learned["score_offset"] is None
    )
    mixed = frame(a=[1.0] * 60, b=list(np.arange(60.0)))
    out = fit_apply(
        mixed, [Step("outliers.isolation_forest_flag", ("a", "b"), {"contamination": 0.05})]
    )
    assert out.reports[0].learned["columns_used"] == ["b"]
    assert out.frame["isolation_forest_outlier"].sum() == pytest.approx(3, abs=1)


def test_highly_correlated_columns_are_handled_and_deterministic() -> None:
    df = iso()  # a and b correlate ~1.0
    step = [Step("outliers.isolation_forest_flag", ("a", "b"), {"contamination": 0.05})]
    flags = [
        fit_apply(df, step, seed=9).frame["isolation_forest_outlier"].tolist() for _ in range(3)
    ]
    assert flags[0] == flags[1] == flags[2]
    assert sum(flags[0]) == 15


def test_repeated_fit_and_transform_calls_are_stable() -> None:
    df = iso()
    executor = PreprocessExecutor(
        validated(df, [Step("outliers.isolation_forest_flag", ("a", "c"))]), seed=3
    )
    first = executor.fit(df.iloc[:200]).transform(df).frame
    again = executor.transform(df).frame  # transform twice
    refit = executor.fit(df.iloc[:200]).transform(df).frame  # fit twice on the same rows
    pd.testing.assert_frame_equal(first, again)
    pd.testing.assert_frame_equal(first, refit)


def test_scoring_runs_under_the_working_memory_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    import sklearn
    from sklearn.ensemble import _iforest

    seen: list[int] = []
    original = _iforest.get_chunk_n_rows

    def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(sklearn.get_config()["working_memory"])
        return original(*args, **kwargs)

    monkeypatch.setattr(_iforest, "get_chunk_n_rows", spy)
    fit_apply(iso(), [Step("outliers.isolation_forest_flag", ("a", "c"))])
    assert seen and all(v == rt.KNN_WORKING_MEMORY_MB for v in seen)


def test_rows_with_missing_values_and_training_only_threshold() -> None:
    df = iso()
    df.loc[[10, 250], "a"] = np.nan
    executor = PreprocessExecutor(
        validated(df, [Step("outliers.isolation_forest_flag", ("a", "c"))])
    )
    executor.fit(df.iloc[:200])
    out = executor.transform(df)
    assert out.frame.loc[[10, 250], "isolation_forest_outlier"].tolist() == [False, False]
    assert out.reports[0].effects["rows_not_scored"] == 2
    assert out.reports[0].learned["n_complete_fit_rows"] == 199  # row 10 excluded; 250 unused


# ==== 5. Capping ================================================================================


def test_integer_rounding_inward_and_outward() -> None:
    assert rt.integer_bounds([1.2, 5.8]) == (2, 5)
    assert rt.integer_bounds([-5.8, -1.2]) == (-5, -2)
    assert rt.integer_bounds([1.07, 1.93]) == (1, 2)  # no whole number inside: outward
    assert rt.integer_bounds([3.0, 3.0]) == (3, 3)
    df = frame(n=pd.array([1, 1, 2, 2, 1, 2, 1, 2, 1, 2, None], dtype="Int64"))
    out, _ = run(
        df, [Step("outliers.cap", ("n",), {"lower_percentile": 5, "upper_percentile": 95})]
    )
    assert out.frame["n"].dtype == "Int64" and out.frame["n"].tolist()[:10] == df["n"].tolist()[:10]
    assert out.frame["n"].isna().tolist()[-1]  # missing remains missing


@pytest.mark.parametrize("method", ["iqr", "zscore", "mad", "percentile"])
def test_cap_constant_and_zero_mad(method: str) -> None:
    for values in ([7.0] * 12, [0.0] * 9 + [1.0, 2.0, 90.0]):
        out, _ = run(frame(x=values + [None]), [Step("outliers.cap", ("x",), {"method": method})])
        if out.reports[0].learned["columns"]["x"]["lower"] is None:
            assert out.frame["x"].tolist()[:-1] == values  # no bounds: unchanged
            assert "ZERO_SPREAD" in [w["code"] for w in out.reports[0].warnings]
        assert np.isnan(out.frame["x"].iloc[-1])


def test_cap_minimum_sample_size() -> None:
    ok, _ = run(
        frame(x=[1.0, 2, 3, 4, 5, 6, 7, 100]), [Step("outliers.cap", ("x",), {"method": "iqr"})]
    )
    assert ok.frame["x"].iloc[-1] < 100
    with pytest.raises(EngineError) as exc:
        run(frame(x=[1.0, 2, 3, 4, 5, 6, 100]), [Step("outliers.cap", ("x",))])
    assert exc.value.code == "FIT_TOO_FEW_VALUES"


def test_cap_extreme_values() -> None:
    huge = frame(x=[1.5e308, -1.5e308] * 5 + [1.0])
    out, _ = run(huge, [Step("outliers.cap", ("x",))])  # percentile bounds are finite
    assert np.isfinite(out.frame["x"]).all()
    for method in ("iqr", "zscore", "mad"):  # their bounds overflow: refused, not ±inf
        with pytest.raises(EngineError) as exc:
            run(huge, [Step("outliers.cap", ("x",), {"method": method})])
        assert exc.value.code == "NUMERIC_OVERFLOW"


def test_cap_skewed_distribution_warns_and_caps_mostly_the_tail() -> None:
    rng = np.random.default_rng(4)
    df = frame(x=rng.lognormal(0, 1.5, 500))
    out, _ = run(df, [Step("outliers.cap", ("x",), {"method": "iqr"})])
    effects = out.reports[0].effects["columns"]["x"]
    assert "SKEWED_COLUMN" in [w["code"] for w in out.reports[0].warnings]
    assert effects["above"] > 10 * max(effects["below"], 1) or effects["below"] == 0


def test_lower_bound_never_exceeds_upper_bound() -> None:
    rng = np.random.default_rng(0)
    for _ in range(200):
        values = rng.choice(
            [rng.normal(size=12), rng.integers(0, 3, 12).astype(float), rng.lognormal(0, 3, 12)]
        )
        for method, params in (
            ("iqr", {"iqr_multiplier": 0.5}),
            ("zscore", {"z_threshold": 1}),
            ("mad", {"mad_threshold": 1}),
            ("percentile", {"lower_percentile": 49, "upper_percentile": 51}),
        ):
            state = rt.fit_bounds(frame(x=values), ["x"], method, params, 8)
            bound = state["bounds"][0]
            if bound is not None:
                assert bound[0] < bound[1]
                low, high = rt.integer_bounds(bound)
                assert low <= high


# ==== 6. Part A + Part B pipelines ==============================================================

SAME_COLUMN = [
    Step("missing.indicator", ("x1",)),
    Step("impute.median", ("x1",)),
    Step("outliers.flag", ("x1",), {"method": "mad"}),
    Step("outliers.cap", ("x1",), {"method": "iqr"}),
    Step("outliers.signed_log", ("x1",)),
    Step("scale.robust", ("x1",)),
]
DIFFERENT_COLUMNS = [
    Step("impute.knn", ("x2",)),
    Step("impute.mode", ("city",)),
    Step("encode.onehot", ("city",)),
    Step("impute.constant", ("age",), {"fill_value": 0}),
    Step("outliers.cap", ("age",), {"method": "percentile"}),
    Step("scale.minmax", ("age",)),
    Step("outliers.isolation_forest_flag", ("x2", "age"), {"contamination": 0.05}),
]


@pytest.mark.parametrize("steps", [SAME_COLUMN, DIFFERENT_COLUMNS], ids=["same", "different"])
def test_pipeline_integrity(steps: list[Step]) -> None:
    df = mixed_frame(n=150, seed=6)
    original = df.copy(deep=True)
    result = validated(df, steps, MIXED_ROLES)
    out = PreprocessExecutor(result, seed=5).fit(df.iloc[:100]).transform(df)
    pd.testing.assert_frame_equal(df, original)  # input never mutated
    assert out.frame[ROW_ID].tolist() == df[ROW_ID].tolist()
    assert [r.rows_removed for r in out.reports] == [0] * len(steps)
    untouched = [c for c in df.columns if c not in {c for s in steps for c in s.columns}]
    for column in untouched:
        pd.testing.assert_series_equal(out.frame[column], df[column])
    if steps is SAME_COLUMN:
        assert list(out.frame.columns) == [*df.columns, "x1__missing", "x1__outlier"]
        assert out.frame["x1"].notna().all() and out.frame["x1"].dtype == "float64"
    else:
        cities = [c for c in out.frame.columns if c.startswith("city=")]
        position = list(out.frame.columns).index(cities[0])
        assert list(out.frame.columns)[position - 1] == "x2"  # one-hot keeps the position
        assert out.frame.columns[-1] == "isolation_forest_outlier"


@pytest.mark.parametrize("steps", [SAME_COLUMN, DIFFERENT_COLUMNS], ids=["same", "different"])
def test_pipeline_learns_nothing_from_test_rows(steps: list[Step]) -> None:
    df = mixed_frame(n=150, seed=6)
    perturbed = df.copy()
    perturbed.loc[100:, ["x1", "x2"]] = 1e6
    perturbed.loc[100:, "age"] = 9999
    perturbed.loc[100:, "city"] = "test-only"
    learned, train_rows = [], []
    for data in (df, perturbed):
        out = (
            PreprocessExecutor(validated(data, steps, MIXED_ROLES), seed=5)
            .fit(data.iloc[:100])
            .transform(data)
        )
        learned.append([r.learned for r in out.reports])
        train_rows.append(out.frame.iloc[:100])
    assert learned[0] == learned[1]
    pd.testing.assert_frame_equal(train_rows[0], train_rows[1])


@pytest.mark.parametrize("steps", [SAME_COLUMN, DIFFERENT_COLUMNS], ids=["same", "different"])
def test_pipeline_generated_code_matches(steps: list[Step], tmp_path: Path) -> None:
    df = mixed_frame(n=150, seed=6)
    result = validated(df, steps, MIXED_ROLES)
    fit_df = df.iloc[:100]
    platform = PreprocessExecutor(result, seed=5).fit(fit_df).transform(df).frame
    assert_equivalent(platform, run_generated(tmp_path, result.steps, fit_df, df, seed=5))


def _record_chunking(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Record (working_memory in effect, rows per chunk) for every scoring chunk plan."""
    import sklearn
    from sklearn.ensemble import _iforest

    calls: list[tuple[int, int]] = []
    original = _iforest.get_chunk_n_rows

    def spy(*args: Any, **kwargs: Any) -> Any:
        rows = original(*args, **kwargs)
        calls.append((sklearn.get_config()["working_memory"], int(rows)))
        return rows

    monkeypatch.setattr(_iforest, "get_chunk_n_rows", spy)
    return calls


def test_memory_cap_covers_fit_and_predict_separately(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression (Part B audit): fit() scores the fitting rows to set the threshold; that
    scoring ran under scikit-learn's default 1024 MB budget before the fix."""
    calls = _record_chunking(monkeypatch)
    df = iso()
    executor = PreprocessExecutor(
        validated(df, [Step("outliers.isolation_forest_flag", ("a", "c"))])
    )
    executor.fit(df.iloc[:200])
    fit_calls = list(calls)
    executor.transform(df)
    predict_calls = calls[len(fit_calls) :]
    assert fit_calls and all(budget == rt.KNN_WORKING_MEMORY_MB for budget, _ in fit_calls)
    assert predict_calls and all(budget == rt.KNN_WORKING_MEMORY_MB for budget, _ in predict_calls)


def test_chunked_scoring_does_not_change_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tiny budget forces many chunks; flags, threshold and KNN estimates must not change."""
    rng = np.random.default_rng(12)
    n = 20_000
    df = frame(a=rng.normal(size=n), c=rng.normal(size=n), x=rng.normal(size=n))
    df.loc[rng.random(n) < 0.05, "x"] = np.nan
    steps = [
        Step("outliers.isolation_forest_flag", ("a", "c"), {"contamination": 0.02}),
        Step("impute.knn", ("x",), {"predictors": ["a", "c"]}),
    ]
    results = []
    # scikit-learn sizes Isolation Forest chunks at 16 bytes per feature per row, so only a
    # tiny budget (0.01 MB = ~330 rows per chunk here) really splits 20,000 rows.
    for budget in (0.01, 1024):
        monkeypatch.setattr(rt, "KNN_WORKING_MEMORY_MB", budget)
        calls = _record_chunking(monkeypatch)
        executor = PreprocessExecutor(validated(df, steps), seed=21).fit(df.iloc[:15_000])
        out = executor.transform(df)
        results.append((out.frame, out.reports[0].learned["score_offset"]))
        if budget < 1:
            assert calls and all(rows < n // 10 for _, rows in calls)  # many chunks were used
    (frame_small, offset_small), (frame_large, offset_large) = results
    assert offset_small == offset_large
    pd.testing.assert_frame_equal(frame_small, frame_large, check_exact=True)

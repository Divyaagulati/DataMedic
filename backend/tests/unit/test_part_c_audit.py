"""Part C independent audit: regression tests for every defect found, and the adversarial
cases the audit checked (grouped by audit area)."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from sklearn.preprocessing import QuantileTransformer

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from tests.unit.preprocess_helpers import frame, run, validated

# ---- 1. Distribution transforms -------------------------------------------------------------------


@pytest.mark.parametrize("dist", ["uniform", "normal"])
def test_quantile_ties_at_the_minimum_get_their_middle_rank(dist: str) -> None:
    """Regression: scikit-learn forces values equal to the fitting minimum to the lower end
    (0, or -5.2 with a normal output). In a column that is 40% zeros every zero became -5.2
    standard deviations. Ties must get their middle rank."""
    values = np.array([0.0] * 40 + list(range(1, 61)), dtype="float64")
    out, _ = run(
        frame(x=values), [Step("transform.quantile", ("x",), {"output_distribution": dist})]
    )
    zeros = out.frame["x"].to_numpy()[:40]
    assert len(set(zeros.tolist())) == 1
    # Middle rank of the lowest 40%: about 0.2 on the uniform scale.
    u = zeros[0] if dist == "uniform" else stats.norm.cdf(zeros[0])
    assert 0.15 < u < 0.25
    reference = QuantileTransformer(n_quantiles=100, output_distribution=dist).fit(
        values.reshape(-1, 1)
    )
    sk = reference.transform(values.reshape(-1, 1))[:40, 0]
    assert sk[0] in (0.0, pytest.approx(-5.1993, abs=1e-3))  # the behaviour we correct


def test_quantile_ties_at_the_maximum_and_single_extremes() -> None:
    values = np.array(list(range(60)) + [100.0] * 40, dtype="float64")
    out, _ = run(frame(x=values), [Step("transform.quantile", ("x",))])
    y = out.frame["x"].to_numpy()
    assert y[60:].tolist() == [pytest.approx(0.8, abs=0.01)] * 40
    assert y[0] == 0.0  # a single minimum keeps the documented end value


def test_quantile_still_matches_scikit_learn_away_from_tied_extremes() -> None:
    rng = np.random.default_rng(3)
    values = rng.lognormal(0, 2, 400)
    for dist in ("uniform", "normal"):
        out, _ = run(
            frame(x=values), [Step("transform.quantile", ("x",), {"output_distribution": dist})]
        )
        reference = QuantileTransformer(n_quantiles=400, output_distribution=dist).fit(
            values.reshape(-1, 1)
        )
        assert np.allclose(
            out.frame["x"].to_numpy(), reference.transform(values.reshape(-1, 1))[:, 0]
        )


def test_yeo_johnson_reports_extrapolation_outside_the_fitting_range() -> None:
    """Regression: values far outside the fitting range produced extreme outputs (-2.3e11)
    with nothing in the report."""
    df = frame(x=list(np.linspace(1, 20, 12)) + [-1e9, 1e9, 5.0])
    out, _ = run(df, [Step("transform.yeo_johnson", ("x",))], fit_rows=df.index[:12])
    report = out.reports[0]
    assert report.effects["outside_fit_range_rows"] == {"x": 2}
    assert "OUTSIDE_FIT_RANGE" in [w["code"] for w in report.warnings]
    assert "fit_range" not in str(report.learned)  # fitting values are never reported


@pytest.mark.parametrize(
    "op", ["transform.log", "transform.log1p", "transform.yeo_johnson", "transform.quantile"]
)
@pytest.mark.parametrize(
    "dtype,values",
    [
        ("Float64", [pd.NA, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ("Int64", [None, 1, 2, 3, 4, 5, 6, 7, 8, 9]),
        ("float64", [np.nan, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        ("object", [None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
    ],
)
def test_every_missing_marker_and_nullable_dtype_stays_missing(
    op: str, dtype: str, values: list
) -> None:  # type: ignore[type-arg]
    df = frame(x=pd.Series(values, dtype=dtype), other=list(range(10)))
    df.index = pd.Index(range(50, 60))
    result = validated(df, [Step(op, ("x",))], types={"x": "numeric"})
    out = PreprocessExecutor(result).fit(df).transform(df)
    assert out.frame["x"].isna().tolist() == [True] + [False] * 9
    assert out.frame.index.equals(df.index) and list(out.frame.columns) == list(df.columns)
    assert str(out.frame["x"].dtype) == "float64"


@pytest.mark.parametrize("bad", [0.0, -0.0, -5e-324])
def test_log_refuses_zero_negative_zero_and_negative_subnormals_at_inference(bad: float) -> None:
    df = frame(x=[1.0, 2.0, 3.0, bad])
    ex = PreprocessExecutor(validated(df, [Step("transform.log", ("x",))])).fit(df.iloc[:3])
    with pytest.raises(EngineError) as exc:
        ex.transform(df)
    assert exc.value.code == "VALUE_OUT_OF_DOMAIN"


def test_log_family_handles_subnormals_and_the_float_limit_without_overflow() -> None:
    values = [5e-324, 1e-310, 1.0, 1.7976931348623157e308] + [2.0] * 6
    for op in ("transform.log", "transform.log1p"):
        out, _ = run(frame(x=values), [Step(op, ("x",))])
        assert np.isfinite(out.frame["x"]).all()
    out, _ = run(frame(x=values), [Step("transform.log", ("x",))])
    assert out.frame["x"].iloc[0] == pytest.approx(-744.44, abs=0.01)


def test_yeo_johnson_special_lambdas_are_continuous() -> None:
    """lambda = 0 and 2 use the log branches; values next to them must agree to rounding."""
    x = np.array([-50.0, -1.0, -1e-12, 0.0, 1e-12, 1.0, 50.0])
    for special in (0.0, 2.0):
        at = rt.yeo_johnson(x, special)
        for delta in (1e-9, -1e-9):
            assert np.allclose(rt.yeo_johnson(x, special + delta), at, rtol=1e-7, atol=1e-12)
    assert np.array_equal(rt.yeo_johnson(x, 1.0), x)  # exact identity


# ---- 2. Feature selection -------------------------------------------------------------------------

from datamedic.engine.preprocess.catalog import Limits, get_op  # noqa: E402
from datamedic.engine.preprocess.split import make_splits  # noqa: E402
from datamedic.engine.preprocess.validate import validate_steps  # noqa: E402
from tests.unit.preprocess_helpers import context  # noqa: E402

ROLES = {"target": ["y"]}


def sel_data(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    return frame(a=rng.normal(size=n) + y, b=rng.normal(size=n), y=y)


@pytest.mark.parametrize("scale", [1e300, 1e-300])
def test_correlation_at_extreme_magnitudes_is_assessed(scale: float) -> None:
    """Regression: standardising overflowed (1e300) or underflowed (1e-300), so perfectly
    correlated pairs were reported as not assessed."""
    df = sel_data()
    df["h"], df["h2"], df["neg"] = df["a"] * scale, df["a"] * scale * 0.5, -df["a"] * scale
    out, _ = run(df, [Step("select.correlation", ("h", "h2", "neg", "b"))], roles=ROLES)
    learned = out.reports[0].learned
    assert learned["pairs_not_assessed"] == 0
    values = sorted(round(p["correlation"], 9) for p in learned["pairs"])
    assert values == [-1.0, -1.0, 1.0]
    assert "LIBRARY_WARNING" not in [w["code"] for w in out.reports[0].warnings]


@pytest.mark.parametrize(
    "op,params",
    [
        ("select.mutual_information", {}),
        ("select.univariate", {"test": "anova_f"}),
        ("select.univariate", {"test": "kruskal"}),
    ],
)
def test_supervised_scores_at_extreme_magnitudes(op: str, params: dict) -> None:  # type: ignore[type-arg]
    """Regression: scikit-learn's internal scaling overflowed (LIBRARY_WARNING) and ANOVA was
    not assessed for values near 1e300; the scores are scale-free and must not change."""
    df = sel_data(seed=1)
    df["h"] = df["a"] * 1e300
    big, _ = run(df, [Step(op, ("h", "b"), params)], roles=ROLES, task="binary", seed=3)
    df["h"] = df["a"]
    small, _ = run(df, [Step(op, ("h", "b"), params)], roles=ROLES, task="binary", seed=3)
    assert "LIBRARY_WARNING" not in [w["code"] for w in big.reports[0].warnings]
    assert big.reports[0].learned["columns"]["h"]["rank"] == 1
    key = "mutual_information" if op.endswith("information") else "p_value"
    assert big.reports[0].learned["columns"]["h"][key] == pytest.approx(
        small.reports[0].learned["columns"]["h"][key], rel=1e-6
    )


def test_selection_that_removes_every_selected_column_warns() -> None:
    steps = [Step("select.variance", ("a", "b"), {"threshold": 1e9, "action": "drop"})]
    out, _ = run(sel_data(), steps, roles=ROLES)
    assert "ALL_SELECTED_REMOVED" in [w["code"] for w in out.reports[0].warnings]


@pytest.mark.parametrize(
    "op,params", [("select.mutual_information", {}), ("select.univariate", {"test": "anova_f"})]
)
def test_tiny_classes_are_flagged_as_unreliable(op: str, params: dict) -> None:  # type: ignore[type-arg]
    df = sel_data()
    df["y"] = 0
    df.loc[df.index[:2], "y"] = 1
    out, _ = run(df, [Step(op, ("a", "b"), params)], roles=ROLES, task="binary")
    assert "SMALL_CLASS_COUNT" in [w["code"] for w in out.reports[0].warnings]


def test_a_copy_of_the_target_is_flagged_as_possible_leakage() -> None:
    df = sel_data()
    df["copy"] = df["y"].astype(float)
    steps = [Step("select.mutual_information", ("copy", "a", "b"))]
    out, _ = run(df, steps, roles=ROLES, task="binary")
    flagged = [
        w["column"] for w in out.reports[0].warnings if w["code"] == "POSSIBLE_TARGET_LEAKAGE"
    ]
    assert flagged == ["copy"]


def test_duplicate_column_names_and_internal_columns_are_refused() -> None:
    df = sel_data()
    result = validated(df, [Step("select.variance", ("b",))], ROLES)
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(result).fit(pd.concat([df, df[["a"]]], axis=1))
    assert exc.value.code == "OP_INVALID"
    df["_dm_sample_weight"] = 1.0
    steps = [Step("select.variance", ("_dm_sample_weight",))]
    codes = [e.code for e in validate_steps(steps, context(df, ROLES)).errors]
    assert codes == ["RESERVED_COLUMN"]


def test_selection_instability_between_folds_is_real_and_documented() -> None:
    """With weak signals, different training folds keep different columns. The catalogue
    says so for every selector (before the audit only correlation mentioned it)."""
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, n)
    cols = {f"f{i}": rng.normal(size=n) + (0.15 if i < 4 else 0) * y for i in range(10)}
    df = frame(**cols, y=y)
    step = Step("select.univariate", tuple(cols), {"test": "anova_f", "k": 4, "action": "drop"})
    v = validated(df, [step], ROLES, task="binary")
    kept = set()
    for fold in make_splits(df, method="stratified", target="y", n_splits=5, seed=0).folds:
        train = df[df[ROW_ID].isin(fold.train_ids)]
        decisions = (
            PreprocessExecutor(v).fit(train).transform(train).reports[0].learned["decisions"]
        )
        kept.add(tuple(sorted(c for c, d in decisions.items() if d == "kept")))
    assert len(kept) > 1
    for op in ("select.variance", "select.mutual_information", "select.univariate"):
        spec = get_op(op)
        assert spec is not None and any("unstable" in text for text in spec.limitations)


# ---- 3. Class imbalance ---------------------------------------------------------------------------


def imb_data(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    return frame(
        pid=[f"p{i // 3}" for i in range(n)],
        a=rng.normal(size=n),
        b=rng.normal(size=n),
        y=np.array([0] * (n - n // 4) + [1] * (n // 4)),
    )


@pytest.mark.parametrize("role", ["bio.patient_id", "id", "group", "time"])
def test_smote_on_dependent_rows_requires_explicit_acknowledgement(role: str) -> None:
    """Audit decision: within a training fold SMOTE interpolates between rows of different
    patients, groups or times, so synthetic rows are not independent observations. This is
    no longer only a validation warning: the user must opt in."""
    df = imb_data()
    if role == "time":
        df["pid"] = np.arange(len(df))
    roles = {"target": ["y"], role: ["pid"]}
    refused = validate_steps([Step("imbalance.smote", ())], context(df, roles, task="binary"))
    assert [e.code for e in refused.errors] == ["DEPENDENT_ROWS_NOT_ACKNOWLEDGED"]
    step = Step("imbalance.smote", (), {"acknowledge_dependent_rows": True})
    accepted = validate_steps([step], context(df, roles, task="binary"))
    assert accepted.valid and "GROUPED_OR_TEMPORAL_DATA" in [w.code for w in accepted.warnings]
    training = PreprocessExecutor(accepted).fit_resample(df)
    assert "GROUPED_OR_TEMPORAL_DATA" in [w["code"] for w in training.reports[0].warnings]


def test_smote_synthetic_rows_do_blend_different_patients() -> None:
    """Documents the scientific implication: synthetic rows lie between rows of different
    patients and carry no patient identifier."""
    df = imb_data()
    roles = {"target": ["y"], "bio.patient_id": ["pid"]}
    params = {"acknowledge_dependent_rows": True, "k_neighbors": 3}
    result = validated(df, [Step("imbalance.smote", (), params)], roles, task="binary")
    out = PreprocessExecutor(result, seed=0).fit_resample(df).frame
    synthetic = out[out[ROW_ID] < 0]
    assert synthetic["pid"].isna().all()
    minority = df[df["y"] == 1]
    blended = 0
    for _, row in synthetic.iterrows():
        distance = ((minority[["a", "b"]] - row[["a", "b"]].astype(float)) ** 2).sum(axis=1)
        nearest = minority.loc[distance.nsmallest(2).index, "pid"]
        blended += int(nearest.nunique() > 1)
    assert blended > 0


def test_smote_without_dependent_roles_needs_no_acknowledgement() -> None:
    df = imb_data().drop(columns=["pid"])
    assert validate_steps([Step("imbalance.smote", ())], context(df, ROLES, task="binary")).valid


def test_negative_real_row_ids_are_refused_before_resampling() -> None:
    """Regression: negative real ids failed late with a misleading 'not unique' error."""
    df = imb_data()
    df[ROW_ID] = -np.arange(1, len(df) + 1)
    roles = {"target": ["y"], "id": ["pid"]}
    result = validated(df, [Step("imbalance.random_oversample", ())], roles, task="binary")
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(result).fit_resample(df)
    assert exc.value.code == "OP_INVALID" and "non-negative" in exc.value.message


@pytest.mark.parametrize("op", ["imbalance.random_oversample", "imbalance.smote"])
def test_output_cell_cap_is_checked_before_resampling(
    op: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: SMOTE at its row and feature caps needed about 2.4 GB (measured 34 bytes
    per output cell), above the 2,000 MB worker cap. The cell cap refuses it up front."""
    import imblearn.over_sampling as ov

    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("the sampler must not run")

    monkeypatch.setattr(ov.SMOTE, "fit_resample", never)
    monkeypatch.setattr(ov.RandomOverSampler, "fit_resample", never)
    monkeypatch.setattr(Limits, "RESAMPLE_MAX_OUTPUT_CELLS", 100)
    df = imb_data().drop(columns=["pid"])
    result = validated(df, [Step(op, ())], ROLES, task="binary")
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(result).fit_resample(df)
    assert exc.value.code == "OP_LIMIT_EXCEEDED" and "cells" in exc.value.message


def test_smote_neighbour_chunking_does_not_change_results(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(4)
    n = 3000
    y = (rng.random(n) < 0.2).astype(int)
    df = frame(a=rng.normal(size=n), b=rng.normal(size=n), y=y)
    result = validated(df, [Step("imbalance.smote", ())], ROLES, task="binary")
    outs = []
    for budget in (1, 1024):
        monkeypatch.setattr(rt, "KNN_WORKING_MEMORY_MB", budget)
        outs.append(PreprocessExecutor(result, seed=1).fit_resample(df).frame)
    pd.testing.assert_frame_equal(outs[0], outs[1], check_exact=True)


# ---- 4. Frozen folds ------------------------------------------------------------------------------


def split_data(n: int = 40) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ROW_ID: np.arange(n),
            "y": (np.arange(n) % 4 == 0).astype(int),
            "g": np.arange(n) // 4,
            "t": np.arange(n) // 2,
        }
    )


def test_folds_report_sizes_and_class_counts_and_flag_missing_classes() -> None:
    """Regression: a class living in one group was absent from most test folds without any
    report or warning."""
    df = split_data().assign(y=[1] * 4 + [0] * 36)
    splits = make_splits(df, method="stratified_group", target="y", group="g", n_splits=3, seed=0)
    assert splits.classes == [0, 1]
    assert sum(s["n_test"] for s in splits.fold_stats) == len(df)
    assert sum(s["test_class_counts"][1] for s in splits.fold_stats) == 4
    assert "CLASS_MISSING_IN_FOLD" in [w["code"] for w in splits.warnings]
    ok = make_splits(split_data(), method="stratified", target="y", n_splits=5)
    assert all(min(s["test_class_counts"]) > 0 for s in ok.fold_stats)
    assert "CLASS_MISSING_IN_FOLD" not in [w["code"] for w in ok.warnings]


def test_very_uneven_group_folds_are_reported() -> None:
    df = split_data().assign(g=[0] * 30 + list(range(1, 11)))
    splits = make_splits(df, method="group", group="g", n_splits=3)
    assert "UNEVEN_FOLDS" in [w["code"] for w in splits.warnings]


@pytest.mark.parametrize(
    "kwargs,code",
    [
        ({"method": "time", "time": "t", "n_splits": 3}, "SPLIT_TIME_INVALID"),
        ({"method": "kfold", "n_splits": 3.0}, "SPLIT_INVALID"),
        ({"method": "stratified", "target": "z", "n_splits": 3}, "SPLIT_INVALID"),
    ],
)
def test_invalid_split_inputs_fail_with_sanitised_codes(kwargs: dict, code: str) -> None:  # type: ignore[type-arg]
    """Regression: mixed time zones and a float fold count raised raw library errors; a
    numeric target gave a misleading 'class too small' error."""
    df = split_data()
    df["t"] = [pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-01-02")] * 20
    df["z"] = np.linspace(0, 1, 40) + 0.001
    with pytest.raises(EngineError) as exc:
        make_splits(df, **kwargs)
    assert exc.value.code == code


def test_unexpected_split_errors_carry_only_the_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from datamedic.engine.preprocess import split as sp

    class Boom:
        def __init__(self, *a: object, **k: object) -> None:
            raise ValueError("secret value 987654")

    monkeypatch.setattr(sp, "KFold", Boom)
    with pytest.raises(EngineError) as exc:
        make_splits(split_data(), method="kfold", n_splits=3)
    assert exc.value.code == "SPLIT_FAILED" and exc.value.details == {"error_type": "ValueError"}
    assert "987654" not in exc.value.message


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "kfold"},
        {"method": "stratified", "target": "y"},
        {"method": "group", "group": "g"},
        {"method": "stratified_group", "target": "y", "group": "g"},
        {"method": "time", "time": "t"},
    ],
)
def test_no_row_is_omitted_or_duplicated_and_no_group_or_future_leaks(kwargs: dict) -> None:  # type: ignore[type-arg]
    df = split_data(80).assign(g=np.r_[np.repeat(np.arange(10), 6), np.arange(10, 30)])
    splits = make_splits(df, n_splits=4, seed=7, **kwargs)
    for fold in splits.folds:
        train, test = splits.frames(df, fold)
        assert len(set(fold.train_ids)) == len(fold.train_ids)
        assert len(set(fold.test_ids)) == len(fold.test_ids)
        assert not set(fold.train_ids) & set(fold.test_ids)
        if "group" in kwargs:
            assert not set(train["g"]) & set(test["g"])
        if kwargs["method"] == "time":
            assert train["t"].max() < test["t"].min()
        else:
            assert len(train) + len(test) == len(df)
    if kwargs["method"] != "time":
        tested = np.sort(np.concatenate([f.test_ids for f in splits.folds]))
        assert tested.tolist() == sorted(df[ROW_ID].tolist())


# ---- 5. Cross-cutting -----------------------------------------------------------------------------


def test_sample_weight_has_lineage_and_is_never_a_model_feature() -> None:
    """Regression: _dm_sample_weight had no lineage entry, and nothing told a consumer that
    it (or _dm_row_id, the target or identifiers) must not be used as a feature."""
    df = imb_data()
    roles = {"target": ["y"], "id": ["pid"]}
    result = validated(df, [Step("imbalance.class_weight", ())], roles, task="binary")
    ex = PreprocessExecutor(result)
    training = ex.fit_resample(df)
    entry = next(e for e in training.lineage if e["column"] == "_dm_sample_weight")
    assert entry["origin"] == "added" and entry["derived_from"] == ["y"]
    assert entry["feature"] is False
    assert ex.feature_columns(training.frame) == ["a", "b"]
    assert all(e["feature"] for e in training.lineage if not e["column"].startswith("_dm_"))


def test_validation_does_not_falsely_refuse_smote_after_a_drop_mode_selector() -> None:
    """Regression: a drop-mode selector hid its possibly-removed columns from SMOTE's
    validation, which then reported NO_FEATURES although k columns are always kept."""
    df = imb_data().drop(columns=["pid"])
    steps = [
        Step("select.mutual_information", ("a", "b"), {"k": 1, "action": "drop"}),
        Step("imbalance.smote", ()),
    ]
    result = validated(df, steps, ROLES, task="binary")
    training = PreprocessExecutor(result).fit_resample(df).frame
    assert len(training) > len(df)
    df["c"] = pd.Series(["u", "v"] * 30, dtype=object)
    steps[0] = Step("select.mutual_information", ("a", "b", "c"), {"k": 1, "action": "drop"})
    codes = [e.code for e in validate_steps(steps, context(df, ROLES, task="binary")).errors]
    assert codes == ["FEATURE_NOT_NUMERIC"]  # a possibly-kept categorical column is still checked


def test_commit_path_refuses_model_stage_steps() -> None:
    """Formerly pinned as the open Part E gap ("fit_transform accepts model-stage steps").
    Closed in Part D: the commit path is now ``commit_transform``, which refuses model-stage
    steps before reading data. ``fit_transform`` stays a general fit-and-apply helper (used
    by code-generation equivalence tests) and is documented as not being a commit path."""
    df = imb_data()
    steps = [
        Step("select.variance", ("a", "b"), {"action": "drop", "threshold": 1e9}),
        Step("imbalance.random_oversample", ()),
    ]
    result = validated(df, steps, {"target": ["y"], "id": ["pid"]}, task="binary")
    assert all(v.spec.stage == "model" for v in result.steps)
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(result).commit_transform(df)
    assert exc.value.code == "MODEL_STAGE_NOT_COMMITTABLE"
    assert exc.value.details["ops"] == ["imbalance.random_oversample", "select.variance"]
    out = PreprocessExecutor(result).fit_transform(df)  # evaluation helper, not a commit
    assert len(out.frame) == len(df)  # no resampled row can reach this path either


def test_combined_pipeline_never_mutates_inputs_and_is_repeatable() -> None:
    df = imb_data().drop(columns=["pid"])
    df.loc[df.index[::7], "a"] = np.nan
    steps = [
        Step("impute.median", ("a",)),
        Step("outliers.cap", ("b",), {"method": "iqr"}),
        Step("transform.yeo_johnson", ("a", "b")),
        Step("select.variance", ("a", "b"), {"action": "drop"}),
        Step("imbalance.smote", (), {"k_neighbors": 3}),
    ]
    result = validated(df, steps, ROLES, task="binary")
    before = df.copy(deep=True)
    runs = []
    for _ in range(2):
        ex = PreprocessExecutor(result, seed=9)
        train, test = df[df[ROW_ID] % 4 != 0], df[df[ROW_ID] % 4 == 0]
        runs.append((ex.fit_resample(train).frame, ex.transform(test).frame))
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(runs[0][0], runs[1][0])
    pd.testing.assert_frame_equal(runs[0][1], runs[1][1])


# ---- 6. Resources ---------------------------------------------------------------------------------

from datamedic.engine.preprocess.report import _changed_count  # noqa: E402

CHANGE_CASES = [
    (pd.Series([1, 2, 3]), pd.Series([1.0, 2.5, 3.0])),
    (
        pd.Series(pd.array([2**60, 2**60 + 1, None], dtype="Int64")),
        pd.Series(pd.array([2**60, 2**60, 5], dtype="Int64")),
    ),
    (pd.Series([True, False, True]), pd.Series([True, True, True])),
    (pd.Series([True, False]), pd.Series([1, 0])),
    (pd.Series([1.0, np.nan, 3.0]), pd.Series([1.0, 2.0, np.nan])),
    (pd.Series(["a", "b"]), pd.Series(["a", "c"])),
    (pd.Series(pd.array([1.5, None], dtype="Float64")), pd.Series([1.5, 9.0])),
]


@pytest.mark.parametrize("before,after", CHANGE_CASES)
def test_vectorised_change_count_equals_the_object_comparison(
    before: pd.Series, after: pd.Series
) -> None:
    """The per-cell Python comparison dominated reports at scale (1M rows: about 14 of 15 s);
    the vectorised path must count exactly the same cells."""
    present = before.notna().to_numpy() & after.notna().to_numpy()
    pairs = zip(before.astype(object)[present], after.astype(object)[present], strict=True)
    slow = sum(x != y for x, y in pairs) + int((before.notna() & after.isna()).sum())
    assert _changed_count(before, after) == slow


def test_correlation_cell_cap_is_checked_before_any_computation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def never(*args: object, **kwargs: object) -> None:
        raise AssertionError("must not compute")

    monkeypatch.setattr(rt, "fit_correlation", never)
    monkeypatch.setattr(Limits, "CORR_MAX_CELLS", 100)
    df = sel_data()
    result = validated(df, [Step("select.correlation", ("a", "b"))], ROLES)
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(result).fit(df)
    assert exc.value.code == "OP_LIMIT_EXCEEDED"

"""Review of Part A: KNN/iterative safety, missingness indicators, and sensitive-data
protection on every output surface (items 2-4)."""

import json
import warnings
from typing import Any

import numpy as np
import pandas as pd
import pytest

from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.base import DMTransformer
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.codegen import render_steps
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.ops.impute import KNNImputeOp, MedianImputer
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import MIXED_ROLES, context, frame, mixed_frame, validated

SECRET = "ZX-SECRET-7731"
SECRET_NUMBER = 987654.321

# ---- KNN and iterative (item 2) ---------------------------------------------------------------


@pytest.mark.parametrize("op", ["impute.knn", "impute.iterative"])
def test_predictors_exclude_every_protected_role(op: str) -> None:
    df = frame(
        x=[1.0, None, 3.0, 4.0],
        target=[0.0, 1.0, 0.0, 1.0],
        rid=[10.0, 11.0, 12.0, 13.0],
        grp=[1.0, 1.0, 2.0, 2.0],
        t=[1.0, 2.0, 3.0, 4.0],
        w=[1.0, 1.0, 1.0, 2.0],
        ex=[5.0, 6.0, 7.0, 8.0],
        ok=[2.0, 3.0, 4.0, 5.0],
    )
    roles = {
        "target": ["target"],
        "id": ["rid"],
        "group": ["grp"],
        "time": ["t"],
        "weight": ["w"],
        "exclude": ["ex"],
    }
    result = validate_steps([Step(op, ("x",))], context(df, roles))
    assert result.steps[0].step.params["predictors"] == ["ok"]


def test_knn_scaling_is_learned_from_training_rows_only() -> None:
    df = frame(p=[1.0, 2.0, 3.0, 1000.0, 2000.0], x=[1.0, 2.0, None, 4.0, None])
    result = validated(df, [Step("impute.knn", ("x",), {"predictors": ["p"]})])
    executor = PreprocessExecutor(result).fit(df.iloc[:3])
    state = executor.transformers()[0].state_
    assert state["mean"] == pytest.approx([1.5, 2.0])  # x then p, training rows only
    assert state["std"] == pytest.approx([0.5, np.std([1.0, 2.0, 3.0])])
    assert state["n_fit_rows"] == 3


def test_missing_predictor_values_are_handled() -> None:
    df = frame(
        p=[1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
        q=[2.0, np.nan, 6.0, 8.0, 10.0, np.nan],
        x=[1.0, 2.0, 3.0, 4.0, np.nan, np.nan],
    )
    for op in ("impute.knn", "impute.iterative"):
        result = validated(df, [Step(op, ("x",))])
        out = PreprocessExecutor(result).fit(df.iloc[:4]).transform(df)
        assert out.frame["x"].isna().sum() == 0
        assert out.frame["p"].isna().sum() == 2  # predictors themselves are never written
        assert out.frame["q"].isna().sum() == 2


@pytest.mark.parametrize("op", ["impute.knn", "impute.iterative"])
def test_repeated_runs_are_bitwise_identical(op: str) -> None:
    df = mixed_frame(n=120, seed=5)
    outs = []
    for _ in range(3):
        result = validated(df, [Step(op, ("x1", "x2"))], MIXED_ROLES)
        outs.append(PreprocessExecutor(result, seed=11).fit(df.iloc[:90]).transform(df).frame)
    for other in outs[1:]:
        pd.testing.assert_frame_equal(outs[0], other, check_exact=True)


def test_knn_transform_work_is_capped_before_computing(monkeypatch: pytest.MonkeyPatch) -> None:
    df = frame(p=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], x=[1.0, 2.0, 3.0, np.nan, np.nan, np.nan])
    executor = PreprocessExecutor(validated(df, [Step("impute.knn", ("x",))])).fit(df.iloc[:3])
    called = []
    monkeypatch.setattr(
        "datamedic.engine.preprocess.runtime.numeric_matrix",
        lambda *a, **k: called.append(1) or (_ for _ in ()).throw(AssertionError("allocated")),
    )
    monkeypatch.setattr(Limits, "KNN_MAX_DISTANCE_PAIRS", 8)  # 3 rows needing x 3 fit rows = 9
    with pytest.raises(EngineError) as exc:
        executor.transform(df)
    assert exc.value.code == "OP_LIMIT_EXCEEDED" and not called


def test_fit_row_cap_is_checked_before_building_matrices(monkeypatch: pytest.MonkeyPatch) -> None:
    df = frame(p=[1.0, 2.0, 3.0], x=[1.0, np.nan, 3.0])
    monkeypatch.setattr(Limits, "ITERATIVE_MAX_FIT_ROWS", 2)
    monkeypatch.setattr(
        "datamedic.engine.preprocess.runtime.numeric_matrix",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("allocated")),
    )
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(validated(df, [Step("impute.iterative", ("x",))])).fit(df)
    assert exc.value.code == "OP_LIMIT_EXCEEDED"


def test_column_cap_is_rechecked_at_fit_without_validation() -> None:
    names = [f"c{i}" for i in range(Limits.KNN_MAX_PREDICTORS + 1)]
    df = frame(**{n: [1.0, np.nan, 3.0] for n in names})
    op = KNNImputeOp(
        columns=("c0",), params={"n_neighbors": 1, "weights": "uniform", "predictors": names[1:]}
    )
    with pytest.raises(EngineError) as exc:
        op.fit(df)
    assert exc.value.code == "OP_LIMIT_EXCEEDED"


# ---- Missingness indicators (item 3) ----------------------------------------------------------


def test_indicator_output_schema_is_identical_for_fit_and_any_transform() -> None:
    df = mixed_frame(n=40)
    result = validated(df, [Step("missing.indicator", ("x2", "age", "city"))], MIXED_ROLES)
    executor = PreprocessExecutor(result).fit(df.iloc[:5])  # rows with few gaps
    schemas = [
        [(c, str(t)) for c, t in executor.transform(part).frame.dtypes.items()]
        for part in (df.iloc[:5], df.iloc[5:], df.iloc[:0], df)
    ]
    assert all(s == schemas[0] for s in schemas)
    assert [c for c, _ in schemas[0]][-3:] == ["x2__missing", "age__missing", "city__missing"]


def test_indicator_refuses_to_overwrite_at_transform_time() -> None:
    df = frame(a=[1.0, None])
    executor = PreprocessExecutor(validated(df, [Step("missing.indicator", ("a",))])).fit(df)
    with pytest.raises(EngineError) as exc:
        executor.transform(df.assign(a__missing=[1, 2]))
    assert exc.value.code == "OUTPUT_COLUMN_CONFLICT"


def test_indicator_names_never_collide_within_or_across_steps() -> None:
    df = frame(a=[1.0, None], b=[None, 2.0])
    ctx = context(df)
    for steps in (
        [Step("missing.indicator", ("a",)), Step("missing.indicator", ("a",))],
        [Step("missing.indicator", ("a", "a"))],
    ):
        assert not validate_steps(steps, ctx).valid
    # An indicator of an indicator is allowed and named predictably.
    ok = validate_steps(
        [Step("missing.indicator", ("a",)), Step("missing.indicator", ("a__missing",))], ctx
    )
    assert [c.name for c in ok.output_columns][-2:] == ["a__missing", "a__missing__missing"]


# ---- Sensitive data on every output surface (item 4) ------------------------------------------


def sensitive_frame() -> pd.DataFrame:
    return frame(
        diag=pd.Series([SECRET, SECRET, None, "other", None], dtype=object),
        lab=[SECRET_NUMBER, SECRET_NUMBER, np.nan, 1.0, 2.0],
        p=[1.0, 2.0, 3.0, 4.0, 5.0],
    )


SENSITIVE_ROLES = {"sensitive": ["diag", "lab"]}
SENSITIVE_STEPS = [
    Step("missing.indicator", ("diag", "lab")),
    Step("impute.mode", ("diag",)),
    Step("impute.median", ("lab",)),
    Step("impute.knn", ("lab",)),
    Step("impute.iterative", ("lab",)),
]


def test_step_reports_never_contain_sensitive_learned_values() -> None:
    df = sensitive_frame()
    result = validated(df, SENSITIVE_STEPS, SENSITIVE_ROLES)
    out = PreprocessExecutor(result).fit_transform(df)
    text = json.dumps(out.report_dicts(), allow_nan=False)  # also proves JSON-safe
    assert SECRET not in text
    assert str(SECRET_NUMBER) not in text and "987654" not in text
    for report in out.reports:
        for column, learned in report.learned.get("columns", {}).items():
            assert learned == {"redacted": True}, (report.op, column)


@pytest.mark.parametrize("op", ["impute.mode", "impute.median", "impute.knn", "impute.iterative"])
def test_generated_code_never_contains_learned_values(op: str) -> None:
    df = sensitive_frame()
    column = "diag" if op == "impute.mode" else "lab"
    result = validated(df, [Step(op, (column,))], SENSITIVE_ROLES)
    source = render_steps(result.steps)
    assert SECRET not in source and "987654" not in source  # code refits; it stores no values


def test_engine_errors_never_contain_values() -> None:
    df = frame(a=pd.Series([np.inf, SECRET_NUMBER, None]), p=[1.0, 2.0, 3.0])
    failures = []
    for steps, fit in (
        ([Step("impute.mean", ("a",))], df),  # NON_FINITE_VALUES
        ([Step("impute.median", ("p",))], df.assign(p=[SECRET_NUMBER, np.inf, 1.0])),
    ):
        try:
            PreprocessExecutor(validated(fit, steps)).fit(fit)
        except EngineError as exc:
            failures.append(exc)
    assert len(failures) == 2
    for exc in failures:
        assert "987654" not in exc.message and "987654" not in json.dumps(exc.details)


class _Boom(MedianImputer):
    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        raise ValueError(f"could not convert string to float: '{SECRET}'")


class _OutOfMemory(MedianImputer):
    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        raise MemoryError


def test_unexpected_library_errors_are_withheld() -> None:
    df = frame(a=[1.0, None])
    with pytest.raises(EngineError) as exc:
        _Boom(columns=("a",)).fit(df)
    assert exc.value.code == "OP_FAILED"
    assert SECRET not in exc.value.message and SECRET not in json.dumps(exc.value.details)
    assert exc.value.details["error_type"] == "ValueError"


def test_memory_errors_pass_through_for_the_worker_limit_handler() -> None:
    with pytest.raises(MemoryError):
        _OutOfMemory(columns=("a",)).fit(frame(a=[1.0, None]))


class _Chatty(MedianImputer):
    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        warnings.warn(f"odd value {SECRET}", RuntimeWarning, stacklevel=1)
        super()._fit(X)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        warnings.warn(f"odd value {SECRET}", UserWarning, stacklevel=1)
        return super()._transform(X)


def test_library_warning_text_never_escapes(recwarn: Any) -> None:
    df = frame(a=[1.0, None])
    op = _Chatty(columns=("a",))
    op.fit(df)
    op.transform(df)
    assert [w["code"] for w in op.warnings_] == ["LIBRARY_WARNING"]
    assert [w["code"] for w in op.transform_warnings_] == ["LIBRARY_WARNING"]
    assert SECRET not in json.dumps(op.warnings_ + op.transform_warnings_)
    assert not any(SECRET in str(w.message) for w in recwarn.list)  # nothing reaches stderr/logs


def test_code_warnings_are_still_emitted() -> None:
    class _Deprecated(MedianImputer):
        def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
            warnings.warn("old API", FutureWarning, stacklevel=1)
            super()._fit(X)

    with pytest.warns(FutureWarning, match="old API"):
        _Deprecated(columns=("a",)).fit(frame(a=[1.0, None]))


def test_repeated_transforms_do_not_duplicate_warnings() -> None:
    df = frame(c=pd.Series(["b", "a", None], dtype=object))
    executor = PreprocessExecutor(validated(df, [Step("impute.mode", ("c",))])).fit(df)
    first = executor.transform(df).reports[0].warnings
    second = executor.transform(df).reports[0].warnings
    assert first == second == [w for w in first if w["code"] == "MODE_TIE"]


def test_transformer_subclasses_cannot_skip_the_base_wrappers() -> None:
    # Every catalogue op inherits fit/transform from DMTransformer, so the error and warning
    # wrappers above cover all of them.
    from datamedic.engine.preprocess.catalog import all_ops
    from datamedic.engine.preprocess.executor import load_transformer

    df = mixed_frame()
    for spec in all_ops():
        cols: tuple[str, ...] = ("x1",) if "numeric" in spec.column_types else ("city",)
        if spec.op == "select.correlation":
            cols = ("x1", "x2")
        if not spec.uses_columns:
            cols = ()
        params = {
            "impute.constant": {"fill_value": 0},
            "encode.ordinal": {"order": ["east", "north", "south"]},
            "select.univariate": {"test": "anova_f"},
            "imbalance.smote": {"acknowledge_dependent_rows": True},
        }.get(spec.op, {})
        data = df
        if spec.op == "imbalance.smote":  # SMOTE needs numeric, complete features
            data = df[["_dm_row_id", "x1", "outcome"]].fillna({"x1": 0.0})
        v = validated(data, [Step(spec.op, cols, params)], MIXED_ROLES, task="binary").steps[0]
        cls = type(load_transformer(v, 0))
        assert cls.fit is DMTransformer.fit and cls.transform is DMTransformer.transform
        assert cls.resample is DMTransformer.resample


# ---- Part B: sensitive data, error texts and limits -------------------------------------------

SECRET_CATEGORY = "DIAG-ZX-SECRET"


def sensitive_b_frame() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    lab = rng.normal(500, 50, 60)
    lab[0] = SECRET_NUMBER
    return frame(
        lab=lab,
        other=rng.normal(0, 1, 60),
        diag=pd.Series([SECRET_CATEGORY, "other", "x"] * 20, dtype=object),
    )


PART_B_SENSITIVE_STEPS = [
    Step(
        "outliers.flag",
        ("lab",),
        {"method": "percentile", "lower_percentile": 0, "upper_percentile": 100},
    ),
    Step("outliers.cap", ("lab",), {"method": "mad"}),
    Step("outliers.isolation_forest_flag", ("lab", "other")),
    Step("encode.frequency", ("diag",)),
    Step("scale.minmax", ("lab",)),
]


def test_part_b_reports_never_contain_sensitive_values() -> None:
    df = sensitive_b_frame()
    roles = {"sensitive": ["lab", "diag"]}
    for steps in (PART_B_SENSITIVE_STEPS, [Step("encode.onehot", ("diag",))]):
        result = validated(df, steps, roles, {"diag": "categorical"})
        out = PreprocessExecutor(result).fit_transform(df)
        text = json.dumps(out.report_dicts(), allow_nan=False)
        assert SECRET_CATEGORY not in text and "987654" not in text
        assert SECRET_CATEGORY not in "".join(out.frame.columns)
        for report in out.reports:
            for learned in report.learned.get("columns", {}).values():
                assert learned == {"redacted": True}


@pytest.mark.parametrize(
    "step",
    PART_B_SENSITIVE_STEPS + [Step("encode.onehot", ("diag",))],
    ids=lambda s: s.op,
)
def test_part_b_generated_code_contains_no_learned_values(step: Step) -> None:
    df = sensitive_b_frame()
    result = validated(df, [step], {"sensitive": ["lab", "diag"]}, {"diag": "categorical"})
    source = render_steps(result.steps)
    assert SECRET_CATEGORY not in source and "987654" not in source


def test_part_b_errors_contain_no_values() -> None:
    df = frame(c=pd.Series(["a", "b", SECRET_CATEGORY], dtype=object))
    cases = [
        (Step("encode.ordinal", ("c",), {"order": ["a", "b"]}), None, "ORDER_INCOMPLETE"),
        (Step("encode.ordinal", ("c",), {"order": ["a", "b"]}), [0, 1], "UNKNOWN_CATEGORY"),
        (Step("encode.onehot", ("c",), {"handle_unknown": "error"}), [0, 1], "UNKNOWN_CATEGORY"),
    ]
    for step, fit_rows, code in cases:
        result = validated(df, [step], types={"c": "categorical"})
        executor = PreprocessExecutor(result)
        with pytest.raises(EngineError) as exc:
            executor.fit(df if fit_rows is None else df.iloc[fit_rows]).transform(df)
        assert exc.value.code == code
        assert SECRET_CATEGORY not in exc.value.message
        assert SECRET_CATEGORY not in json.dumps(exc.value.details)


def test_onehot_runtime_refuses_colliding_output_names() -> None:
    """Defence in depth behind validation: an output name that already exists."""
    from datamedic.engine.preprocess import runtime as rt

    df = frame(c=pd.Series(["a", "b"], dtype=object), **{"c=a": [1.0, 2.0]})
    state = rt.fit_onehot(df, ["c"], 10, "value")
    with pytest.raises(rt.PreprocessError) as exc:
        rt.apply_onehot(df, ["c"], state, "ignore")
    assert exc.value.code == "OUTPUT_COLUMN_CONFLICT"


def test_isolation_forest_limits_are_checked_before_building_matrices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = sensitive_b_frame()
    monkeypatch.setattr(
        "datamedic.engine.preprocess.runtime.numeric_matrix",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("allocated")),
    )
    monkeypatch.setattr(Limits, "IF_MAX_FIT_ROWS", 10)
    step = Step("outliers.isolation_forest_flag", ("lab", "other"))
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(validated(df, [step])).fit(df)
    assert exc.value.code == "OP_LIMIT_EXCEEDED"
    monkeypatch.setattr(Limits, "IF_MAX_FIT_ROWS", 10_000)
    from datamedic.engine.preprocess.ops.outliers import IsolationForestFlagOp

    too_wide = IsolationForestFlagOp(
        columns=tuple(f"c{i}" for i in range(Limits.IF_MAX_COLUMNS + 1)),
        params={"n_estimators": 10, "contamination": None, "output_column": "o"},
    )
    wide = frame(**{f"c{i}": [1.0, 2.0] for i in range(Limits.IF_MAX_COLUMNS + 1)})
    with pytest.raises(EngineError) as exc:
        too_wide.fit(wide)
    assert exc.value.code == "OP_LIMIT_EXCEEDED"


def test_runtime_constants_match_the_catalogue() -> None:
    """The runtime is standalone (embedded in generated code), so it repeats a few names."""
    from datamedic.engine.preprocess import catalog
    from datamedic.engine.preprocess import runtime as rt

    assert rt.INDICATOR_SUFFIX == catalog.INDICATOR_SUFFIX
    assert rt.OUTLIER_SUFFIX == catalog.OUTLIER_SUFFIX
    assert rt.ONEHOT_SEPARATOR == catalog.ONEHOT_SEPARATOR
    assert rt.ONEHOT_OTHER == catalog.ONEHOT_OTHER

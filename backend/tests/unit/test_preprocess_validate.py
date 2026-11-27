"""Step validation: invalid configurations fail clearly; protected columns are never touched."""

import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import ValidationContext, validate_steps
from tests.unit.preprocess_helpers import MIXED_ROLES, ROLE_SPECS, context, frame, mixed_frame

DF = mixed_frame()
CTX = context(DF, MIXED_ROLES)


def codes(steps: list[Step], ctx: ValidationContext = CTX) -> list[tuple[str, str | None]]:
    return [(e.code, e.column) for e in validate_steps(steps, ctx).errors]


def test_defaults_are_filled_and_predictors_resolved() -> None:
    result = validate_steps([Step("impute.knn", ("x1",))], CTX)
    assert result.valid
    params = result.steps[0].step.params
    assert params["n_neighbors"] == 5 and params["weights"] == "uniform"
    # Numeric features except the imputed column; never the target or the id.
    assert params["predictors"] == ["age", "x2"]
    assert result.spec_hash is not None


@pytest.mark.parametrize("op", ["impute.mean", "impute.median", "impute.mode", "missing.indicator"])
def test_target_and_identifier_columns_are_protected(op: str) -> None:
    assert ("COLUMN_PROTECTED", "outcome") in codes([Step(op, ("outcome",))])
    assert ("COLUMN_PROTECTED", "patient") in codes([Step(op, ("patient",))])


@pytest.mark.parametrize("role", ["group", "time", "weight", "exclude"])
def test_every_non_feature_role_is_protected(role: str) -> None:
    ctx = context(DF, {**MIXED_ROLES, role: ["x2"]})
    assert ("COLUMN_PROTECTED", "x2") in codes([Step("impute.median", ("x2",))], ctx)


def test_unknown_role_is_treated_as_protected() -> None:
    ctx = ValidationContext.from_schema(
        [{"name": "a", "inferred_type": "numeric", "dtype": "float"}],
        {"made.up": ["a"]},
        ROLE_SPECS,
    )
    assert codes([Step("impute.mean", ("a",))], ctx) == [("COLUMN_PROTECTED", "a")]


def test_sensitive_role_is_a_feature_but_marked_sensitive() -> None:
    ctx = context(DF, {**MIXED_ROLES, "sensitive": ["age"]})
    result = validate_steps([Step("impute.median", ("age",))], ctx)
    assert result.valid and result.sensitive == frozenset({"age"})


def test_target_is_never_a_predictor() -> None:
    assert ("COLUMN_PROTECTED", "outcome") in codes(
        [Step("impute.iterative", ("x1",), {"predictors": ["outcome"]})]
    )


def test_row_id_cannot_be_selected_or_used() -> None:
    assert ("RESERVED_COLUMN", ROW_ID) in codes([Step("impute.mean", (ROW_ID,))])
    assert ("RESERVED_COLUMN", ROW_ID) in codes(
        [Step("impute.knn", ("x1",), {"predictors": [ROW_ID]})]
    )


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (Step("impute.nope", ("x1",)), "UNKNOWN_OP"),
        (Step("impute.mean", ("x1",), {"strategy": "x"}), "UNKNOWN_PARAM"),
        (Step("impute.mean", ()), "COLUMNS_REQUIRED"),
        (Step("impute.mean", ("x1", "x1")), "DUPLICATE_COLUMN"),
        (Step("impute.mean", ("nope",)), "COLUMN_NOT_FOUND"),
        (Step("impute.mean", ("city",)), "COLUMN_TYPE_UNSUPPORTED"),
        (Step("impute.median", ("flag",)), "COLUMN_TYPE_UNSUPPORTED"),
        (Step("impute.knn", ("city",)), "COLUMN_TYPE_UNSUPPORTED"),
        (Step("impute.constant", ("x1",)), "PARAM_REQUIRED"),
        (Step("impute.constant", ("x1",), {"fill_value": "zero"}), "PARAM_INVALID"),
        (Step("impute.constant", ("x1",), {"fill_value": True}), "PARAM_INVALID"),
        (Step("impute.constant", ("x1",), {"fill_value": float("inf")}), "PARAM_INVALID"),
        (Step("impute.constant", ("city",), {"fill_value": 3}), "PARAM_INVALID"),
        (Step("impute.constant", ("city",), {"fill_value": "   "}), "PARAM_INVALID"),
        (Step("impute.constant", ("city",), {"fill_value": "x" * 201}), "PARAM_INVALID"),
        (Step("impute.constant", ("flag",), {"fill_value": 1}), "PARAM_INVALID"),
        (Step("impute.constant", ("x1",), {"fill_value": [1]}), "PARAM_INVALID"),
        (Step("impute.knn", ("x1",), {"n_neighbors": 0}), "PARAM_INVALID"),
        (Step("impute.knn", ("x1",), {"n_neighbors": 51}), "PARAM_INVALID"),
        (Step("impute.knn", ("x1",), {"n_neighbors": 2.5}), "PARAM_INVALID"),
        (Step("impute.knn", ("x1",), {"n_neighbors": True}), "PARAM_INVALID"),
        (Step("impute.knn", ("x1",), {"weights": "cosine"}), "PARAM_INVALID"),
        (Step("impute.knn", ("x1",), {"predictors": "x2"}), "PARAM_INVALID"),
        (Step("impute.knn", ("x1",), {"predictors": ["x2", "x2"]}), "PARAM_INVALID"),
        (Step("impute.knn", ("x1",), {"predictors": ["city"]}), "COLUMN_TYPE_UNSUPPORTED"),
        (Step("impute.knn", ("x1",), {"predictors": ["ghost"]}), "COLUMN_NOT_FOUND"),
        (Step("impute.iterative", ("x1",), {"clip_to_fit_range": "yes"}), "PARAM_INVALID"),
        (Step("impute.iterative", ("x1",), {"max_iter": 0}), "PARAM_INVALID"),
    ],
)
def test_invalid_configurations(step: Step, expected: str) -> None:
    errors = validate_steps([step], CTX).errors
    assert expected in {e.code for e in errors}, [e.as_dict() for e in errors]


def test_error_messages_contain_no_data_values() -> None:
    df = frame(code=["SECRET-VALUE-1", None, "SECRET-VALUE-1"])
    ctx = context(df, types={"code": "categorical"})
    errors = validate_steps([Step("impute.constant", ("code",), {"fill_value": 7})], ctx).errors
    assert errors
    assert all("SECRET" not in e.message for e in errors)


def test_datetime_and_text_columns_cannot_be_imputed() -> None:
    df = frame(when=[None, "2024-01-01"], note=[None, "x"])
    ctx = context(df, types={"when": "datetime", "note": "text"})
    found = {e.code for e in validate_steps([Step("impute.mode", ("when", "note"))], ctx).errors}
    assert found == {"COLUMN_TYPE_UNSUPPORTED"}
    # ... but a missing-value indicator works for any feature column.
    assert validate_steps([Step("missing.indicator", ("when", "note"))], ctx).valid


def test_schema_is_tracked_through_steps() -> None:
    result = validate_steps(
        [
            Step("missing.indicator", ("x1",)),
            Step("impute.median", ("x1_missing_typo",)),
        ],
        CTX,
    )
    assert ("COLUMN_NOT_FOUND", "x1_missing_typo") in [(e.code, e.column) for e in result.errors]
    ok = validate_steps(
        [Step("missing.indicator", ("x1",)), Step("impute.mode", ("x1__missing",))], CTX
    )
    assert ok.valid  # an added column can be used by a later step
    assert {c.name: c.dtype for c in ok.output_columns}["x1__missing"] == "boolean"


def test_indicator_name_conflicts_are_rejected() -> None:
    df = frame(a=[1.0, None], a__missing=[True, False])
    ctx = context(df)
    assert ("OUTPUT_COLUMN_CONFLICT", "a") in codes([Step("missing.indicator", ("a",))], ctx)
    twice = [Step("missing.indicator", ("x1",)), Step("missing.indicator", ("x1",))]
    assert ("OUTPUT_COLUMN_CONFLICT", "x1") in codes(twice)


def test_dtype_tracking_and_no_missing_warning() -> None:
    result = validate_steps([Step("impute.median", ("age",)), Step("impute.mean", ("age",))], CTX)
    assert result.valid
    assert {c.name: c.dtype for c in result.output_columns}["age"] == "float"
    warned = [(w.code, w.step) for w in result.warnings]
    assert ("NO_MISSING_VALUES", 1) in warned  # the second imputation has nothing left to do


def test_predictor_cap_is_enforced() -> None:
    wide = frame(**{f"c{i}": [1.0, None, 3.0] for i in range(Limits.ITERATIVE_MAX_PREDICTORS + 1)})
    ctx = context(wide)
    assert "OP_LIMIT_EXCEEDED" in {
        e.code for e in validate_steps([Step("impute.iterative", ("c0",))], ctx).errors
    }


def test_too_many_steps() -> None:
    steps = [Step("impute.mean", ("x1",))] * (Limits.MAX_STEPS + 1)
    assert [e.code for e in validate_steps(steps, CTX).errors] == ["TOO_MANY_STEPS"]


def test_executor_refuses_invalid_pipelines() -> None:
    result = validate_steps([Step("impute.mean", ("outcome",))], CTX)
    with pytest.raises(EngineError) as exc:
        PreprocessExecutor(result)
    assert exc.value.code == "OP_INVALID"

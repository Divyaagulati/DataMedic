"""Encoding ops (Part B): unseen categories, high cardinality, nominal versus ordinal, stable
names and order, collisions, frequency encoding."""

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import context, frame, validated

CAT = {"c": "categorical"}


def fit_apply(
    df: pd.DataFrame,
    steps: list[Step],
    fit_rows: list[int] | None = None,
    roles: dict[str, list[str]] | None = None,
    types: dict[str, str] | None = None,
):
    result = validated(df, steps, roles, types or CAT)
    executor = PreprocessExecutor(result)
    executor.fit(df if fit_rows is None else df[df[ROW_ID].isin(fit_rows)])
    return executor.transform(df)


# ---- One-hot ----------------------------------------------------------------------------------


def test_onehot_replaces_the_column_in_place_with_sorted_categories() -> None:
    df = frame(a=[1.0, 2.0, 3.0], c=["red", "blue", None], z=[0, 0, 0])
    out = fit_apply(df, [Step("encode.onehot", ("c",))])
    assert list(out.frame.columns) == [ROW_ID, "a", "c=blue", "c=red", "z"]
    assert out.frame["c=blue"].tolist() == [False, True, False]
    assert out.frame["c=red"].tolist() == [True, False, False]  # missing -> all zeros
    report = out.reports[0]
    assert report.columns_removed == ["c"] and report.columns_added == ["c=blue", "c=red"]
    assert report.effects["output_columns"] == {"c": 2}


def test_onehot_unseen_categories_ignore_or_error() -> None:
    df = frame(c=["a", "b", "a", "zzz"])
    ignore = fit_apply(df, [Step("encode.onehot", ("c",))], fit_rows=[0, 1, 2])
    assert ignore.frame[["c=a", "c=b"]].iloc[3].tolist() == [False, False]
    assert ignore.reports[0].effects["unseen_category_rows"] == {"c": 1}
    assert "UNSEEN_CATEGORIES" in [w["code"] for w in ignore.reports[0].warnings]
    with pytest.raises(EngineError) as exc:
        fit_apply(df, [Step("encode.onehot", ("c",), {"handle_unknown": "error"})], [0, 1, 2])
    assert exc.value.code == "UNKNOWN_CATEGORY"
    assert "zzz" not in exc.value.message


def test_onehot_high_cardinality_groups_rare_categories() -> None:
    values = ["a"] * 5 + ["b"] * 4 + ["c"] * 3 + ["d", "e", "f"]
    df = frame(c=values + ["never-seen"])
    out = fit_apply(
        df,
        [Step("encode.onehot", ("c",), {"max_categories": 3})],
        fit_rows=list(range(len(values))),
    )
    assert [c for c in out.frame.columns if c.startswith("c=")] == [
        "c=a",
        "c=b",
        "c=c",
        "c=__other__",
    ]
    other = out.frame["c=__other__"].tolist()
    assert other == [False] * 12 + [True, True, True, True]  # rare and unseen share 'other'
    learned = out.reports[0].learned["columns"]["c"]
    assert learned["grouped_as_other"] == 3 and learned["n_categories_seen"] == 6
    assert "RARE_GROUPED" in [w["code"] for w in out.reports[0].warnings]


def test_onehot_output_is_stable_across_row_order_and_folds() -> None:
    df = frame(c=["b", "a", "c", "a", "b", "c"])
    first = fit_apply(df, [Step("encode.onehot", ("c",))], fit_rows=[0, 1, 2])
    shuffled = df.iloc[::-1].reset_index(drop=True)
    second = fit_apply(shuffled, [Step("encode.onehot", ("c",))], fit_rows=[3, 4, 5])
    assert list(first.frame.columns) == list(second.frame.columns)


def test_onehot_name_collisions_are_refused() -> None:
    existing = frame(c=["a", "b"], **{"c=a": [1.0, 2.0]})
    errors = validate_steps([Step("encode.onehot", ("c",))], context(existing, types=CAT)).errors
    assert ("OUTPUT_COLUMN_CONFLICT", "c") in [(e.code, e.column) for e in errors]
    # A later step cannot add a column inside a one-hot namespace...
    df = frame(c=["a", "b"], x=[1.0, None])
    later = validate_steps(
        [
            Step("encode.onehot", ("c",)),
            Step("outliers.isolation_forest_flag", ("x",), {"output_column": "c=a"}),
        ],
        context(df, types=CAT),
    )
    assert "OUTPUT_COLUMN_CONFLICT" in {e.code for e in later.errors}
    # ...and the removed source column can no longer be used.
    gone = validate_steps(
        [Step("encode.onehot", ("c",)), Step("missing.indicator", ("c",))],
        context(df, types=CAT),
    )
    assert ("COLUMN_NOT_FOUND", "c") in [(e.code, e.column) for e in gone.errors]


def test_onehot_runtime_collision_between_overlapping_names() -> None:
    # Columns 'a' (category 'b=x') and 'a=b' (category 'x') would both produce 'a=b=x'.
    df = frame(a=["b=x", "y"], **{"a=b": ["x", "z"]})
    result = validate_steps(
        [Step("encode.onehot", ("a", "a=b"))],
        context(df, types={"a": "categorical", "a=b": "categorical"}),
    )
    assert not result.valid  # the namespaces overlap, refused before any data is read


def test_onehot_sensitive_columns_are_named_by_position() -> None:
    df = frame(c=["HIV+", "neg", "HIV+"])
    result = validate_steps(
        [Step("encode.onehot", ("c",))], context(df, {"sensitive": ["c"]}, types=CAT)
    )
    assert result.steps[0].step.params["name_by"] == "position"
    assert "SENSITIVE_POSITIONAL_NAMES" in [w.code for w in result.warnings]
    out = PreprocessExecutor(result).fit_transform(df)
    assert [c for c in out.frame.columns if c.startswith("c=")] == ["c=1", "c=2"]
    assert "HIV" not in repr(out.report_dicts())


def test_onehot_boolean_columns() -> None:
    df = frame(b=pd.array([True, None, False], dtype="boolean"))
    out = fit_apply(df, [Step("encode.onehot", ("b",))], types={"b": "boolean"})
    assert out.frame["b=True"].tolist() == [True, False, False]
    assert out.frame["b=False"].tolist() == [False, False, True]


def test_onehot_output_cap() -> None:
    df = frame(**{f"c{i}": ["a", "b"] for i in range(10)})
    types = {f"c{i}": "categorical" for i in range(10)}
    errors = validate_steps(
        [Step("encode.onehot", tuple(types), {"max_categories": 100})], context(df, types=types)
    ).errors
    assert "OP_LIMIT_EXCEEDED" in {e.code for e in errors}  # 10 x 101 > 1000


# ---- Ordinal ----------------------------------------------------------------------------------


def test_ordinal_requires_an_explicit_order() -> None:
    df = frame(c=["low", "high"])
    errors = validate_steps([Step("encode.ordinal", ("c",))], context(df, types=CAT)).errors
    assert [e.code for e in errors] == ["PARAM_REQUIRED"]  # never an invented order
    for bad in ([], ["low"], ["low", "low"], ["low", " "], "low,high", [1, 2]):
        errors = validate_steps(
            [Step("encode.ordinal", ("c",), {"order": bad})], context(df, types=CAT)
        ).errors
        assert errors, bad


def test_ordinal_maps_in_the_given_order() -> None:
    df = frame(c=["medium", "low", None, "high"])
    out = fit_apply(df, [Step("encode.ordinal", ("c",), {"order": ["low", "medium", "high"]})])
    assert out.frame["c"].tolist() == [1, 0, pd.NA, 2]
    assert out.frame["c"].dtype == "Int64"


def test_ordinal_incomplete_order_and_unknown_values() -> None:
    df = frame(c=["low", "high", "low", "extreme"])
    order = {"order": ["low", "high"]}
    with pytest.raises(EngineError) as exc:  # the training rows contain 'extreme'
        fit_apply(df, [Step("encode.ordinal", ("c",), order)])
    assert exc.value.code == "ORDER_INCOMPLETE"
    with pytest.raises(EngineError) as exc:  # unknown only in other rows; default 'error'
        fit_apply(df, [Step("encode.ordinal", ("c",), order)], fit_rows=[0, 1, 2])
    assert exc.value.code == "UNKNOWN_CATEGORY"
    out = fit_apply(
        df,
        [Step("encode.ordinal", ("c",), {**order, "handle_unknown": "missing"})],
        fit_rows=[0, 1, 2],
    )
    assert out.frame["c"].tolist() == [0, 1, 0, pd.NA]
    assert out.reports[0].effects == {"unknown_to_missing_rows": 1}


def test_ordinal_is_one_column_at_a_time() -> None:
    df = frame(c=["a"], d=["b"])
    errors = validate_steps(
        [Step("encode.ordinal", ("c", "d"), {"order": ["a", "b"]})],
        context(df, types={"c": "categorical", "d": "categorical"}),
    ).errors
    assert "TOO_MANY_COLUMNS" in {e.code for e in errors}


# ---- Frequency --------------------------------------------------------------------------------


def test_frequency_learned_from_training_rows_with_unseen_as_zero() -> None:
    df = frame(c=["a", "a", "a", "b", "b", "c", None, "z"])
    out = fit_apply(df, [Step("encode.frequency", ("c",))], fit_rows=[0, 1, 2, 3, 4, 5, 6])
    got = out.frame["c"].tolist()
    assert got[:6] == pytest.approx([0.5, 0.5, 0.5, 2 / 6, 2 / 6, 1 / 6])
    assert np.isnan(got[6]) and got[7] == 0.0  # missing stays missing; unseen -> 0
    assert out.reports[0].effects == {"unseen_category_rows": {"c": 1}}


def test_frequency_ties_are_reported() -> None:
    out = fit_apply(frame(c=["a", "b", "c", "c"]), [Step("encode.frequency", ("c",))])
    assert out.reports[0].learned["columns"]["c"] == {"n_categories": 3, "tied_categories": 2}
    assert "FREQUENCY_TIES" in [w["code"] for w in out.reports[0].warnings]


# ---- Roles ------------------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["encode.onehot", "encode.ordinal", "encode.frequency"])
def test_encoders_never_touch_target_or_identifier(op: str) -> None:
    df = frame(y=["yes", "no"], pid=["p1", "p2"])
    ctx = context(
        df, {"target": ["y"], "id": ["pid"]}, types={"y": "categorical", "pid": "categorical"}
    )
    params = {"order": ["no", "yes"]} if op == "encode.ordinal" else {}
    for column in ("y", "pid"):
        codes = {e.code for e in validate_steps([Step(op, (column,), params)], ctx).errors}
        assert "COLUMN_PROTECTED" in codes


def test_text_and_identifier_columns_are_not_encoded() -> None:
    df = frame(t=["long free text"], i=["id-1"])
    ctx = context(df, types={"t": "text", "i": "identifier"})
    for op in ("encode.onehot", "encode.frequency"):
        for column in ("t", "i"):
            codes = {e.code for e in validate_steps([Step(op, (column,))], ctx).errors}
            assert codes == {"COLUMN_TYPE_UNSUPPORTED"}

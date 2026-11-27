"""Column type detection rules (conservative and documented in engine/table.py)."""

import pandas as pd
import pytest

from datamedic.engine.table import (
    BOOLEAN,
    CATEGORICAL,
    DATETIME,
    EMPTY,
    IDENTIFIER,
    NUMERIC,
    TEXT,
    RawColumns,
    build_canonical,
    canonical_names,
    infer_text_column,
    normalise_text_column,
)


def infer(values: list[str | None]) -> tuple[pd.Series, str]:
    return infer_text_column(normalise_text_column(pd.Series(values, dtype=object)))


@pytest.mark.parametrize(
    "values,expected",
    [
        (["1", "2", "3"], NUMERIC),
        ([" 1.5", "2", "-3e2"], NUMERIC),
        (["true", "False", "YES", "no"], BOOLEAN),
        (["2024-01-01", "2024-02-29T10:30:00"], DATETIME),
        (["a", "b", "a"], CATEGORICAL),
        (["", "NA", None], EMPTY),
    ],
)
def test_semantic_types(values: list[str | None], expected: str) -> None:
    assert infer(values)[1] == expected


def test_missing_tokens_are_documented_set() -> None:
    series, _ = infer(["1", "NA", "N/A", "null", "#N/A", "  ", "2"])
    assert series.isna().sum() == 5
    # Values that can be meaningful are not treated as missing.
    kept, kind = infer(["None", "-", "?"])
    assert kept.notna().all() and kind == CATEGORICAL


def test_integers_use_nullable_dtype() -> None:
    series, _ = infer(["1", None, "3"])
    assert str(series.dtype) == "Int64" and series.isna().sum() == 1


@pytest.mark.parametrize(
    "values",
    [
        ["007", "010", "123"],  # leading zeros: codes, not numbers
        ["12345678901234567890", "1"],  # too long for exact numbers
        ["1", "two"],
        ["inf", "1"],
        ["1,234", "5"],  # thousands separators are not guessed
    ],
)
def test_numeric_like_text_stays_text(values: list[str]) -> None:
    series, kind = infer(values)
    assert kind != NUMERIC and series.dtype == object


@pytest.mark.parametrize(
    "values",
    [
        ["01/03/2024", "02/03/2024"],  # day/month order is ambiguous
        ["2024-01-01", "not a date"],
        ["2024-01-01T10:00:00Z", "2024-01-02T10:00:00"],  # zoned mixed with naive
        ["2024-02-30"],
    ],
)
def test_ambiguous_or_mixed_dates_stay_text(values: list[str]) -> None:
    assert infer(values)[1] != DATETIME


def test_zoned_dates_are_converted_to_utc() -> None:
    series, kind = infer(["2024-01-01T10:00:00+01:00", "2024-01-01T10:00:00+02:00"])
    assert kind == DATETIME
    assert str(series.dt.tz) == "UTC"
    assert series.iloc[0].hour == 9 and series.iloc[1].hour == 8


def test_identifier_and_text_detection() -> None:
    ids = [f"P{i:04d}" for i in range(30)]
    assert infer(ids)[1] == IDENTIFIER
    notes = [f"free text note number {i} with several words in it" for i in range(30)]
    assert infer(notes)[1] == TEXT


def test_unique_integers_over_many_rows_look_like_identifiers() -> None:
    table, _ = build_canonical(
        RawColumns(names=["id"], columns=[pd.Series([str(i) for i in range(25)])])
    )
    assert table.columns[0].inferred_type == IDENTIFIER
    assert table.columns[0].dtype == "integer"


def test_canonical_names() -> None:
    assert canonical_names(["a", "a", "A", None, 3, "x\ty", "_dm_x"]) == [
        ("a", "a"),
        ("a_2", "a"),
        ("A", "A"),
        ("column_4", ""),
        ("3", "3"),
        ("x y", "x\ty"),
        ("col_dm_x", "_dm_x"),
    ]


def test_long_names_are_truncated() -> None:
    ((name, original),) = canonical_names(["n" * 500])
    assert len(name) == 200 and len(original) == 500


@pytest.mark.parametrize(
    "values,expected",
    [
        (["F", "F", "F"], CATEGORICAL),  # one letter only: a code (e.g. sex), not all-false
        (["F", "M", "F"], CATEGORICAL),
        (["N", "S", "E", "W"], CATEGORICAL),  # compass points, not booleans
        (["Y", "N", "y"], BOOLEAN),  # both letters of one pair
        (["T", "F"], BOOLEAN),
        (["t", "n"], CATEGORICAL),  # letters from different pairs
        (["yes", "no", "Y"], CATEGORICAL),  # a lone letter beside words stays ambiguous
        (["yes", "yes"], BOOLEAN),  # whole words are unambiguous
        (["TRUE", "false", "Yes"], BOOLEAN),
    ],
)
def test_boolean_rules_are_conservative(values: list[str], expected: str) -> None:
    assert infer(values)[1] == expected


def test_boolean_values_map_correctly() -> None:
    series, _ = infer(["Y", "N", None, "y"])
    assert series.tolist() == [True, False, pd.NA, True]


def _int_column(values: list[int]) -> str:
    table, _ = build_canonical(
        RawColumns(names=["v"], columns=[pd.Series([str(v) for v in values])])
    )
    return table.columns[0].inferred_type


def test_dense_distinct_integers_are_identifiers() -> None:
    assert _int_column(list(range(1001, 1031))) == IDENTIFIER  # record numbers, shuffled or not
    assert _int_column(list(reversed(range(30)))) == IDENTIFIER


def test_spread_out_distinct_integers_stay_measurements() -> None:
    incomes = [21000 + 1379 * i for i in range(30)]  # all distinct, wide range
    assert _int_column(incomes) == NUMERIC
    assert _int_column(list(range(1, 11))) == NUMERIC  # too few rows to call it an ID


@pytest.mark.parametrize(
    "values,expected",
    [
        (["007", "010", "123"], CATEGORICAL),  # zero-padded codes stay text
        (["0.5", "1.25"], NUMERIC),
        (["-0", "0", "3"], NUMERIC),
        (["20240101", "20240102"], NUMERIC),  # compact dates are not guessed
        (["2024-01-01", "2024-01-02 10:00"], DATETIME),
        (["1", "2", "x"], CATEGORICAL),  # mixed types are classified as text-like, not coerced
    ],
)
def test_numeric_and_date_boundaries(values: list[str], expected: str) -> None:
    assert infer(values)[1] == expected

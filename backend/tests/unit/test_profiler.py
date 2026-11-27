"""BuiltinProfiler: golden statistics on synthetic data, honest not-run reasons, privacy caps."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.adapters.builtin_profiler import (
    CORR_MAX_ROWS,
    MAX_COMPLETE_VALUES,
    BuiltinProfiler,
)
from datamedic.engine.table import RawColumns, build_canonical


def profile_of(columns: dict[str, list[object]]) -> dict[str, object]:
    raw = RawColumns(
        names=list(columns),
        columns=[
            pd.Series([None if v is None else str(v) for v in vals], dtype=object)
            for vals in columns.values()
        ],
    )
    table, _ = build_canonical(raw)
    schema = [c.as_dict() for c in table.columns]
    before = table.frame.copy()
    result = BuiltinProfiler().profile(table.frame, schema, seed=0)
    pd.testing.assert_frame_equal(table.frame, before)  # never mutates its input
    json.dumps(result, allow_nan=False)  # JSON-safe: no NaN or infinities
    return result


def col(result: dict[str, object], name: str) -> dict[str, object]:
    return next(c for c in result["columns"] if c["name"] == name)  # type: ignore[union-attr, index]


def test_numeric_statistics_match_numpy() -> None:
    values = [1, 2, 2, 3, 4, 7, 9, 15, None]
    result = profile_of({"x": values})
    stats = col(result, "x")["numeric"]
    present = np.array([v for v in values if v is not None], dtype=float)
    assert stats["mean"] == pytest.approx(present.mean())  # type: ignore[index]
    assert stats["median"] == pytest.approx(np.median(present))  # type: ignore[index]
    assert stats["std"] == pytest.approx(present.std(ddof=1))  # type: ignore[index]
    assert stats["min"] == 1 and stats["max"] == 15  # type: ignore[index]
    q1, q3 = np.quantile(present, [0.25, 0.75])
    assert stats["q1"] == pytest.approx(q1) and stats["q3"] == pytest.approx(q3)  # type: ignore[index]
    assert stats["iqr"] == pytest.approx(q3 - q1)  # type: ignore[index]
    assert stats["skewness"] == pytest.approx(pd.Series(present).skew())  # type: ignore[index]
    assert stats["skewness"] > 1  # type: ignore[index, operator]
    hist = col(result, "x")["histogram"]
    assert sum(hist["counts"]) == 8 and len(hist["edges"]) == 21  # type: ignore[index]


def test_missing_unique_and_constant() -> None:
    result = profile_of({"a": [1, None, 3, None], "c": ["k", "k", "k", None], "e": [None] * 4})
    a = col(result, "a")
    assert a["n_missing"] == 2 and a["missing_pct"] == 50.0 and a["n_unique"] == 2
    c = col(result, "c")
    assert c["is_constant"] and c["top_value_pct"] == 100.0
    e = col(result, "e")
    assert e["is_empty"] and e["inferred_type"] == "empty" and e["frequencies"] is None
    summary = result["summary"]
    assert summary["constant_columns"] == ["c"]  # type: ignore[index]
    assert summary["empty_columns"] == ["e"]  # type: ignore[index]
    assert summary["missing_cells"] == 7 and summary["n_cells"] == 12  # type: ignore[index]
    assert summary["rows_with_missing"] == 4  # type: ignore[index]


def test_skewness_needs_three_values() -> None:
    stats = col(profile_of({"x": [1, 2]}), "x")["numeric"]
    assert stats["skewness"] is None and "3 values" in stats["skewness_note"]  # type: ignore[index, operator]
    stats = col(profile_of({"x": [5, 5, 5, 5]}), "x")["numeric"]
    assert stats["skewness"] is None and "undefined" in stats["skewness_note"]  # type: ignore[index, operator]


def test_duplicate_rows_ignore_row_id() -> None:
    result = profile_of({"a": [1, 1, 2, 1], "b": ["x", "x", "y", "x"]})
    summary = result["summary"]
    assert summary["duplicate_rows"] == 2  # type: ignore[index]
    assert summary["rows_in_duplicate_groups"] == 3  # type: ignore[index]


def test_class_distribution_frequencies_are_complete_and_sorted() -> None:
    labels = ["yes"] * 6 + ["no"] * 3 + ["maybe"]
    freq = col(profile_of({"y": labels}), "y")["frequencies"]
    assert freq["complete"] and freq["other_count"] == 0  # type: ignore[index]
    assert [(v["value"], v["count"]) for v in freq["values"]] == [  # type: ignore[index]
        ("yes", 6),
        ("no", 3),
        ("maybe", 1),
    ]
    assert freq["values"][0]["pct"] == 60.0  # type: ignore[index]


def test_numeric_binary_target_gets_full_counts() -> None:
    freq = col(profile_of({"outcome": [0, 1, 0, 0, 1]}), "outcome")["frequencies"]
    assert freq["complete"] and {v["value"] for v in freq["values"]} == {0, 1}  # type: ignore[index]


def test_frequency_tables_are_capped_for_privacy() -> None:
    n = 200
    ids = [f"ID{i:05d}" for i in range(n)]
    many_cats = [f"c{i % 60}" for i in range(n)]
    long_values = ["v" * 100 if i % 2 else "w" * 100 for i in range(n)]
    result = profile_of({"id": ids, "cat": many_cats, "long": long_values})
    assert col(result, "id")["frequencies"]["values"] == []  # type: ignore[index]
    cat = col(result, "cat")["frequencies"]
    assert not cat["complete"] and len(cat["values"]) == 10  # type: ignore[index, arg-type]
    assert cat["other_count"] == n - sum(v["count"] for v in cat["values"])  # type: ignore[index, union-attr]
    long_freq = col(result, "long")["frequencies"]
    assert all(len(v["value"]) <= 64 for v in long_freq["values"])  # type: ignore[index, union-attr]
    for c in result["columns"]:  # type: ignore[union-attr]
        freq = c["frequencies"]  # type: ignore[index]
        assert freq is None or len(freq["values"]) <= MAX_COMPLETE_VALUES


def test_correlations() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    result = profile_of(
        {
            "x": list(x.round(6)),
            "y": list((2 * x + rng.normal(scale=0.1, size=200)).round(6)),
            "z": list(rng.normal(size=200).round(6)),
            "flag": [bool(v > 0) for v in x],
            "label": ["a"] * 200,  # categorical and constant: excluded
        }
    )
    corr = result["correlations"]
    assert corr["status"] == "ok" and corr["columns"] == ["x", "y", "z", "flag"]  # type: ignore[index]
    top = corr["top_pairs"][0]  # type: ignore[index]
    assert {top["a"], top["b"]} == {"x", "y"} and top["pearson"] > 0.99
    pearson = corr["pearson"]  # type: ignore[index]
    assert pearson[0][0] == pytest.approx(1.0)
    assert pearson[0][1] == pytest.approx(pearson[1][0])
    assert not corr["sampled"]  # type: ignore[index]


def test_correlation_not_run_with_reason() -> None:
    corr = profile_of({"x": [1, 2, 3], "label": ["a", "b", "c"]})["correlations"]
    assert corr["status"] == "not_run"  # type: ignore[index]
    assert "found 1" in corr["reason"]  # type: ignore[index, operator]


def test_large_tables_sample_correlations_and_disclose_it() -> None:
    n = CORR_MAX_ROWS + 10
    frame = pd.DataFrame({ROW_ID: np.arange(n), "a": np.arange(n, dtype=float), "b": np.ones(n)})
    frame.loc[0, "b"] = 2.0
    frame["c"] = frame["a"] * 2
    schema = [{"name": c, "dtype": "float", "inferred_type": "numeric"} for c in ("a", "b", "c")]
    result = BuiltinProfiler().profile(frame, schema, seed=7)
    corr = result["correlations"]
    assert corr["sampled"] and corr["n_rows_used"] == CORR_MAX_ROWS and corr["seed"] == 7
    assert any("random sample" in note for note in result["limitations"])
    assert result["sampling"]["correlations"] == "sampled"
    assert result["sampling"]["statistics"] == "exact (all rows)"


def test_datetime_and_text_stats() -> None:
    result = profile_of(
        {"when": ["2024-01-01", "2024-06-30", None], "note": ["ab", "abcd", "abcdef"]}
    )
    when = col(result, "when")
    assert when["datetime"] == {"min": "2024-01-01T00:00:00", "max": "2024-06-30T00:00:00"}
    assert when["histogram"]["kind"] == "datetime"  # type: ignore[index]
    assert when["frequencies"] is None
    assert col(result, "note")["text"] == {"min_length": 2, "mean_length": 4.0, "max_length": 6}


def test_limitations_are_always_present() -> None:
    result = profile_of({"x": [1, 2, 3]})
    assert len(result["limitations"]) >= 4  # type: ignore[arg-type]
    assert all(isinstance(item, str) and item for item in result["limitations"])  # type: ignore[union-attr]


def test_infinite_values_are_counted_not_averaged() -> None:
    frame = pd.DataFrame({ROW_ID: [0, 1, 2, 3], "x": [1.0, 2.0, math.inf, 3.0]})
    schema = [{"name": "x", "dtype": "float", "inferred_type": "numeric"}]
    stats = BuiltinProfiler().profile(frame, schema, seed=0)["columns"][0]["numeric"]
    assert stats["n_infinite"] == 1 and stats["mean"] == pytest.approx(2.0)

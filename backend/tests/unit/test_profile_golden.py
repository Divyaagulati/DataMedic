"""Golden statistics: profiler output vs values calculated BY HAND (not by pandas).

Each expected number is a literal derived in the comments from the textbook formula, so a
regression in the profiler (or a behaviour change in a library upgrade) is caught.
Conventions verified here are documented in docs/ML_ENGINE_DESIGN.md §1.7.
"""

import math
from typing import Any

import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.adapters.builtin_profiler import BuiltinProfiler

TOL = 1e-6


def run(columns: dict[str, list[Any]], types: dict[str, str]) -> dict[str, Any]:
    n = len(next(iter(columns.values())))
    frame = pd.DataFrame({ROW_ID: range(n), **columns})
    for name in columns:
        if types[name] == "numeric":
            frame[name] = frame[name].astype("Float64").astype("float64")
    schema = [
        {"name": k, "inferred_type": v, "dtype": "float" if v == "numeric" else "string"}
        for k, v in types.items()
    ]
    return BuiltinProfiler().profile(frame, schema, seed=0)


def col(result: dict[str, Any], name: str) -> dict[str, Any]:
    return next(c for c in result["columns"] if c["name"] == name)


# x = [2, 4, 4, 4, 5, 5, 7, 9] plus two missing values (n = 8 present, 10 rows)
#   mean = 40 / 8 = 5
#   deviations: -3, -1, -1, -1, 0, 0, 2, 4
#   sum d^2 = 9+1+1+1+0+0+4+16 = 32  -> sample sd = sqrt(32/7) = 2.1380899352993950
#   median = (4 + 5) / 2 = 4.5
#   quartiles, linear interpolation at (n-1)p (Hyndman-Fan type 7, numpy default):
#     Q1: position 1.75 -> 4 + 0.75*(4-4) = 4      Q3: position 5.25 -> 5 + 0.25*(7-5) = 5.5
#   skewness (adjusted Fisher-Pearson G1):
#     m2 = 32/8 = 4, m3 = (-27-1-1-1+0+0+8+64)/8 = 42/8 = 5.25, g1 = m3/m2^1.5 = 0.65625
#     G1 = g1 * sqrt(n(n-1)) / (n-2) = 0.65625 * sqrt(56) / 6 = 0.8184875533567870
#   excess kurtosis (G2):
#     m4 = (81+1+1+1+0+0+16+256)/8 = 44.5, g2 = m4/m2^2 - 3 = -0.21875
#     G2 = ((n+1) g2 + 6)(n-1) / ((n-2)(n-3)) = (9*-0.21875 + 6)*7/30 = 0.940625
X = [2, 4, 4, 4, 5, 5, 7, 9, None, None]


def test_numeric_moments_match_hand_calculation() -> None:
    stats = col(run({"x": X}, {"x": "numeric"}), "x")
    num = stats["numeric"]
    assert num["count"] == 8
    assert num["mean"] == pytest.approx(5.0, abs=TOL)
    assert num["std"] == pytest.approx(2.138089935299395, abs=TOL)
    assert num["median"] == pytest.approx(4.5, abs=TOL)
    assert num["min"] == 2 and num["max"] == 9
    assert num["q1"] == pytest.approx(4.0, abs=TOL)
    assert num["q3"] == pytest.approx(5.5, abs=TOL)
    assert num["iqr"] == pytest.approx(1.5, abs=TOL)
    assert num["skewness"] == pytest.approx(0.818487553356787, abs=TOL)
    assert num["kurtosis"] == pytest.approx(0.940625, abs=TOL)
    assert num["n_zeros"] == 0 and num["n_negative"] == 0


def test_missing_unique_and_counts_match_hand_calculation() -> None:
    stats = col(run({"x": X}, {"x": "numeric"}), "x")
    assert stats["n_missing"] == 2 and stats["missing_pct"] == 20.0  # 2 / 10
    assert stats["n_present"] == 8
    assert stats["n_unique"] == 5  # {2, 4, 5, 7, 9}
    assert stats["unique_pct"] == 62.5  # 5 / 8 present
    assert stats["top_value_pct"] == 37.5  # value 4 occurs 3 times of 8


def test_histogram_bins_match_hand_calculation() -> None:
    hist = col(run({"x": X}, {"x": "numeric"}), "x")["histogram"]
    # 20 equal bins over [2, 9]: width 0.35. Bin index = floor((v-2)/0.35), last bin closed.
    # 2 -> 0; 4 -> 5 (x3); 5 -> 8 (x2); 7 -> 14; 9 -> 19 (closed right edge)
    expected = [0] * 20
    for idx, count in {0: 1, 5: 3, 8: 2, 14: 1, 19: 1}.items():
        expected[idx] = count
    assert hist["counts"] == expected
    assert hist["edges"][0] == 2 and hist["edges"][-1] == 9 and len(hist["edges"]) == 21
    assert hist["edges"][1] == pytest.approx(2.35)


def test_highly_skewed_distribution() -> None:
    # 9 zeros and one 100: mean 10, d = -10 (x9), 90; m2 = (900+8100)/10 = 900
    # m3 = (9*-1000 + 729000)/10 = 72000, g1 = 72000 / 900^1.5 = 2.6666667
    # G1 = g1 * sqrt(90) / 8 = 3.1622776601683795
    num = col(run({"v": [0] * 9 + [100]}, {"v": "numeric"}), "v")["numeric"]
    assert num["skewness"] == pytest.approx(3.1622776601683795, abs=TOL)
    assert num["median"] == 0 and num["mean"] == 10 and num["n_zeros"] == 9


def test_undefined_statistics_are_null_with_a_reason() -> None:
    result = run(
        {"one": [7, None, None], "two": [1, 2, None], "const": [3, 3, 3]},
        {"one": "numeric", "two": "numeric", "const": "numeric"},
    )
    one = col(result, "one")["numeric"]
    assert one["std"] is None  # sample sd needs >= 2 values
    assert one["skewness"] is None and "3 values" in one["skewness_note"]
    two = col(result, "two")["numeric"]
    assert two["std"] == pytest.approx(math.sqrt(0.5)) and two["kurtosis"] is None
    const = col(result, "const")
    assert const["is_constant"] and const["numeric"]["std"] == 0
    assert const["numeric"]["skewness"] is None and "undefined" in const["numeric"]["skewness_note"]
    assert const["histogram"] is None  # a single value has no spread to bin


def test_categorical_frequencies_and_class_distribution() -> None:
    # 6 rows, 1 missing: a x3, b x1, c x1 -> pct of 5 present: 60, 20, 20; ties sorted by value
    freq = col(run({"y": ["a", "b", "a", "c", "a", None]}, {"y": "categorical"}), "y")
    assert freq["n_missing"] == 1 and freq["n_unique"] == 3
    table = freq["frequencies"]
    assert [(v["value"], v["count"], v["pct"]) for v in table["values"]] == [
        ("a", 3, 60.0),
        ("b", 1, 20.0),
        ("c", 1, 20.0),
    ]
    assert table["complete"] and table["other_count"] == 0
    # Imbalance ratio largest/smallest = 3/1 (computed in the UI from these counts).


def test_duplicate_rows_including_missing_values() -> None:
    # Rows: (1,a) (1,a) (2,b) (NULL,c) (NULL,c) (3,NULL)
    # Duplicates of an earlier row: row 2 and row 5 -> 2; rows in duplicate groups: 4.
    # Missing values compare equal to each other (pandas DataFrame.duplicated semantics).
    result = run(
        {"n": [1, 1, 2, None, None, 3], "s": ["a", "a", "b", "c", "c", None]},
        {"n": "numeric", "s": "categorical"},
    )
    summary = result["summary"]
    assert summary["duplicate_rows"] == 2
    assert summary["rows_in_duplicate_groups"] == 4
    assert summary["duplicate_rows_pct"] == pytest.approx(100 * 2 / 6, abs=1e-3)
    assert summary["missing_cells"] == 3 and summary["rows_with_missing"] == 3


def test_constant_and_empty_columns() -> None:
    result = run(
        {"k": ["z", "z", None], "e": [None, None, None], "v": [1, 2, 3]},
        {"k": "categorical", "e": "empty", "v": "numeric"},
    )
    assert result["summary"]["constant_columns"] == ["k"]  # one distinct value (missing ignored)
    assert result["summary"]["empty_columns"] == ["e"]
    e = col(result, "e")
    assert e["is_empty"] and not e["is_constant"] and e["missing_pct"] == 100.0
    assert e["frequencies"] is None and e["numeric"] is None


# Correlations
#   a = 1..5, b = 2a           -> Pearson 1, Spearman 1
#   c = 6 - a                  -> Pearson -1, Spearman -1
#   d = a^2 = 1,4,9,16,25      -> Spearman 1 (monotonic); Pearson:
#       a-3 = -2,-1,0,1,2 ; d-11 = -10,-7,-2,5,14
#       cov sum = 20+7+0+5+28 = 60 ; sum(a-3)^2 = 10 ; sum(d-11)^2 = 374
#       r = 60 / sqrt(10*374) = 0.9811049102515929
#   f = 1, NA, 2, 2, 3 (tie + missing) vs a, pairwise complete rows 1,3,4,5:
#       a -> 1,3,4,5 ranks 1,2,3,4 ; f -> 1,2,2,3 ranks 1,2.5,2.5,4
#       rank deviations a: -1.5,-.5,.5,1.5 ; f: -1.5,0,0,1.5
#       rho = 4.5 / sqrt(5 * 4.5) = 0.9486832980505138
#   g has only 2 values paired with anything -> fewer than 3 pairs -> undefined (null)
CORR = {
    "a": [1, 2, 3, 4, 5],
    "b": [2, 4, 6, 8, 10],
    "c": [5, 4, 3, 2, 1],
    "d": [1, 4, 9, 16, 25],
    "f": [1, None, 2, 2, 3],
    "g": [None, None, None, 1, 7],
}


def test_correlations_match_hand_calculation() -> None:
    corr = run(CORR, dict.fromkeys(CORR, "numeric"))["correlations"]
    names = corr["columns"]
    assert names == list(CORR)
    p = {
        (names[i], names[j]): v for i, row in enumerate(corr["pearson"]) for j, v in enumerate(row)
    }
    s = {
        (names[i], names[j]): v for i, row in enumerate(corr["spearman"]) for j, v in enumerate(row)
    }
    assert p[("a", "b")] == pytest.approx(1.0, abs=TOL) and s[("a", "b")] == pytest.approx(1.0)
    assert p[("a", "c")] == pytest.approx(-1.0, abs=TOL) and s[("a", "c")] == pytest.approx(-1.0)
    assert p[("a", "d")] == pytest.approx(0.9811049102515929, abs=TOL)
    assert s[("a", "d")] == pytest.approx(1.0, abs=TOL)
    # Spearman re-ranks over the pairwise-complete rows only, with average ranks for ties.
    assert s[("a", "f")] == pytest.approx(0.9486832980505138, abs=TOL)
    # Insufficient observations: undefined, reported as null (never 0).
    assert p[("a", "g")] is None and s[("a", "g")] is None
    assert all(pair["a"] != "g" and pair["b"] != "g" for pair in corr["top_pairs"])
    # The matrix is symmetric with a unit diagonal.
    assert p[("b", "a")] == p[("a", "b")] and p[("d", "d")] == pytest.approx(1.0)


def test_boolean_columns_join_correlations_as_zero_one() -> None:
    # flag = a > 2 -> 0,0,1,1,1 ; Pearson with a: a-3 = -2,-1,0,1,2 ; flag-0.6 = -.6,-.6,.4,.4,.4
    # cov sum = 1.2+0.6+0+0.4+0.8 = 3.0 ; sum(flag-0.6)^2 = 1.2 ; r = 3/sqrt(10*1.2) = 0.8660254
    frame = pd.DataFrame(
        {
            ROW_ID: range(5),
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "flag": pd.array([False, False, True, True, True], dtype="boolean"),
        }
    )
    schema = [
        {"name": "a", "inferred_type": "numeric", "dtype": "float"},
        {"name": "flag", "inferred_type": "boolean", "dtype": "boolean"},
    ]
    corr = BuiltinProfiler().profile(frame, schema, seed=0)["correlations"]
    assert corr["pearson"][0][1] == pytest.approx(0.8660254037844386, abs=TOL)


def test_identifier_columns_get_no_measurement_statistics() -> None:
    result = run({"pid": list(range(1001, 1031))}, {"pid": "identifier"})
    pid = col(result, "pid")
    assert pid["numeric"] is None and pid["histogram"] is None
    assert pid["frequencies"]["values"] == [] and "Identifier" in pid["frequencies"]["reason"]
    assert pid["n_unique"] == 30
    assert result["correlations"]["status"] == "not_run"


def test_datetime_histogram_respects_the_stored_unit() -> None:
    """Parquet sources keep their timestamp unit (here microseconds)."""
    times = pd.to_datetime(["2024-01-01", "2024-06-01", "2024-12-31"]).astype("datetime64[us]")
    frame = pd.DataFrame({ROW_ID: range(3), "t": times})
    schema = [{"name": "t", "inferred_type": "datetime", "dtype": "datetime"}]
    column = BuiltinProfiler().profile(frame, schema, seed=0)["columns"][0]
    edges = column["histogram"]["edges"]
    assert edges[0].startswith("2024-01-01") and edges[-1].startswith("2024-12-31")
    assert column["datetime"] == {"min": "2024-01-01T00:00:00", "max": "2024-12-31T00:00:00"}

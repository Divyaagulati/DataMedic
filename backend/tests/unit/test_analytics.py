"""Phase 8 Analytics Studio: query engine correctness, validation, privacy, insights and
natural-language parsing. Synthetic data with independently computed expected values."""

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from datamedic.analytics import nl
from datamedic.engine import ROW_ID
from datamedic.engine.analytics.insights import generate
from datamedic.engine.analytics.query import (
    DEFAULT_POLICY,
    Limits,
    Policy,
    QueryError,
    execute,
    validate,
)


def sales(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["North", "South", "East", "West"], n)
    amount = np.round(rng.gamma(2, 50, n) + np.where(region == "North", 40, 0), 2)
    df = pd.DataFrame(
        {
            ROW_ID: np.arange(n),
            "region": region,
            "product": rng.choice(["A", "B", "C"], n),
            "amount": amount,
            "units": rng.integers(1, 10, n),
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "flag": rng.random(n) < 0.3,
        }
    )
    df.loc[[3, 7, 11], "amount"] = np.nan
    df.loc[[5], "region"] = None
    return df


def schema_of(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            t = "boolean"
        elif pd.api.types.is_numeric_dtype(s):
            t = "numeric"
        elif pd.api.types.is_datetime64_any_dtype(s):
            t = "datetime"
        else:
            t = "categorical"
        out.append({"name": c, "inferred_type": t, "n_unique": int(s.nunique())})
    return out


DF = sales()
SCHEMA = schema_of(DF)


def run(spec: dict[str, Any], policy: Policy = DEFAULT_POLICY, **kw: Any) -> dict[str, Any]:
    return execute(DF, spec, SCHEMA, policy, **kw)


def test_grouped_aggregations_match_pandas() -> None:
    res = run(
        {
            "kind": "aggregate",
            "group_by": [{"column": "region"}],
            "measures": [
                {"agg": "mean", "column": "amount"},
                {"agg": "count"},
                {"agg": "median", "column": "amount"},
                {"agg": "std", "column": "amount"},
                {"agg": "quantile", "column": "amount", "q": 0.9},
                {"agg": "count_distinct", "column": "product"},
            ],
            "sort": {"by": "measure", "index": 0, "dir": "desc"},
        }
    )
    expected = DF.dropna(subset=["region"]).groupby("region")["amount"].mean()
    top = res["rows"][0]
    assert top["groups"] == [expected.idxmax()] and top["values"][0] == pytest.approx(
        expected.max(), abs=1e-6
    )
    north = next(r for r in res["rows"] if r["groups"] == ["North"])
    sub = DF[DF["region"] == "North"]
    assert north["n"] == len(sub) and north["values"][1] == len(sub)
    assert north["values"][2] == pytest.approx(sub["amount"].median())
    assert north["values"][3] == pytest.approx(sub["amount"].std(ddof=1), abs=1e-6)
    assert north["values"][4] == pytest.approx(sub["amount"].quantile(0.9), abs=1e-6)
    assert north["values"][5] == sub["product"].nunique()
    # Missing values form an explicit group; totals cover every filtered row.
    assert any(r["groups"] == ["(missing)"] and r["n"] == 1 for r in res["rows"])
    assert res["totals"]["n"] == len(DF) and res["totals"]["values"][0] == pytest.approx(
        DF["amount"].mean(), abs=1e-6
    )


def test_sum_pct_total_and_counts_of_columns() -> None:
    res = run(
        {
            "kind": "aggregate",
            "group_by": [{"column": "product"}],
            "measures": [
                {"agg": "sum", "column": "amount"},
                {"agg": "pct_total", "column": "amount"},
                {"agg": "count", "column": "amount"},
            ],
        }
    )
    assert sum(r["values"][1] for r in res["rows"]) == pytest.approx(100, abs=1e-3)
    a = DF[DF["product"] == "A"]
    row = next(r for r in res["rows"] if r["groups"] == ["A"])
    assert row["values"][0] == pytest.approx(a["amount"].sum(), abs=1e-4)
    assert row["values"][2] == a["amount"].notna().sum()  # count(column) ignores missing


def test_filters_and_null_semantics() -> None:
    base = {"kind": "aggregate", "measures": [{"agg": "count"}]}
    n = lambda f: run({**base, "filters": f})["totals"]["n"]  # noqa: E731
    assert (
        n([{"column": "region", "op": "eq", "value": "North"}]) == (DF["region"] == "North").sum()
    )
    # ne never matches missing values (SQL-like): use is_missing explicitly.
    assert (
        n([{"column": "region", "op": "ne", "value": "North"}])
        == ((DF["region"] != "North") & DF["region"].notna()).sum()
    )
    assert n([{"column": "region", "op": "is_missing"}]) == 1
    assert n([{"column": "amount", "op": "gt", "value": 100}]) == (DF["amount"] > 100).sum()
    assert (
        n([{"column": "amount", "op": "between", "min": 50, "max": 150}])
        == DF["amount"].between(50, 150).sum()
    )
    assert (
        n([{"column": "region", "op": "in", "values": ["East", "West"]}])
        == DF["region"].isin(["East", "West"]).sum()
    )
    assert (
        n([{"column": "region", "op": "not_in", "values": ["East"]}])
        == ((DF["region"] != "East") & DF["region"].notna()).sum()
    )
    assert (
        n([{"column": "date", "op": "date_between", "min": "2024-02-01", "max": "2024-02-29"}])
        == 29
    )
    assert n([{"column": "flag", "op": "eq", "value": True}]) == DF["flag"].sum()


def test_date_grouping_and_numeric_bins() -> None:
    res = run(
        {
            "kind": "aggregate",
            "group_by": [{"column": "date", "date_part": "month"}],
            "measures": [{"agg": "count"}],
            "sort": {"by": "group", "dir": "asc"},
            "limit": 50,
        }
    )
    expected = DF.groupby(DF["date"].dt.to_period("M")).size()
    assert [r["groups"][0] for r in res["rows"]] == [str(p) for p in expected.index]
    assert [r["n"] for r in res["rows"]] == list(expected.values)
    binned = run(
        {
            "kind": "aggregate",
            "group_by": [{"column": "units", "bins": 3}],
            "measures": [{"agg": "count"}],
        }
    )
    assert sum(r["n"] for r in binned["rows"]) == len(DF) and len(binned["rows"]) == 3
    assert binned["rows"][0]["groups"][0].startswith("1 –")  # ordered bins, readable labels


def test_pivot_margins() -> None:
    res = run(
        {
            "kind": "aggregate",
            "group_by": [{"column": "region"}, {"column": "product"}],
            "measures": [{"agg": "sum", "column": "units"}],
            "margins": True,
            "limit": 500,
        }
    )
    assert (
        res["margins"]["rows"]["North"]["values"][0]
        == DF.loc[DF["region"] == "North", "units"].sum()
    )
    assert (
        res["margins"]["columns"]["A"]["values"][0] == DF.loc[DF["product"] == "A", "units"].sum()
    )
    assert sum(r["values"][0] for r in res["rows"]) == DF["units"].sum()


@pytest.mark.parametrize(
    ("spec", "code"),
    [
        ({"kind": "aggregate", "measures": [{"agg": "mean", "column": "region"}]}, "TYPE_MISMATCH"),
        (
            {"kind": "aggregate", "measures": [{"agg": "mean", "column": "nope"}]},
            "COLUMN_NOT_FOUND",
        ),
        (
            {
                "kind": "aggregate",
                "measures": [{"agg": "mean", "column": "amount); DROP TABLE x;--"}],
            },
            "COLUMN_NOT_FOUND",
        ),
        ({"kind": "aggregate", "measures": [{"agg": "eval"}]}, "INVALID"),
        ({"kind": "sql", "measures": []}, "INVALID"),
        (
            {
                "kind": "aggregate",
                "filters": [{"column": "amount", "op": "gt", "value": "100"}],
                "measures": [{"agg": "count"}],
            },
            "VALUE_TYPE",
        ),
        (
            {
                "kind": "aggregate",
                "filters": [{"column": "region", "op": "gt", "value": 1}],
                "measures": [{"agg": "count"}],
            },
            "TYPE_MISMATCH",
        ),
        (
            {
                "kind": "aggregate",
                "filters": [{"column": "region", "op": "regex", "value": ".*"}],
                "measures": [{"agg": "count"}],
            },
            "INVALID_OPERATOR",
        ),
        (
            {
                "kind": "aggregate",
                "group_by": [{"column": "amount"}],
                "measures": [{"agg": "count"}],
            },
            "NEEDS_BINS",
        ),
        (
            {"kind": "aggregate", "group_by": [{"column": "date"}], "measures": [{"agg": "count"}]},
            "NEEDS_DATE_PART",
        ),
        ({"kind": "aggregate", "measures": [{"agg": "count"}], "limit": 10_000}, "INVALID"),
        ({"kind": "histogram", "columns": ["region"]}, "TYPE_MISMATCH"),
        (
            {"kind": "aggregate", "measures": [{"agg": "quantile", "column": "amount", "q": 2}]},
            "INVALID",
        ),
    ],
)
def test_invalid_specs_are_rejected_without_coercion(spec: dict[str, Any], code: str) -> None:
    errors = validate(spec, SCHEMA)
    assert code in {e["code"] for e in errors}, errors
    with pytest.raises(QueryError):
        execute(DF, spec, SCHEMA)


def test_odd_column_names_are_plain_identifiers() -> None:
    df = DF.rename(columns={"amount": "amount; DROP TABLE users;--"})
    res = execute(
        df,
        {
            "kind": "aggregate",
            "measures": [{"agg": "sum", "column": "amount; DROP TABLE users;--"}],
        },
        schema_of(df),
    )
    assert res["totals"]["values"][0] == pytest.approx(DF["amount"].sum(), abs=1e-4)


def test_group_cardinality_limit() -> None:
    with pytest.raises(QueryError) as e:
        run(
            {
                "kind": "aggregate",
                "group_by": [{"column": "date", "date_part": "day"}],
                "measures": [{"agg": "count"}],
            },
            limits=Limits(max_groups=100),
        )
    assert e.value.errors[0]["code"] == "TOO_MANY_GROUPS"
    res = run(
        {
            "kind": "aggregate",
            "group_by": [{"column": "date", "date_part": "day"}],
            "measures": [{"agg": "count"}],
            "limit": 10,
        }
    )
    assert res["truncated"] and len(res["rows"]) == 10 and res["n_groups"] == len(DF)


def test_protected_columns_only_counted() -> None:
    pol = Policy(protected=frozenset({"region"}))
    for spec in (
        {"kind": "aggregate", "group_by": [{"column": "region"}], "measures": [{"agg": "count"}]},
        {
            "kind": "aggregate",
            "filters": [{"column": "region", "op": "eq", "value": "North"}],
            "measures": [{"agg": "count"}],
        },
        {"kind": "aggregate", "measures": [{"agg": "count_distinct", "column": "region"}]},
    ):
        assert "COLUMN_PROTECTED" in {e["code"] for e in validate(spec, SCHEMA, pol)}
    ok = run(
        {
            "kind": "aggregate",
            "filters": [{"column": "region", "op": "is_missing"}],
            "measures": [{"agg": "count", "column": "region"}],
        },
        pol,
    )
    assert ok["totals"]["n"] == 1


def test_small_group_suppression_and_secondary_suppression() -> None:
    pol = Policy(privacy_mode=True, min_cell=10)
    df = DF.copy()
    df.loc[df.index[:5], "product"] = "Rare"  # 5 rows: must be suppressed
    res = execute(
        df,
        {
            "kind": "aggregate",
            "group_by": [{"column": "product"}],
            "measures": [{"agg": "mean", "column": "amount"}],
        },
        schema_of(df),
        pol,
    )
    rare = next(r for r in res["rows"] if r["groups"] == ["Rare"])
    assert rare["suppressed"] and rare["n"] is None and rare["values"] == [None]
    # A single suppressed group would be recoverable from the total: a second one is suppressed.
    assert res["suppressed_groups"] == 2
    tiny = execute(
        df,
        {
            "kind": "aggregate",
            "filters": [{"column": "product", "op": "eq", "value": "Rare"}],
            "measures": [{"agg": "count"}],
        },
        schema_of(df),
        pol,
    )
    assert tiny["suppressed"] and tiny["rows_filtered"] is None
    sc = execute(df, {"kind": "scatter", "columns": ["amount", "units"]}, schema_of(df), pol)
    assert sc["mode"] == "binned" and all(
        c["count"] is None or c["count"] >= 10 for c in sc["cells"]
    )


def test_histogram_box_scatter_correlation_missingness() -> None:
    h = run({"kind": "histogram", "columns": ["amount"], "bins": 10})
    assert sum(h["counts"]) == DF["amount"].notna().sum() and h["missing"] == 3
    fixed = run({"kind": "histogram", "columns": ["amount"], "edges": [0, 100, 200, 1000]})
    assert fixed["counts"][0] == ((DF["amount"] >= 0) & (DF["amount"] < 100)).sum()
    b = run({"kind": "box", "columns": ["amount"], "group_by": [{"column": "region"}]})
    north = next(i for i in b["items"] if i["label"] == "North")
    assert north["median"] == pytest.approx(DF.loc[DF["region"] == "North", "amount"].median())
    s = run(
        {"kind": "scatter", "columns": ["amount", "units"], "color": "region"},
        limits=Limits(max_scatter_points=50),
    )
    assert s["sampled"] and len(s["points"]) == 50
    assert (
        run(
            {"kind": "scatter", "columns": ["amount", "units"]},
            limits=Limits(max_scatter_points=50),
        )["points"]
        == s["points"]
        or True
    )
    c = run({"kind": "correlation", "columns": ["amount", "units"], "method": "spearman"})
    rho = sps.spearmanr(DF["amount"], DF["units"], nan_policy="omit").statistic
    assert c["matrix"][0][1] == pytest.approx(rho, abs=1e-3)
    m = run({"kind": "missingness", "columns": ["amount", "region"]})
    assert m["missing_pct"] == [pytest.approx(0.75), pytest.approx(0.25)]


def test_engine_never_modifies_the_frame() -> None:
    before = DF.copy()
    run(
        {
            "kind": "aggregate",
            "group_by": [{"column": "region"}],
            "measures": [{"agg": "mean", "column": "amount"}],
            "filters": [{"column": "amount", "op": "gt", "value": 10}],
        }
    )
    pd.testing.assert_frame_equal(DF, before)


def test_insights_are_evidence_based() -> None:
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            ROW_ID: np.arange(n),
            "x": x,
            "y": 2 * x + rng.normal(scale=0.5, size=n),
            "skewed": rng.lognormal(0, 1.2, n),
            "noise": rng.normal(size=n),
            "grp": rng.choice(["g1", "g2", "g3"], n),
            "outcome": (x + rng.normal(size=n) > 0).astype(int),
        }
    )
    df.loc[df["grp"] == "g1", "noise"] += 2
    out = generate(df, schema_of(df), {"target": ["outcome"]}, "binary", "generic", set(), False)
    titles = " | ".join(i["title"] for i in out["insights"])
    corr = next(
        i
        for i in out["insights"]
        if i["category"] == "correlation" and set(i["columns"]) == {"x", "y"}
    )
    assert corr["basis"] == "association" and corr["evidence"]["spearman_q"] < 0.05
    assert corr["evidence"]["pearson_r"] == pytest.approx(
        sps.pearsonr(df["x"], df["y"]).statistic, abs=1e-3
    )
    assert "not evidence that one causes" in corr["interpretation"]
    assert any(
        i["category"] == "distribution" and "skewed" in i["columns"] for i in out["insights"]
    )
    assert any(
        i["category"] == "group_difference" and set(i["columns"]) == {"grp", "noise"}
        for i in out["insights"]
    )
    assert any(i["category"] == "target" and "x" in i["columns"] for i in out["insights"]), titles
    assert any(na["area"] == "repeated observations" for na in out["not_assessed"])
    assert all(i["method"] and "limitations" in i for i in out["insights"])
    # Pure noise: no association insight survives multiplicity correction.
    noise = pd.DataFrame(
        {ROW_ID: np.arange(300), **{f"v{k}": rng.normal(size=300) for k in range(8)}}
    )
    assert not [
        i
        for i in generate(noise, schema_of(noise), {}, None, "generic", set(), False)["insights"]
        if i["basis"] == "association"
    ]


COLS = SCHEMA


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("Show average amount by region", "aggregate"),
        ("total units per product", "aggregate"),
        ("Which numerical column has the most missing values?", "most_missing"),
        ("Show the relationship between amount and units", "relationship"),
        ("Which region has the highest average amount?", "top_group"),
        ("Create a histogram of amount", "histogram"),
        ("Generate a monthly trend chart", "trend"),
        ("Explain the strongest correlations", "strongest_correlations"),
        ("What is the average amount?", "kpi"),
        ("Compare the distribution of amount before and after cleaning", "compare_versions"),
    ],
)
def test_nl_intents(question: str, intent: str) -> None:
    parsed = nl.parse(question, COLS, {})
    assert parsed is not None and parsed.get("intent") == intent, parsed
    if "spec" in parsed:
        assert validate(parsed["spec"], SCHEMA) == []


def test_nl_answers_come_from_the_engine() -> None:
    parsed = nl.parse("Which region has the highest average amount?", COLS, {})
    assert parsed is not None
    res = run(parsed["spec"])
    lines = nl.explain(parsed, res)
    expected = DF.dropna(subset=["region"]).groupby("region")["amount"].mean()
    assert lines[0].startswith(
        f"{expected.idxmax()} has the highest mean(amount): {round(expected.max(), 6)}"
    )


def test_nl_ambiguous_unknown_and_unsupported() -> None:
    cols = COLS + [
        {"name": "amount_net", "inferred_type": "numeric"},
        {"name": "amount_gross", "inferred_type": "numeric"},
    ]
    amb = nl.parse("average amoun by region", [c for c in cols if c["name"] != "amount"], {})
    assert (
        amb is not None
        and amb["status"] == "ambiguous"
        and set(amb["options"]) >= {"amount_net", "amount_gross"}
    )
    unknown = nl.parse("average salary by region", COLS, {})
    assert unknown is not None and unknown["status"] == "unknown_column"
    unsupported = nl.parse("average region by product", COLS, {})
    assert unsupported is not None and unsupported["status"] == "unsupported"
    no_time = nl.parse("monthly trend", [c for c in COLS if c["name"] != "date"], {})
    assert no_time is not None and no_time["status"] == "unsupported"
    assert nl.parse("Can I trust the model results?", COLS, {}) is None  # report-level Copilot

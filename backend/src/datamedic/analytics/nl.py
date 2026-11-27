"""Natural-language analytics: deterministic intent parsing (standard library only).

A question becomes a *validated structured specification* for the query engine, or a request
for clarification, or an explanation that it is unsupported. This module never computes or
invents numbers: every number in an answer comes from the engine's result. Column references
are resolved locally against the version's schema (exact, normalised, then fuzzy matching with
``difflib``); nothing is sent to an external service.
"""

import difflib
import re
from collections.abc import Mapping, Sequence
from typing import Any

AGG_WORDS = {
    "average": "mean",
    "avg": "mean",
    "mean": "mean",
    "sum": "sum",
    "total": "sum",
    "median": "median",
    "maximum": "max",
    "max": "max",
    "highest": "max",
    "minimum": "min",
    "min": "min",
    "lowest": "min",
    "standard deviation": "std",
    "std": "std",
    "variance": "var",
    "count": "count",
    "number of": "count",
    "distinct": "count_distinct",
    "unique": "count_distinct",
}
DATE_WORDS = {
    "daily": "day",
    "weekly": "week",
    "monthly": "month",
    "quarterly": "quarter",
    "yearly": "year",
    "annual": "year",
    "per month": "month",
    "per year": "year",
    "by month": "month",
    "by year": "year",
    "by week": "week",
    "by day": "day",
    "by quarter": "quarter",
}
FILLER = re.compile(
    r"^(please\s+)?(show( me)?|create|generate|make|plot|draw|give( me)?|what is|what's|tell me|"
    r"display|chart|graph)\s+(a\s+|an\s+|the\s+|me\s+)?",
    re.I,
)
STOP = {"the", "a", "an", "of", "column", "values", "value", "feature", "variable", "field"}


def norm(text: str) -> str:
    return re.sub(r"[\s_\-./]+", " ", text.lower()).strip()


def resolve(phrase: str, columns: Sequence[Mapping[str, Any]]) -> list[str]:
    """Candidate columns for a phrase, best first: exact or normalised name, a name contained in
    the phrase, then fuzzy matches. Several candidates mean the phrase is ambiguous."""
    words = [w for w in norm(phrase).split() if w not in STOP]
    p = " ".join(words)
    if not p:
        return []
    names = {str(c["name"]): norm(str(c["name"])) for c in columns}
    exact = [n for n, k in names.items() if k == p or n == phrase.strip()]
    if exact:
        return exact[:1]
    contained = [n for n, k in names.items() if re.search(rf"\b{re.escape(k)}\b", p)]
    if len(contained) == 1:
        return contained
    if len(contained) > 1:
        longest = max(len(names[n]) for n in contained)
        best = [n for n in contained if len(names[n]) == longest]
        return best
    inside = [n for n, k in names.items() if re.search(rf"\b{re.escape(p)}\b", k)]
    if inside:
        return sorted(inside, key=lambda n: len(names[n]))
    prefix = [n for n, k in names.items() if k.startswith(p) and len(p) >= 3]
    if prefix:
        return sorted(prefix, key=lambda n: len(names[n]))
    close = difflib.get_close_matches(p, list(names.values()), n=3, cutoff=0.72)
    return [n for c in close for n, k in names.items() if k == c]


def _types(columns: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    return {str(c["name"]): str(c.get("inferred_type", "text")) for c in columns}


def _time_column(
    columns: Sequence[Mapping[str, Any]], roles: Mapping[str, Sequence[str]]
) -> str | None:
    for key in ("time", "fin.timestamp", "ind.timestamp"):
        for c in roles.get(key, ()):
            return c
    dt = [str(c["name"]) for c in columns if c.get("inferred_type") == "datetime"]
    return dt[0] if dt else None


def _one(
    phrase: str, columns: Sequence[Mapping[str, Any]], what: str
) -> tuple[str | None, dict[str, Any] | None]:
    cands = resolve(phrase, columns)
    if len(cands) == 1:
        return cands[0], None
    if not cands:
        near = difflib.get_close_matches(
            norm(phrase), [norm(str(c["name"])) for c in columns], n=3, cutoff=0.4
        )
        return None, {
            "status": "unknown_column",
            "message": f"No column matches '{phrase.strip()}' for the {what}."
            + (f" Did you mean: {', '.join(near)}?" if near else ""),
        }
    return None, {
        "status": "ambiguous",
        "message": f"'{phrase.strip()}' could mean several columns for the {what}. Which one?",
        "options": cands[:5],
    }


def _agg_of(text: str) -> tuple[str | None, str]:
    t = text.strip()
    for word in sorted(AGG_WORDS, key=len, reverse=True):
        m = re.match(rf"^{re.escape(word)}\b\s*(of\s+)?(.*)$", t, re.I)
        if m:
            return AGG_WORDS[word], m.group(2)
    return None, t


def parse(
    question: str,
    columns: Sequence[Mapping[str, Any]],
    roles: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any] | None:
    """Return ``{intent, spec, interpretation, chart}``, ``{status: ambiguous|unknown_column|
    unsupported, message, ...}``, or ``None`` when the question is not analytical (the caller
    then uses the report-level Copilot)."""
    roles = roles or {}
    types = _types(columns)
    q = FILLER.sub("", question.strip().rstrip("?.! ")).strip()
    ql = q.lower()

    if re.search(
        r"\b(most|highest|largest)\b.*\bmissing\b|\bmissing values?\b.*\b(most|which)\b", ql
    ):
        numeric_only = "numeric" in ql or "numerical" in ql
        cols = [
            c
            for c, t in types.items()
            if not c.startswith("_dm_") and (t == "numeric" or not numeric_only)
        ]
        return {
            "intent": "most_missing",
            "spec": {"kind": "missingness", "columns": cols[:60]},
            "interpretation": "Missing-value share of each "
            + ("numerical " if numeric_only else "")
            + "column.",
            "chart": {"type": "bar", "x": "column", "y": "missing_pct"},
        }

    m = re.search(
        r"\b(before and after|raw (vs|versus) clean|before cleaning|after cleaning)\b", ql
    )
    if m:
        col_phrase = re.sub(
            r"\b(compare|the|distribution|of|before and "
            r"after|cleaning|raw|vs|versus|cleaned|clean)\b",
            " ",
            ql,
        )
        col, problem = _one(col_phrase, columns, "comparison")
        if problem:
            return problem
        return {
            "intent": "compare_versions",
            "column": col,
            "interpretation": f"Distribution of '{col}' in this version and its parent version, on "
            f"shared bins.",
        }

    m = re.search(
        r"(relationship|correlation|association|scatter( "
        r"plot)?)\s+(between|of)\s+(?P<a>.+?)\s+(and|vs\.?|versus|with)\s+(?P<b>.+)$",
        ql,
    )
    if m:
        a, pa = _one(m.group("a"), columns, "first variable")
        b, pb = _one(m.group("b"), columns, "second variable")
        if pa or pb:
            return pa or pb
        assert a is not None and b is not None
        if types.get(a) != "numeric" or types.get(b) != "numeric":
            return {
                "status": "unsupported",
                "message": f"A relationship chart needs two numerical columns; '{a}' is "
                f"{types.get(a)} and '{b}' is {types.get(b)}. Try a grouped bar chart instead.",
            }
        return {
            "intent": "relationship",
            "spec": {"kind": "scatter", "columns": [a, b]},
            "correlation_spec": {"kind": "correlation", "columns": [a, b], "method": "spearman"},
            "interpretation": f"Scatter plot of '{b}' against '{a}' with their rank correlation.",
            "chart": {"type": "scatter", "x": a, "y": b},
        }

    if re.search(
        r"\bstrongest correlations?\b|\bmost correlated\b|\bexplain (the )?correlations?\b", ql
    ):
        num = [c for c, t in types.items() if t == "numeric" and not c.startswith("_dm_")][:20]
        return {
            "intent": "strongest_correlations",
            "spec": {"kind": "correlation", "columns": num, "method": "spearman"},
            "interpretation": "Spearman correlations between numerical columns; the strongest "
            "pairs are listed.",
            "chart": {"type": "heatmap"},
        }

    m = re.search(r"\b(histogram|distribution|spread)\s+(of|for)\s+(?P<col>.+)$", ql) or re.search(
        r"^(?P<col>.+?)\s+(histogram|distribution)$", ql
    )
    if m:
        col, problem = _one(m.group("col"), columns, "histogram")
        if problem:
            return problem
        assert col is not None
        if types.get(col) != "numeric":
            return {
                "intent": "category_counts",
                "spec": {
                    "kind": "aggregate",
                    "group_by": [{"column": col}],
                    "measures": [{"agg": "count"}],
                    "sort": {"by": "measure", "index": 0, "dir": "desc"},
                    "limit": 30,
                },
                "interpretation": f"'{col}' is {types.get(col)}: counts per category instead of a "
                f"histogram.",
                "chart": {"type": "bar", "x": col, "y": "count(rows)"},
            }
        return {
            "intent": "histogram",
            "spec": {"kind": "histogram", "columns": [col], "bins": 20},
            "box_spec": {"kind": "box", "columns": [col]},
            "interpretation": f"Histogram of '{col}' (20 equal-width bins) with its five-number "
            f"summary.",
            "chart": {"type": "histogram", "x": col},
        }

    m = re.search(r"\b(trend|over time|time series)\b", ql)
    date_part = next(
        (v for k, v in sorted(DATE_WORDS.items(), key=lambda kv: -len(kv[0])) if k in ql), None
    )
    if m or (date_part and re.search(r"\b(chart|trend|series)\b", ql)):
        tcol = _time_column(columns, roles)
        if tcol is None:
            return {
                "status": "unsupported",
                "message": "A trend needs a date or time column; this version has none.",
            }
        rest = re.sub(
            r"\b(monthly|weekly|daily|yearly|quarterly|annual|trend|over time|time "
            r"series|chart|of|in|per|by|month|year|week|day|quarter)\b",
            " ",
            ql,
        ).strip()
        agg, rest = _agg_of(rest)
        measure: dict[str, Any] = {"agg": "count"}
        label = "number of rows"
        if rest.strip() and rest.strip() not in ("rows", "records", "observations"):
            col, problem = _one(rest, columns, "trend measure")
            if problem:
                return problem
            assert col is not None
            if types.get(col) != "numeric":
                return {
                    "status": "unsupported",
                    "message": f"'{col}' is not numerical; a trend can show counts or a numerical "
                    f"column.",
                }
            measure = {"agg": agg if agg and agg != "count" else "mean", "column": col}
            label = f"{measure['agg']} of '{col}'"
        part = date_part or "month"
        return {
            "intent": "trend",
            "spec": {
                "kind": "aggregate",
                "group_by": [{"column": tcol, "date_part": part}],
                "measures": [measure],
                "sort": {"by": "group", "dir": "asc"},
                "limit": 500,
            },
            "interpretation": f"{label.capitalize()} per {part} of '{tcol}'.",
            "chart": {"type": "line", "x": tcol, "y": label},
        }

    m = re.search(
        r"^which\s+(?P<dim>.+?)\s+(has|have|shows?)\s+the\s+(?P<dir>highest|lowest|largest|smallest|most|least)\s+(?P<measure>.+)$",
        ql,
    )
    if m:
        dim, problem = _one(m.group("dim"), columns, "grouping")
        if problem:
            return problem
        agg, mphrase = _agg_of(m.group("measure"))
        mphrase = re.sub(r"^(value of|number of)\s+", "", mphrase)
        if mphrase.strip() in ("target", "target value", "outcome") and roles.get("target"):
            mcol: str | None = roles["target"][0]
        elif mphrase.strip() in ("rows", "records", "observations", "count", ""):
            mcol, agg = None, "count"
        else:
            mcol, problem = _one(mphrase, columns, "measure")
            if problem:
                return problem
        measure = {"agg": agg or ("mean" if mcol else "count")} | ({"column": mcol} if mcol else {})
        desc = m.group("dir") in ("highest", "largest", "most")
        return {
            "intent": "top_group",
            "spec": {
                "kind": "aggregate",
                "group_by": [{"column": dim}],
                "measures": [measure],
                "sort": {"by": "measure", "index": 0, "dir": "desc" if desc else "asc"},
                "limit": 10,
            },
            "interpretation": f"{measure['agg']} of {('' + repr(mcol)) if mcol else 'rows'} for "
            f"each '{dim}', {'highest' if desc else 'lowest'} first.",
            "chart": {"type": "bar", "x": dim, "y": measure["agg"]},
        }

    m = re.search(
        r"^(?P<measure>.+?)\s+(by|per|for each|across|grouped by|split by)\s+(?P<dim>.+)$", ql
    )
    if m:
        agg, mphrase = _agg_of(m.group("measure"))
        dim_phrase = m.group("dim")
        group: dict[str, Any] = {}
        dp = next((v for k, v in DATE_WORDS.items() if k.split()[-1] == dim_phrase.strip()), None)
        if dp:
            tcol = _time_column(columns, roles)
            if tcol is None:
                return {"status": "unsupported", "message": "Grouping by time needs a date column."}
            group = {"column": tcol, "date_part": dp}
        else:
            dim, problem = _one(dim_phrase, columns, "grouping")
            if problem:
                return problem
            assert dim is not None
            group = {"column": dim}
            if types.get(dim) == "numeric":
                group["bins"] = 10
            elif types.get(dim) == "datetime":
                group["date_part"] = "month"
        if agg is None and mphrase.strip() in ("rows", "records", "observations", "count"):
            agg = "count"
        mcol = None
        if mphrase.strip() not in ("rows", "records", "observations", "count", ""):
            mcol, problem = _one(mphrase, columns, "measure")
            if problem:
                return problem
        if agg is None:
            agg = "mean" if mcol and types.get(mcol) == "numeric" else "count"
        if mcol and agg not in ("count", "count_distinct") and types.get(mcol) != "numeric":
            return {
                "status": "unsupported",
                "message": f"'{mcol}' is {types.get(mcol)}: only counts are possible, not a {agg}.",
            }
        measure = {"agg": agg} | ({"column": mcol} if mcol else {})
        return {
            "intent": "aggregate",
            "spec": {
                "kind": "aggregate",
                "group_by": [group],
                "measures": [measure],
                "sort": {"by": "measure", "index": 0, "dir": "desc"}
                if "date_part" not in group
                else {"by": "group", "dir": "asc"},
                "limit": 50,
            },
            "interpretation": f"{agg} of {repr(mcol) if mcol else 'rows'} for each "
            f"{group.get('date_part', '')} '{group['column']}'.".replace("  ", " "),
            "chart": {
                "type": "line" if "date_part" in group else "bar",
                "x": group["column"],
                "y": agg,
            },
        }
    if re.search(r"\b(average|mean|sum|total|median|count|how many|number of)\b", ql):
        agg, mphrase = _agg_of(re.sub(r"^how many\s+", "count ", ql))
        if mphrase.strip() in ("rows", "records", "observations", "") or ql.startswith(
            "how many rows"
        ):
            return {
                "intent": "kpi",
                "spec": {"kind": "aggregate", "measures": [{"agg": "count"}]},
                "interpretation": "Number of rows.",
                "chart": {"type": "kpi"},
            }
        col, problem = _one(mphrase, columns, "measure")
        if problem:
            return problem
        assert col is not None
        agg = agg or "mean"
        if agg not in ("count", "count_distinct") and types.get(col) != "numeric":
            return {
                "status": "unsupported",
                "message": f"'{col}' is {types.get(col)}: a {agg} needs a numerical column.",
            }
        return {
            "intent": "kpi",
            "spec": {"kind": "aggregate", "measures": [{"agg": agg, "column": col}]},
            "interpretation": f"{agg} of '{col}' over all rows.",
            "chart": {"type": "kpi"},
        }
    return None


def explain(
    parsed: Mapping[str, Any], result: Mapping[str, Any], extra: Mapping[str, Any] | None = None
) -> list[str]:
    """Deterministic sentences built only from engine results."""
    lines: list[str] = []
    if result.get("suppressed"):
        return [str(result.get("note"))]
    intent = parsed["intent"]
    filt = result.get("rows_filtered")
    if intent == "most_missing":
        pairs = sorted(
            zip(result["columns"], result["missing_pct"], strict=True), key=lambda t: -(t[1] or 0)
        )
        top = [p for p in pairs if (p[1] or 0) > 0][:5]
        if not top:
            return ["No column has missing values."]
        lines.append(
            f"'{top[0][0]}' has the most missing values: {top[0][1]:.2f}% of {result['n']} rows."
        )
        if len(top) > 1:
            lines.append("Next: " + ", ".join(f"'{c}' ({p:.2f}%)" for c, p in top[1:]) + ".")
    elif intent in ("aggregate", "top_group", "trend", "category_counts"):
        rows = result.get("rows", [])
        label = result["measure_labels"][0]
        shown = [r for r in rows if not r.get("suppressed")]
        if not shown:
            return ["No group can be shown (all groups are empty or suppressed)."]
        if intent == "trend":
            first, last = shown[0], shown[-1]
            lines.append(
                f"{label} goes from {first['values'][0]} ({first['groups'][0]}) to "
                f"{last['values'][0]} ({last['groups'][0]}) over {len(shown)} periods."
            )
        else:
            top = shown[0]
            desc = (parsed["spec"].get("sort") or {}).get("dir", "desc") == "desc"
            word = "highest" if desc else "lowest"
            lines.append(
                f"{' / '.join(top['groups'])} has the {word} {label}: {top['values'][0]} "
                f"(n = {top['n']}).",
            )
            if len(shown) > 1:
                lines.append(
                    "Others: "
                    + "; ".join(f"{' / '.join(r['groups'])}: {r['values'][0]}" for r in shown[1:5])
                    + ("…" if len(shown) > 5 else "")
                    + "."
                )
        lines.append(
            f"Overall {label}: {result['totals']['values'][0]} across {result['totals']['n']} rows."
        )
        if result.get("suppressed_groups"):
            lines.append(
                f"{result['suppressed_groups']} small group(s) are suppressed for privacy."
            )
        if result.get("truncated"):
            lines.append(f"Showing the first {len(rows)} of {result['n_groups']} groups.")
    elif intent == "kpi":
        lines.append(
            f"{result['measure_labels'][0]} = {result['totals']['values'][0]} over "
            f"{result['totals']['n']} rows."
        )
    elif intent == "histogram":
        lines.append(
            f"{result['n']} non-missing values of '{result['column']}' ({result['missing']} "
            f"missing)."
        )
        box = (extra or {}).get("box")
        if box and box.get("items") and not box["items"][0].get("suppressed"):
            b = box["items"][0]
            lines.append(
                f"Median {b['median']}, interquartile range {b['q1']} to {b['q3']}, mean "
                f"{b['mean']}; {b['outliers']} value(s) beyond 1.5 × IQR."
            )
    elif intent == "relationship":
        corr = (extra or {}).get("correlation")
        if corr:
            rho, n = corr["matrix"][0][1], corr["n"][0][1]
            if rho is None:
                lines.append("Too few complete pairs to estimate a correlation.")
            else:
                strength = (
                    "strong" if abs(rho) >= 0.7 else "moderate" if abs(rho) >= 0.3 else "weak"
                )
                lines.append(
                    f"Spearman ρ = {rho:.3f} over {n} complete pairs: a {strength} "
                    f"{'positive' if rho > 0 else 'negative'} monotonic association. "
                    "Correlation does not imply causation."
                )
        if result.get("sampled"):
            lines.append(str(result["note"]))
        if result.get("mode") == "binned":
            lines.append(str(result["note"]))
    elif intent == "strongest_correlations":
        cols, mat = result["columns"], result["matrix"]
        cpairs = [
            (cols[i], cols[j], mat[i][j])
            for i in range(len(cols))
            for j in range(i + 1, len(cols))
            if mat[i][j] is not None
        ]
        cpairs.sort(key=lambda t: -abs(t[2]))
        if not cpairs:
            return ["Fewer than two numerical columns with enough data."]
        lines.append(
            "Strongest rank correlations: "
            + "; ".join(f"'{a}' & '{b}' ρ = {r:.2f}" for a, b, r in cpairs[:5])
            + "."
        )
        lines.append(
            "These are associations in this sample, not causal effects; see Insights for "
            "multiplicity-adjusted tests."
        )
    if filt is not None and intent not in ("most_missing",):
        pass
    return lines

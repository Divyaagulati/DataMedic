"""Analytics Studio query engine: validated structured specifications over one immutable
dataset version (pure pandas/NumPy; read-only).

There is no SQL and no expression language: a specification names columns of the version's
schema, operators from fixed lists and literal values. Validation rejects unknown columns,
incompatible types and out-of-range parameters with explanations; nothing is coerced silently.

Null semantics (documented in docs/ARCHITECTURE.md "Analytics Studio"):
- Comparison filters (eq, ne, gt, gte, lt, lte, between, in, not_in, date_between) never match
  missing values; use ``is_missing`` / ``not_missing`` explicitly.
- Grouping keeps missing values as an explicit "(missing)" group.
- Aggregations ignore missing values; ``count`` without a column counts rows, ``count`` of a
  column counts its non-missing values. Every group reports its row count ``n``.
- ``pct_total`` is the group's share (in %) of the measure's total over all filtered rows (of
  the row count when no column is given).

Privacy (``Policy``): protected (sensitive) columns can only be counted or tested for
missingness; they cannot be grouped, filtered by value, plotted or aggregated. In privacy mode
(biomedical datasets or datasets with protected columns) groups and cells with fewer than
``min_cell`` rows are suppressed, with secondary suppression when a single suppressed group
could be recovered from the total, the whole result is suppressed when the filtered cohort is
smaller than ``min_cell``, and row-level scatter points are replaced by binned counts.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID

QUERY_VERSION = "1.0.0"
KINDS = ("aggregate", "histogram", "box", "scatter", "correlation", "missingness")
AGGS = (
    "count",
    "count_distinct",
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "std",
    "var",
    "quantile",
    "pct_total",
)
NUMERIC_AGGS = {"sum", "mean", "median", "min", "max", "std", "var", "quantile"}
OPS = (
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "between",
    "in",
    "not_in",
    "is_missing",
    "not_missing",
    "date_between",
)
DATE_PARTS = ("year", "quarter", "month", "week", "day")
MISSING = "(missing)"
SUPPRESSED = "suppressed"


@dataclass(frozen=True)
class Limits:
    max_filters: int = 20
    max_group_by: int = 2
    max_measures: int = 6
    max_in_values: int = 200
    max_groups: int = 1000
    max_result_rows: int = 500
    max_bins: int = 60
    max_scatter_points: int = 2000
    max_matrix_columns: int = 20
    max_missingness_columns: int = 60
    max_box_groups: int = 20
    max_unbinned_numeric_levels: int = 50


@dataclass(frozen=True)
class Policy:
    protected: frozenset[str] = frozenset()
    privacy_mode: bool = False
    min_cell: int = 10


DEFAULT_POLICY = Policy()
DEFAULT_LIMITS = Limits()


@dataclass
class QueryError(Exception):
    errors: list[dict[str, str]] = field(default_factory=list)

    def __str__(self) -> str:
        return "; ".join(e["message"] for e in self.errors)


def _r(x: Any, d: int = 6) -> Any:
    if x is None:
        return None
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(x).isoformat()
    if isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_)):
        return int(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return round(v, d) if np.isfinite(v) else None


# ---- Validation ----------------------------------------------------------------------------------


class _Validator:
    def __init__(
        self,
        spec: Mapping[str, Any],
        schema: Sequence[Mapping[str, Any]],
        policy: Policy,
        limits: Limits,
    ) -> None:
        self.spec, self.policy, self.limits = spec, policy, limits
        self.types = {
            str(c["name"]): str(c.get("inferred_type", "text"))
            for c in schema
            if not str(c["name"]).startswith("_dm_")
        }
        self.n_unique = {str(c["name"]): c.get("n_unique") for c in schema}
        self.errors: list[dict[str, str]] = []

    def err(self, pointer: str, code: str, message: str) -> None:
        self.errors.append({"pointer": pointer, "code": code, "message": message})

    def column(self, name: Any, pointer: str, *, allow_protected: bool = False) -> str | None:
        if not isinstance(name, str) or name not in self.types:
            self.err(pointer, "COLUMN_NOT_FOUND", "This column is not in the dataset version.")
            return None
        if name in self.policy.protected and not allow_protected:
            self.err(
                pointer,
                "COLUMN_PROTECTED",
                f"'{name}' is a sensitive column: it can only be counted or checked for "
                "missing values.",
            )
            return None
        return name

    def numeric(self, name: str, pointer: str) -> bool:
        if self.types[name] not in ("numeric", "boolean"):
            self.err(
                pointer,
                "TYPE_MISMATCH",
                f"'{name}' is {self.types[name]}, not numeric; this operation needs numbers.",
            )
            return False
        return True

    def value_ok(self, name: str, value: Any, pointer: str) -> bool:
        t = self.types[name]
        if t == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                self.err(pointer, "VALUE_TYPE", f"'{name}' is numeric: give a number.")
                return False
        elif t == "boolean":
            if not isinstance(value, bool):
                self.err(pointer, "VALUE_TYPE", f"'{name}' is boolean: give true or false.")
                return False
        elif t == "datetime":
            if not isinstance(value, str) or pd.isna(pd.to_datetime(value, errors="coerce")):
                self.err(pointer, "VALUE_TYPE", f"'{name}' is a date: give an ISO date.")
                return False
        elif not isinstance(value, (str, int, float)) or isinstance(value, bool):
            self.err(pointer, "VALUE_TYPE", f"'{name}' is text: give a text value.")
            return False
        return True

    def filters(self, filters: Any, base: str = "/filters") -> None:
        if not isinstance(filters, list):
            self.err(base, "INVALID", "Filters must be a list.")
            return
        if len(filters) > self.limits.max_filters:
            self.err(base, "LIMIT_EXCEEDED", f"At most {self.limits.max_filters} filters.")
        for i, f in enumerate(filters):
            p = f"{base}/{i}"
            if not isinstance(f, Mapping):
                self.err(p, "INVALID", "A filter must be an object.")
                continue
            op = f.get("op")
            if op not in OPS:
                self.err(p + "/op", "INVALID_OPERATOR", f"Operators: {', '.join(OPS)}.")
                continue
            missing_op = op in ("is_missing", "not_missing")
            col = self.column(f.get("column"), p + "/column", allow_protected=missing_op)
            if col is None or missing_op:
                continue
            t = self.types[col]
            if op in ("gt", "gte", "lt", "lte", "between") and t not in ("numeric", "datetime"):
                self.err(p + "/op", "TYPE_MISMATCH", f"'{op}' needs a numeric or date column.")
                continue
            if op == "date_between" and t != "datetime":
                self.err(p + "/op", "TYPE_MISMATCH", "'date_between' needs a date column.")
                continue
            if op in ("between", "date_between"):
                lo, hi = f.get("min"), f.get("max")
                if lo is None and hi is None:
                    self.err(p, "VALUE_REQUIRED", "Give min and/or max.")
                for k, v in (("min", lo), ("max", hi)):
                    if v is not None:
                        self.value_ok(col, v, f"{p}/{k}")
            elif op in ("in", "not_in"):
                values = f.get("values")
                if not isinstance(values, list) or not values:
                    self.err(p + "/values", "VALUE_REQUIRED", "Give a non-empty list of values.")
                elif len(values) > self.limits.max_in_values:
                    self.err(
                        p + "/values", "LIMIT_EXCEEDED", f"At most {self.limits.max_in_values}."
                    )
                else:
                    for j, v in enumerate(values):
                        self.value_ok(col, v, f"{p}/values/{j}")
            else:
                if "value" not in f:
                    self.err(p + "/value", "VALUE_REQUIRED", "Give a value.")
                else:
                    self.value_ok(col, f["value"], p + "/value")

    def group_by(self, groups: Any) -> None:
        if not isinstance(groups, list):
            self.err("/group_by", "INVALID", "group_by must be a list.")
            return
        if len(groups) > self.limits.max_group_by:
            self.err("/group_by", "LIMIT_EXCEEDED", f"At most {self.limits.max_group_by} groups.")
        for i, g in enumerate(groups):
            p = f"/group_by/{i}"
            if not isinstance(g, Mapping):
                self.err(p, "INVALID", "A grouping must be an object.")
                continue
            col = self.column(g.get("column"), p + "/column")
            if col is None:
                continue
            t = self.types[col]
            date_part, bins = g.get("date_part"), g.get("bins")
            if date_part is not None:
                if t != "datetime":
                    self.err(
                        p + "/date_part", "TYPE_MISMATCH", "Date grouping needs a date column."
                    )
                elif date_part not in DATE_PARTS:
                    self.err(p + "/date_part", "INVALID", f"Use one of {', '.join(DATE_PARTS)}.")
            elif bins is not None:
                if t != "numeric":
                    self.err(p + "/bins", "TYPE_MISMATCH", "Binning needs a numeric column.")
                elif not isinstance(bins, int) or not 2 <= bins <= self.limits.max_bins:
                    self.err(p + "/bins", "INVALID", f"Use 2 to {self.limits.max_bins} bins.")
            elif t == "numeric":
                nu = self.n_unique.get(col)
                if nu is not None and nu > self.limits.max_unbinned_numeric_levels:
                    self.err(
                        p,
                        "NEEDS_BINS",
                        f"'{col}' has {nu} distinct numbers: group it into bins instead.",
                    )
            elif t == "datetime":
                self.err(p, "NEEDS_DATE_PART", "Group a date column by year, month, week or day.")

    def measures(self, measures: Any) -> None:
        if not isinstance(measures, list) or not measures:
            self.err("/measures", "VALUE_REQUIRED", "Give at least one measure.")
            return
        if len(measures) > self.limits.max_measures:
            self.err("/measures", "LIMIT_EXCEEDED", f"At most {self.limits.max_measures} measures.")
        for i, m in enumerate(measures):
            p = f"/measures/{i}"
            if not isinstance(m, Mapping) or m.get("agg") not in AGGS:
                self.err(p + "/agg", "INVALID", f"Aggregations: {', '.join(AGGS)}.")
                continue
            agg, col = m["agg"], m.get("column")
            if col is None:
                if agg not in ("count", "pct_total"):
                    self.err(p + "/column", "VALUE_REQUIRED", f"'{agg}' needs a column.")
                continue
            name = self.column(col, p + "/column", allow_protected=agg == "count")
            if name is None:
                continue
            if agg in NUMERIC_AGGS or (agg == "pct_total"):
                self.numeric(name, p + "/column")
            if agg == "quantile":
                q = m.get("q")
                if not isinstance(q, (int, float)) or isinstance(q, bool) or not 0 <= q <= 1:
                    self.err(p + "/q", "INVALID", "Give a quantile q between 0 and 1.")

    def run(self) -> None:
        s = self.spec
        kind = s.get("kind", "aggregate")
        if kind not in KINDS:
            self.err("/kind", "INVALID", f"Kinds: {', '.join(KINDS)}.")
            return
        self.filters(s.get("filters", []))
        limit = s.get("limit", 100)
        if not isinstance(limit, int) or not 1 <= limit <= self.limits.max_result_rows:
            self.err("/limit", "INVALID", f"Use a limit from 1 to {self.limits.max_result_rows}.")
        cols = s.get("columns", [])
        if not isinstance(cols, list):
            self.err("/columns", "INVALID", "columns must be a list.")
            return
        if kind == "aggregate":
            self.group_by(s.get("group_by", []))
            self.measures(s.get("measures", []))
            sort = s.get("sort")
            if sort is not None:
                if not isinstance(sort, Mapping) or sort.get("by") not in ("group", "measure"):
                    self.err("/sort", "INVALID", "Sort by 'group' or 'measure'.")
                elif sort.get("dir", "desc") not in ("asc", "desc"):
                    self.err("/sort/dir", "INVALID", "Use asc or desc.")
        elif kind in ("histogram", "box"):
            if len(cols) != 1:
                self.err("/columns", "VALUE_REQUIRED", f"A {kind} needs exactly one column.")
            else:
                c = self.column(cols[0], "/columns/0")
                if c is not None:
                    self.numeric(c, "/columns/0")
            if kind == "histogram":
                bins = s.get("bins", 20)
                if not isinstance(bins, int) or not 2 <= bins <= self.limits.max_bins:
                    self.err("/bins", "INVALID", f"Use 2 to {self.limits.max_bins} bins.")
                edges = s.get("edges")
                if edges is not None and (
                    not isinstance(edges, list)
                    or not 3 <= len(edges) <= self.limits.max_bins + 1
                    or any(isinstance(e, bool) or not isinstance(e, (int, float)) for e in edges)
                    or any(b <= a for a, b in zip(edges, edges[1:], strict=False))
                ):
                    self.err("/edges", "INVALID", "Edges must be increasing numbers.")
            groups = s.get("group_by", [])
            if groups:
                if kind != "box" or len(groups) != 1:
                    self.err("/group_by", "INVALID", "Only a box plot takes one grouping.")
                else:
                    self.group_by(groups)
        elif kind == "scatter":
            if len(cols) != 2:
                self.err("/columns", "VALUE_REQUIRED", "A scatter plot needs x and y columns.")
            for i, c in enumerate(cols[:2]):
                name = self.column(c, f"/columns/{i}")
                if name is not None:
                    self.numeric(name, f"/columns/{i}")
            for key in ("color", "size"):
                if s.get(key) is not None:
                    name = self.column(s[key], f"/{key}")
                    if name is not None and key == "size":
                        self.numeric(name, "/size")
        elif kind == "correlation":
            if s.get("method", "pearson") not in ("pearson", "spearman"):
                self.err("/method", "INVALID", "Use pearson or spearman.")
            if len(cols) > self.limits.max_matrix_columns:
                self.err("/columns", "LIMIT_EXCEEDED", f"At most {self.limits.max_matrix_columns}.")
            for i, c in enumerate(cols):
                name = self.column(c, f"/columns/{i}")
                if name is not None:
                    self.numeric(name, f"/columns/{i}")
        elif kind == "missingness":
            if len(cols) > self.limits.max_missingness_columns:
                self.err("/columns", "LIMIT_EXCEEDED", "Too many columns.")
            for i, c in enumerate(cols):
                self.column(c, f"/columns/{i}", allow_protected=True)
            groups = s.get("group_by", [])
            if groups:
                if len(groups) != 1:
                    self.err("/group_by", "INVALID", "Missingness takes at most one grouping.")
                else:
                    self.group_by(groups)


def validate(
    spec: Mapping[str, Any],
    schema: Sequence[Mapping[str, Any]],
    policy: Policy = DEFAULT_POLICY,
    limits: Limits = DEFAULT_LIMITS,
) -> list[dict[str, str]]:
    v = _Validator(spec, schema, policy, limits)
    v.run()
    return v.errors


def needed_columns(spec: Mapping[str, Any]) -> list[str]:
    cols: list[str] = []
    for f in spec.get("filters", []):
        cols.append(f["column"])
    for g in spec.get("group_by", []):
        cols.append(g["column"])
    for m in spec.get("measures", []):
        if m.get("column"):
            cols.append(m["column"])
    cols += list(spec.get("columns", []))
    for key in ("color", "size"):
        if spec.get(key):
            cols.append(spec[key])
    return list(dict.fromkeys(cols))


# ---- Execution -----------------------------------------------------------------------------------


def _filter(frame: pd.DataFrame, filters: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for f in filters:
        s = frame[f["column"]]
        op = f["op"]
        present = s.notna()
        if op == "is_missing":
            mask &= ~present
            continue
        if op == "not_missing":
            mask &= present
            continue
        is_date = pd.api.types.is_datetime64_any_dtype(s)

        def lit(v: Any, *, date: bool = is_date) -> Any:
            return pd.Timestamp(v) if date else v

        if op in ("in", "not_in"):
            if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                hit = s.astype(str).isin([str(v) for v in f["values"]])
            else:
                hit = s.isin([lit(v) for v in f["values"]])
            mask &= present & (hit if op == "in" else ~hit)
        elif op in ("between", "date_between"):
            cond = present.copy()
            if f.get("min") is not None:
                cond &= s >= lit(f["min"], date=is_date or op == "date_between")
            if f.get("max") is not None:
                cond &= s <= lit(f["max"], date=is_date or op == "date_between")
            mask &= cond.fillna(False)
        else:
            v = lit(f["value"])
            if pd.api.types.is_object_dtype(s) or isinstance(s.dtype, pd.CategoricalDtype):
                left: Any = (
                    s.astype(object)
                    .where(present, None)
                    .map(lambda x: None if x is None else str(x))
                )
                v = str(v)
            else:
                left = s
            import operator

            fn = {
                "eq": operator.eq,
                "ne": operator.ne,
                "gt": operator.gt,
                "gte": operator.ge,
                "lt": operator.lt,
                "lte": operator.le,
            }[op]
            cmp = fn(left, v)
            mask &= present & pd.Series(cmp, index=frame.index).fillna(False).astype(bool)
    return frame[mask]


def _key(frame: pd.DataFrame, g: Mapping[str, Any]) -> pd.Series:
    s = frame[g["column"]]
    if g.get("date_part"):
        d = pd.to_datetime(s, errors="coerce")
        part = g["date_part"]
        if part == "year":
            out = d.dt.year.astype("Int64").astype(str)
        elif part == "quarter":
            out = d.dt.to_period("Q").astype(str)
        elif part == "month":
            out = d.dt.to_period("M").astype(str)
        elif part == "week":
            out = d.dt.to_period("W").astype(str)
        else:
            out = d.dt.strftime("%Y-%m-%d")
        return out.where(d.notna(), MISSING)
    if g.get("bins"):
        x = pd.to_numeric(s, errors="coerce")
        lo, hi = x.min(), x.max()
        if pd.isna(lo):
            return pd.Series(MISSING, index=frame.index)
        if hi <= lo:
            hi = lo + 1
        edges = np.linspace(lo, hi, int(g["bins"]) + 1)
        idx = np.clip(
            np.searchsorted(edges, x.to_numpy(dtype=float), side="right") - 1, 0, len(edges) - 2
        )
        labels = np.array([f"{edges[i]:.4g} – {edges[i + 1]:.4g}" for i in range(len(edges) - 1)])
        # Sortable: prefix each label with its bin index; stripped in the output.
        keyed = pd.Series([f"{i:03d}|{labels[i]}" for i in idx], index=frame.index)
        return keyed.where(x.notna(), MISSING)
    return s.astype(object).where(s.notna(), MISSING).map(str)


def _measure(series: pd.Series | None, agg: str, q: float | None, n_rows: int) -> Any:
    if series is None:
        return n_rows
    if agg == "count":
        return int(series.notna().sum())
    if agg == "count_distinct":
        return int(series.nunique(dropna=True))
    x = pd.to_numeric(series, errors="coerce") if series.dtype == bool else series
    x = x.dropna()
    if not len(x):
        return None
    if agg in ("sum", "pct_total"):
        return float(x.sum())
    if agg == "mean":
        return float(x.mean())
    if agg == "median":
        return float(x.median())
    if agg == "min":
        return x.min()
    if agg == "max":
        return x.max()
    if agg == "std":
        return float(x.std(ddof=1)) if len(x) > 1 else None
    if agg == "var":
        return float(x.var(ddof=1)) if len(x) > 1 else None
    if agg == "quantile":
        return float(x.quantile(float(q if q is not None else 0.5)))
    raise ValueError(agg)


def _label(m: Mapping[str, Any]) -> str:
    agg, col = m["agg"], m.get("column")
    if agg == "quantile":
        return f"p{round(float(m.get('q', 0.5)) * 100)}({col})"
    return f"{agg}({col})" if col else ("count(rows)" if agg == "count" else "pct_total(rows)")


def _suppress(rows: list[dict[str, Any]], policy: Policy) -> int:
    """Primary (n < min_cell) and secondary suppression; returns the number suppressed."""
    if not policy.privacy_mode:
        for r in rows:
            r["small"] = r["n"] < policy.min_cell
        return 0
    small = [r for r in rows if r["n"] < policy.min_cell]
    for r in small:
        r["suppressed"] = True
    if len(small) == 1 and len(rows) > 1:
        # One suppressed group could be recovered from the total: suppress the next smallest.
        others = sorted((r for r in rows if not r.get("suppressed")), key=lambda r: r["n"])
        if others:
            others[0]["suppressed"] = True
    count = 0
    for r in rows:
        if r.get("suppressed"):
            r["values"] = [None] * len(r["values"])
            r["n"] = None
            count += 1
    return count


def _aggregate(
    data: pd.DataFrame, spec: Mapping[str, Any], policy: Policy, limits: Limits
) -> dict[str, Any]:
    groups = list(spec.get("group_by", []))
    measures = list(spec["measures"])
    labels = [_label(m) for m in measures]
    totals_raw = [
        _measure(data[m["column"]] if m.get("column") else None, m["agg"], m.get("q"), len(data))
        for m in measures
    ]

    def finish(values: list[Any]) -> list[Any]:
        out = []
        for m, v, tot in zip(measures, values, totals_raw, strict=True):
            if m["agg"] == "pct_total":
                out.append(None if v is None or not tot else _r(100 * float(v) / float(tot), 4))
            else:
                out.append(_r(v))
        return out

    result: dict[str, Any] = {
        "kind": "aggregate",
        "group_columns": [g["column"] for g in groups],
        "measure_labels": labels,
        "rows": [],
        "totals": {"n": len(data), "values": finish(totals_raw)},
    }
    if not groups:
        return result
    keys = pd.DataFrame({f"g{i}": _key(data, g) for i, g in enumerate(groups)}, index=data.index)
    grouped = keys.groupby(list(keys.columns), sort=False, dropna=False).indices
    if len(grouped) > limits.max_groups:
        raise QueryError(
            [
                {
                    "pointer": "/group_by",
                    "code": "TOO_MANY_GROUPS",
                    "message": f"The grouping produces {len(grouped)} groups (limit "
                    f"{limits.max_groups}); use bins, a coarser date part or filters.",
                }
            ]
        )
    rows = []
    for key, idx in grouped.items():
        key = key if isinstance(key, tuple) else (key,)
        sub = data.iloc[idx]
        vals = [
            _measure(sub[m["column"]] if m.get("column") else None, m["agg"], m.get("q"), len(sub))
            for m in measures
        ]
        rows.append(
            {
                "groups": [str(k).split("|", 1)[-1] if "|" in str(k) else str(k) for k in key],
                "_sort": [str(k) for k in key],
                "n": len(sub),
                "values": finish(vals),
            }
        )
    sort = spec.get("sort") or {"by": "group", "dir": "asc"}
    rev = sort.get("dir", "desc") == "desc"
    if sort["by"] == "measure":
        i = int(sort.get("index", 0))

        def mkey(r: dict[str, Any]) -> tuple[int, float]:
            v = r["values"][i] if i < len(r["values"]) else None
            return (0, float(v)) if isinstance(v, (int, float)) else (1, 0.0)

        rows.sort(key=lambda r: mkey(r)[1], reverse=rev)
        rows.sort(key=lambda r: mkey(r)[0])  # missing values last
    else:
        rows.sort(key=lambda r: str(r["_sort"]), reverse=rev)
    for r in rows:
        del r["_sort"]
    if spec.get("margins") and len(groups) == 2:
        result["margins"] = {"rows": {}, "columns": {}}
        for j, side in ((0, "rows"), (1, "columns")):
            for key, idx in keys.groupby(f"g{j}", dropna=False).indices.items():
                sub = data.iloc[idx]
                vals = [
                    _measure(
                        sub[m["column"]] if m.get("column") else None,
                        m["agg"],
                        m.get("q"),
                        len(sub),
                    )
                    for m in measures
                ]
                label = str(key).split("|", 1)[-1]
                n = len(sub)
                hidden = policy.privacy_mode and n < policy.min_cell
                result["margins"][side][label] = {
                    "n": None if hidden else n,
                    "values": [None] * len(vals) if hidden else finish(vals),
                    "suppressed": hidden,
                }
    result["n_groups"] = len(rows)
    result["suppressed_groups"] = _suppress(rows, policy)
    limit = int(spec.get("limit", 100))
    result["truncated"] = len(rows) > limit
    result["rows"] = rows[:limit]
    return result


def _histogram(data: pd.DataFrame, spec: Mapping[str, Any], policy: Policy) -> dict[str, Any]:
    col = spec["columns"][0]
    x = pd.to_numeric(data[col], errors="coerce").dropna().to_numpy(dtype=float)
    if spec.get("edges"):
        edges = np.asarray(spec["edges"], dtype=float)
    elif len(x):
        lo, hi = float(x.min()), float(x.max())
        edges = np.linspace(lo, hi if hi > lo else lo + 1, int(spec.get("bins", 20)) + 1)
    else:
        edges = np.array([0.0, 1.0])
    counts, _ = np.histogram(x, bins=edges)
    outside = int(((x < edges[0]) | (x > edges[-1])).sum())
    out_counts: list[int | None] = [int(c) for c in counts]
    suppressed = 0
    if policy.privacy_mode:
        for i, c in enumerate(out_counts):
            if c is not None and 0 < c < policy.min_cell:
                out_counts[i], suppressed = None, suppressed + 1
    return {
        "kind": "histogram",
        "column": col,
        "edges": [_r(e) for e in edges],
        "counts": out_counts,
        "n": int(len(x)),
        "missing": int(data[col].isna().sum()),
        "outside_edges": outside,
        "suppressed_bins": suppressed,
    }


def _five(x: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(x, errors="coerce").dropna()
    if not len(x):
        return {"n": 0}
    q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    inside = x[(x >= q1 - 1.5 * iqr) & (x <= q3 + 1.5 * iqr)]
    return {
        "n": int(len(x)),
        "mean": _r(x.mean()),
        "q1": _r(q1),
        "median": _r(med),
        "q3": _r(q3),
        "whisker_low": _r(inside.min()),
        "whisker_high": _r(inside.max()),
        "outliers": int(len(x) - len(inside)),
    }


def _box(
    data: pd.DataFrame, spec: Mapping[str, Any], policy: Policy, limits: Limits
) -> dict[str, Any]:
    col = spec["columns"][0]
    groups = spec.get("group_by", [])
    items = []
    if groups:
        keys = _key(data, groups[0])
        counts = keys.value_counts()
        if len(counts) > limits.max_box_groups:
            raise QueryError(
                [
                    {
                        "pointer": "/group_by",
                        "code": "TOO_MANY_GROUPS",
                        "message": f"At most {limits.max_box_groups} box groups.",
                    }
                ]
            )
        for k in sorted(counts.index, key=str):
            stats = _five(data.loc[keys == k, col])
            label = str(k).split("|", 1)[-1]
            hidden = policy.privacy_mode and stats["n"] < policy.min_cell
            items.append({"label": label, **({"n": None, "suppressed": True} if hidden else stats)})
    else:
        stats = _five(data[col])
        hidden = policy.privacy_mode and stats["n"] < policy.min_cell
        items.append({"label": col, **({"n": None, "suppressed": True} if hidden else stats)})
    return {
        "kind": "box",
        "column": col,
        "items": items,
        "note": "Outliers are counted (1.5 × IQR), not listed.",
    }


def _scatter(
    data: pd.DataFrame, spec: Mapping[str, Any], policy: Policy, limits: Limits, seed: int
) -> dict[str, Any]:
    xcol, ycol = spec["columns"]
    color, size = spec.get("color"), spec.get("size")
    use = [xcol, ycol] + [c for c in (color, size) if c]
    d = data[list(dict.fromkeys(use))].copy()
    d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
    d[ycol] = pd.to_numeric(d[ycol], errors="coerce")
    d = d.dropna(subset=[xcol, ycol])
    n = len(d)
    if policy.privacy_mode:
        bins = 20
        if not n:
            return {"kind": "scatter", "mode": "binned", "n": 0, "cells": []}
        h, xe, ye = np.histogram2d(d[xcol], d[ycol], bins=bins)
        cells = []
        for i in range(bins):
            for j in range(bins):
                c = int(h[i, j])
                if c == 0:
                    continue
                cells.append(
                    {
                        "x0": _r(xe[i]),
                        "x1": _r(xe[i + 1]),
                        "y0": _r(ye[j]),
                        "y1": _r(ye[j + 1]),
                        "count": None if c < policy.min_cell else c,
                    }
                )
        return {
            "kind": "scatter",
            "mode": "binned",
            "n": n,
            "cells": cells,
            "note": f"Privacy mode: points are shown as binned counts; cells with fewer than "
            f"{policy.min_cell} rows are suppressed.",
        }
    sampled = n > limits.max_scatter_points
    if sampled:
        d = d.sample(n=limits.max_scatter_points, random_state=seed)
    pts = []
    for _, r in d.iterrows():
        p: dict[str, Any] = {"x": _r(r[xcol]), "y": _r(r[ycol])}
        if color:
            p["color"] = MISSING if pd.isna(r[color]) else str(r[color])[:48]
        if size:
            p["size"] = _r(r[size])
        pts.append(p)
    return {
        "kind": "scatter",
        "mode": "points",
        "n": n,
        "points": pts,
        "sampled": sampled,
        "note": f"Deterministic random sample of {limits.max_scatter_points} of {n} rows "
        f"(seed {seed})."
        if sampled
        else None,
    }


def _correlation(data: pd.DataFrame, spec: Mapping[str, Any]) -> dict[str, Any]:
    cols = list(spec["columns"])
    method = spec.get("method", "pearson")
    num = data[cols].apply(pd.to_numeric, errors="coerce")
    matrix = num.corr(method=method, min_periods=10)
    present = num.notna().astype(int)
    n = present.T @ present
    return {
        "kind": "correlation",
        "method": method,
        "columns": cols,
        "matrix": [[_r(matrix.loc[a, b], 4) for b in cols] for a in cols],
        "n": [[int(n.loc[a, b]) for b in cols] for a in cols],
        "note": "Pairwise-complete observations; at least 10 pairs per cell. Correlation is "
        "not causation.",
    }


def _missingness(data: pd.DataFrame, spec: Mapping[str, Any], policy: Policy) -> dict[str, Any]:
    cols = list(spec["columns"])
    miss = data[cols].isna()
    out: dict[str, Any] = {
        "kind": "missingness",
        "columns": cols,
        "n": len(data),
        "missing_pct": [_r(100 * miss[c].mean(), 3) for c in cols],
    }
    groups = spec.get("group_by", [])
    if groups:
        keys = _key(data, groups[0])
        rows = []
        for k, idx in keys.groupby(keys).indices.items():
            sub = miss.iloc[idx]
            hidden = policy.privacy_mode and len(idx) < policy.min_cell
            rows.append(
                {
                    "label": str(k).split("|", 1)[-1],
                    "n": None if hidden else len(idx),
                    "missing_pct": [None if hidden else _r(100 * sub[c].mean(), 3) for c in cols],
                    "suppressed": hidden,
                }
            )
        out["groups"] = sorted(rows, key=lambda r: str(r["label"]))
    with_missing = [c for c in cols if miss[c].any()][:20]
    if len(with_missing) >= 2:
        co = miss[with_missing].astype(float).corr()
        out["co_missing"] = {
            "columns": with_missing,
            "matrix": [[_r(co.loc[a, b], 3) for b in with_missing] for a in with_missing],
        }
    return out


def execute(
    frame: pd.DataFrame,
    spec: Mapping[str, Any],
    schema: Sequence[Mapping[str, Any]],
    policy: Policy = DEFAULT_POLICY,
    limits: Limits = DEFAULT_LIMITS,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate and run ``spec`` on ``frame`` (never modified). Raises ``QueryError``."""
    errors = validate(spec, schema, policy, limits)
    if errors:
        raise QueryError(errors)
    kind = spec.get("kind", "aggregate")
    spec = dict(spec)
    if kind == "correlation" and not spec.get("columns"):
        spec["columns"] = [
            c["name"]
            for c in schema
            if c.get("inferred_type") == "numeric"
            and c["name"] not in policy.protected
            and not str(c["name"]).startswith("_dm_")
        ][: limits.max_matrix_columns]
    if kind == "missingness" and not spec.get("columns"):
        spec["columns"] = [c["name"] for c in schema if not str(c["name"]).startswith("_dm_")][
            : limits.max_missingness_columns
        ]
    needed = [c for c in needed_columns(spec) if c in frame.columns]
    data = _filter(
        frame[needed] if needed else frame.drop(columns=[ROW_ID], errors="ignore"),
        spec.get("filters", []),
    )
    meta = {
        "query_version": QUERY_VERSION,
        "rows_total": len(frame),
        "rows_filtered": len(data),
        "privacy_mode": policy.privacy_mode,
        "min_cell": policy.min_cell,
    }
    if policy.privacy_mode and 0 < len(data) < policy.min_cell:
        return {
            **meta,
            "kind": kind,
            "suppressed": True,
            "rows_filtered": None,
            "note": f"The filtered cohort has fewer than {policy.min_cell} rows; results are "
            "suppressed.",
        }
    if kind == "aggregate":
        res = _aggregate(data, spec, policy, limits)
    elif kind == "histogram":
        res = _histogram(data, spec, policy)
    elif kind == "box":
        res = _box(data, spec, policy, limits)
    elif kind == "scatter":
        res = _scatter(data, spec, policy, limits, seed)
    elif kind == "correlation":
        res = _correlation(data, spec)
    else:
        res = _missingness(data, spec, policy)
    return {**meta, **res}

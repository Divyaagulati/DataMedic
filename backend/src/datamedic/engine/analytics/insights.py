"""Automated Data Intelligence: evidence-based insights about one dataset version (worker only).

Every insight states its ``basis``:
- ``descriptive``: an observed fact about this sample (distribution, missingness, concentration);
- ``association``: a statistical association with a test, effect size and multiplicity-adjusted
  q-value (Benjamini–Hochberg within each family of tests); never a causal claim;
- ``suspicion``: a pattern worth checking (possible nonlinearity, change point, unusual values);
- ``configuration``: a fact about declared roles or the dataset structure.

Thresholds such as |rho| ≥ 0.3 or epsilon² ≥ 0.06 select what is worth showing; significance is
only ever claimed through q-values. When an entity role (patient, customer, account, asset,
group) is declared, associations use one value per entity (its mean), so repeated observations
do not inflate the evidence; otherwise rows are treated as independent (disclosed). Large
versions are analysed on a deterministic sample (disclosed). Target associations reuse the
Biomedical Intelligence Lite comparison statistics (``engine.bio.summary``) so both agree.

Sensitive (protected) columns are never analysed by value; groups below ``min_cell`` are not
reported in privacy mode.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from datamedic.engine import ROW_ID
from datamedic.engine.bio.summary import MIN_CELL, _bh, summarise

INSIGHTS_VERSION = "1.0.0"
MAX_ROWS = 50_000
MAX_NUMERIC = 30
MAX_CATEGORICAL = 10
MAX_LEVELS = 20
ENTITY_ROLES = ("bio.patient_id", "group", "biz.customer_id", "fin.account_id", "ind.asset_id")
TIME_ROLES = ("time", "fin.timestamp", "ind.timestamp")
NON_FEATURE = {
    "target",
    "id",
    "group",
    "time",
    "weight",
    "exclude",
    "bio.patient_id",
    "bio.sample_id",
    "bio.biomarker_id",
    "biz.customer_id",
    "fin.account_id",
    "fin.transaction_id",
    "ind.asset_id",
    "fin.timestamp",
    "ind.timestamp",
}
LIMITATIONS = [
    "Insights describe this dataset version only; associations are not causal and may reflect "
    "confounding, selection or measurement artefacts.",
    "q-values control the false discovery rate within each family of tests; thresholds used to "
    "pick what is shown are not significance tests.",
    "Rows are treated as independent unless an entity role is declared, in which case one value "
    "per entity is used.",
    "Temporal tests assume independent periods; autocorrelation can make trends and change "
    "points look stronger than they are.",
]


def _r(x: Any, d: int = 4) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, d) if np.isfinite(v) else None


def _first(roles: Mapping[str, Sequence[str]], keys: Sequence[str], cols: Any) -> str | None:
    for k in keys:
        for c in roles.get(k, ()):
            if c in cols:
                return c
    return None


def _insight(
    category: str,
    basis: str,
    title: str,
    interpretation: str,
    evidence: dict[str, Any],
    *,
    n: int | None,
    method: str,
    limitations: list[str] | None = None,
    chart: dict[str, Any] | None = None,
    next_action: str | None = None,
    columns: list[str] | None = None,
    importance: float = 0.0,
) -> dict[str, Any]:
    return {
        "category": category,
        "basis": basis,
        "title": title,
        "interpretation": interpretation,
        "evidence": evidence,
        "n": n,
        "method": method,
        "limitations": limitations or [],
        "chart": chart,
        "next_action": next_action,
        "columns": columns or [],
        "importance": round(float(importance), 4),
    }


class _Ctx:
    def __init__(
        self,
        frame: pd.DataFrame,
        schema: Sequence[Mapping[str, Any]],
        roles: Mapping[str, Sequence[str]],
        task_type: str | None,
        domain: str,
        protected: set[str],
        privacy_mode: bool,
        seed: int,
    ) -> None:
        data = frame.drop(columns=[ROW_ID], errors="ignore")
        self.n_rows = len(data)
        self.sampled = len(data) > MAX_ROWS
        self.data = data.sample(n=MAX_ROWS, random_state=seed) if self.sampled else data
        self.full = data
        self.types = {str(c["name"]): str(c.get("inferred_type")) for c in schema}
        self.roles, self.task_type, self.domain = roles, task_type, domain
        self.protected, self.privacy_mode, self.seed = protected, privacy_mode, seed
        declared = {c for k, cs in roles.items() if k in NON_FEATURE for c in cs}
        self.entity = _first(roles, ENTITY_ROLES, data.columns)
        self.time = _first(roles, TIME_ROLES, data.columns)
        self.time_declared = self.time is not None
        if self.time is None:
            dt = [c for c in data.columns if self.types.get(c) == "datetime"]
            self.time = dt[0] if dt else None
        target = list(roles.get("target", ()))
        self.target = target[0] if len(target) == 1 and target[0] in data.columns else None
        usable = [
            c
            for c in data.columns
            if c not in protected and c not in declared and not str(c).startswith("_dm_")
        ]
        self.numeric = [c for c in usable if self.types.get(c) == "numeric"][:MAX_NUMERIC]
        self.categorical = [
            c
            for c in usable
            if self.types.get(c) in ("categorical", "boolean")
            and 2 <= data[c].nunique(dropna=True) <= MAX_LEVELS
        ][:MAX_CATEGORICAL]

    @property
    def classification_target(self) -> bool:
        return self.task_type in ("binary", "multiclass")

    @property
    def unit(self) -> str:
        return "entity" if self.entity else "row"

    def units(self, numeric: list[str], extra: list[str] | None = None) -> pd.DataFrame:
        """One row per entity (mean of numerics; categoricals kept only when constant)."""
        cols = list(dict.fromkeys(numeric + (extra or [])))
        if self.entity is None:
            return self.data[cols]
        d = self.data[[self.entity] + cols].dropna(subset=[self.entity])
        out = (
            d[[self.entity] + [c for c in cols if c in numeric]]
            .groupby(self.entity)
            .mean(numeric_only=True)
        )
        for c in extra or []:
            if c in numeric:
                continue
            per = d.groupby(self.entity)[c].agg(lambda s: s.iloc[0] if s.nunique() <= 1 else None)
            out[c] = per
        return out.reset_index(drop=True)


def _distributions(c: _Ctx) -> list[dict[str, Any]]:
    out = []
    for col in c.numeric:
        x = pd.to_numeric(c.full[col], errors="coerce").dropna()
        if len(x) < 20 or x.nunique() < 3:
            continue
        skew = float(sps.skew(x, bias=False))
        se = float(np.sqrt(6 / len(x)))
        if abs(skew) >= 1:
            side = (
                "right (a long tail of large values)"
                if skew > 0
                else "left (a long tail of small values)"
            )
            out.append(
                _insight(
                    "distribution",
                    "descriptive",
                    f"'{col}' is strongly skewed to the {side.split(' ')[0]}",
                    f"The distribution of '{col}' is skewed to the {side}; means are pulled by the "
                    f"tail, "
                    "so medians and quantiles summarise it better.",
                    {
                        "skewness": _r(skew),
                        "skewness_se": _r(se),
                        "median": _r(x.median()),
                        "mean": _r(x.mean()),
                        "q1": _r(x.quantile(0.25)),
                        "q3": _r(x.quantile(0.75)),
                    },
                    n=len(x),
                    method="Sample skewness (adjusted Fisher–Pearson) with approximate SE √(6/n).",
                    chart={"kind": "histogram", "columns": [col], "bins": 30},
                    next_action="Compare a log or rank transformation in the Experiment Lab if it "
                    "is a model input.",
                    columns=[col],
                    importance=min(abs(skew) / 5, 1) * 0.5,
                )
            )
    return out


def _missingness(c: _Ctx) -> list[dict[str, Any]]:
    out = []
    miss = c.full.isna().mean().sort_values(ascending=False)
    top = [(col, float(p)) for col, p in miss.items() if p > 0 and not str(col).startswith("_dm_")][
        :5
    ]
    if top:
        out.append(
            _insight(
                "missingness",
                "descriptive",
                f"{len(miss[miss > 0])} column(s) have missing values",
                "Most missing: " + ", ".join(f"'{k}' ({100 * v:.1f}%)" for k, v in top) + ".",
                {
                    "columns": [{"column": k, "missing_pct": _r(100 * v, 2)} for k, v in top],
                    "rows_with_any_missing_pct": _r(100 * c.full.isna().any(axis=1).mean(), 2),
                },
                n=c.n_rows,
                method="Share of missing cells per column.",
                chart={"kind": "missingness"},
                columns=[k for k, _ in top],
                next_action="Check whether values are missing at random before imputing; compare "
                "imputation strategies in the Experiment Lab.",
                importance=0.4 + 0.4 * top[0][1],
            )
        )
    if c.target is not None:
        tests = []
        y = c.data[c.target]
        for col in [
            k
            for k, v in miss.items()
            if 0.05 <= v <= 0.95
            and k != c.target
            and k not in c.protected
            and not str(k).startswith("_dm_")
        ][:15]:
            m = c.data[col].isna()
            pair = pd.DataFrame({"m": m, "y": y}).dropna(subset=["y"])
            if c.task_type in ("binary", "multiclass"):
                table = pd.crosstab(pair["m"], pair["y"].astype(object))
                if table.shape[0] < 2 or table.shape[1] < 2 or table.values.min() < 5:
                    continue
                chi2, p, _, _ = sps.chi2_contingency(table)
                eff = float(np.sqrt(chi2 / (table.values.sum() * (min(table.shape) - 1))))
                tests.append((col, p, eff, "cramers_v"))
            elif c.task_type == "regression":
                a = pd.to_numeric(pair.loc[pair["m"], "y"], errors="coerce").dropna()
                b = pd.to_numeric(pair.loc[~pair["m"], "y"], errors="coerce").dropna()
                if min(len(a), len(b)) < 10:
                    continue
                u = sps.mannwhitneyu(a, b)
                tests.append(
                    (
                        col,
                        float(u.pvalue),
                        float(2 * u.statistic / (len(a) * len(b)) - 1),
                        "rank_biserial",
                    )
                )
        qs = _bh([t[1] for t in tests])
        for (col, p, eff, name), q in zip(tests, qs, strict=True):
            if q is not None and q < 0.05:
                out.append(
                    _insight(
                        "missingness",
                        "association",
                        f"Missingness of '{col}' is associated with the target",
                        f"Rows where '{col}' is missing have a different target distribution, so "
                        f"the "
                        "values are probably not missing completely at random. Dropping or naively "
                        "imputing them can bias results.",
                        {
                            "effect": {"name": name, "value": _r(eff)},
                            "p_value": _r(p, 6),
                            "q_value": q,
                        },
                        n=len(c.data),
                        method="Chi-squared test (classification) or Mann–Whitney U "
                        "(regression) of missing vs present rows; Benjamini–Hochberg q-values.",
                        columns=[col, c.target],
                        importance=0.6,
                        next_action="Keep a missing-value indicator as a feature and impute inside "
                        "folds.",
                    )
                )
    return out


def _outliers(c: _Ctx) -> list[dict[str, Any]]:
    out = []
    for col in c.numeric:
        x = pd.to_numeric(c.full[col], errors="coerce").dropna()
        if len(x) < 30:
            continue
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        iqr = q3 - q1
        if iqr <= 0:
            continue
        far = int(((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr)).sum())
        if far:
            out.append(
                _insight(
                    "outliers",
                    "suspicion",
                    f"{far} extreme value(s) in '{col}'",
                    f"{far} value(s) of '{col}' lie more than 3 IQR outside the quartiles. They "
                    f"can be "
                    "genuine extremes or errors; check them before removing anything.",
                    {
                        "extreme_count": far,
                        "extreme_pct": _r(100 * far / len(x), 3),
                        "q1": _r(q1),
                        "q3": _r(q3),
                        "fence_low": _r(q1 - 3 * iqr),
                        "fence_high": _r(q3 + 3 * iqr),
                    },
                    n=len(x),
                    method="Tukey's far-out fences (Q1 − 3·IQR, Q3 + 3·IQR).",
                    chart={"kind": "box", "columns": [col]},
                    columns=[col],
                    next_action="Inspect the source of these values; compare capping vs keeping "
                    "them in the Lab.",
                    importance=min(far / len(x) * 20, 1) * 0.5,
                )
            )
    return out


def _correlations(c: _Ctx) -> list[dict[str, Any]]:
    cols = c.numeric
    if len(cols) < 2:
        return []
    d = c.units(cols)
    tests = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            pair = d[[a, b]].dropna()
            if len(pair) < 20 or pair[a].nunique() < 3 or pair[b].nunique() < 3:
                continue
            r, pr = sps.pearsonr(pair[a], pair[b])
            s, ps = sps.spearmanr(pair[a], pair[b])
            tests.append((a, b, float(r), float(pr), float(s), float(ps), len(pair)))
    if not tests:
        return []
    q_p = _bh([t[3] for t in tests])
    q_s = _bh([t[5] for t in tests])
    out = []
    ranked = sorted(zip(tests, q_p, q_s, strict=True), key=lambda t: -abs(t[0][4]))
    shown = 0
    for (a, b, r, _pr, s, _ps, n), qp, qs in ranked:
        if shown >= 8:
            break
        if qs is not None and qs < 0.05 and abs(s) >= 0.3:
            z, se = np.arctanh(np.clip(r, -0.9999, 0.9999)), 1 / np.sqrt(max(n - 3, 1))
            lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
            strength = "strong" if abs(s) >= 0.7 else "moderate"
            out.append(
                _insight(
                    "correlation",
                    "association",
                    f"{strength.capitalize()} {'positive' if s > 0 else 'negative'} association: "
                    f"'{a}' and '{b}'",
                    f"Higher '{a}' tends to go with {'higher' if s > 0 else 'lower'} '{b}' "
                    f"(Spearman ρ = {s:.2f}). This is an association, not evidence that one causes "
                    f"the other.",
                    {
                        "spearman_rho": _r(s),
                        "spearman_q": qs,
                        "pearson_r": _r(r),
                        "pearson_q": qp,
                        "pearson_ci95": [_r(lo), _r(hi)],
                    },
                    n=n,
                    method=f"Pearson and Spearman correlation per {c.unit}; Fisher-z 95% CI for "
                    "Pearson; Benjamini–Hochberg q-values across all pairs.",
                    chart={"kind": "scatter", "columns": [a, b]},
                    columns=[a, b],
                    next_action="Inspect the scatter plot; strongly correlated inputs share "
                    "importance in models.",
                    importance=0.5 + 0.5 * abs(s),
                )
            )
            shown += 1
            if qs < 0.05 and abs(s) - abs(r) >= 0.15:
                out.append(
                    _insight(
                        "correlation",
                        "suspicion",
                        f"Possible nonlinear relationship: '{a}' and '{b}'",
                        "The rank correlation is clearly stronger than the linear one, which "
                        "suggests a "
                        "monotonic but nonlinear relationship (or the influence of extreme "
                        "values).",
                        {"spearman_rho": _r(s), "pearson_r": _r(r)},
                        n=n,
                        method="Difference between |Spearman ρ| and |Pearson r| ≥ 0.15.",
                        chart={"kind": "scatter", "columns": [a, b]},
                        columns=[a, b],
                        importance=0.45,
                        next_action="Consider a transformation or a nonlinear model.",
                    )
                )
    return out


def _group_differences(c: _Ctx) -> list[dict[str, Any]]:
    tests = []
    for g in c.categorical:
        d = c.units(c.numeric[:15], [g])
        if c.entity is not None and d[g].isna().mean() > 0.5:
            continue  # the category changes within entities: not assessable at entity level
        for col in c.numeric[:15]:
            if col == g:
                continue
            samples = [pd.to_numeric(s, errors="coerce").dropna() for _, s in d.groupby(g)[col]]
            samples = [s for s in samples if len(s) >= MIN_CELL]
            if len(samples) < 2:
                continue
            h = sps.kruskal(*samples)
            n, k = sum(len(s) for s in samples), len(samples)
            tests.append((g, col, float(h.pvalue), float((h.statistic - k + 1) / (n - k)), n))
    qs = _bh([t[2] for t in tests])
    out = []
    for (g, col, p, eps, n), q in sorted(zip(tests, qs, strict=True), key=lambda t: -t[0][3])[:8]:
        if q is not None and q < 0.05 and eps >= 0.06:
            out.append(
                _insight(
                    "group_difference",
                    "association",
                    f"'{col}' differs between '{g}' groups",
                    f"The distribution of '{col}' differs between levels of '{g}' (epsilon² = "
                    f"{eps:.2f}, "
                    "a moderate or larger effect). Descriptive: other variables may explain it.",
                    {
                        "epsilon_squared": _r(eps),
                        "p_value": _r(p, 6),
                        "q_value": q,
                        "groups": int(c.data[g].nunique()),
                    },
                    n=n,
                    method=f"Kruskal–Wallis test per {c.unit} (groups with ≥ {MIN_CELL}); "
                    "epsilon² effect size; Benjamini–Hochberg q-values.",
                    chart={"kind": "box", "columns": [col], "group_by": [{"column": g}]},
                    columns=[g, col],
                    importance=0.4 + min(eps, 0.6),
                    next_action=f"Compare '{col}' by '{g}' in a box plot; check for confounders.",
                )
            )
    return out


def _concentration(c: _Ctx) -> list[dict[str, Any]]:
    out = []
    for g in c.categorical:
        counts = c.full[g].dropna().astype(object).value_counts()
        if counts.sum() == 0:
            continue
        share = counts / counts.sum()
        hhi = float((share**2).sum())
        if share.iloc[0] >= 0.5 and len(counts) > 1:
            label = (
                "the largest group"
                if c.privacy_mode or g in c.protected
                else f"'{counts.index[0]}'"
            )
            out.append(
                _insight(
                    "concentration",
                    "descriptive",
                    f"'{g}' is dominated by one category",
                    f"{label} accounts for {100 * share.iloc[0]:.1f}% of rows; comparisons "
                    f"involving "
                    "the other categories rest on fewer observations.",
                    {
                        "top_share_pct": _r(100 * share.iloc[0], 2),
                        "levels": int(len(counts)),
                        "hhi": _r(hhi),
                    },
                    n=int(counts.sum()),
                    method="Category shares and the Herfindahl–Hirschman index (Σ share²).",
                    chart={
                        "kind": "aggregate",
                        "group_by": [{"column": g}],
                        "measures": [{"agg": "count"}],
                    },
                    columns=[g],
                    importance=0.3 + 0.3 * float(share.iloc[0]),
                )
            )
    return out


def _period(dt: pd.Series) -> tuple[pd.Series, str]:
    span = (dt.max() - dt.min()).days if dt.notna().any() else 0
    if span > 3 * 365:
        return dt.dt.to_period("Q").astype(str), "quarter"
    if span > 120:
        return dt.dt.to_period("M").astype(str), "month"
    if span > 21:
        return dt.dt.to_period("W").astype(str), "week"
    return dt.dt.strftime("%Y-%m-%d"), "day"


def _change_point(
    values: np.ndarray, seed: int, perms: int = 500
) -> tuple[int, float, float] | None:
    n = len(values)
    if n < 12:
        return None

    def best(v: np.ndarray) -> tuple[int, float]:
        csum = np.cumsum(v)
        total = csum[-1]
        k = np.arange(3, n - 2)
        left = csum[k - 1] / k
        right = (total - csum[k - 1]) / (n - k)
        stat = np.abs(left - right) * np.sqrt(k * (n - k) / n)
        i = int(np.argmax(stat))
        return int(k[i]), float(stat[i])

    k, s = best(values)
    rng = np.random.default_rng(seed)
    exceed = sum(1 for _ in range(perms) if best(rng.permutation(values))[1] >= s)
    return k, s, (exceed + 1) / (perms + 1)


def _temporal(c: _Ctx) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if c.time is None:
        return [], [{"area": "temporal", "reason": "No date or time column is available."}]
    dt = pd.to_datetime(c.full[c.time], errors="coerce")
    if dt.notna().sum() < 30:
        return [], [{"area": "temporal", "reason": "Fewer than 30 rows with a valid date."}]
    period, unit = _period(dt)
    frame = pd.DataFrame({"p": period[dt.notna()]})
    series: dict[str, pd.Series] = {"row count": frame.groupby("p").size()}
    measure_cols = [
        x
        for x in ([c.target] if c.target and c.task_type == "regression" else [])
        + list(c.roles.get("fin.amount", ()))
        + list(c.roles.get("ind.sensor", ()))
        if x in c.full.columns and x not in c.protected
    ][:3]
    for col in measure_cols:
        series[f"mean {col}"] = (
            pd.to_numeric(c.full.loc[dt.notna(), col], errors="coerce")
            .groupby(period[dt.notna()])
            .mean()
        )
    tests, out = [], []
    for name, s in series.items():
        s = s.sort_index().dropna()
        if len(s) < 6:
            continue
        tau, p = sps.kendalltau(np.arange(len(s)), s.to_numpy())
        tests.append((name, s, float(tau), float(p)))
    qs = _bh([t[3] for t in tests])
    note = (
        []
        if c.time_declared
        else [f"'{c.time}' is not declared as a time role; it was used as the date column."]
    )
    for (name, s, tau, p), q in zip(tests, qs, strict=True):
        if q is not None and q < 0.05 and abs(tau) >= 0.3:
            out.append(
                _insight(
                    "temporal",
                    "association",
                    f"{'Upward' if tau > 0 else 'Downward'} trend in {name} per {unit}",
                    f"{name.capitalize()} tends to {'increase' if tau > 0 else 'decrease'} over "
                    f"time "
                    f"(Kendall τ = {tau:.2f} across {len(s)} {unit}s).",
                    {
                        "kendall_tau": _r(tau),
                        "p_value": _r(p, 6),
                        "q_value": q,
                        "periods": len(s),
                        "first": _r(s.iloc[0]),
                        "last": _r(s.iloc[-1]),
                    },
                    n=int(dt.notna().sum()),
                    method=f"Mann–Kendall-type trend test (Kendall τ vs time) on {unit}ly values; "
                    f"Benjamini–Hochberg q-values.",
                    limitations=note + ["Autocorrelated series can overstate trend evidence."],
                    chart={
                        "kind": "aggregate",
                        "group_by": [{"column": c.time, "date_part": unit}],
                        "measures": [{"agg": "count"}]
                        if name == "row count"
                        else [{"agg": "mean", "column": name.removeprefix("mean ")}],
                    },
                    columns=[c.time],
                    importance=0.5 + 0.4 * abs(tau),
                    next_action="Check whether the trend reflects a real change or a change in "
                    "data collection.",
                )
            )
        cp = _change_point(s.to_numpy(dtype=float), c.seed)
        if cp is not None and cp[2] < 0.01:
            k = cp[0]
            before, after = float(s.iloc[:k].mean()), float(s.iloc[k:].mean())
            out.append(
                _insight(
                    "temporal",
                    "suspicion",
                    f"Possible level shift in {name} around {s.index[k]}",
                    f"The average {name} changes from {before:.4g} before {s.index[k]} to "
                    f"{after:.4g} "
                    "after it. This may be a real change, a data-collection change or a seasonal "
                    "effect.",
                    {
                        "change_at": str(s.index[k]),
                        "mean_before": _r(before),
                        "mean_after": _r(after),
                        "permutation_p": _r(cp[2], 4),
                        "periods": len(s),
                    },
                    n=len(s),
                    method="Single change in mean (CUSUM-type statistic), permutation test "
                    "with 500 seeded permutations.",
                    limitations=note
                    + ["Permutation tests assume exchangeable periods (no autocorrelation)."],
                    columns=[c.time],
                    importance=0.55,
                    next_action="Compare the periods before and after; check for process or data "
                    "changes.",
                )
            )
    if c.domain in ("business", "financial") and len(series["row count"]) >= 2:
        s = series["row count"].sort_index()
        last, prev = float(s.iloc[-1]), float(s.iloc[-2])
        if prev > 0:
            out.append(
                _insight(
                    "temporal",
                    "descriptive",
                    f"Latest {unit} vs previous: {100 * (last - prev) / prev:+.1f}% rows",
                    f"The latest {unit} ({s.index[-1]}) has {last:.0f} rows against {prev:.0f} in "
                    f"{s.index[-2]}. The latest period may be incomplete.",
                    {
                        "latest": last,
                        "previous": prev,
                        "change_pct": _r(100 * (last - prev) / prev, 2),
                    },
                    n=int(last + prev),
                    method="Period-over-period comparison of row counts.",
                    limitations=note,
                    columns=[c.time],
                    importance=0.35,
                )
            )
    return out, []


def _target(c: _Ctx) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if c.target is None or c.task_type is None:
        return [], [{"area": "target", "reason": "No target with a task type is declared."}]
    bio_roles = dict(c.roles)
    if c.entity and "bio.patient_id" not in bio_roles and "group" not in bio_roles:
        bio_roles = {**bio_roles, "group": [c.entity]}
    s = summarise(c.full, bio_roles, c.task_type, c.protected, c.numeric + [c.target])
    out = []
    t = s["target"]
    imb = t.get("imbalance") or {}
    if imb.get("minority_share") is not None and imb["minority_share"] < 0.2:
        out.append(
            _insight(
                "target",
                "descriptive",
                f"Class imbalance: minority class is {100 * imb['minority_share']:.1f}%",
                "Accuracy can look high by predicting the majority class; balanced accuracy and "
                "PR-AUC "
                "are more informative.",
                imb,
                n=c.n_rows,
                method="Class shares of the declared target.",
                chart={
                    "kind": "aggregate",
                    "group_by": [{"column": c.target}],
                    "measures": [{"agg": "count"}],
                }
                if c.target not in c.protected
                else None,
                columns=[c.target],
                importance=0.7,
            )
        )
    comp = s["comparisons"]
    if comp.get("status") == "assessed":
        for item in comp["items"][:6]:
            if (
                item.get("status") != "assessed"
                or item.get("q_value") is None
                or item["q_value"] >= 0.05
            ):
                continue
            eff = item["effect"]
            out.append(
                _insight(
                    "target",
                    "association",
                    f"'{item['column']}' is associated with the target",
                    f"'{item['column']}' differs with the target ({eff['name'].replace('_', ' ')} "
                    f"{eff['value']}). An association to investigate, not a causal or validated "
                    f"predictor.",
                    {
                        "effect": eff,
                        "p_value": item.get("p_value"),
                        "q_value": item["q_value"],
                        "unit": comp["unit"],
                    },
                    n=item.get("n") or sum((g.get("n") or 0) for g in item.get("groups", [])),
                    method=f"Biomedical Intelligence Lite comparison statistics per {comp['unit']} "
                    "(Hedges' g with Mann–Whitney, Kruskal–Wallis epsilon², or Spearman ρ); "
                    "Benjamini–Hochberg q-values.",
                    chart={
                        "kind": "box",
                        "columns": [item["column"]],
                        "group_by": [{"column": c.target}],
                    }
                    if c.classification_target and c.target not in c.protected
                    else {"kind": "scatter", "columns": [item["column"], c.target]}
                    if c.target not in c.protected
                    else None,
                    columns=[item["column"], c.target],
                    importance=0.5 + min(abs(eff.get("value") or 0), 1) * 0.4,
                    next_action="Evaluate predictive value with leakage-controlled training in "
                    "Models.",
                )
            )
    return out, []


def _domain(
    c: _Ctx, bio: Mapping[str, Any] | None
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    out: list[dict[str, Any]] = []
    na: list[dict[str, str]] = []
    if c.domain == "biomedical":
        if bio is None:
            na.append(
                {
                    "area": "biomedical",
                    "reason": "No biomedical summary is stored; run it on the Biomedical tab.",
                }
            )
        else:
            pats = bio["cohort"].get("patients", {})
            if pats.get("status") == "assessed":
                out.append(
                    _insight(
                        "biomedical",
                        "configuration",
                        f"{pats['unique']} patients, {pats['repeated_patients']} with repeated "
                        f"observations",
                        "Repeated observations of one patient are not independent; patient-grouped "
                        "splits and patient-level comparisons are used.",
                        {"observations_per_patient": pats["observations_per_patient"]},
                        n=pats["unique"],
                        method="Biomedical Intelligence Lite cohort summary.",
                        columns=[pats["column"]],
                        importance=0.6,
                    )
                )
            for w in bio.get("confounding", []):
                out.append(
                    _insight(
                        "biomedical",
                        "suspicion",
                        w["message"].split(".")[0],
                        w["message"],
                        {"kind": w["kind"]},
                        n=None,
                        method="Biomedical Intelligence Lite batch and covariate checks.",
                        columns=[w.get("column")] if w.get("column") else [],
                        importance=0.75,
                        limitations=[bio["disclaimer"]],
                    )
                )
    if c.domain == "financial":
        for col in [
            x for x in c.roles.get("fin.amount", ()) if x in c.full.columns and x not in c.protected
        ]:
            x = pd.to_numeric(c.full[col], errors="coerce").dropna()
            if len(x) < 30 or x.sum() == 0:
                continue
            top = x.sort_values(ascending=False)
            k = max(1, int(np.ceil(0.01 * len(x))))
            share = float(top.iloc[:k].sum() / x.sum())
            med = float(x.median())
            mad = float(np.median(np.abs(x - med))) * 1.4826
            unusual = int((np.abs(x - med) > 5 * mad).sum()) if mad > 0 else 0
            out.append(
                _insight(
                    "financial",
                    "descriptive",
                    f"Top 1% of transactions carry {100 * share:.1f}% of '{col}'",
                    f"The largest {k} transaction(s) account for {100 * share:.1f}% of the total. "
                    f"{unusual} value(s) are more than 5 robust SDs from the median: descriptive "
                    f"anomalies, "
                    "not evidence of fraud and not financial advice.",
                    {
                        "top1pct_share": _r(share),
                        "unusual_count": unusual,
                        "median": _r(med),
                        "robust_sd": _r(mad),
                    },
                    n=len(x),
                    method="Concentration of the largest 1% of values; robust z-score (median/MAD) "
                    "> 5.",
                    chart={"kind": "histogram", "columns": [col], "bins": 40},
                    columns=[col],
                    importance=0.55,
                )
            )
    if c.domain == "industrial":
        sensors = [
            x
            for k in ("ind.sensor", "ind.sensor_nonneg", "ind.sensor_percent")
            for x in c.roles.get(k, ())
            if x in c.full.columns and x not in c.protected
        ]
        asset = _first(c.roles, ("ind.asset_id",), c.full.columns)
        for col in sensors[:10]:
            x = pd.to_numeric(c.full[col], errors="coerce").dropna()
            if len(x) < 30:
                continue
            med = float(x.median())
            mad = float(np.median(np.abs(x - med))) * 1.4826
            abnormal = int((np.abs(x - med) > 5 * mad).sum()) if mad > 0 else 0
            if abnormal:
                out.append(
                    _insight(
                        "industrial",
                        "suspicion",
                        f"{abnormal} abnormal reading(s) of '{col}'",
                        "Readings far from the typical range can indicate faults, calibration "
                        "issues "
                        "or genuine events; units are not inferred.",
                        {"abnormal_count": abnormal, "median": _r(med), "robust_sd": _r(mad)},
                        n=len(x),
                        method="Robust z-score (median/MAD) > 5.",
                        chart={"kind": "box", "columns": [col]}
                        if asset is None
                        else {"kind": "box", "columns": [col], "group_by": [{"column": asset}]},
                        columns=[col],
                        importance=0.5,
                    )
                )
        if asset is None and sensors:
            na.append(
                {
                    "area": "industrial",
                    "reason": "No asset role is declared: equipment-level comparisons are not "
                    "assessed.",
                }
            )
    if c.domain == "business":
        cust = _first(c.roles, ("biz.customer_id",), c.full.columns)
        if cust is not None and cust not in c.protected:
            counts = c.full[cust].dropna().astype(object).value_counts()
            if len(counts) >= 10:
                k = max(1, int(np.ceil(0.1 * len(counts))))
                share = float(counts.iloc[:k].sum() / counts.sum())
                out.append(
                    _insight(
                        "business",
                        "descriptive",
                        f"Top 10% of customers account for {100 * share:.1f}% of rows",
                        "Customer activity is concentrated; aggregate results are driven by the "
                        "most "
                        "active customers.",
                        {"top10pct_share": _r(share), "customers": int(len(counts))},
                        n=int(counts.sum()),
                        method="Share of rows from the most active 10% of customers.",
                        columns=[cust],
                        importance=0.45,
                    )
                )
        elif cust is None:
            na.append(
                {
                    "area": "business",
                    "reason": "No customer role is declared: customer concentration is not "
                    "assessed.",
                }
            )
    return out, na


def raw_vs_cleaned(
    current: pd.DataFrame, parent: pd.DataFrame, protected: set[str]
) -> list[dict[str, Any]]:
    a, b = (
        parent.drop(columns=[ROW_ID], errors="ignore"),
        current.drop(columns=[ROW_ID], errors="ignore"),
    )
    added = sorted(set(b.columns) - set(a.columns))
    removed = sorted(set(a.columns) - set(b.columns))
    common = [c for c in a.columns if c in b.columns]
    ma, mb = (
        float(a[common].isna().mean().mean() * 100) if common else 0.0,
        float(b[common].isna().mean().mean() * 100) if common else 0.0,
    )
    ids_a = set(parent[ROW_ID]) if ROW_ID in parent else set()
    ids_b = set(current[ROW_ID]) if ROW_ID in current else set()
    shifts = []
    for col in common:
        if (
            col in protected
            or not pd.api.types.is_numeric_dtype(a[col])
            or not pd.api.types.is_numeric_dtype(b[col])
        ):
            continue
        xa, xb = a[col].dropna(), b[col].dropna()
        if len(xa) and len(xb) and xa.std() > 0:
            d = float((xb.mean() - xa.mean()) / xa.std())
            if abs(d) >= 0.1:
                shifts.append(
                    {
                        "column": col,
                        "mean_before": _r(xa.mean()),
                        "mean_after": _r(xb.mean()),
                        "std_shift": _r(d),
                    }
                )
    return [
        _insight(
            "versions",
            "descriptive",
            "Changes from the parent version",
            f"Rows {len(a)} → {len(b)}, columns {a.shape[1]} → {b.shape[1]}, mean missing share of "
            f"shared "
            f"columns {ma:.2f}% → {mb:.2f}%. Changes in values do not by themselves show better "
            f"predictive "
            "performance; only evaluated models can show that.",
            {
                "rows_before": len(a),
                "rows_after": len(b),
                "rows_in_both": len(ids_a & ids_b),
                "columns_added": added,
                "columns_removed": removed,
                "missing_pct_before": _r(ma, 3),
                "missing_pct_after": _r(mb, 3),
                "mean_shifts": shifts[:10],
            },
            n=len(b),
            method="Direct comparison of the two stored versions (row identity by _dm_row_id).",
            columns=added + [s["column"] for s in shifts[:5]],
            importance=0.5,
            next_action="Open the version comparison in Analytics Studio for distributions side by "
            "side.",
        )
    ]


def generate(
    frame: pd.DataFrame,
    schema: Sequence[Mapping[str, Any]],
    roles: Mapping[str, Sequence[str]],
    task_type: str | None,
    domain: str,
    protected: set[str],
    privacy_mode: bool,
    *,
    parent: pd.DataFrame | None = None,
    bio_summary: Mapping[str, Any] | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    c = _Ctx(frame, schema, roles, task_type, domain, protected, privacy_mode, seed)
    insights: list[dict[str, Any]] = []
    not_assessed: list[dict[str, str]] = []
    sections = [
        _distributions,
        _missingness,
        _outliers,
        _correlations,
        _group_differences,
        _concentration,
    ]
    for fn in sections:
        try:
            insights += fn(c)
        except (ValueError, FloatingPointError, ZeroDivisionError):
            not_assessed.append(
                {"area": fn.__name__.strip("_"), "reason": "Could not be computed."}
            )
    for fn2 in (_temporal, _target):
        try:
            got, na = fn2(c)
            insights += got
            not_assessed += na
        except (ValueError, FloatingPointError, ZeroDivisionError):
            not_assessed.append(
                {"area": fn2.__name__.strip("_"), "reason": "Could not be computed."}
            )
    got, na = _domain(c, bio_summary)
    insights += got
    not_assessed += na
    if parent is not None:
        insights += raw_vs_cleaned(frame, parent, protected)
    if c.entity is None:
        not_assessed.append(
            {
                "area": "repeated observations",
                "reason": "No entity role (patient, "
                "customer, account, asset or group) is declared: rows are treated as independent.",
            }
        )
    insights.sort(key=lambda i: -i["importance"])
    for i, item in enumerate(insights):
        item["id"] = f"i{i + 1}"
    return {
        "version": INSIGHTS_VERSION,
        "rows": c.n_rows,
        "sampled": c.sampled,
        "sample_rows": len(c.data),
        "unit": c.unit,
        "entity_column": c.entity,
        "time_column": c.time,
        "insights": insights,
        "counts": {
            b: sum(1 for i in insights if i["basis"] == b)
            for b in ("descriptive", "association", "suspicion", "configuration")
        },
        "not_assessed": not_assessed,
        "limitations": LIMITATIONS,
    }

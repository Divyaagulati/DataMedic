"""Before/after comparison of two versions of a dataset (Phase 4 Part E). Read-only; counts
and statistics only, never row values.

Rows are aligned by ``_dm_row_id``. For every column present in both versions the comparison
reports missing values, values changed and rows affected (on aligned rows), outlier counts
(IQR x 1.5), a distribution summary, and a two-sample test:
- numeric: Kolmogorov-Smirnov (statistic D is also the effect size) plus the standardised
  mean difference;
- categorical / boolean: chi-square on the category counts, with Cramer's V.

Tests are ``not_assessed`` (with a reason) for constant columns, fewer than 8 observed values
on either side, changed types, or too many categories. The two versions describe the same
rows, so they are not independent samples: p-values are descriptive, and a significant
difference says nothing about practical importance, causation or scientific validity.
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from datamedic.engine import ROW_ID

MIN_VALUES = 8
MAX_CATEGORIES = 50
MAX_TEST_ROWS = 200_000  # seeded sample per side above this, disclosed
MAX_COLUMNS = 500

LIMITATIONS = [
    "Both versions describe the same rows, so they are not independent samples: p-values "
    "are descriptive only.",
    "Statistical significance does not establish practical importance, causation or "
    "scientific validity; large datasets make tiny differences significant.",
    "A changed distribution is expected after a transformation; it is not an error by itself.",
    f"Tests use at most {MAX_TEST_ROWS:,} values per side (a seeded sample above that).",
]


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if _numeric(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    return "categorical"


def _num(value: Any) -> float | None:
    if value is None:
        return None
    f = float(value)
    return round(f, 6) if np.isfinite(f) else None


def _outliers(values: np.ndarray) -> int | None:
    if values.size < 4:
        return None
    q1, q3 = np.quantile(values, [0.25, 0.75])
    iqr = q3 - q1
    if iqr <= 0:
        return 0
    return int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())


def _sample(values: np.ndarray, seed: int) -> tuple[np.ndarray, bool]:
    if values.size <= MAX_TEST_ROWS:
        return values, False
    rng = np.random.default_rng(seed)
    return values[rng.choice(values.size, MAX_TEST_ROWS, replace=False)], True


def _describe(s: pd.Series) -> dict[str, Any]:
    out: dict[str, Any] = {
        "kind": _kind(s),
        "missing": int(s.isna().sum()),
        "missing_pct": _num(100 * s.isna().mean()) if len(s) else None,
    }
    present = s.dropna()
    if _numeric(s) and len(present):
        v = present.to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            q1, med, q3 = np.quantile(v, [0.25, 0.5, 0.75])
            out.update(
                mean=_num(v.mean()),
                std=_num(v.std(ddof=1)) if v.size > 1 else None,
                min=_num(v.min()),
                q1=_num(q1),
                median=_num(med),
                q3=_num(q3),
                max=_num(v.max()),
                outliers_iqr_1_5=_outliers(v),
                skewness=_num(pd.Series(v).skew()) if v.size >= 3 else None,
            )
    else:
        out["distinct"] = int(present.nunique())
    return out


def _test(a: pd.Series, b: pd.Series, seed: int) -> dict[str, Any]:
    ka, kb = _kind(a), _kind(b)
    if ka != kb:
        return {
            "status": "not_assessed",
            "reason": f"The type changed ({ka} → {kb}); distributions are not comparable.",
        }
    pa, pb = a.dropna(), b.dropna()
    if len(pa) < MIN_VALUES or len(pb) < MIN_VALUES:
        return {
            "status": "not_assessed",
            "reason": f"Fewer than {MIN_VALUES} observed values in one of the versions.",
        }
    if ka == "datetime":
        return {"status": "not_assessed", "reason": "Date and time columns are not tested."}
    if ka == "numeric":
        va = pa.to_numpy(dtype=float)
        vb = pb.to_numpy(dtype=float)
        va, vb = va[np.isfinite(va)], vb[np.isfinite(vb)]
        if np.ptp(va) == 0 and np.ptp(vb) == 0:
            return {"status": "not_assessed", "reason": "The column is constant in both versions."}
        sa, sampled_a = _sample(va, seed)
        sb, sampled_b = _sample(vb, seed + 1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stats.ks_2samp(sa, sb)
        approximate = any(issubclass(w.category, RuntimeWarning) for w in caught)
        pooled = (
            np.sqrt((va.var(ddof=1) + vb.var(ddof=1)) / 2) if va.size > 1 and vb.size > 1 else 0.0
        )
        smd = (vb.mean() - va.mean()) / pooled if pooled > 0 else None
        return {
            "status": "assessed",
            "test": "kolmogorov_smirnov",
            "statistic": _num(result.statistic),
            "p_value": _num(result.pvalue),
            "effect_size": {
                "ks_d": _num(result.statistic),
                "standardised_mean_difference": _num(smd),
            },
            "sampled": sampled_a or sampled_b,
            "p_value_approximate": approximate,  # SciPy fell back to the asymptotic method
        }
    levels = sorted(set(pa.astype(str)) | set(pb.astype(str)))
    if len(levels) < 2:
        return {"status": "not_assessed", "reason": "Only one category in both versions."}
    if len(levels) > MAX_CATEGORIES:
        return {"status": "not_assessed", "reason": f"More than {MAX_CATEGORIES} categories."}
    ca = pa.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy()
    cb = pb.astype(str).value_counts().reindex(levels, fill_value=0).to_numpy()
    table = np.vstack([ca, cb])
    table = table[:, table.sum(axis=0) > 0]
    chi2, p, dof, expected = stats.chi2_contingency(table)
    n = table.sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1))) if n and min(table.shape) > 1 else None
    sparse = bool((expected < 5).mean() > 0.2)
    return {
        "status": "assessed",
        "test": "chi_square",
        "statistic": _num(chi2),
        "p_value": _num(p),
        "dof": int(dof),
        "effect_size": {"cramers_v": _num(v)},
        "sparse": sparse,
        "note": "More than 20% of expected counts are below 5; the chi-square "
        "approximation is unreliable."
        if sparse
        else None,
    }


def compare(a: pd.DataFrame, b: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    cols_a = [c for c in a.columns if c != ROW_ID]
    cols_b = [c for c in b.columns if c != ROW_ID]
    common = [c for c in cols_a if c in cols_b]
    truncated = max(0, len(common) - MAX_COLUMNS)
    common = common[:MAX_COLUMNS]
    ids_a, ids_b = a[ROW_ID], b[ROW_ID]
    shared = np.intersect1d(ids_a.to_numpy(), ids_b.to_numpy())
    ia = pd.Series(np.arange(len(a)), index=ids_a.to_numpy()).loc[shared].to_numpy()
    ib = pd.Series(np.arange(len(b)), index=ids_b.to_numpy()).loc[shared].to_numpy()
    affected = np.zeros(len(shared), dtype=bool)
    columns = []
    for name in common:
        sa, sb = a[name], b[name]
        xa, xb = sa.iloc[ia].reset_index(drop=True), sb.iloc[ib].reset_index(drop=True)
        na, nb = xa.isna().to_numpy(), xb.isna().to_numpy()
        if _numeric(xa) and _numeric(xb):
            differ = xa.to_numpy(dtype=float, na_value=np.nan) != xb.to_numpy(
                dtype=float, na_value=np.nan
            )
        else:
            differ = xa.astype(object).to_numpy() != xb.astype(object).to_numpy()
        changed = (na != nb) | (~na & ~nb & differ)
        affected |= changed
        columns.append(
            {
                "column": name,
                "type_before": _kind(sa),
                "type_after": _kind(sb),
                "values_changed": int((~na & ~nb & differ).sum()),
                "filled": int((na & ~nb).sum()),
                "emptied": int((~na & nb).sum()),
                "before": _describe(sa),
                "after": _describe(sb),
                "test": _test(sa, sb, seed),
            }
        )
    added = [c for c in cols_b if c not in cols_a]
    removed = [c for c in cols_a if c not in cols_b]
    loss = []
    if removed:
        loss.append("Columns were removed.")
    if any(c["values_changed"] for c in columns):
        loss.append("Observed values differ between the versions.")
    if any(c["filled"] for c in columns):
        loss.append("Missing values were filled (estimates).")
    if len(b) < len(a):
        loss.append("Rows were removed.")
    return {
        "rows_before": len(a),
        "rows_after": len(b),
        "rows_aligned": int(len(shared)),
        "rows_only_before": int(len(a) - len(shared)),
        "rows_only_after": int(len(b) - len(shared)),
        "rows_affected": int(affected.sum()),
        "columns_before": len(cols_a),
        "columns_after": len(cols_b),
        "columns_added": added,
        "columns_removed": removed,
        "columns_type_changed": [
            c["column"] for c in columns if c["type_before"] != c["type_after"]
        ],
        "columns_not_compared": truncated,
        "columns": columns,
        "information_loss": loss,
        "limitations": LIMITATIONS,
    }

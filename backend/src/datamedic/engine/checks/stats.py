"""Statistical helpers shared by checks. Every function documents its method."""

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import association


def as_float(series: pd.Series) -> np.ndarray:
    """Numeric or boolean column as float64 (missing -> NaN)."""
    return np.asarray(
        series.astype("Float64").to_numpy(dtype="float64", na_value=np.nan), dtype=np.float64
    )


def num(value: Any, digits: int = 6) -> float | None:
    """JSON-safe rounded float (NaN/inf -> None)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float | None, float | None, int]:
    """Cramer's V (scipy ``association``, no bias correction) and the chi-square p-value
    on rows where both are present. Returns (V, p, n); V None when either side has < 2 levels."""
    mask = x.notna() & y.notna()
    if mask.sum() < 2:
        return None, None, int(mask.sum())
    table = pd.crosstab(x[mask].astype(str), y[mask].astype(str))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None, None, int(mask.sum())
    v = float(association(table.to_numpy(), method="cramer"))
    p = float(stats.chi2_contingency(table.to_numpy()).pvalue)
    return v, p, int(mask.sum())


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """ROC AUC via the Mann-Whitney statistic: P(score_pos > score_neg) + 0.5 P(tie).
    ``labels`` are booleans. None if a class is empty."""
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return None
    ranks = stats.rankdata(scores)
    u = ranks[labels].sum() - pos * (pos + 1) / 2
    return float(u / (pos * neg))


def bh_adjust(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values (``scipy.stats.false_discovery_control``)."""
    if not pvalues:
        return []
    return [float(q) for q in stats.false_discovery_control(np.asarray(pvalues), method="bh")]


def kruskal_by_group(values: np.ndarray, groups: np.ndarray, min_size: int = 3) -> dict[str, Any]:
    """Kruskal-Wallis H across groups with >= ``min_size`` values, plus epsilon-squared
    effect size H / (n - 1) (Tomczak and Tomczak 2014). Empty dict if < 2 usable groups."""
    mask = ~np.isnan(values)
    frame = pd.DataFrame({"v": values[mask], "g": groups[mask]})
    samples = [g["v"].to_numpy() for _, g in frame.groupby("g") if len(g) >= min_size]
    n = sum(len(s) for s in samples)
    if len(samples) < 2 or n < 3 or all(np.ptp(s) == 0 for s in samples):
        return {}
    try:
        h, p = stats.kruskal(*samples)
    except ValueError:  # all values identical
        return {}
    return {"h": float(h), "p": float(p), "epsilon_sq": float(h) / (n - 1), "groups": len(samples)}


def robust_z(values: np.ndarray) -> np.ndarray | None:
    """Modified z-score 0.6745 (x - median) / MAD (Iglewicz and Hoaglin 1993); None if MAD=0."""
    present = values[~np.isnan(values)]
    if present.size == 0:
        return None
    median = np.median(present)
    mad = np.median(np.abs(present - median))
    if mad == 0:
        return None
    return np.asarray(0.6745 * (values - median) / mad)

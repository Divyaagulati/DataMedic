"""Industrial check implementations (worker only)."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from datamedic.domains.industrial import SENSOR_ROLES
from datamedic.engine.checks.framework import (
    CheckContext,
    Finding,
    detected,
    not_assessed,
    not_detected,
)
from datamedic.engine.checks.spec import CheckSpec
from datamedic.engine.checks.stats import as_float, bh_adjust, num


def _sensors(ctx: CheckContext) -> list[tuple[str, str]]:
    """(column, sensor role) pairs; a column may carry several sensor roles."""
    return [(c, role) for role in sorted(SENSOR_ROLES) for c in ctx.roles.all(role)]


def _sensor_columns(ctx: CheckContext) -> list[str]:
    """Each sensor column once (selecting a column twice would duplicate it in the frame)."""
    return list(dict.fromkeys(c for c, _ in _sensors(ctx)))


def _time_ordered(ctx: CheckContext, columns: list[str]) -> pd.DataFrame:
    time_col = ctx.roles.first("ind.timestamp")
    assert time_col is not None
    frame = ctx.data[[time_col, *columns]].dropna(subset=[time_col])
    return frame.sort_values(time_col, kind="stable")


def impossible_values(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    findings = []
    for column, role in _sensors(ctx):
        values = as_float(ctx.data[column])
        present = values[~np.isnan(values)]
        if present.size == 0:
            continue
        finite = present[np.isfinite(present)]
        non_finite = int(present.size - finite.size)
        out_of_range = 0
        rule = None
        if role == "ind.sensor_nonneg":
            out_of_range, rule = int((finite < 0).sum()), "never negative"
        elif role == "ind.sensor_percent":
            out_of_range, rule = int(((finite < 0) | (finite > 100)).sum()), "between 0 and 100"
        saturated = {}
        if finite.size >= 20 and np.ptp(finite) > 0:
            for edge, value in (("at_minimum", finite.min()), ("at_maximum", finite.max())):
                share = float((finite == value).mean())
                if share >= 0.05:
                    saturated[edge] = {"value": num(value), "share_pct": num(share * 100, 2)}
        evidence = {
            "sensor_type": role,
            "rule": rule,
            "non_finite": non_finite,
            "outside_physical_range": out_of_range,
            "saturation": saturated,
        }
        if not (non_finite or out_of_range or saturated):
            findings.append(
                not_detected(
                    spec,
                    f"'{column}' has no impossible or saturated readings.",
                    evidence,
                    columns=[column],
                )
            )
            continue
        severity = "high" if out_of_range else "medium" if non_finite else "low"
        parts = []
        if out_of_range:
            parts.append(f"{out_of_range} readings break the rule '{rule}'")
        if non_finite:
            parts.append(f"{non_finite} readings are not finite")
        if saturated:
            parts.append(
                "many readings sit exactly at the "
                + " and ".join(k.replace("at_", "") for k in saturated)
                + ", suggesting the sensor is clipping"
            )
        findings.append(
            detected(
                spec,
                ctx,
                severity,
                f"Sensor '{column}' has impossible or saturated readings",
                "; ".join(parts).capitalize() + ".",
                evidence,
                columns=[column],
                affected_rows=(out_of_range + non_finite) or None,
                recommendation="Check sensor calibration and range settings. Mark impossible "
                "readings as missing in a reviewed step; treat clipped readings as censored.",
            )
        )
    return findings


def irregular_sampling(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    time_col = ctx.roles.first("ind.timestamp")
    asset = ctx.roles.first("ind.asset_id")
    assert time_col is not None
    frame = _time_ordered(ctx, [asset] if asset else [])
    groups = frame.groupby(asset) if asset else [("all", frame)]
    deltas: list[np.ndarray] = []
    duplicates = 0
    for _, group in groups:
        times = group[time_col]
        duplicates += int(times.duplicated().sum())
        if len(times) >= 2:
            d = times.diff().dropna().dt.total_seconds().to_numpy()
            deltas.append(d[d > 0])
    all_d = np.concatenate(deltas) if deltas else np.array([])
    if all_d.size < 5:
        return [not_assessed(spec, "fewer than 5 intervals between distinct timestamps.")]
    median = float(np.median(all_d))
    gaps = int((all_d > 5 * median).sum())
    cv = float(all_d.std() / all_d.mean()) if all_d.mean() > 0 else 0.0
    evidence = {
        "median_interval_seconds": num(median, 3),
        "gaps_over_5x_median": gaps,
        "interval_cv": num(cv, 3),
        "duplicate_timestamps": duplicates,
        "max_interval_seconds": num(all_d.max(), 3),
        "per_asset": bool(asset),
    }
    if not (gaps or duplicates or cv > 0.5):
        return [
            not_detected(
                spec, "Measurements are taken at regular intervals.", evidence, columns=[time_col]
            )
        ]
    return [
        detected(
            spec,
            ctx,
            "medium" if gaps or duplicates else "low",
            "Measurement intervals are irregular",
            f"{gaps} gaps exceed five times the typical interval, {duplicates} timestamps are "
            f"repeated, and the interval variability (CV) is {cv:.2f}. Rolling features and "
            "lags assume regular sampling.",
            evidence,
            columns=[time_col, *([asset] if asset else [])],
            affected_rows=(gaps + duplicates) or None,
            recommendation="Resample to a fixed frequency or use time-aware features; confirm "
            "whether gaps are downtime or data loss.",
        )
    ]


def _early_late(ctx: CheckContext) -> tuple[pd.DataFrame, pd.DataFrame, list[str]] | None:
    sensors = _sensor_columns(ctx)
    frame = _time_ordered(ctx, sensors)
    if len(frame) < 30:
        return None
    third = len(frame) // 3
    return frame.iloc[:third], frame.iloc[-third:], sensors


def distribution_shift(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    split = _early_late(ctx)
    if split is None:
        return [not_assessed(spec, "at least 30 time-stamped rows are needed.")]
    early, late, sensors = split
    tests = {}
    for column in sensors:
        a, b = as_float(early[column]), as_float(late[column])
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        if a.size < 10 or b.size < 10:
            continue
        iqr = np.subtract(*np.quantile(np.concatenate([a, b]), [0.75, 0.25]))
        shift = abs(np.median(b) - np.median(a)) / iqr if iqr > 0 else 0.0
        tests[column] = {
            "ks": float(stats.ks_2samp(a, b).statistic),
            "p": float(stats.ks_2samp(a, b).pvalue),
            "median_shift_iqr": shift,
            "early_median": float(np.median(a)),
            "late_median": float(np.median(b)),
        }
    if not tests:
        return [not_assessed(spec, "no sensor has 10 or more readings in both periods.")]
    qvalues = bh_adjust([t["p"] for t in tests.values()])
    shifted: list[dict[str, Any]] = []
    for (column, t), q in zip(tests.items(), qvalues, strict=True):
        if q < 0.01 and t["median_shift_iqr"] >= 0.5:
            shifted.append(
                {
                    "sensor": column,
                    "ks_statistic": num(t["ks"], 4),
                    "q_value": num(q, 8),
                    "median_shift_in_iqr": num(t["median_shift_iqr"], 3),
                    "early_median": num(t["early_median"]),
                    "late_median": num(t["late_median"]),
                }
            )
    if not shifted:
        return [
            not_detected(
                spec,
                "No sensor's distribution shifts between the earliest and "
                "latest third of the period.",
                {"sensors_tested": list(tests)},
            )
        ]
    return [
        detected(
            spec,
            ctx,
            "medium",
            f"{len(shifted)} sensors shift between early and late periods",
            "The distribution of these sensors differs between the start and end of the "
            "recording. Models trained on early data may not hold later.",
            {"shifted": shifted, "sensors_tested": len(tests)},
            columns=[s["sensor"] for s in shifted],
            recommendation="Find out whether the process, the sensor or its calibration "
            "changed. Evaluate models with time-ordered splits.",
        )
    ]


def sensor_drift(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    time_col = ctx.roles.first("ind.timestamp")
    assert time_col is not None
    sensors = _sensor_columns(ctx)
    frame = _time_ordered(ctx, sensors)
    if len(frame) < 30:
        return [not_assessed(spec, "at least 30 time-stamped rows are needed.")]
    t = frame[time_col].astype("int64").to_numpy(dtype=float)
    results = {}
    for column in sensors:
        v = as_float(frame[column])
        ok = np.isfinite(v)
        if ok.sum() < 30 or np.ptp(v[ok]) == 0:
            continue
        res = stats.spearmanr(t[ok], v[ok])
        results[column] = (float(res.statistic), float(res.pvalue))
    if not results:
        return [not_assessed(spec, "no sensor has 30 or more varying readings.")]
    qvalues = bh_adjust([p for _, p in results.values()])
    drifting: list[dict[str, Any]] = [
        {"sensor": c, "spearman_with_time": num(rho, 4), "q_value": num(q, 8)}
        for (c, (rho, _)), q in zip(results.items(), qvalues, strict=True)
        if abs(rho) >= 0.5 and q < 0.01
    ]
    if not drifting:
        return [
            not_detected(
                spec, "No sensor trends strongly over time.", {"sensors_tested": list(results)}
            )
        ]
    return [
        detected(
            spec,
            ctx,
            "medium",
            f"{len(drifting)} sensors trend steadily over time",
            "These readings rise or fall consistently with time. This can indicate sensor "
            "drift (needing recalibration), wear, or a genuine process trend.",
            {"drifting": drifting, "sensors_tested": len(results)},
            columns=[d["sensor"] for d in drifting],
            recommendation="Compare against reference measurements or maintenance logs; "
            "evaluate models on later periods than they were trained on.",
        )
    ]

"""Financial check implementations (worker only)."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from datamedic.engine.checks.framework import (
    CheckContext,
    Finding,
    detected,
    not_assessed,
    not_detected,
)
from datamedic.engine.checks.spec import CheckSpec
from datamedic.engine.checks.stats import as_float, num, robust_z


def _naive_utc(series: pd.Series) -> pd.Series:
    return series.dt.tz_convert("UTC").dt.tz_localize(None) if series.dt.tz is not None else series


def timestamp_consistency(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    column = ctx.roles.first("fin.timestamp")
    assert column is not None
    times = _naive_utc(ctx.data[column])
    now = pd.Timestamp(ctx.now).tz_convert("UTC").tz_localize(None)
    missing = int(times.isna().sum())
    future = int((times > now).sum())
    early = int((times < pd.Timestamp("1970-01-01")).sum())
    conflicting = 0
    txn = ctx.roles.first("fin.transaction_id")
    if txn:
        per_id = (
            pd.DataFrame({"id": ctx.data[txn], "t": times}).dropna().groupby("id")["t"].nunique()
        )
        conflicting = int((per_id > 1).sum())
    evidence = {
        "missing": missing,
        "future": future,
        "before_1970": early,
        "transaction_ids_with_several_times": conflicting,
        "checked_against": now.isoformat(timespec="seconds"),
    }
    issues = missing + future + early + conflicting
    if issues == 0:
        return [
            not_detected(
                spec, "All transaction times are present and plausible.", evidence, columns=[column]
            )
        ]
    return [
        detected(
            spec,
            ctx,
            "high" if future or conflicting else "medium",
            f"Transaction times in '{column}' are inconsistent",
            f"{future} times are in the future, {early} before 1970, {missing} missing, and "
            f"{conflicting} transaction IDs appear with different times.",
            evidence,
            columns=[column, *([txn] if txn else [])],
            affected_rows=missing + future + early,
            recommendation="Check time zones, clock errors and default dates at the source. "
            "Time-based features and splits are unreliable until these are resolved.",
        )
    ]


def duplicate_transactions(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    txn = ctx.roles.first("fin.transaction_id")
    if txn:
        ids = ctx.data[txn].dropna()
        counts = ids.value_counts()
        repeated = counts[counts > 1]
        if repeated.empty:
            return [
                not_detected(
                    spec,
                    "Every transaction ID is unique.",
                    {"transactions": int(counts.size)},
                    columns=[txn],
                )
            ]
        rows = int(repeated.sum())
        exact = int(ctx.data[ctx.data[txn].isin(repeated.index)].duplicated(keep="first").sum())
        return [
            detected(
                spec,
                ctx,
                "high",
                f"{repeated.size} transaction IDs occur more than once",
                f"{rows} rows share a transaction ID; {exact} of the repeats are identical in "
                "every column (likely double-loaded records).",
                {
                    "repeated_ids": int(repeated.size),
                    "rows": rows,
                    "exact_copies": exact,
                    "max_repeats": int(repeated.max()),
                },
                columns=[txn],
                affected_rows=rows,
                recommendation="Deduplicate in a reviewed preprocessing step after confirming "
                "with the source; amounts summed over duplicates are overstated.",
            )
        ]
    keys = [ctx.roles.first(r) for r in ("fin.account_id", "fin.amount", "fin.timestamp")]
    if any(k is None for k in keys):
        return [
            not_assessed(
                spec,
                "without a transaction ID, the account, amount and "
                "transaction-time roles are all needed.",
            )
        ]
    cols = [k for k in keys if k]
    dup = ctx.data[cols].dropna().duplicated(keep=False)
    rows = int(dup.sum())
    if rows == 0:
        return [
            not_detected(
                spec,
                "No account has the same amount at the same time twice.",
                {"key_columns": cols},
                columns=cols,
            )
        ]
    return [
        detected(
            spec,
            ctx,
            "medium",
            f"{rows} rows repeat the same account, amount and time",
            "These rows look like the same transaction recorded more than once, though split "
            "or repeated payments are possible.",
            {"rows": rows, "key_columns": cols},
            columns=cols,
            affected_rows=rows,
            recommendation="Confirm with the source system before removing anything.",
        )
    ]


def temporal_leakage(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    column = ctx.roles.first("fin.timestamp")
    target = ctx.target()
    assert column is not None and target is not None
    frame = pd.DataFrame({"t": ctx.data[column], "y": ctx.data[target]}).dropna()
    if len(frame) < 50 or frame["t"].nunique() < 5:
        return [
            not_assessed(
                spec,
                "at least 50 rows with both a time and a target, over 5 "
                "distinct times, are needed.",
            )
        ]
    frame = frame.sort_values("t", kind="stable")
    frame["period"] = pd.qcut(frame["t"].rank(method="first"), 5, labels=False)
    if ctx.roles.task_type == "regression":
        samples = [as_float(g["y"]) for _, g in frame.groupby("period")]
        p = float(stats.kruskal(*samples).pvalue)
        by_period: dict[int, Any] = {
            int(k): num(as_float(g["y"]).mean(), 4) for k, g in frame.groupby("period")
        }
        measure = "mean_target_by_period"
    else:
        table = pd.crosstab(frame["period"], frame["y"].astype(str))
        p = float(stats.chi2_contingency(table.to_numpy()).pvalue) if table.shape[1] > 1 else 1.0
        rates = pd.crosstab(frame["period"], frame["y"].astype(str), normalize="index")
        by_period = {
            int(k): {str(c): num(v * 100, 2) for c, v in row.items()}
            for k, row in rates.iloc[:, :5].iterrows()
        }
        measure = "target_share_pct_by_period"
    evidence = {
        measure: by_period,
        "p_value": num(p, 8),
        "periods": 5,
        "time_range": [frame["t"].min().isoformat(), frame["t"].max().isoformat()],
    }
    if p >= 0.01:
        return [
            not_detected(
                spec,
                "The target does not change detectably over time; "
                "time-ordered evaluation is still the safer choice.",
                evidence,
                columns=[column, target],
            )
        ]
    return [
        detected(
            spec,
            ctx,
            "high",
            "The target changes over time: random splits would leak the future",
            "The outcome distribution differs between time periods. A random train/test split "
            "lets the model learn from later periods to predict earlier ones, which overstates "
            "real-world performance.",
            evidence,
            columns=[column, target],
            recommendation="Evaluate with time-ordered splits (train on the past, test on the "
            "future, e.g. TimeSeriesSplit with a gap) and check that features use only "
            "information available at transaction time.",
        )
    ]


def unusual_amounts(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    column = ctx.roles.first("fin.amount")
    assert column is not None
    values = as_float(ctx.data[column])
    present = values[~np.isnan(values)]
    if present.size == 0:
        return [not_assessed(spec, f"'{column}' has no values.")]
    finite = present[np.isfinite(present)]
    non_finite = int(present.size - finite.size)
    zeros = int((finite == 0).sum())
    negative = int((finite < 0).sum())
    rz = robust_z(np.log1p(np.abs(finite)))
    extreme = int((np.abs(rz) > 3.5).sum()) if rz is not None else 0
    nonzero = finite[finite != 0]
    round_share = float((np.mod(nonzero, 100) == 0).mean()) if nonzero.size else 0.0
    evidence = {
        "non_finite": non_finite,
        "zero": zeros,
        "negative": negative,
        "extreme_by_log_modified_z": extreme,
        "round_hundreds_share_pct": num(round_share * 100, 2),
        "median": num(np.median(finite)) if finite.size else None,
        "max": num(finite.max()) if finite.size else None,
    }
    notes = []
    if non_finite:
        notes.append(f"{non_finite} non-finite")
    if extreme:
        notes.append(f"{extreme} extreme")
    if zeros:
        notes.append(f"{zeros} zero")
    if negative:
        notes.append(f"{negative} negative")
    if round_share >= 0.5 and nonzero.size >= 20:
        notes.append(f"{round_share * 100:.0f}% exact multiples of 100")
    if not notes:
        return [not_detected(spec, "Amounts show no unusual values.", evidence, columns=[column])]
    severity = "medium" if non_finite or extreme > 0.01 * finite.size else "low"
    return [
        detected(
            spec,
            ctx,
            severity,
            f"'{column}' has unusual amounts",
            "Amounts include " + ", ".join(notes) + " values. These may be refunds, fees, test "
            "transactions, entry errors or genuine large payments.",
            evidence,
            columns=[column],
            affected_rows=non_finite + extreme + zeros + negative,
            recommendation="Review the categories with the data owner (e.g. are negatives "
            "refunds?). Model them explicitly rather than deleting them.",
        )
    ]

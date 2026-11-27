"""Business check implementations (worker only)."""

import re

import pandas as pd

from datamedic.engine.checks.framework import (
    CheckContext,
    Finding,
    detected,
    not_assessed,
    not_detected,
)
from datamedic.engine.checks.spec import CheckSpec
from datamedic.engine.checks.stats import num

PLACEHOLDER_IDS = frozenset(
    {
        "0",
        "-1",
        "00000",
        "000000",
        "test",
        "dummy",
        "unknown",
        "none",
        "null",
        "n/a",
        "na",
        "xxx",
        "tbd",
        "999999",
        "123456",
        "temp",
    }
)


def _pattern(value: str) -> str:
    """Shape of an ID: letters -> A, digits -> 9, runs collapsed (e.g. 'CU-0042' -> 'A-9')."""
    shape = re.sub(r"[A-Za-z]", "A", re.sub(r"\d", "9", value))
    return re.sub(r"(.)\1+", r"\1", shape)


def duplicate_customers(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    column = ctx.roles.first("biz.customer_id")
    assert column is not None
    ids = ctx.data[column].dropna().astype(str)
    counts = ids.value_counts()
    repeated = counts[counts > 1]
    normalised = ids.str.strip().str.lower().str.lstrip("0")
    collisions = normalised.groupby(normalised).agg(lambda s: ids[s.index].nunique())
    collided = collisions[collisions > 1]
    if repeated.empty and collided.empty:
        return [
            not_detected(
                spec,
                "Every customer ID is unique, including after normalising "
                "case, spaces and leading zeros.",
                {"customers": int(counts.size)},
                columns=[column],
            )
        ]
    rows = int(repeated.sum())
    return [
        detected(
            spec,
            ctx,
            "medium",
            f"Customer IDs in '{column}' repeat or collide",
            f"{repeated.size} IDs occur more than once ({rows} rows) and {collided.size} IDs "
            "differ only by case, spaces or leading zeros.",
            {
                "repeated_ids": int(repeated.size),
                "rows_with_repeated_id": rows,
                "near_duplicate_id_groups": int(collided.size),
            },
            columns=[column],
            affected_rows=rows or None,
            recommendation="If the table should have one row per customer, consolidate the "
            "records with the source system. Otherwise evaluate models grouped by customer.",
        )
    ]


def suspicious_ids(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    column = ctx.roles.first("biz.customer_id")
    assert column is not None
    ids = ctx.data[column].dropna().astype(str)
    if ids.empty:
        return [not_assessed(spec, f"'{column}' has no values.")]
    placeholders = int(ids.str.strip().str.lower().isin(PLACEHOLDER_IDS).sum())
    spaced = int((ids != ids.str.strip()).sum())
    shapes = ids.str.strip().map(_pattern).value_counts()
    dominant_share = float(shapes.iloc[0] / shapes.sum())
    mixed = dominant_share < 0.9 and ids.size >= 20
    evidence = {
        "placeholders": placeholders,
        "with_surrounding_spaces": spaced,
        "formats": int(shapes.size),
        "dominant_format_share_pct": num(dominant_share * 100, 2),
        "top_formats": {k: int(v) for k, v in shapes.head(5).items()},
    }
    if not (placeholders or spaced or mixed):
        return [not_detected(spec, "Customer IDs look consistent.", evidence, columns=[column])]
    return [
        detected(
            spec,
            ctx,
            "medium" if placeholders else "low",
            f"Customer IDs in '{column}' look suspicious",
            f"{placeholders} IDs are placeholders, {spaced} have surrounding spaces"
            + (f", and {shapes.size} different ID formats are mixed." if mixed else "."),
            evidence,
            columns=[column],
            affected_rows=placeholders + spaced or None,
            recommendation="Placeholder IDs usually merge unrelated customers: replace them "
            "with real IDs or treat them as unknown. Trim spaces before joining tables.",
        )
    ]


def post_outcome_fields(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    target = ctx.target()
    assert target is not None
    y = ctx.data[target]
    labels = sorted(y.dropna().astype(str).unique())
    if len(labels) != 2:
        return [not_assessed(spec, "the target must have exactly two classes.")]
    has_y = y.notna()
    positive = (y[has_y].astype(str) == labels[1]).to_numpy()
    findings = []
    for column in ctx.features():
        if column == target:
            continue
        present = ctx.data.loc[has_y, column].notna().to_numpy()
        if present.all() or not present.any():
            continue
        agreement = max((present == positive).mean(), (present == ~positive).mean())
        if agreement >= 0.95:
            findings.append(
                detected(
                    spec,
                    ctx,
                    "critical" if agreement >= 0.999 else "high",
                    f"'{column}' is filled in almost only for one outcome",
                    f"Whether '{column}' has a value matches the target in {agreement * 100:.1f}% "
                    "of rows. Fields completed only after an event (e.g. a cancellation date or "
                    "exit reason) reveal the outcome and will not exist at prediction time.",
                    {
                        "agreement_pct": num(agreement * 100, 3),
                        "classes": labels,
                        "present_share_pct": num(present.mean() * 100, 2),
                    },
                    columns=[column, target],
                    recommendation="Confirm when this field is recorded. If it is written after "
                    "the outcome, exclude it (and anything derived from it) from features.",
                )
            )
    return findings or [
        not_detected(
            spec,
            "No feature's missing-value pattern matches the outcome.",
            {"features_checked": len(ctx.features())},
            columns=[target],
        )
    ]


def segment_imbalance(ctx: CheckContext, spec: CheckSpec) -> list[Finding]:
    findings = []
    target = ctx.target()
    for column in ctx.roles.all("biz.segment"):
        seg = ctx.data[column].astype(str).where(ctx.data[column].notna())
        counts = seg.dropna().value_counts()
        if counts.size < 2:
            findings.append(not_assessed(spec, f"'{column}' has fewer than two segments."))
            continue
        total = int(counts.sum())
        small = counts[(counts < 30) | (counts < 0.01 * total)]
        dominant = counts.iloc[0] / total
        evidence: dict[str, object] = {
            "segments": int(counts.size),
            "small_segments": {k: int(v) for k, v in small.head(20).items()},
            "largest_segment_share_pct": num(dominant * 100, 2),
        }
        if target and ctx.roles.task_type in ("binary", "multiclass"):
            labels = sorted(ctx.data[target].dropna().astype(str).unique())
            if len(labels) == 2:
                frame = pd.DataFrame({"s": seg, "y": ctx.data[target].astype(str)}).dropna()
                rates = (frame["y"] == labels[1]).groupby(frame["s"]).mean() * 100
                evidence["target_rate_pct_range"] = [num(rates.min(), 2), num(rates.max(), 2)]
        if small.empty and dominant <= 0.9:
            findings.append(
                not_detected(
                    spec,
                    f"Segments of '{column}' are reasonably balanced.",
                    evidence,
                    columns=[column],
                )
            )
            continue
        findings.append(
            detected(
                spec,
                ctx,
                "medium" if dominant > 0.9 else "low",
                f"Segments of '{column}' are imbalanced",
                f"{small.size} segments are very small and the largest holds "
                f"{dominant * 100:.0f}% of rows. Models and metrics will mostly reflect the "
                "large segments.",
                evidence,
                columns=[column],
                affected_rows=int(small.sum()) or None,
                recommendation="Report metrics per segment, stratify splits by segment, and be "
                "cautious about conclusions for small segments.",
            )
        )
    return findings

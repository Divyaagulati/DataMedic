"""Transparent ranking of recommendations (standard library only).

The rank is an ordering aid, not a prediction of improvement. Each recommendation gets an
integer score that is the sum of named factors, and every factor carries its points and the
reason for them, so the order can always be explained and reproduced:

| Factor | Points |
|---|---|
| Evidence strength | strong +3, moderate +2, weak +1 |
| Relevance to the task | relevant +2, indirect +1 |
| Domain compatibility | compatible 0, caution -1 |
| Risk | low 0, medium -1, high -2 |
| Information loss | none/low 0, moderate -1, high -2 |
| Sample size | -1 below 100 rows (learned parameters are noisy) |
| Resources | -1 for slow methods above 20,000 rows |
| Statistical assumptions | -1 when an assumption is doubtful |

Ties are broken by pipeline position, then by rule, then by the recommendation id, so the
same inputs always give the same order. Compatibility with other recommendations is not a
score: options that compete for the same decision are grouped as a conflict, with trade-offs.
"""

from typing import Any

from datamedic.engine.preprocess.recommend.rules import RESOURCE_HEAVY_ROWS, SMALL_DATA_ROWS

EVIDENCE_POINTS = {"strong": 3, "moderate": 2, "weak": 1}
TASK_POINTS = {"relevant": 2, "indirect": 1}
DOMAIN_POINTS = {"compatible": 0, "caution": -1}
RISK_POINTS = {"low": 0, "medium": -1, "high": -2}
LOSS_POINTS = {"none": 0, "low": 0, "moderate": -1, "high": -2}

# Position of an op family in a pipeline (also used to order candidate steps).
STEP_ORDER = {
    "missing": 0,
    "impute": 1,
    "outliers": 2,
    "transform": 3,
    "encode": 4,
    "scale": 5,
    "select": 6,
    "imbalance": 7,
}


def step_position(op: str) -> int:
    return STEP_ORDER.get(op.split(".")[0], 99)


def factors(rec: dict[str, Any], n_rows: int, fitted: bool) -> list[dict[str, Any]]:
    strength = rec["evidence_strength"]["level"]
    task = rec["task_compatibility"]["status"]
    domain = rec["domain_compatibility"]["status"]
    loss = rec["information_loss"]["level"]
    out = [
        {
            "factor": "evidence",
            "points": EVIDENCE_POINTS[strength],
            "detail": f"{strength.capitalize()} evidence: "
            + rec["evidence_strength"]["explanation"],
        },
        {
            "factor": "task",
            "points": TASK_POINTS[task],
            "detail": rec["task_compatibility"]["detail"],
        },
        {
            "factor": "domain",
            "points": DOMAIN_POINTS[domain],
            "detail": rec["domain_compatibility"]["detail"],
        },
        {
            "factor": "risk",
            "points": RISK_POINTS[rec["risk"]],
            "detail": f"{rec['risk'].capitalize()}-risk operation"
            + (" (destructive)." if rec["destructive"] else "."),
        },
        {
            "factor": "information_loss",
            "points": LOSS_POINTS[loss],
            "detail": rec["information_loss"]["detail"],
        },
    ]
    if fitted and n_rows < SMALL_DATA_ROWS:
        out.append(
            {
                "factor": "sample_size",
                "points": -1,
                "detail": f"Only {n_rows} rows: parameters learned per fold are noisy.",
            }
        )
    if rec.get("_heavy") and n_rows > RESOURCE_HEAVY_ROWS:
        out.append(
            {
                "factor": "resources",
                "points": -1,
                "detail": f"Slow on {n_rows:,} rows (fitted once per fold).",
            }
        )
    if rec.get("assumption_note"):
        out.append({"factor": "assumptions", "points": -1, "detail": rec["assumption_note"]})
    return out


def rank(recs: list[dict[str, Any]], n_rows: int) -> list[dict[str, Any]]:
    for rec in recs:
        fs = factors(rec, n_rows, fitted=rec["kind"] != "stateless_column")
        rec["ranking"] = {"score": sum(f["points"] for f in fs), "factors": fs}
    ordered = sorted(
        recs,
        key=lambda r: (
            -r["ranking"]["score"],
            step_position(r["op"]),
            r["_rule_order"],
            r["id"],
        ),
    )
    for i, rec in enumerate(ordered, start=1):
        rec["rank"] = i
    return ordered

"""Recommendation engine entry point (standard library only).

``recommend(inp)`` runs every rule, checks each drafted step against the domain policy and
the step validator (an operation whose roles, column types or parameters do not fit is never
recommended), ranks the result transparently, groups competing options into conflicts with
their trade-offs, proposes candidate pipelines for the Experiment Lab and a frozen-fold
split. The output is plain JSON and depends only on the inputs: the same version, profile,
findings and roles always give the same recommendations, ids and order.

Recommendations use whole-version descriptive statistics (as an analyst's exploration
would). They only choose *which* operations to compare; every learned parameter is fitted
inside training folds by the Experiment Lab, never from these statistics.
"""

import hashlib
from typing import Any

from datamedic.engine.preprocess.catalog import get_op
from datamedic.engine.preprocess.recommend.domain import assess
from datamedic.engine.preprocess.recommend.facts import Facts, RecommendInput
from datamedic.engine.preprocess.recommend.ranking import rank, step_position
from datamedic.engine.preprocess.recommend.rules import (
    NO_GUARANTEE,
    RULES,
    RULES_VERSION,
    Collector,
    Draft,
)
from datamedic.engine.preprocess.recommend.split_plan import recommend_split
from datamedic.engine.preprocess.spec import Step, canonical_json
from datamedic.engine.preprocess.validate import validate_steps

MAX_SUGGESTED_ALTERNATIVES = 3

GLOBAL_LIMITATIONS = (
    "Recommendations are deterministic rules over diagnostics, profile statistics and "
    "declared roles. They are hypotheses to test, not guaranteed improvements.",
    "They use statistics of the whole version (exploration); they only choose which "
    "operations to compare. Every learned parameter is fitted inside training folds.",
    "Associations in the data are not causes; nothing here establishes causal or clinical "
    "relevance.",
    "Semantic types are inferred heuristically at ingest; review them if a recommendation "
    "looks wrong for a column.",
    "Nothing is removed automatically: outlier and feature removal are never recommended, "
    "and selection steps run in report mode.",
)

# Trade-offs shown for groups of competing options. "allow_none": "do nothing" is a real
# alternative candidate (an add-on decision), not just a consequence to state.
TRADE_OFFS: dict[str, dict[str, Any]] = {
    "impute.numeric": {
        "topic": "How to fill missing numeric values",
        "trade_off": "The median is robust and simple but ignores relationships between "
        "columns; KNN and iterative imputation use them but are slower and can make imputed "
        "values look more certain than they are.",
        "none": "Without an imputation step only models that handle gaps natively (gradient "
        "boosting) use the incomplete rows as they are; the linear baseline falls back to its "
        "own in-fold median.",
        "allow_none": False,
    },
    "missing.indicator": {
        "topic": "Imputation alone or with missing-value indicators",
        "trade_off": "Indicators preserve the information that a value was missing (useful "
        "when missingness is informative) at the cost of extra columns and a dependence on "
        "how data are collected.",
        "none": "Imputation alone hides which values were estimated.",
        "allow_none": True,
    },
    "outliers.treatment": {
        "topic": "Extreme values: flag, cap or scale robustly",
        "trade_off": "Flagging keeps every value and adds information; capping limits "
        "leverage but overwrites real values (destructive); robust scaling changes no value "
        "and only reduces their influence on the scale.",
        "none": "Extreme values stay as observed; scale-sensitive models may be dominated by them.",
        "allow_none": True,
    },
    "transform": {
        "topic": "Reshape skewed distributions or preserve them",
        "trade_off": "A log keeps an interpretable scale; Yeo-Johnson fits any sign but is "
        "less interpretable; a quantile transform removes tail influence entirely but keeps "
        "only the order of values. Preserving the original distribution keeps the "
        "measurement scale, and tree models do not need a transformation.",
        "none": "Original units and distributions are preserved.",
        "allow_none": True,
    },
    "encode.cardinality": {
        "topic": "Encoding a high-cardinality category",
        "trade_off": "Frequency encoding stays compact but loses category identity; grouped "
        "one-hot keeps the frequent categories identifiable at the cost of more columns.",
        "none": "The baselines one-hot encode up to 20 categories themselves (linear) or use "
        "ordinal codes (gradient boosting).",
        "allow_none": False,
    },
    "imbalance": {
        "topic": "Handling class imbalance",
        "trade_off": "Class weights change no rows; random oversampling repeats real minority "
        "rows; SMOTE synthesises new ones (not real observations); undersampling discards "
        "majority rows. All shift precision towards recall and de-calibrate probabilities. "
        "Only one can be used, as the last step, inside training folds.",
        "none": "The model optimises overall accuracy and may ignore the minority class; use "
        "balanced metrics either way.",
        "allow_none": True,
    },
    "redundancy": {
        "topic": "Correlation-based reduction or preserving every measurement",
        "trade_off": "Reviewing correlated pairs can stabilise linear models, but correlated "
        "measurements (e.g. related biomarkers) can each carry meaning. DataMedic only "
        "proposes; removing a column is an explicit decision.",
        "none": "All correlated columns are kept (no information is lost).",
        "allow_none": True,
    },
    "selection": {
        "topic": "Ranking features or keeping all",
        "trade_off": "Rankings show which features carry signal and how stable that is; acting "
        "on unstable rankings removes information arbitrarily.",
        "none": "Every feature is used.",
        "allow_none": True,
    },
}


def _trade_off_key(group: str) -> str:
    for key in TRADE_OFFS:
        if group == key or group.startswith(key + ".") or group.startswith(key + ":"):
            return key
    return group.split(".")[0].split(":")[0]


def _rec_id(draft: Draft) -> str:
    payload = canonical_json(
        {
            "rule": draft.rule_id,
            "op": draft.op,
            "columns": list(draft.columns),
            "params": draft.params,
        }
    )
    return "rec_" + hashlib.sha256(payload.encode()).hexdigest()[:16]


def _finalise(
    facts: Facts, out: Collector, draft: Draft, order: int, prefix: list[Step]
) -> dict[str, Any] | None:
    """``prefix``: imputation steps already recommended. A training-only op (e.g. SMOTE needs
    complete numeric inputs) is validated after them, as it would run in a candidate."""
    spec = get_op(draft.op)
    if spec is None:  # a rule referencing an unknown op is a programming error
        raise ValueError(f"rule {draft.rule_id} uses unknown op {draft.op}")
    columns = [facts.columns[c] for c in draft.columns]
    domain = assess(facts, draft.op, columns)
    if domain.blocked:
        for name, reason in sorted(domain.blocked.items()):
            out.reject(draft.rule_id, draft.op, reason, [name])
        draft.columns = tuple(c for c in draft.columns if c not in domain.blocked)
        if spec.uses_columns and not draft.columns:
            return None
        draft.evidence = [e for e in draft.evidence if e.get("column") not in domain.blocked]
    step = Step(draft.op, draft.columns, dict(draft.params))
    before = prefix if spec.kind == "training_only" else []
    check = validate_steps([*before, step], facts.context)
    if not check.valid:
        out.skip(
            draft.rule_id,
            spec.title,
            "The operation does not fit the current schema and roles: "
            + "; ".join(sorted({e.message for e in check.errors})),
            draft.columns,
        )
        return None
    limitations = list(dict.fromkeys(list(draft.limitations) + list(spec.limitations)))
    return {
        "id": _rec_id(draft),
        "rule_id": draft.rule_id,
        "rule_version": RULES_VERSION,
        "title": draft.title,
        "op": spec.op,
        "op_version": spec.version,
        "op_title": spec.title,
        "family": spec.family,
        "params": dict(draft.params),
        "columns": list(draft.columns),
        "applies_to": f"Selected columns: {', '.join(draft.columns)}."
        if spec.uses_columns
        else f"Whole training rows of each fold, by the class of the target "
        f"'{facts.target}'; validation and test rows are never changed.",
        "stage": spec.stage,
        "kind": spec.kind,
        "risk": spec.risk,
        "destructive": spec.destructive,
        "what_it_does": spec.description,
        "reason": draft.reason,
        "evidence": draft.evidence,
        "expected_benefit": draft.expected_benefit,
        "risks": list(draft.risks) + list(spec.downsides),
        "information_loss": {
            "level": draft.information_loss[0],
            "detail": draft.information_loss[1],
        },
        "domain_compatibility": domain.as_dict(),
        "task_compatibility": {"status": draft.task_fit[0], "detail": draft.task_fit[1]},
        "required_decisions": list(draft.required_decisions),
        "evidence_strength": {"level": draft.strength[0], "explanation": draft.strength[1]},
        "limitations": limitations,
        "assumption_note": draft.assumption_note,
        "not_a_guarantee": NO_GUARANTEE,
        "conflict_group": draft.conflict_group,
        "finding_ids": sorted(set(draft.finding_ids)),
        "alternatives": [],
        "_heavy": draft.heavy,
        "_rule_order": order,
    }


def _conflicts(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for rec in recs:  # already in rank order
        if rec["conflict_group"]:
            groups.setdefault(rec["conflict_group"], []).append(rec)
    out = []
    for group, members in groups.items():
        info = TRADE_OFFS[_trade_off_key(group)]
        if len(members) < 2 and not info["allow_none"]:
            continue
        ids = [m["id"] for m in members]
        for m in members:
            m["alternatives"] = [i for i in ids if i != m["id"]]
        options = [
            {
                "recommendation_id": m["id"],
                "label": m["title"],
                "risk": m["risk"],
                "information_loss": m["information_loss"]["level"],
                "rank": m["rank"],
            }
            for m in members
        ]
        options.append(
            {
                "recommendation_id": None,
                "label": "Do nothing",
                "risk": "low",
                "information_loss": "none",
                "rank": None,
                "consequence": info["none"],
            }
        )
        out.append(
            {
                "id": "conflict_" + hashlib.sha256(group.encode()).hexdigest()[:12],
                "group": group,
                "topic": info["topic"],
                "trade_off": info["trade_off"],
                "options": options,
                "allow_none": info["allow_none"],
                "how_to_decide": "There is no universal answer. Run these options as separate "
                "candidates in the Experiment Lab (same folds), and weigh fold-to-fold "
                "variation, information loss and interpretability, not only the mean score.",
            }
        )
    return out


def _assemble(
    facts: Facts, key: str, label: str, rationale: str, recs: list[dict[str, Any]]
) -> dict[str, Any]:
    """Add steps in pipeline order, keeping each only if the pipeline stays valid."""
    ordered = sorted(recs, key=lambda r: (step_position(r["op"]), r["rank"]))
    steps: list[Step] = []
    used: list[str] = []
    skipped: list[dict[str, str]] = []
    for rec in ordered:
        step = Step(rec["op"], tuple(rec["columns"]), dict(rec["params"]))
        check = validate_steps([*steps, step], facts.context)
        if check.valid:
            steps.append(step)
            used.append(rec["id"])
        else:
            skipped.append(
                {
                    "recommendation_id": rec["id"],
                    "reason": "; ".join(sorted({e.message for e in check.errors})),
                }
            )
    return {
        "key": key,
        "label": label,
        "rationale": rationale,
        "steps": [s.as_dict() for s in steps],
        "recommendation_ids": used,
        "skipped": skipped,
    }


def _candidates(
    facts: Facts, recs: list[dict[str, Any]], conflicts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {r["id"]: r for r in recs}
    chosen = {c["group"]: c["options"][0]["recommendation_id"] for c in conflicts}
    grouped = {
        r["id"]
        for c in conflicts
        for o in c["options"]
        if (r := by_id.get(o["recommendation_id"] or ""))
    }
    base = [r for r in recs if r["id"] not in grouped]
    core = base + [by_id[i] for i in chosen.values() if i]
    out = [
        {
            "key": "baseline",
            "label": "Baseline: no preprocessing",
            "rationale": "Reference: "
            "only the model's own minimal in-fold handling (documented per model).",
            "steps": [],
            "recommendation_ids": [],
            "skipped": [],
        }
    ]
    if not core:
        return out
    out.append(
        _assemble(
            facts,
            "core",
            "Recommended core",
            "The top-ranked option of every decision plus every uncontested recommendation.",
            core,
        )
    )
    for conflict in conflicts:
        if len(out) - 2 >= MAX_SUGGESTED_ALTERNATIVES:
            break
        top = conflict["options"][0]["recommendation_id"]
        alternative = conflict["options"][1]["recommendation_id"]
        if alternative is None and not conflict["allow_none"]:
            continue
        swapped = [r for r in core if r["id"] != top]
        if alternative is not None:
            swapped.append(by_id[alternative])
            label = f"Core, with: {by_id[alternative]['op_title']}"
        else:
            label = f"Core, without: {by_id[top]['op_title']}"
        out.append(
            _assemble(
                facts,
                f"alt_{len(out) - 1}",
                label,
                f"Isolates the decision '{conflict['topic']}'.",
                swapped,
            )
        )
    return out


def recommend(inp: RecommendInput) -> dict[str, Any]:
    facts = Facts(inp)
    out = Collector()
    recs: list[dict[str, Any]] = []
    imputation: list[Step] = []
    groups_seen: set[str] = set()
    for order, rule in enumerate(RULES):
        before = len(out.drafts)
        rule(facts, out)
        for draft in out.drafts[before:]:
            rec = _finalise(facts, out, draft, order, imputation)
            if rec is None:
                continue
            recs.append(rec)
            group = rec["conflict_group"] or rec["id"]
            if rec["op"].startswith("impute.") and group not in groups_seen:
                groups_seen.add(group)
                imputation.append(Step(rec["op"], tuple(rec["columns"]), dict(rec["params"])))
    if not facts.has_diagnostics:
        out.skip(
            "diagnostics",
            "finding-based rules (outliers, imbalance, redundancy, constant columns, leakage)",
            "Diagnostics have not run for this version.",
        )
    ranked = rank(recs, facts.n_rows)
    conflicts = _conflicts(ranked)
    candidates = _candidates(facts, ranked, conflicts)
    for rec in ranked:
        rec.pop("_heavy", None)
        rec.pop("_rule_order", None)
    features = facts.features()
    return {
        "rules_version": RULES_VERSION,
        "inputs": {
            "n_rows": facts.n_rows,
            "n_features": len(features),
            "domain": inp.domain,
            "target": facts.target,
            "task_type": facts.task,
            "profile_available": facts.has_profile,
            "diagnostics_available": facts.has_diagnostics,
            "correlations_sampled": facts.correlations_sampled,
            "roles_declared": bool(inp.roles),
        },
        "recommendations": ranked,
        "conflicts": conflicts,
        "candidates": candidates,
        "split_plan": recommend_split(facts),
        "not_assessed": out.not_assessed,
        "not_recommended": out.not_recommended,
        "advisories": out.advisories,
        "limitations": list(GLOBAL_LIMITATIONS),
    }

"""Recommended frozen-fold split from declared roles (standard library only).

| Roles | Task | Method |
|---|---|---|
| time role | any | ``time`` (forward chaining) |
| entity role (group, patient, account, asset, customer) | classification | ``stratified_group`` |
| entity role | regression / none | ``group`` |
| none | classification | ``stratified`` |
| none | regression / none | ``kfold`` |

Exact duplicate rows (identical in every feature and the target) are always kept on the
same side of each split. Rows without a target are excluded from supervised experiments.
"""

from typing import Any

from datamedic.engine.preprocess.recommend.facts import ENTITY_ROLES, TIME_ROLES, Facts

DEFAULT_SPLITS = 5
MIN_SPLITS = 2


def recommend_split(facts: Facts) -> dict[str, Any]:
    entity = facts.role_columns(ENTITY_ROLES)
    time = facts.role_columns(TIME_ROLES)
    warnings: list[dict[str, str]] = []
    n_splits = DEFAULT_SPLITS
    group: str | None = None
    time_col: str | None = None
    if time:
        time_col = time[0]
        method = "time"
        reason = (
            "A time role is declared: forward-chaining folds always train on earlier rows and "
            "test on later ones, as the model would be used."
        )
        if entity:
            warnings.append(
                {
                    "code": "GROUPS_NOT_ENFORCED_WITH_TIME",
                    "message": "Time-ordered folds cannot also keep entities together; the same "
                    "entity can appear in training and test rows of a fold.",
                }
            )
        distinct = facts.columns[time_col].n_unique
        if distinct is not None:
            n_splits = min(n_splits, distinct - 1)
    elif entity:
        group = entity[0]
        method = "stratified_group" if facts.classification else "group"
        reason = (
            "Rows belong to entities (a declared group or patient/account/asset/customer role): "
            "all rows of an entity stay on one side of every split, so the model is tested on "
            "unseen entities."
        )
        if len(entity) > 1:
            warnings.append(
                {
                    "code": "SEVERAL_ENTITY_ROLES",
                    "message": "Several entity columns are declared; folds group by the first.",
                }
            )
        distinct = facts.columns[group].n_unique
        if distinct is not None:
            n_splits = min(n_splits, distinct)
    elif facts.classification:
        method = "stratified"
        reason = (
            "Classification target without entity or time roles: stratified folds keep "
            "class proportions similar in every fold."
        )
    else:
        method = "kfold"
        reason = "No entity, time or class structure is declared: shuffled K-fold."
        if facts.target is None:
            warnings.append(
                {
                    "code": "NO_TARGET",
                    "message": "No target is declared; only clustering experiments are possible.",
                }
            )
    if facts.classification and method in ("stratified", "stratified_group"):
        minority = _minority_count(facts)
        if minority is not None:
            n_splits = min(n_splits, minority)
    target_missing = facts.columns[facts.target].n_missing if facts.target else 0
    if target_missing:
        warnings.append(
            {
                "code": "ROWS_WITHOUT_TARGET",
                "message": f"{target_missing} rows have no target value and are excluded from "
                "evaluation.",
            }
        )
    if n_splits < MIN_SPLITS:
        return {
            "status": "not_assessed",
            "reason": "Too few classes, entities or distinct times for at least two folds.",
            "method": method,
            "warnings": warnings,
        }
    if n_splits < DEFAULT_SPLITS:
        warnings.append(
            {
                "code": "FEWER_FOLDS",
                "message": f"Reduced to {n_splits} folds (smallest class, number of entities or "
                "distinct times).",
            }
        )
    duplicate_columns = [c.name for c in facts.features()]
    if facts.target:
        duplicate_columns.append(facts.target)
    return {
        "status": "ok",
        "method": method,
        "n_splits": n_splits,
        "seed": 0,
        "target": facts.target,
        "group": group,
        "time": time_col,
        "duplicate_columns": duplicate_columns,
        "reason": reason,
        "warnings": warnings,
    }


def _minority_count(facts: Facts) -> int | None:
    """Smallest class count from the class-imbalance diagnostic (detected or not)."""
    for f in facts.findings_of("universal.target.imbalance"):
        classes = f.evidence.get("classes") or {}
        if classes:
            return min(int(v) for v in classes.values())
    return None

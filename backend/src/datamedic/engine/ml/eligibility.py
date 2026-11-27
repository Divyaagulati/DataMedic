"""Evaluation eligibility of a dataset version (standard library only; API and worker).

A version whose committed steps learned from all of its rows (``fitted_on_full_data``) carries
information from rows that would later be test rows. Training directly on it overstates
performance. Eligibility decides how a version may be evaluated:

- ``unbiased_direct``: the version is not fitted on full data; evaluate it as stored.
- ``unbiased_replay``: evaluate the rows of the *eligible ancestor* (the latest version in the
  lineage that is not fitted on full data) and re-run the committed data-stage steps inside
  every training fold. This requires that the replay is *faithful* (every step re-validates to
  its recorded hash; the worker additionally proves that refitting the steps on all ancestor
  rows reproduces the version's content hash) **and** *safe* (checked here): no step removes
  rows, no step transforms the current target, roles needed for splitting exist on the
  ancestor, and no step is of a kind that cannot run in a fold.
- ``blocked``: faithful and safe replay cannot be established. Ordinary evaluation is refused;
  the user may explicitly choose an exploratory run on the version as stored, which is
  permanently labelled and audited as a confirmed critical violation.

A matching content hash proves fidelity, never safety; both are required.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from datamedic.engine.checks.spec import RoleSpec
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.state import ValidationContext
from datamedic.engine.preprocess.validate import validate_steps

DIRECT = "unbiased_direct"
REPLAY = "unbiased_replay"
BLOCKED = "blocked"
EXPLORATORY = "exploratory"
EVALUATION_MODES = (DIRECT, REPLAY, EXPLORATORY)

ROW_REMOVING_KINDS = ("row_filter_rule", "row_filter_fitted", "training_only")


@dataclass(frozen=True)
class LineageRun:
    """One committed transformation between the eligible ancestor and the selected version,
    in lineage order (oldest first). ``source_*`` describe the version it was applied to."""

    run_id: str
    result_version_id: str
    source_version_id: str
    source_schema: Sequence[Mapping[str, Any]]
    source_n_rows: int
    data_steps: Sequence[Mapping[str, Any]]
    data_step_hashes: Sequence[str]
    commit_roles: Mapping[str, Any]  # roles snapshot used at commit time
    seed: int


@dataclass(frozen=True)
class Lineage:
    """Facts gathered by the service from the database (no data read)."""

    selected_version_id: str
    selected_fitted: bool
    ancestor_version_id: str | None  # None: the walk failed (``problem`` says why)
    ancestor_schema: Sequence[Mapping[str, Any]] = ()
    ancestor_n_rows: int = 0
    selected_n_rows: int = 0
    runs: Sequence[LineageRun] = ()
    problem: str | None = None


@dataclass
class Eligibility:
    mode: str
    evaluation_version_id: str
    reasons: list[dict[str, str]] = field(default_factory=list)
    replay: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "evaluation_version_id": self.evaluation_version_id,
            "reasons": list(self.reasons),
            "replay": self.replay,
        }


def _blocked(lineage: Lineage, code: str, message: str) -> Eligibility:
    return Eligibility(
        BLOCKED,
        lineage.selected_version_id,
        [{"code": code, "message": message}],
        None,
    )


def assess(
    lineage: Lineage,
    current_roles: Mapping[str, Any],
    role_specs: Mapping[str, RoleSpec],
) -> Eligibility:
    """Decide the evaluation mode from lineage facts and the current roles snapshot
    (``{"roles": {role: [columns]}, "task_type": ...}``)."""
    if not lineage.selected_fitted:
        return Eligibility(DIRECT, lineage.selected_version_id)
    if lineage.ancestor_version_id is None:
        return _blocked(
            lineage,
            "LINEAGE_INCOMPLETE",
            lineage.problem or "The version's lineage cannot be traced to an earlier version.",
        )
    if not lineage.runs:
        return _blocked(
            lineage, "LINEAGE_INCOMPLETE", "No committed transformation to replay was found."
        )
    if lineage.ancestor_n_rows != lineage.selected_n_rows:
        return _blocked(
            lineage,
            "REPLAY_ROWS_DIFFER",
            "The version has a different number of rows than its eligible ancestor, so the "
            "same rows cannot be evaluated.",
        )
    roles: Mapping[str, Sequence[str]] = current_roles.get("roles", {})
    targets = list(roles.get("target", ()))
    target = targets[0] if len(targets) == 1 else None
    ancestor_columns = {str(c["name"]) for c in lineage.ancestor_schema}
    if target is not None and target not in ancestor_columns:
        return _blocked(
            lineage,
            "ROLE_ON_DERIVED_COLUMN",
            "The target column does not exist in the eligible ancestor version; it was "
            "created by a committed step, so it cannot be evaluated without that step's "
            "full-data fit.",
        )
    for role in ("group", "time", "bio.patient_id"):
        for column in roles.get(role, ()):
            if column not in ancestor_columns:
                return _blocked(
                    lineage,
                    "ROLE_ON_DERIVED_COLUMN",
                    "A group or time role is declared on a column created by a committed "
                    "step; the evaluation split cannot be built on the ancestor's rows.",
                )
    replayed: list[dict[str, Any]] = []
    target_using: list[str] = []
    for run in lineage.runs:
        ctx = ValidationContext.from_schema(
            run.source_schema,
            run.commit_roles.get("roles", {}),
            role_specs,
            n_rows=run.source_n_rows,
            task_type=run.commit_roles.get("task_type"),
        )
        result = validate_steps(
            [
                Step(str(s["op"]), tuple(s.get("columns") or ()), dict(s.get("params") or {}))
                for s in run.data_steps
            ],
            ctx,
        )
        if not result.valid or [v.step_hash for v in result.steps] != list(run.data_step_hashes):
            return _blocked(
                lineage,
                "REPLAY_HASH_MISMATCH",
                "A committed step no longer validates to its recorded hash (the operation "
                "catalogue or engine changed), so the replay would not be faithful.",
            )
        for v in result.steps:
            if v.spec.kind in ROW_REMOVING_KINDS:
                return _blocked(
                    lineage,
                    "REPLAY_UNSUPPORTED_STEP",
                    "A committed step removes or adds rows; it cannot be replayed on "
                    "identical evaluation rows.",
                )
            touched = set(v.step.columns)
            if target is not None and target in touched:
                return _blocked(
                    lineage,
                    "TARGET_TRANSFORMED_IN_LINEAGE",
                    "A committed step changed the column that is now the target. Replaying "
                    "it would alter test-row labels, so leakage-controlled evaluation is not "
                    "possible.",
                )
            if v.spec.needs_target:
                target_using.append(v.spec.op)
            replayed.append(
                {
                    "run_id": run.run_id,
                    "result_version_id": run.result_version_id,
                    "op": v.spec.op,
                    "op_version": v.spec.version,
                    "kind": v.spec.kind,
                    "columns": list(v.step.columns),
                    "step_hash": v.step_hash,
                }
            )
    notes = []
    if target_using:
        notes.append(
            "Some replayed steps read the target; inside folds they read training labels only."
        )
    return Eligibility(
        REPLAY,
        lineage.ancestor_version_id,
        [],
        {
            "ancestor_version_id": lineage.ancestor_version_id,
            "run_ids": [r.run_id for r in lineage.runs],
            "steps": replayed,
            "notes": notes,
            "equivalence": None,  # proven by the worker before training
        },
    )

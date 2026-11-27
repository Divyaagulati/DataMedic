"""Pipeline stages for committing (Phase 4 Part E). Standard library only: the API plans and
hashes pipelines with this module; ``commit.py`` executes plans in the worker.

A pipeline step carries a *chosen stage*:
- ``data`` steps are materialised into the new version;
- ``model`` steps are stored with the version as its ``model_pipeline_spec`` and run only
  inside model training folds (owner decision D1). They never change the saved data.

Ops whose catalogue stage is ``data`` or ``model`` must keep it; ``either`` ops need an
explicit choice, which is never changed silently. Data-stage steps must all come before the
first model-stage step, so the committed table is exactly what the model-stage steps were
validated against. The chosen stage is part of each pipeline step's hash, so changing it
invalidates approvals like any other edit.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Any

from datamedic.engine.preprocess.catalog import get_op
from datamedic.engine.preprocess.spec import Step, canonical_json, spec_hash
from datamedic.engine.preprocess.state import Issue, ValidationContext
from datamedic.engine.preprocess.validate import ValidationResult, validate_steps

CHOSEN_STAGES = ("data", "model")


@dataclass(frozen=True)
class PlannedStep:
    step: Step
    stage: str  # chosen stage: data | model


@dataclass
class CommitPlan:
    steps: list[PlannedStep]
    full: ValidationResult
    data: ValidationResult
    errors: list[Issue]
    warnings: list[Issue]
    step_hashes: list[str | None] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def spec_hash(self) -> str | None:
        if not self.valid or any(h is None for h in self.step_hashes):
            return None
        return spec_hash([h for h in self.step_hashes if h is not None])

    @property
    def model_steps(self) -> list[PlannedStep]:
        return [s for s in self.steps if s.stage == "model"]

    def model_spec(self) -> list[dict[str, Any]]:
        """The modelling specification stored with the version (submitted steps)."""
        out = []
        for i, s in enumerate(self.steps):
            if s.stage == "model":
                v = self.full.steps[i]
                out.append(
                    {
                        **s.step.as_dict(),
                        "stage": "model",
                        "op_version": v.spec.version,
                        "engine_step_hash": v.step_hash,
                    }
                )
        return out

    def report(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "errors": [e.as_dict() for e in self.errors],
            "warnings": [w.as_dict() for w in self.warnings],
            "data_steps": sum(1 for s in self.steps if s.stage == "data"),
            "model_steps": sum(1 for s in self.steps if s.stage == "model"),
        }


def pipeline_step_hash(engine_step_hash: str, stage: str) -> str:
    """Hash of a pipeline step: the engine step hash plus its chosen stage."""
    return hashlib.sha256(canonical_json([engine_step_hash, stage]).encode()).hexdigest()


def draft_hash(steps: list[PlannedStep]) -> str:
    """Hash of an invalid pipeline (no approval or commit is possible for it)."""
    payload = [{**s.step.as_dict(), "stage": s.stage} for s in steps]
    return "draft:" + hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def default_stage(op: str) -> str:
    """Stage assigned when none is chosen: the catalogue stage, ``data`` for ``either``."""
    spec = get_op(op)
    if spec is None:
        return "data"
    return "data" if spec.stage == "either" else spec.stage


def plan(steps: list[PlannedStep], ctx: ValidationContext) -> CommitPlan:
    full = validate_steps([s.step for s in steps], ctx)
    errors = list(full.errors)
    warnings = list(full.warnings)
    seen_model = False
    for i, s in enumerate(steps):
        spec = get_op(s.step.op)
        if spec is None:
            continue  # reported by the validator
        if s.stage not in CHOSEN_STAGES:
            errors.append(Issue("STAGE_INVALID", "Choose 'data' or 'model' for this step.", i))
        elif spec.stage != "either" and s.stage != spec.stage:
            errors.append(
                Issue(
                    "STAGE_NOT_ALLOWED",
                    f"This operation is {spec.stage}-stage only and cannot be used as a "
                    f"{s.stage}-stage step.",
                    i,
                )
            )
        if s.stage == "model":
            seen_model = True
        elif seen_model:
            errors.append(
                Issue("STAGE_ORDER", "Data-stage steps must come before every model-stage step.", i)
            )
    data_steps = [s.step for s in steps if s.stage == "data"]
    data = validate_steps(data_steps, ctx) if not errors else full
    if full.valid:
        hashes: list[str | None] = [
            pipeline_step_hash(v.step_hash, s.stage) for v, s in zip(full.steps, steps, strict=True)
        ]
    else:
        hashes = [None] * len(steps)
    return CommitPlan(steps, full, data, errors, warnings, hashes)

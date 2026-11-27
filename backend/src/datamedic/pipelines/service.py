"""Pipelines, impact previews, approvals, commits, reverts and comparisons (API side).

Authority rules (CLAUDE.md invariant 7):
- risk, destructive flag, kind and op version come from the catalogue; the chosen stage is
  explicit; step and spec hashes are recomputed by the service on every change (and again
  by the worker), never taken from the client;
- an approval is valid only while its hashes equal the pipeline's *current* hashes and the
  preview it was given against is the pipeline's current preview of its current source
  version. Edits never rewrite approvals; they make them inapplicable;
- ``commit_readiness`` is the single authoritative check, used by the API before queueing
  and by the worker before reading data and again under the dataset lock before writing.
Every lookup is scoped to the owner (and RLS); another user's resource is a 404.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.datasets.service import get_owned
from datamedic.db.models import (
    Approval,
    Artifact,
    Dataset,
    DatasetVersion,
    Experiment,
    ExperimentCandidate,
    Job,
    Pipeline,
    PipelineStep,
    Recommendation,
    TransformationRun,
)
from datamedic.engine.preprocess.catalog import get_op
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.stages import CommitPlan, PlannedStep, default_stage, draft_hash
from datamedic.engine.preprocess.stages import plan as plan_steps
from datamedic.errors import (
    ApprovalRequired,
    ApprovalStale,
    Conflict,
    DatasetNotReady,
    DomainError,
    NotFound,
    QuotaExceeded,
    ValidationFailed,
)
from datamedic.jobs.service import create_job
from datamedic.preprocessing.context import roles_snapshot, validation_context
from datamedic.storage import ObjectNotFound, ObjectStorage
from datamedic.versions.service import get_owned_version

MAX_STEPS = 50
EDITABLE = ("draft", "validated")


def _fail(message: str, pointer: str, code: str) -> DomainError:
    return ValidationFailed(
        message, errors=[{"pointer": pointer, "code": code, "message": message}]
    )


# ---- Loading ----------------------------------------------------------------------------------


def get_pipeline(
    db: Session, owner_id: uuid.UUID, pipeline_id: uuid.UUID, *, for_update: bool = False
) -> Pipeline:
    q = select(Pipeline).where(Pipeline.id == pipeline_id, Pipeline.owner_id == owner_id)
    pipeline = db.scalar(q.with_for_update() if for_update else q)
    if pipeline is None:
        raise NotFound()
    get_owned(db, owner_id, pipeline.dataset_id)  # dataset not deleted
    return pipeline


def steps_of(db: Session, pipeline: Pipeline) -> list[PipelineStep]:
    return list(
        db.scalars(
            select(PipelineStep)
            .where(
                PipelineStep.owner_id == pipeline.owner_id, PipelineStep.pipeline_id == pipeline.id
            )
            .order_by(PipelineStep.position)
        )
    )


def approvals_of(db: Session, pipeline: Pipeline) -> list[Approval]:
    return list(
        db.scalars(
            select(Approval)
            .where(Approval.owner_id == pipeline.owner_id, Approval.pipeline_id == pipeline.id)
            .order_by(Approval.created_at)
        )
    )


def source_version(db: Session, pipeline: Pipeline) -> DatasetVersion:
    version, _ = get_owned_version(db, pipeline.owner_id, pipeline.source_version_id)
    return version


def plan_for(db: Session, pipeline: Pipeline, steps: list[PipelineStep]) -> CommitPlan:
    """The authoritative plan: current steps validated against the source version's schema
    and current roles."""
    version = source_version(db, pipeline)
    ctx = validation_context(version, roles_snapshot(db, pipeline.owner_id, version.id))
    return plan_steps(
        [PlannedStep(Step(s.op, tuple(s.columns), dict(s.params)), s.stage) for s in steps], ctx
    )


def _refresh(db: Session, pipeline: Pipeline, steps: list[PipelineStep]) -> CommitPlan:
    """Recompute catalogue fields, hashes and validation after any change. State returns to
    draft (or validated when valid); approvals and previews become inapplicable through the
    changed hashes, never by rewriting them."""
    for position, s in enumerate(steps):
        spec = get_op(s.op)
        if spec is not None:
            s.op_version, s.risk, s.destructive, s.kind = (
                spec.version,
                spec.risk,
                spec.destructive,
                spec.kind,
            )
        s.position = position
    plan = plan_for(db, pipeline, steps)
    for s, h in zip(steps, plan.step_hashes, strict=True):
        s.step_hash = h
        s.updated_at = datetime.now(UTC)
    pipeline.spec_hash = plan.spec_hash or draft_hash(plan.steps)
    pipeline.state = "validated" if plan.valid else "draft"
    pipeline.validation_report = plan.report()
    pipeline.updated_at = datetime.now(UTC)
    db.flush()
    return plan


def _check_step_body(
    body: dict[str, Any], index: int
) -> tuple[str, list[str], dict[str, Any], str]:
    op = str(body["op"])
    spec = get_op(op)
    if spec is None:
        raise _fail("Unknown operation.", f"/steps/{index}/op", "UNKNOWN_OP")
    stage = body.get("stage") or default_stage(op)
    return op, list(body.get("columns") or []), dict(body.get("params") or {}), stage


def _new_step(
    pipeline: Pipeline,
    position: int,
    op: str,
    columns: list[str],
    params: dict[str, Any],
    stage: str,
    rec_id: uuid.UUID | None,
) -> PipelineStep:
    spec = get_op(op)
    assert spec is not None
    return PipelineStep(
        owner_id=pipeline.owner_id,
        dataset_id=pipeline.dataset_id,
        pipeline_id=pipeline.id,
        position=position,
        op=op,
        op_version=spec.version,
        params=params,
        columns=columns,
        stage=stage,
        risk=spec.risk,
        destructive=spec.destructive,
        kind=spec.kind,
        from_recommendation_id=rec_id,
    )


def _editable(pipeline: Pipeline) -> None:
    if pipeline.state not in EDITABLE:
        raise Conflict(
            "This pipeline was committed and can no longer change. Create a new "
            "pipeline for further changes."
        )


# ---- Creating and editing ---------------------------------------------------------------------


def create(
    db: Session,
    owner_id: uuid.UUID,
    dataset_id: uuid.UUID,
    name: str,
    steps: list[dict[str, Any]],
    *,
    from_candidate: ExperimentCandidate | None = None,
) -> Pipeline:
    ds = get_owned(db, owner_id, dataset_id)
    if ds.status != "ready" or ds.head_version_id is None:
        raise DatasetNotReady("The dataset is not ready.")
    if len(steps) > MAX_STEPS:
        raise _fail(f"At most {MAX_STEPS} steps.", "/steps", "TOO_MANY_STEPS")
    pipeline = Pipeline(
        owner_id=owner_id,
        dataset_id=ds.id,
        source_version_id=ds.head_version_id,
        name=name.strip() or "Pipeline",
        state="draft",
        spec_hash="draft:",
        from_experiment_candidate_id=from_candidate.id if from_candidate else None,
        created_by=owner_id,
    )
    db.add(pipeline)
    db.flush()
    rows = []
    for i, body in enumerate(steps):
        op, columns, params, stage = _check_step_body(body, i)
        rec = body.get("recommendation_id")
        rec_id = _owned_recommendation(db, owner_id, ds.id, rec) if rec else None
        rows.append(_new_step(pipeline, i, op, columns, params, stage, rec_id))
    db.add_all(rows)
    db.flush()
    _refresh(db, pipeline, rows)
    db.commit()
    return pipeline


def _owned_recommendation(
    db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID, rec_id: Any
) -> uuid.UUID:
    rid = uuid.UUID(str(rec_id))
    found = db.scalar(
        select(Recommendation.id).where(
            Recommendation.id == rid,
            Recommendation.owner_id == owner_id,
            Recommendation.dataset_id == dataset_id,
        )
    )
    if found is None:
        raise _fail("Unknown recommendation for this dataset.", "/steps", "UNKNOWN_RECOMMENDATION")
    return rid


def from_candidate(db: Session, owner_id: uuid.UUID, candidate_id: uuid.UUID) -> Pipeline:
    """An editable pipeline from an Experiment Lab candidate, against the dataset's current
    head. Steps, parameters and order are kept. Catalogue model-stage ops stay model-stage
    (stored as the version's modelling specification, D1/E3). ``either`` ops become
    data-stage, except after the first model-stage step, where they must stay model-stage to
    keep the evaluated order; the stage of every step is shown and can be changed. The
    experiment and its results are not modified; nothing is approved or committed."""
    cand = db.scalar(
        select(ExperimentCandidate).where(
            ExperimentCandidate.id == candidate_id, ExperimentCandidate.owner_id == owner_id
        )
    )
    if cand is None:
        raise NotFound()
    exp = db.scalar(
        select(Experiment).where(
            Experiment.id == cand.experiment_id, Experiment.owner_id == owner_id
        )
    )
    if exp is None:
        raise NotFound()
    get_owned_version(db, owner_id, exp.dataset_version_id)
    if cand.is_baseline or not cand.steps:
        raise _fail("The baseline has no steps to convert.", "/candidate", "EMPTY_CANDIDATE")
    steps, seen_model = [], False
    for s in cand.steps:
        stage = default_stage(s["op"])
        if stage == "model":
            seen_model = True
        elif seen_model and (spec := get_op(s["op"])) is not None and spec.stage == "either":
            stage = "model"
        steps.append(
            {"op": s["op"], "columns": s["columns"], "params": s["params"], "stage": stage}
        )
    return create(
        db,
        owner_id,
        exp.dataset_id,
        f"From experiment: {cand.label}"[:200],
        steps,
        from_candidate=cand,
    )


def add_step(
    db: Session, owner_id: uuid.UUID, pipeline_id: uuid.UUID, body: dict[str, Any]
) -> PipelineStep:
    pipeline = get_pipeline(db, owner_id, pipeline_id, for_update=True)
    _editable(pipeline)
    steps = steps_of(db, pipeline)
    if len(steps) >= MAX_STEPS:
        raise _fail(f"At most {MAX_STEPS} steps.", "/op", "TOO_MANY_STEPS")
    op, columns, params, stage = _check_step_body(body, len(steps))
    rec = body.get("recommendation_id")
    rec_id = _owned_recommendation(db, owner_id, pipeline.dataset_id, rec) if rec else None
    position = body.get("position")
    position = len(steps) if position is None else max(0, min(int(position), len(steps)))
    row = _new_step(pipeline, position, op, columns, params, stage, rec_id)
    db.add(row)
    steps.insert(position, row)
    db.flush()
    _refresh(db, pipeline, steps)
    db.commit()
    return row


def _owned_step(steps: list[PipelineStep], step_id: uuid.UUID) -> PipelineStep:
    row = next((s for s in steps if s.id == step_id), None)
    if row is None:
        raise NotFound()
    return row


def edit_step(
    db: Session,
    owner_id: uuid.UUID,
    pipeline_id: uuid.UUID,
    step_id: uuid.UUID,
    body: dict[str, Any],
) -> PipelineStep:
    pipeline = get_pipeline(db, owner_id, pipeline_id, for_update=True)
    _editable(pipeline)
    steps = steps_of(db, pipeline)
    row = _owned_step(steps, step_id)
    if body.get("params") is not None:
        row.params = dict(body["params"])
    if body.get("columns") is not None:
        row.columns = list(body["columns"])
    if body.get("stage") is not None:
        row.stage = str(body["stage"])
    if body.get("position") is not None:
        steps.remove(row)
        steps.insert(max(0, min(int(body["position"]), len(steps))), row)
    _refresh(db, pipeline, steps)
    db.commit()
    return row


def delete_step(
    db: Session, owner_id: uuid.UUID, pipeline_id: uuid.UUID, step_id: uuid.UUID
) -> None:
    pipeline = get_pipeline(db, owner_id, pipeline_id, for_update=True)
    _editable(pipeline)
    steps = steps_of(db, pipeline)
    row = _owned_step(steps, step_id)
    steps.remove(row)
    db.delete(row)
    db.flush()
    _refresh(db, pipeline, steps)
    db.commit()


def validate(db: Session, owner_id: uuid.UUID, pipeline_id: uuid.UUID) -> CommitPlan:
    """Re-validate against the current roles (they may have changed since the last edit)."""
    pipeline = get_pipeline(db, owner_id, pipeline_id, for_update=True)
    steps = steps_of(db, pipeline)
    if pipeline.state in EDITABLE:
        plan = _refresh(db, pipeline, steps)
    else:
        plan = plan_for(db, pipeline, steps)
    db.commit()
    return plan


def list_pipelines(db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID) -> list[Pipeline]:
    get_owned(db, owner_id, dataset_id)
    return list(
        db.scalars(
            select(Pipeline)
            .where(Pipeline.owner_id == owner_id, Pipeline.dataset_id == dataset_id)
            .order_by(Pipeline.created_at.desc())
            .limit(100)
        )
    )


# ---- Impact preview ---------------------------------------------------------------------------


def start_preview(
    db: Session, settings: Settings, owner_id: uuid.UUID, pipeline_id: uuid.UUID
) -> Job:
    pipeline = get_pipeline(db, owner_id, pipeline_id, for_update=True)
    _editable(pipeline)
    plan = plan_for(db, pipeline, steps_of(db, pipeline))
    if not plan.valid or plan.spec_hash != pipeline.spec_hash:
        raise ValidationFailed(
            "The pipeline is not valid; fix its steps first.",
            errors=[
                {"pointer": "/steps", "code": e.code, "message": e.message} for e in plan.errors
            ],
        )
    job = create_job(
        db,
        settings,
        owner_id,
        "impact_preview",
        pipeline.dataset_id,
        {"pipeline_id": str(pipeline.id), "spec_hash": pipeline.spec_hash},
    )
    pipeline.impact_preview_job_id = job.id
    db.commit()
    return job


def read_preview(storage: ObjectStorage, db: Session, pipeline: Pipeline) -> dict[str, Any] | None:
    if pipeline.impact_preview_artifact_id is None:
        return None
    art = db.scalar(
        select(Artifact).where(
            Artifact.id == pipeline.impact_preview_artifact_id,
            Artifact.owner_id == pipeline.owner_id,
        )
    )
    if art is None:
        return None
    try:
        with storage.open_reader(art.object_key) as fh:
            raw = fh.read()
    except ObjectNotFound:
        return None
    if hashlib.sha256(raw).digest() != art.sha256:
        raise Conflict("The stored impact preview failed its integrity check; run it again.")
    payload: dict[str, Any] = json.loads(raw)
    return payload


def preview_current(db: Session, pipeline: Pipeline) -> bool:
    ds = db.get(Dataset, pipeline.dataset_id)
    return (
        pipeline.impact_preview_artifact_id is not None
        and pipeline.impact_preview_spec_hash == pipeline.spec_hash
        and ds is not None
        and ds.head_version_id == pipeline.source_version_id
        and not pipeline.spec_hash.startswith("draft:")
    )


# ---- Approvals --------------------------------------------------------------------------------


def _approval_valid(a: Approval, pipeline: Pipeline, step: PipelineStep | None) -> bool:
    if (
        a.pipeline_id != pipeline.id
        or a.spec_hash != pipeline.spec_hash
        or a.preview_artifact_id != pipeline.impact_preview_artifact_id
        or a.source_version_id != pipeline.source_version_id
    ):
        return False
    if step is None:
        return a.scope == "pipeline" and a.approved_hash == pipeline.spec_hash
    return (
        a.scope == "step"
        and a.pipeline_step_id == step.id
        and a.approved_hash == step.step_hash
        and (a.acknowledge_destructive or not step.destructive)
    )


@dataclass
class Readiness:
    ready: bool
    preview_current: bool
    steps: list[dict[str, Any]]  # high-risk steps and their approval state
    confirmation_required: bool
    confirmation_valid: bool
    stale_approvals: int
    problems: list[str]
    approval_ids: list[str]


def readiness(
    db: Session, pipeline: Pipeline, steps: list[PipelineStep], plan: CommitPlan
) -> Readiness:
    approvals = approvals_of(db, pipeline)
    current = preview_current(db, pipeline) and plan.valid and plan.spec_hash == pipeline.spec_hash
    problems = []
    if not plan.valid:
        problems.append("The pipeline has validation errors.")
    if not current:
        problems.append("Generate an impact preview for the current pipeline.")
    high = []
    used: list[str] = []
    for s in steps:
        if s.risk != "high":
            continue
        valid = [a for a in approvals if _approval_valid(a, pipeline, s)]
        high.append(
            {
                "step_id": str(s.id),
                "position": s.position,
                "op": s.op,
                "destructive": s.destructive,
                "approved": bool(valid) and current,
                "approval_id": str(valid[-1].id) if valid else None,
            }
        )
        if valid:
            used.append(str(valid[-1].id))
        else:
            problems.append("A high-risk step needs its own approval.")
    need_confirm = any(s.risk == "medium" for s in steps)
    confirms = [a for a in approvals if _approval_valid(a, pipeline, None)]
    if need_confirm and not confirms:
        problems.append("Medium-risk steps need a pipeline confirmation.")
    if confirms:
        used.append(str(confirms[-1].id))
    valid_ids = set(used)
    stale = sum(1 for a in approvals if str(a.id) not in valid_ids)
    ready = not problems
    return Readiness(
        ready,
        current,
        high,
        need_confirm,
        bool(confirms) and current,
        stale,
        sorted(set(problems)),
        used if ready else [],
    )


def _impact_for(payload: dict[str, Any], position: int | None) -> dict[str, Any]:
    summary = payload["summary"]
    if position is None:
        return {
            "totals": summary["totals"],
            "fitted_on_full_data": summary["fitted_on_full_data"],
            "steps": len(summary["steps"]),
            "model_steps": summary["model_steps"],
        }
    data_steps = summary["steps"]
    if position < len(data_steps):
        s = data_steps[position]
        return {
            k: s[k]
            for k in (
                "op",
                "rows_affected",
                "rows_listed",
                "rows_truncated",
                "columns_added",
                "columns_removed",
                "information_loss",
            )
        } | {
            "columns": [
                {
                    k: c[k]
                    for k in (
                        "column",
                        "rows_changed",
                        "estimates_filled",
                        "observed_overwritten",
                        "observed_re_expressed",
                    )
                }
                for c in s["columns"]
            ]
        }
    return {
        "stage": "model",
        "note": "Model-stage step: stored as the version's modelling "
        "specification; does not change the saved data.",
    }


def _statement(step: PipelineStep | None, impact: dict[str, Any]) -> str:
    if step is None:
        t = impact["totals"]
        return (
            f"I confirm this pipeline: {t['estimates_filled']} missing values will be "
            f"replaced by estimates, {t['observed_overwritten']} observed values "
            f"overwritten and {t['observed_re_expressed']} re-expressed, in a new version. "
            "The current version stays unchanged."
        )
    spec = get_op(step.op)
    title = spec.title if spec else step.op
    if step.stage == "model":
        return (
            f"I approve '{title}' as a model-stage step: it will be stored with the new "
            "version and run only inside model training; it will not change the saved data."
        )
    text = (
        f"I approve '{title}', which affects {impact.get('rows_affected', 0)} rows of the "
        "new version."
    )
    if step.destructive:
        text += (
            " Destructive: it removes or overwrites observed values, which cannot be "
            "recovered from the new version (earlier versions stay unchanged)."
        )
    return text


def approve(
    db: Session,
    storage: ObjectStorage,
    owner_id: uuid.UUID,
    pipeline_id: uuid.UUID,
    step_id: uuid.UUID | None,
    step_hash: str | None,
    spec_hash: str,
    acknowledge_destructive: bool,
) -> Approval:
    pipeline = get_pipeline(db, owner_id, pipeline_id, for_update=True)
    _editable(pipeline)
    steps = steps_of(db, pipeline)
    plan = plan_for(db, pipeline, steps)
    step = _owned_step(steps, step_id) if step_id else None
    if not plan.valid or plan.spec_hash != pipeline.spec_hash:
        raise ApprovalStale("The pipeline changed or is invalid; review it again.")
    if spec_hash != pipeline.spec_hash:
        raise ApprovalStale("The pipeline changed after it was shown to you; review it again.")
    if step is not None:
        if step.risk != "high":
            raise _fail(
                "Only high-risk steps are approved one by one; medium-risk steps are "
                "covered by the pipeline confirmation.",
                "/step",
                "APPROVAL_NOT_NEEDED",
            )
        if step_hash != step.step_hash:
            raise ApprovalStale("This step changed after it was shown to you; review it again.")
        if step.destructive and not acknowledge_destructive:
            raise _fail(
                "This step is destructive: acknowledge that observed values will be "
                "overwritten or removed.",
                "/acknowledge_destructive",
                "ACKNOWLEDGEMENT_REQUIRED",
            )
    if not preview_current(db, pipeline):
        raise ApprovalStale(
            "There is no impact preview for the current pipeline and source "
            "version; generate one first."
        )
    payload = read_preview(storage, db, pipeline)
    if payload is None or payload["summary"]["spec_hash"] != pipeline.spec_hash:
        raise ApprovalStale("The impact preview is not available for the current pipeline.")
    impact = _impact_for(payload, step.position if step else None)
    assert pipeline.impact_preview_artifact_id is not None
    approval = Approval(
        owner_id=owner_id,
        dataset_id=pipeline.dataset_id,
        pipeline_id=pipeline.id,
        pipeline_step_id=step.id if step else None,
        scope="step" if step else "pipeline",
        approved_hash=step.step_hash if step and step.step_hash else pipeline.spec_hash,
        spec_hash=pipeline.spec_hash,
        source_version_id=pipeline.source_version_id,
        preview_artifact_id=pipeline.impact_preview_artifact_id,
        impact_shown=impact,
        statement=_statement(step, impact),
        acknowledge_destructive=bool(acknowledge_destructive and step and step.destructive),
        approved_by=owner_id,
    )
    db.add(approval)
    db.commit()
    return approval


# ---- Commit, revert, comparison ---------------------------------------------------------------


def version_count(db: Session, dataset_id: uuid.UUID) -> int:
    return int(
        db.scalar(
            select(func.count())
            .select_from(DatasetVersion)
            .where(DatasetVersion.dataset_id == dataset_id)
        )
        or 0
    )


def check_quota(db: Session, settings: Settings, dataset_id: uuid.UUID) -> None:
    if version_count(db, dataset_id) >= settings.max_versions_per_dataset:
        raise QuotaExceeded(
            f"A dataset can have at most {settings.max_versions_per_dataset} "
            "versions. No version is deleted to make room."
        )


def commit_readiness(
    db: Session, settings: Settings, pipeline: Pipeline, expected_spec_hash: str
) -> tuple[CommitPlan, Readiness]:
    """Authoritative commit checks (API before queueing; worker before reading data and again
    under the dataset lock). Raises the domain error describing the first failure."""
    if pipeline.state not in EDITABLE:
        raise Conflict("This pipeline was already committed.")
    ds = db.get(Dataset, pipeline.dataset_id)
    if ds is None or ds.deleted_at is not None:
        raise NotFound()
    if ds.head_version_id != pipeline.source_version_id:
        raise Conflict(
            "The dataset has a newer version than this pipeline's source. Create a "
            "new pipeline from the current version."
        )
    steps = steps_of(db, pipeline)
    plan = plan_for(db, pipeline, steps)
    if not plan.valid:
        raise ValidationFailed(
            "The pipeline has validation errors.",
            errors=[
                {"pointer": "/steps", "code": e.code, "message": e.message} for e in plan.errors
            ],
        )
    if plan.spec_hash != pipeline.spec_hash or expected_spec_hash != pipeline.spec_hash:
        raise ApprovalStale("The pipeline changed after it was reviewed; review it again.")
    ready = readiness(db, pipeline, steps, plan)
    if not ready.ready:
        exc = ApprovalStale if ready.stale_approvals else ApprovalRequired
        raise exc(" ".join(ready.problems))
    check_quota(db, settings, pipeline.dataset_id)
    return plan, ready


def _mutation_job(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    job_type: str,
    dataset_id: uuid.UUID,
    params: dict[str, Any],
    key: str | None,
) -> Job:
    try:
        with db.begin_nested():
            job = create_job(db, settings, owner_id, job_type, dataset_id, params)
            job.idempotency_key = key
            db.flush()
    except IntegrityError as exc:
        raise Conflict(
            "Another commit or revert is already queued or running for this dataset."
        ) from exc
    return job


def _replay(db: Session, owner_id: uuid.UUID, key: str | None, job_type: str) -> Job | None:
    if not key:
        return None
    prior = db.scalar(select(Job).where(Job.owner_id == owner_id, Job.idempotency_key == key))
    if prior is not None and prior.type != job_type:
        raise _fail(
            "This Idempotency-Key was used for another request.",
            "/headers/Idempotency-Key",
            "IDEMPOTENCY_KEY_REUSED",
        )
    return prior


def commit(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    pipeline_id: uuid.UUID,
    spec_hash: str,
    key: str | None,
) -> tuple[Job, bool]:
    pipeline = get_pipeline(db, owner_id, pipeline_id)  # ownership first: others get 404
    if not key:
        raise _fail(
            "Committing requires an Idempotency-Key header.",
            "/headers/Idempotency-Key",
            "IDEMPOTENCY_KEY_REQUIRED",
        )
    prior = _replay(db, owner_id, key, "commit")
    if prior is not None:
        return prior, False
    db.execute(select(Dataset.id).where(Dataset.id == pipeline.dataset_id).with_for_update())
    db.refresh(pipeline, with_for_update=True)
    _, ready = commit_readiness(db, settings, pipeline, spec_hash)
    job = _mutation_job(
        db,
        settings,
        owner_id,
        "commit",
        pipeline.dataset_id,
        {
            "pipeline_id": str(pipeline.id),
            "spec_hash": pipeline.spec_hash,
            "source_version_id": str(pipeline.source_version_id),
            "version_id": str(uuid.uuid4()),
            "transformation_run_id": str(uuid.uuid4()),
            "approval_ids": ready.approval_ids,
        },
        key,
    )
    db.commit()
    return job, True


def revert(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
    key: str | None,
) -> tuple[Job, bool]:
    prior = _replay(db, owner_id, key, "revert")
    if prior is not None:
        return prior, False
    ds = get_owned(db, owner_id, dataset_id, for_update=True)
    target, target_ds = get_owned_version(db, owner_id, version_id)
    if target_ds.id != ds.id:
        raise NotFound()
    if ds.head_version_id == target.id:
        raise Conflict("This is already the current version.")
    check_quota(db, settings, ds.id)
    job = _mutation_job(
        db,
        settings,
        owner_id,
        "revert",
        ds.id,
        {
            "target_version_id": str(target.id),
            "expected_head": str(ds.head_version_id),
            "version_id": str(uuid.uuid4()),
        },
        key,
    )
    db.commit()
    return job, True


def compare(
    db: Session, settings: Settings, owner_id: uuid.UUID, a: uuid.UUID, b: uuid.UUID
) -> Job:
    va, ds_a = get_owned_version(db, owner_id, a)
    vb, ds_b = get_owned_version(db, owner_id, b)
    if ds_a.id != ds_b.id:
        raise _fail(
            "Both versions must belong to the same dataset.", "/version_id", "DIFFERENT_DATASETS"
        )
    job = create_job(
        db,
        settings,
        owner_id,
        "compare_versions",
        ds_a.id,
        {"version_a": str(va.id), "version_b": str(vb.id)},
    )
    db.commit()
    return job


def comparison(
    db: Session, storage: ObjectStorage, owner_id: uuid.UUID, job_id: uuid.UUID
) -> tuple[Job, dict[str, Any] | None]:
    job = db.scalar(
        select(Job).where(
            Job.id == job_id, Job.owner_id == owner_id, Job.type == "compare_versions"
        )
    )
    if job is None:
        raise NotFound()
    if job.dataset_id is not None:
        get_owned(db, owner_id, job.dataset_id)
    artifact_id = (job.result or {}).get("artifact_id")
    if job.status != "succeeded" or not artifact_id:
        return job, None
    art = db.scalar(
        select(Artifact).where(Artifact.id == uuid.UUID(artifact_id), Artifact.owner_id == owner_id)
    )
    if art is None:
        return job, None
    with storage.open_reader(art.object_key) as fh:
        raw = fh.read()
    if hashlib.sha256(raw).digest() != art.sha256:
        raise Conflict("The stored comparison failed its integrity check.")
    result: dict[str, Any] = json.loads(raw)
    return job, result


def get_run(
    db: Session, owner_id: uuid.UUID, run_id: uuid.UUID
) -> tuple[TransformationRun, list[Approval]]:
    run = db.scalar(
        select(TransformationRun).where(
            TransformationRun.id == run_id, TransformationRun.owner_id == owner_id
        )
    )
    if run is None:
        raise NotFound()
    get_owned(db, owner_id, run.dataset_id)
    ids = [uuid.UUID(i) for i in run.approval_ids]
    approvals = (
        list(
            db.scalars(select(Approval).where(Approval.owner_id == owner_id, Approval.id.in_(ids)))
        )
        if ids
        else []
    )
    return run, approvals

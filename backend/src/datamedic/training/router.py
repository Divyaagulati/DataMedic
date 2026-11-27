import uuid
from typing import Annotated

from fastapi import APIRouter, Header, Response, status

from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.db.models import Artifact
from datamedic.experiments.schemas import SplitOut
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut
from datamedic.training import privacy, service
from datamedic.training.schemas import (
    AuditAcceptedOut,
    AuditOut,
    ModelCatalogueOut,
    ModelEvaluationOut,
    TrainingAcceptedOut,
    TrainingListOut,
    TrainingPlanOut,
    TrainingRunIn,
    TrainingRunOut,
    TrainingRunSummaryOut,
)

router = APIRouter(prefix="/api/v1", tags=["training"])

IdempotencyKey = Annotated[uuid.UUID | None, Header(alias="Idempotency-Key")]


@router.get("/ml/models")
def model_catalogue(principal: PrincipalDep) -> ModelCatalogueOut:
    """Models, fixed tuning grids, metric definitions and the reliability-audit checks."""
    del principal
    return ModelCatalogueOut(**service.catalogue())


@router.post("/versions/{version_id}/training-plan")
def training_plan(
    version_id: uuid.UUID, body: TrainingRunIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> TrainingPlanOut:
    """Advisory check of a training configuration: task and target, eligible features,
    evaluation eligibility (direct, replay from an ancestor, or blocked), split, preprocessing,
    and the fit estimate. Nothing is stored or queued; the worker re-checks everything."""
    return TrainingPlanOut(
        **service.plan(
            db, ctx.settings, principal.user.id, version_id, body.model_dump(mode="json")
        )
    )


@router.post("/versions/{version_id}/training-runs", status_code=status.HTTP_202_ACCEPTED)
def create_training_run(
    version_id: uuid.UUID,
    body: TrainingRunIn,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
    response: Response,
    idempotency_key: IdempotencyKey = None,
) -> TrainingAcceptedOut:
    """Queue a training run. Training never changes the dataset. Leakage-controlled
    evaluation of a version fitted on all rows is refused (409 EVALUATION_BLOCKED) unless it
    can be replayed from an eligible ancestor, or ``allow_exploratory`` is set (a permanently
    labelled run)."""
    run, job, created = service.create(
        db,
        ctx.settings,
        principal.user.id,
        version_id,
        body.model_dump(mode="json"),
        str(idempotency_key) if idempotency_key else None,
    )
    if created:
        publish_committed(db, ctx.require_publisher(), job.id, job.type)
        db.refresh(job)
        db.refresh(run)
    else:
        response.status_code = status.HTTP_200_OK
    return TrainingAcceptedOut(
        training_run=TrainingRunSummaryOut.build(run, job), job=JobOut.from_job(job)
    )


@router.get("/datasets/{dataset_id}/training-runs")
def list_training_runs(
    dataset_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> TrainingListOut:
    rows = service.list_for_dataset(db, principal.user.id, dataset_id)
    return TrainingListOut(items=[TrainingRunSummaryOut.build(r, j, a, e) for r, j, a, e in rows])


@router.get("/training-runs/{training_run_id}")
def get_training_run(
    training_run_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> TrainingRunOut:
    run, job, evals, split = service.detail(db, principal.user.id, training_run_id)
    audit = service.latest_audit_run(db, principal.user.id, run.id)
    base = TrainingRunSummaryOut.build(run, job, audit, evals)
    artifact = db.get(Artifact, run.oof_artifact_id) if run.oof_artifact_id else None
    # Sensitive-column protection is re-applied on every read (training-time or current
    # sensitivity, and derived columns); stored rows are never modified.
    protected = privacy.for_run(db, run)
    target_protected = run.target_column in protected
    withheld: list[str] = []
    evaluations = []
    for e in evals:
        out = ModelEvaluationOut.build(e)
        details, hidden = privacy.redact_details(e.details, protected, target_protected)
        withheld += hidden
        evaluations.append(out.model_copy(update={"details": details}))
    return TrainingRunOut(
        **base.model_dump(),
        config=run.config,
        eligibility=run.eligibility,
        replay=run.replay,
        split_config=run.split_config,
        split=SplitOut.build(split) if split else None,
        tuning=run.tuning,
        preprocess_steps=run.preprocess_steps,
        spec_hash=run.spec_hash,
        source_content_hash=run.source_content_hash.hex(),
        summary=privacy.redact_summary(run.summary, target_protected),
        library_versions=run.library_versions,
        engine_version=run.engine_version,
        oof_artifact_sha256=artifact.sha256.hex() if artifact is not None else None,
        evaluations=evaluations,
        privacy=privacy.disclosure(protected, withheld, run.target_column),
    )


@router.post("/training-runs/{training_run_id}/rerun", status_code=status.HTTP_202_ACCEPTED)
def rerun_training_run(
    training_run_id: uuid.UUID,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
    response: Response,
    idempotency_key: IdempotencyKey = None,
) -> TrainingAcceptedOut:
    """Re-run exactly the stored configuration (roles snapshot, split, seeds, models, tuning,
    preprocessing). An exploratory run stays exploratory."""
    run, job, created = service.rerun(
        db,
        ctx.settings,
        principal.user.id,
        training_run_id,
        str(idempotency_key) if idempotency_key else None,
    )
    if created:
        publish_committed(db, ctx.require_publisher(), job.id, job.type)
        db.refresh(job)
        db.refresh(run)
    else:
        response.status_code = status.HTTP_200_OK
    return TrainingAcceptedOut(
        training_run=TrainingRunSummaryOut.build(run, job), job=JobOut.from_job(job)
    )


@router.post("/training-runs/{training_run_id}/audit", status_code=status.HTTP_202_ACCEPTED)
def run_audit(
    training_run_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> AuditAcceptedOut:
    """Run the reliability audit of a completed run again (e.g. after a role change)."""
    job = service.start_audit(db, ctx.settings, principal.user.id, training_run_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return AuditAcceptedOut(job=JobOut.from_job(job))


@router.get("/training-runs/{training_run_id}/audit")
def get_audit(training_run_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> AuditOut:
    run, audit, findings, job, stale = service.audit(db, principal.user.id, training_run_id)
    return AuditOut.build(run, audit, findings, job, stale)

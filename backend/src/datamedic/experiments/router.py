import uuid
from typing import Annotated

from fastapi import APIRouter, Header, Response, status

from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.experiments import service
from datamedic.experiments.schemas import (
    CandidateOut,
    ExperimentAcceptedOut,
    ExperimentIn,
    ExperimentListOut,
    ExperimentOut,
    ExperimentSummaryOut,
    SplitOut,
)
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut

router = APIRouter(prefix="/api/v1", tags=["experiments"])

IdempotencyKey = Annotated[uuid.UUID | None, Header(alias="Idempotency-Key")]


@router.post("/versions/{version_id}/experiments", status_code=status.HTTP_202_ACCEPTED)
def create_experiment(
    version_id: uuid.UUID,
    body: ExperimentIn,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
    response: Response,
    idempotency_key: IdempotencyKey = None,
) -> ExperimentAcceptedOut:
    """Queue an experiment: the baseline plus up to 6 candidate pipelines, all evaluated on
    the same frozen folds. Candidates may contain high-risk steps; they are labelled, never
    committed. Experiments never change the dataset."""
    exp, job, created = service.create(
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
        db.refresh(exp)
    else:
        response.status_code = status.HTTP_200_OK
    return ExperimentAcceptedOut(
        experiment=ExperimentSummaryOut.build(exp, job), job=JobOut.from_job(job)
    )


@router.get("/versions/{version_id}/experiments")
def list_experiments(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> ExperimentListOut:
    rows = service.list_for_version(db, principal.user.id, version_id)
    return ExperimentListOut(items=[ExperimentSummaryOut.build(e, j) for e, j in rows])


@router.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> ExperimentOut:
    exp, job, cands, results, split = service.detail(db, principal.user.id, experiment_id)
    base = ExperimentSummaryOut.build(exp, job)
    return ExperimentOut(
        **base.model_dump(),
        config=exp.config,
        recommendation_set_id=exp.recommendation_set_id,
        split=SplitOut.build(split) if split else None,
        summary=exp.summary,
        library_versions=exp.library_versions,
        candidates=[CandidateOut.build(c, results.get(c.id)) for c in cands],
    )


@router.post("/experiments/{experiment_id}/rerun", status_code=status.HTTP_202_ACCEPTED)
def rerun_experiment(
    experiment_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> ExperimentAcceptedOut:
    """Re-run exactly the stored configuration (same roles snapshot, folds, seeds and
    candidates) on the same source version."""
    exp, job = service.rerun(db, ctx.settings, principal.user.id, experiment_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    db.refresh(exp)
    return ExperimentAcceptedOut(
        experiment=ExperimentSummaryOut.build(exp, job), job=JobOut.from_job(job)
    )

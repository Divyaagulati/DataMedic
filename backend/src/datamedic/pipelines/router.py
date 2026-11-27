import uuid
from typing import Annotated

from fastapi import APIRouter, Header, Query, Response, status
from sqlalchemy import select

from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.datasets.service import get_owned
from datamedic.db.models import Job, Pipeline, VersionEvent
from datamedic.errors import NotFound
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut
from datamedic.pipelines import service
from datamedic.pipelines.schemas import (
    ApprovalOut,
    ApproveIn,
    CommitAcceptedOut,
    CommitIn,
    ComparisonOut,
    ConfirmIn,
    ImpactPreviewOut,
    ImpactPreviewRowsOut,
    JobAcceptedOut,
    PipelineIn,
    PipelineListOut,
    PipelineOut,
    PipelineStepIn,
    PipelineStepOut,
    PipelineStepPatchIn,
    PipelineSummaryOut,
    ReadinessOut,
    RevertIn,
    TransformationRunOut,
    ValidationOut,
    VersionEventListOut,
    VersionEventOut,
)
from datamedic.pipelines.versioning import render_code
from datamedic.versions.service import get_owned_version

router = APIRouter(prefix="/api/v1", tags=["pipelines"])

IdempotencyKey = Annotated[uuid.UUID | None, Header(alias="Idempotency-Key")]


def _out(db: DbDep, pipeline: Pipeline) -> PipelineOut:
    steps = service.steps_of(db, pipeline)
    plan = service.plan_for(db, pipeline, steps)
    ready = service.readiness(db, pipeline, steps, plan)
    approved = {s["step_id"] for s in ready.steps if s["approved"]}
    used = set(ready.approval_ids) | {s["approval_id"] for s in ready.steps if s["approval_id"]}
    ds = get_owned(db, pipeline.owner_id, pipeline.dataset_id)
    job = db.get(Job, pipeline.impact_preview_job_id) if pipeline.impact_preview_job_id else None
    return PipelineOut(
        **PipelineSummaryOut.build(pipeline).model_dump(),
        source_is_head=ds.head_version_id == pipeline.source_version_id,
        spec_hash=pipeline.spec_hash,
        validation=pipeline.validation_report,
        steps=[PipelineStepOut.build(s, str(s.id) in approved) for s in steps],
        approvals=[
            ApprovalOut.build(a, valid=str(a.id) in used and ready.preview_current)
            for a in service.approvals_of(db, pipeline)
        ],
        readiness=ReadinessOut(
            ready=ready.ready,
            preview_current=ready.preview_current,
            high_risk_steps=ready.steps,
            confirmation_required=ready.confirmation_required,
            confirmation_valid=ready.confirmation_valid,
            stale_approvals=ready.stale_approvals,
            problems=ready.problems,
        ),
        preview_job=JobOut.from_job(job) if job else None,
        from_experiment_candidate_id=pipeline.from_experiment_candidate_id,
    )


def _reload(db: DbDep, owner_id: uuid.UUID, pipeline_id: uuid.UUID) -> PipelineOut:
    return _out(db, service.get_pipeline(db, owner_id, pipeline_id))


# ---- Pipelines and steps ----------------------------------------------------------------------


@router.get("/datasets/{dataset_id}/pipelines")
def list_pipelines(dataset_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> PipelineListOut:
    return PipelineListOut(
        items=[
            PipelineSummaryOut.build(p)
            for p in service.list_pipelines(db, principal.user.id, dataset_id)
        ]
    )


@router.post("/datasets/{dataset_id}/pipelines", status_code=status.HTTP_201_CREATED)
def create_pipeline(
    dataset_id: uuid.UUID, body: PipelineIn, principal: PrincipalDep, db: DbDep
) -> PipelineOut:
    """A pipeline against the current head. Nothing changes the dataset until a commit."""
    pipeline = service.create(
        db,
        principal.user.id,
        dataset_id,
        body.name,
        [s.model_dump(mode="json") for s in body.steps],
    )
    return _out(db, pipeline)


@router.post(
    "/experiment-candidates/{candidate_id}/to-pipeline", status_code=status.HTTP_201_CREATED
)
def candidate_to_pipeline(
    candidate_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> PipelineOut:
    """An editable pipeline from a lab candidate. Experiment results are evidence, not
    approval: the pipeline needs its own preview and approvals."""
    return _out(db, service.from_candidate(db, principal.user.id, candidate_id))


@router.get("/pipelines/{pipeline_id}")
def get_pipeline(pipeline_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> PipelineOut:
    return _reload(db, principal.user.id, pipeline_id)


@router.post("/pipelines/{pipeline_id}/steps", status_code=status.HTTP_201_CREATED)
def add_step(
    pipeline_id: uuid.UUID, body: PipelineStepIn, principal: PrincipalDep, db: DbDep
) -> PipelineOut:
    service.add_step(db, principal.user.id, pipeline_id, body.model_dump(mode="json"))
    return _reload(db, principal.user.id, pipeline_id)


@router.patch("/pipelines/{pipeline_id}/steps/{step_id}")
def edit_step(
    pipeline_id: uuid.UUID,
    step_id: uuid.UUID,
    body: PipelineStepPatchIn,
    principal: PrincipalDep,
    db: DbDep,
) -> PipelineOut:
    """Any edit (parameters, columns, stage, position) changes the hashes, so earlier
    previews and approvals stop applying."""
    service.edit_step(
        db,
        principal.user.id,
        pipeline_id,
        step_id,
        body.model_dump(mode="json", exclude_unset=True),
    )
    return _reload(db, principal.user.id, pipeline_id)


@router.delete("/pipelines/{pipeline_id}/steps/{step_id}")
def delete_step(
    pipeline_id: uuid.UUID, step_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> PipelineOut:
    service.delete_step(db, principal.user.id, pipeline_id, step_id)
    return _reload(db, principal.user.id, pipeline_id)


@router.post("/pipelines/{pipeline_id}/validate")
def validate_pipeline(pipeline_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> ValidationOut:
    plan = service.validate(db, principal.user.id, pipeline_id)
    pipeline = service.get_pipeline(db, principal.user.id, pipeline_id)
    return ValidationOut(**plan.report(), spec_hash=pipeline.spec_hash)


# ---- Impact preview and approvals -------------------------------------------------------------


@router.post("/pipelines/{pipeline_id}/impact-preview", status_code=status.HTTP_202_ACCEPTED)
def start_preview(
    pipeline_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> JobAcceptedOut:
    job = service.start_preview(db, ctx.settings, principal.user.id, pipeline_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return JobAcceptedOut(job=JobOut.from_job(job))


@router.get("/pipelines/{pipeline_id}/impact-preview")
def get_preview(
    pipeline_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> ImpactPreviewOut:
    pipeline = service.get_pipeline(db, principal.user.id, pipeline_id)
    payload = service.read_preview(ctx.storage, db, pipeline)
    job = db.get(Job, pipeline.impact_preview_job_id) if pipeline.impact_preview_job_id else None
    return ImpactPreviewOut(
        current=payload is not None and service.preview_current(db, pipeline),
        job=JobOut.from_job(job) if job else None,
        summary=payload["summary"] if payload else None,
    )


@router.get("/pipelines/{pipeline_id}/impact-preview/rows")
def preview_rows(
    pipeline_id: uuid.UUID,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
    step: Annotated[int, Query(ge=0, le=1000)] = 0,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> ImpactPreviewRowsOut:
    """Before/after values of affected rows (owner only, never cached; sensitive columns
    redacted; at most the configured number of rows per step)."""
    pipeline = service.get_pipeline(db, principal.user.id, pipeline_id)
    payload = service.read_preview(ctx.storage, db, pipeline)
    if payload is None or step >= len(payload["rows"]):
        raise NotFound()
    listed, summary = payload["rows"][step], payload["summary"]["steps"][step]
    rows = listed["rows"][offset : offset + limit]
    nxt = offset + limit if offset + limit < len(listed["rows"]) else None
    return ImpactPreviewRowsOut(
        step_index=step,
        columns=listed["columns"],
        redacted_columns=listed["redacted_columns"],
        rows=rows,
        rows_affected=summary["rows_affected"],
        rows_listed=summary["rows_listed"],
        rows_truncated=summary["rows_truncated"],
        next_offset=nxt,
    )


@router.post(
    "/pipelines/{pipeline_id}/steps/{step_id}/approve", status_code=status.HTTP_201_CREATED
)
def approve_step(
    pipeline_id: uuid.UUID,
    step_id: uuid.UUID,
    body: ApproveIn,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
) -> ApprovalOut:
    approval = service.approve(
        db,
        ctx.storage,
        principal.user.id,
        pipeline_id,
        step_id,
        body.step_hash,
        body.spec_hash,
        body.acknowledge_destructive,
    )
    return ApprovalOut.build(approval, valid=True)


@router.post("/pipelines/{pipeline_id}/confirm", status_code=status.HTTP_201_CREATED)
def confirm_pipeline(
    pipeline_id: uuid.UUID, body: ConfirmIn, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> ApprovalOut:
    approval = service.approve(
        db, ctx.storage, principal.user.id, pipeline_id, None, None, body.spec_hash, False
    )
    return ApprovalOut.build(approval, valid=True)


# ---- Commit, runs, revert, comparison, code ---------------------------------------------------


@router.post("/pipelines/{pipeline_id}/commit", status_code=status.HTTP_202_ACCEPTED)
def commit_pipeline(
    pipeline_id: uuid.UUID,
    body: CommitIn,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
    response: Response,
    idempotency_key: IdempotencyKey = None,
) -> CommitAcceptedOut:
    """Queue the commit of the approved pipeline into a new version (requires
    ``Idempotency-Key``). Every check is repeated by the worker."""
    job, created = service.commit(
        db,
        ctx.settings,
        principal.user.id,
        pipeline_id,
        body.spec_hash,
        str(idempotency_key) if idempotency_key else None,
    )
    run_id = uuid.UUID(job.params["transformation_run_id"])
    if created:
        publish_committed(db, ctx.require_publisher(), job.id, job.type)
        db.refresh(job)
    else:
        response.status_code = status.HTTP_200_OK
    return CommitAcceptedOut(job=JobOut.from_job(job), transformation_run_id=run_id)


@router.get("/transformation-runs/{transformation_run_id}")
def get_run(
    transformation_run_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> TransformationRunOut:
    run, approvals = service.get_run(db, principal.user.id, transformation_run_id)
    return TransformationRunOut.build(run, approvals)


@router.post("/datasets/{dataset_id}/revert", status_code=status.HTTP_202_ACCEPTED)
def revert_dataset(
    dataset_id: uuid.UUID,
    body: RevertIn,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
    response: Response,
    idempotency_key: IdempotencyKey = None,
) -> JobAcceptedOut:
    """Create a new version whose content is a server-side copy of an earlier version. No
    version is deleted or changed."""
    job, created = service.revert(
        db,
        ctx.settings,
        principal.user.id,
        dataset_id,
        body.version_id,
        str(idempotency_key) if idempotency_key else None,
    )
    if created:
        publish_committed(db, ctx.require_publisher(), job.id, job.type)
        db.refresh(job)
    else:
        response.status_code = status.HTTP_200_OK
    return JobAcceptedOut(job=JobOut.from_job(job))


@router.get("/datasets/{dataset_id}/version-events")
def version_events(
    dataset_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> VersionEventListOut:
    ds = get_owned(db, principal.user.id, dataset_id)
    rows = db.scalars(
        select(VersionEvent)
        .where(VersionEvent.owner_id == principal.user.id, VersionEvent.dataset_id == ds.id)
        .order_by(VersionEvent.created_at)
    )
    return VersionEventListOut(items=[VersionEventOut.build(e) for e in rows])


@router.post(
    "/versions/{version_id}/compare/{other_version_id}", status_code=status.HTTP_202_ACCEPTED
)
def compare_versions(
    version_id: uuid.UUID,
    other_version_id: uuid.UUID,
    principal: PrincipalDep,
    db: DbDep,
    ctx: CtxDep,
) -> JobAcceptedOut:
    """Before/after comparison (counts and statistics only, no row values)."""
    job = service.compare(db, ctx.settings, principal.user.id, version_id, other_version_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return JobAcceptedOut(job=JobOut.from_job(job))


@router.get("/version-comparisons/{job_id}")
def get_comparison(
    job_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> ComparisonOut:
    job, result = service.comparison(db, ctx.storage, principal.user.id, job_id)
    return ComparisonOut(job=JobOut.from_job(job), result=result)


@router.get(
    "/versions/{version_id}/code",
    response_class=Response,
    responses={200: {"content": {"text/x-python": {}}}},
)
def version_code(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> Response:
    """``pipeline.py`` reproducing the version from its original file (owner only, never
    cached). Contains no data values, credentials or storage paths."""
    version, _ = get_owned_version(db, principal.user.id, version_id)
    code = render_code(db, version)
    return Response(
        content=code,
        media_type="text/x-python",
        headers={
            "Content-Disposition": f'attachment; filename="pipeline_v{version.version_number}.py"'
        },
    )

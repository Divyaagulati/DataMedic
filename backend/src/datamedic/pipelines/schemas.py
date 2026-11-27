import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from datamedic.db.models import Approval, Pipeline, PipelineStep, TransformationRun, VersionEvent
from datamedic.engine.preprocess.catalog import get_op
from datamedic.jobs.schemas import JobOut

Stage = Literal["data", "model"]


class PipelineStepIn(BaseModel):
    op: str = Field(min_length=1, max_length=64)
    columns: list[str] = Field(default_factory=list, max_length=500)
    params: dict[str, Any] = Field(default_factory=dict)
    stage: Stage | None = None  # default: the catalogue stage ('data' for 'either' ops)
    position: int | None = Field(default=None, ge=0, le=1000)
    recommendation_id: uuid.UUID | None = None


class PipelineStepPatchIn(BaseModel):
    columns: list[str] | None = Field(default=None, max_length=500)
    params: dict[str, Any] | None = None
    stage: Stage | None = None
    position: int | None = Field(default=None, ge=0, le=1000)


class PipelineIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    steps: list[PipelineStepIn] = Field(default_factory=list, max_length=50)


class ApproveIn(BaseModel):
    step_hash: str = Field(min_length=1, max_length=128)
    spec_hash: str = Field(min_length=1, max_length=128)
    acknowledge_destructive: bool = False


class ConfirmIn(BaseModel):
    spec_hash: str = Field(min_length=1, max_length=128)


class CommitIn(BaseModel):
    spec_hash: str = Field(min_length=1, max_length=128)


class RevertIn(BaseModel):
    version_id: uuid.UUID


class ApprovalOut(BaseModel):
    id: uuid.UUID
    scope: str
    pipeline_step_id: uuid.UUID | None
    approved_hash: str
    spec_hash: str
    impact_shown: dict[str, Any]
    statement: str
    acknowledge_destructive: bool
    created_at: datetime
    valid: bool = False

    @classmethod
    def build(cls, a: Approval, valid: bool = False) -> "ApprovalOut":
        return cls(
            id=a.id,
            scope=a.scope,
            pipeline_step_id=a.pipeline_step_id,
            approved_hash=a.approved_hash,
            spec_hash=a.spec_hash,
            impact_shown=a.impact_shown,
            statement=a.statement,
            acknowledge_destructive=a.acknowledge_destructive,
            created_at=a.created_at,
            valid=valid,
        )


class PipelineStepOut(BaseModel):
    id: uuid.UUID
    position: int
    op: str
    op_version: str
    op_title: str
    catalog_stage: str
    stage: str
    columns: list[str]
    params: dict[str, Any]
    risk: str
    destructive: bool
    kind: str
    step_hash: str | None
    approved: bool
    from_recommendation_id: uuid.UUID | None

    @classmethod
    def build(cls, s: PipelineStep, approved: bool) -> "PipelineStepOut":
        spec = get_op(s.op)
        return cls(
            id=s.id,
            position=s.position,
            op=s.op,
            op_version=s.op_version,
            op_title=spec.title if spec else s.op,
            catalog_stage=spec.stage if spec else s.stage,
            stage=s.stage,
            columns=s.columns,
            params=s.params,
            risk=s.risk,
            destructive=s.destructive,
            kind=s.kind,
            step_hash=s.step_hash,
            approved=approved,
            from_recommendation_id=s.from_recommendation_id,
        )


class ReadinessOut(BaseModel):
    ready: bool
    preview_current: bool
    high_risk_steps: list[dict[str, Any]]
    confirmation_required: bool
    confirmation_valid: bool
    stale_approvals: int
    problems: list[str]


class PipelineSummaryOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    name: str
    state: str
    source_version_id: uuid.UUID
    committed_version_id: uuid.UUID | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def build(cls, p: Pipeline) -> "PipelineSummaryOut":
        return cls(
            id=p.id,
            dataset_id=p.dataset_id,
            name=p.name,
            state=p.state,
            source_version_id=p.source_version_id,
            committed_version_id=p.committed_version_id,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )


class PipelineOut(PipelineSummaryOut):
    source_is_head: bool
    spec_hash: str
    validation: dict[str, Any] | None
    steps: list[PipelineStepOut]
    approvals: list[ApprovalOut]
    readiness: ReadinessOut
    preview_job: JobOut | None
    from_experiment_candidate_id: uuid.UUID | None


class PipelineListOut(BaseModel):
    items: list[PipelineSummaryOut]


class ValidationOut(BaseModel):
    valid: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    data_steps: int
    model_steps: int
    spec_hash: str


class ImpactPreviewOut(BaseModel):
    """The impact preview. ``current`` is false when the pipeline, its source version or the
    dataset head changed after it was computed (it cannot back an approval then)."""

    current: bool
    job: JobOut | None
    summary: dict[str, Any] | None


class ImpactPreviewRowsOut(BaseModel):
    step_index: int
    columns: list[str]
    redacted_columns: list[str]
    rows: list[dict[str, Any]]
    rows_affected: int
    rows_listed: int
    rows_truncated: bool
    next_offset: int | None


class CommitAcceptedOut(BaseModel):
    job: JobOut
    transformation_run_id: uuid.UUID


class JobAcceptedOut(BaseModel):
    job: JobOut


class ComparisonOut(BaseModel):
    job: JobOut
    result: dict[str, Any] | None


class TransformationRunOut(BaseModel):
    id: uuid.UUID
    pipeline_id: uuid.UUID
    source_version_id: uuid.UUID
    result_version_id: uuid.UUID
    spec_hash: str
    steps: list[dict[str, Any]]
    data_ops: list[dict[str, Any]]
    model_pipeline_spec: list[dict[str, Any]]
    step_reports: list[dict[str, Any]]
    roles_report: dict[str, Any]
    approvals: list[ApprovalOut]
    seeds: dict[str, Any]
    engine_version: str
    library_versions: dict[str, Any]
    created_at: datetime

    @classmethod
    def build(cls, r: TransformationRun, approvals: list[Approval]) -> "TransformationRunOut":
        return cls(
            id=r.id,
            pipeline_id=r.pipeline_id,
            source_version_id=r.source_version_id,
            result_version_id=r.result_version_id,
            spec_hash=r.spec_hash,
            steps=r.spec_snapshot.get("steps", []),
            data_ops=r.spec_snapshot.get("data_ops", []),
            model_pipeline_spec=r.model_pipeline_spec,
            step_reports=r.step_reports,
            roles_report=r.roles_report,
            approvals=[ApprovalOut.build(a, valid=True) for a in approvals],
            seeds=r.seeds,
            engine_version=r.engine_version,
            library_versions=r.library_versions,
            created_at=r.created_at,
        )


class VersionEventOut(BaseModel):
    id: uuid.UUID
    event: str
    version_id: uuid.UUID
    related_version_id: uuid.UUID | None
    details: dict[str, Any]
    created_at: datetime

    @classmethod
    def build(cls, e: VersionEvent) -> "VersionEventOut":
        return cls(
            id=e.id,
            event=e.event,
            version_id=e.version_id,
            related_version_id=e.related_version_id,
            details=e.details,
            created_at=e.created_at,
        )


class VersionEventListOut(BaseModel):
    items: list[VersionEventOut]

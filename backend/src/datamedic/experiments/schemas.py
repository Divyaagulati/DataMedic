import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from datamedic.db.models import (
    Experiment,
    ExperimentCandidate,
    ExperimentResult,
    Job,
    SplitDefinition,
)
from datamedic.jobs.schemas import JobOut


class StepIn(BaseModel):
    op: str = Field(min_length=1, max_length=64)
    columns: list[str] = Field(default_factory=list, max_length=500)
    params: dict[str, Any] = Field(default_factory=dict)


class CandidateIn(BaseModel):
    label: str = Field(min_length=1, max_length=100)
    steps: list[StepIn] = Field(max_length=50)
    recommendation_ids: list[uuid.UUID] = Field(default_factory=list, max_length=100)


class SplitIn(BaseModel):
    """Group and time columns always come from the declared roles."""

    method: Literal["kfold", "stratified", "group", "stratified_group", "time"] | None = None
    n_splits: int = Field(default=5, ge=2, le=10)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class ExperimentIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    candidates: list[CandidateIn] = Field(min_length=1, max_length=20)
    models: list[Literal["dummy", "linear", "hgb", "kmeans"]] | None = None
    task: Literal["clustering"] | None = None  # otherwise the target role's task type
    n_clusters: int | None = Field(default=None, ge=2, le=20)
    split: SplitIn = Field(default_factory=SplitIn)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    recommendation_set_id: uuid.UUID | None = None


class CandidateOut(BaseModel):
    id: uuid.UUID
    key: str
    label: str
    is_baseline: bool
    steps: list[dict[str, Any]]
    spec_hash: str
    max_risk: str
    has_destructive: bool
    stages: list[str]
    recommendation_ids: list[str]
    result: dict[str, Any] | None

    @classmethod
    def build(cls, c: ExperimentCandidate, r: ExperimentResult | None) -> "CandidateOut":
        result = None
        if r is not None:
            result = {
                "status": r.status,
                "comparable": r.comparable,
                "comparability_reason": r.comparability_reason,
                "metrics": r.metrics,
                "folds": r.fold_results,
                "data_quality": r.data_quality,
                "selection_stability": r.selection_stability,
                "failures": r.failures,
                **r.details,
            }
        return cls(
            id=c.id,
            key=c.key,
            label=c.label,
            is_baseline=c.is_baseline,
            steps=c.steps,
            spec_hash=c.spec_hash,
            max_risk=c.max_risk,
            has_destructive=c.has_destructive,
            stages=c.stages,
            recommendation_ids=c.recommendation_ids,
            result=result,
        )


class SplitOut(BaseModel):
    id: uuid.UUID
    method: str
    n_splits: int
    seed: int
    target_column: str | None
    group_column: str | None
    time_column: str | None
    assignment_sha256: str
    cohort: dict[str, Any]
    fold_stats: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    created_at: datetime

    @classmethod
    def build(cls, s: SplitDefinition) -> "SplitOut":
        return cls(
            id=s.id,
            method=s.method,
            n_splits=s.n_splits,
            seed=s.seed,
            target_column=s.target_column,
            group_column=s.group_column,
            time_column=s.time_column,
            assignment_sha256=s.assignment_sha256.hex(),
            cohort=s.cohort,
            fold_stats=s.fold_stats,
            warnings=s.warnings,
            created_at=s.created_at,
        )


class ExperimentSummaryOut(BaseModel):
    id: uuid.UUID
    version_id: uuid.UUID
    name: str
    task_type: str
    target_column: str | None
    models: list[str]
    seed: int
    config_hash: str
    rerun_of_id: uuid.UUID | None
    created_at: datetime
    completed_at: datetime | None
    job: JobOut | None

    @classmethod
    def build(cls, e: Experiment, job: Job | None) -> "ExperimentSummaryOut":
        return cls(
            id=e.id,
            version_id=e.dataset_version_id,
            name=e.name,
            task_type=e.task_type,
            target_column=e.target_column,
            models=e.models,
            seed=e.seed,
            config_hash=e.config_hash.hex(),
            rerun_of_id=e.rerun_of_id,
            created_at=e.created_at,
            completed_at=e.completed_at,
            job=JobOut.from_job(job) if job else None,
        )


class ExperimentOut(ExperimentSummaryOut):
    config: dict[str, Any]
    recommendation_set_id: uuid.UUID | None
    split: SplitOut | None
    summary: dict[str, Any] | None
    library_versions: dict[str, Any] | None
    candidates: list[CandidateOut]


class ExperimentListOut(BaseModel):
    items: list[ExperimentSummaryOut]


class ExperimentAcceptedOut(BaseModel):
    experiment: ExperimentSummaryOut
    job: JobOut

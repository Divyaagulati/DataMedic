import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from datamedic.analyses.schemas import FindingOut
from datamedic.db.models import AnalysisRun, FindingRow, Job, ModelEvaluation, TrainingRun
from datamedic.engine.ml import spec as ml
from datamedic.experiments.schemas import SplitOut
from datamedic.jobs.schemas import JobOut

ModelKey = Literal["dummy", "linear", "random_forest", "hgb"]


class TrainingStepIn(BaseModel):
    op: str = Field(min_length=1, max_length=64)
    columns: list[str] = Field(default_factory=list, max_length=500)
    params: dict[str, Any] = Field(default_factory=dict)


class TrainingSplitIn(BaseModel):
    """Group and time columns always come from the declared roles."""

    method: Literal["kfold", "stratified", "group", "stratified_group", "time"] | None = None
    n_splits: int = Field(default=5, ge=2, le=10)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class TuningIn(BaseModel):
    mode: Literal["nested", "defaults"] = "nested"


class TrainingRunIn(BaseModel):
    name: str | None = Field(default=None, max_length=200)
    models: list[ModelKey] | None = Field(default=None, max_length=4)
    tuning: TuningIn = Field(default_factory=TuningIn)
    split: TrainingSplitIn = Field(default_factory=TrainingSplitIn)
    # None: the version's stored modelling specification. Steps run inside training folds.
    preprocess_steps: list[TrainingStepIn] | None = Field(default=None, max_length=50)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    allow_exploratory: bool = False


class TrainingPlanOut(BaseModel):
    valid: bool
    errors: list[dict[str, Any]]
    warnings: list[dict[str, Any]]
    task: str | None
    target: str | None
    features: list[str]
    excluded_columns: list[dict[str, Any]]
    datetime_columns: list[str]
    sensitive_columns: list[str]
    models: list[str]
    tuning: dict[str, Any]
    estimated_fits: int
    fit_limit: int
    seed: int
    split: dict[str, Any] | None
    eligibility: dict[str, Any]
    evaluation_mode: str
    evaluation_version_id: uuid.UUID
    evaluation_version_number: int
    preprocess_steps: list[dict[str, Any]]
    preprocess_source: str
    spec_hash: str | None
    allow_exploratory: bool


class ModelCatalogueOut(BaseModel):
    models: list[dict[str, Any]]
    tasks: list[str]
    tuning_modes: list[str]
    inner_folds: int
    metrics: dict[str, Any]
    task_metrics: dict[str, list[str]]
    primary_metric: dict[str, str]
    selection_metric: dict[str, str]
    metric_conventions: list[str]
    audit_checks: list[dict[str, Any]]
    evaluation_status_rules: list[str]
    evaluation_status_limitation: str


class AuditBriefOut(BaseModel):
    id: uuid.UUID
    status: str
    created_at: datetime

    @classmethod
    def build(cls, a: AnalysisRun) -> "AuditBriefOut":
        return cls(
            id=a.id, status=str(a.summary["evaluation_status"]["status"]), created_at=a.created_at
        )


HEADLINE_NOTE = (
    "Highest observed mean of the primary metric over the outer test folds among the compared "
    "models (the dummy baseline excluded). Choosing the highest of several scores on the same "
    "folds is optimistic; this is not a declared winner."
)
EXPLORATORY_NOTE = (
    "Exploratory: evaluated on a version whose committed steps learned from every row, "
    "including test rows. The score is likely optimistic and is not a leakage-controlled "
    "estimate."
)
ACTIVE_JOB = ("queued", "running", "cancel_requested")


def headline_of(
    r: TrainingRun, job: Job | None, evals: list[ModelEvaluation] | None
) -> dict[str, Any]:
    """The run's headline for lists: the highest observed fold mean of the primary metric,
    respecting the metric's direction, or an explicit state with a reason. Never a winner."""
    primary = str((r.summary or {}).get("primary_metric") or ml.PRIMARY.get(r.task_type, ""))
    info = ml.METRICS.get(primary)
    higher = bool(info[1]) if info else True
    out: dict[str, Any] = {
        "state": "unavailable",
        "metric": primary,
        "label": info[0] if info else primary,
        "higher_is_better": higher,
        "model": None,
        "mean": None,
        "sd": None,
        "baseline_mean": None,
        "exploratory": r.evaluation_mode == "exploratory",
        "note": HEADLINE_NOTE,
        "exploratory_note": EXPLORATORY_NOTE if r.evaluation_mode == "exploratory" else None,
        "reason": None,
        "n_folds": None,
        "total_folds": None,
    }
    if r.completed_at is None or r.summary is None:
        status = job.status if job is not None else None
        if status in ACTIVE_JOB:
            out.update(state="pending", reason="The run has not finished.")
        elif status == "cancelled":
            out.update(state="cancelled", reason="The run was cancelled; no results were stored.")
        else:
            out.update(state="failed", reason="The run did not complete; no results were stored.")
        return out
    evals = evals or []
    for e in evals:
        if e.model_key == "dummy" and e.comparable:
            out["baseline_mean"] = (e.metrics.get("folds", {}).get(primary) or {}).get("mean")
    trained = [e for e in evals if e.model_key != "dummy"]
    compared = [e for e in trained if e.comparable and e.status == "completed"]
    scored = [
        (e, (e.metrics.get("folds", {}).get(primary) or {}))
        for e in compared
        if (e.metrics.get("folds", {}).get(primary) or {}).get("mean") is not None
    ]
    if not scored:
        if not trained:
            reason = "Only the dummy baseline was trained."
        elif not compared:
            reason = "No model other than the dummy completed every fold."
        else:
            reason = f"{out['label']} could not be computed for any compared model."
        out["reason"] = reason
        return out
    order = {m: i for i, m in enumerate(r.models)}
    scored.sort(
        key=lambda t: (
            -float(t[1]["mean"]) if higher else float(t[1]["mean"]),
            order.get(t[0].model_key, 99),
        )
    )
    best, stats = scored[0]
    out.update(
        state="available",
        model=best.model_key,
        mean=stats["mean"],
        sd=stats.get("sd"),
        n_folds=stats.get("n_folds"),
        total_folds=len(stats.get("values") or []),
    )
    return out


class TrainingRunSummaryOut(BaseModel):
    id: uuid.UUID
    dataset_id: uuid.UUID
    version_id: uuid.UUID
    evaluation_version_id: uuid.UUID
    name: str
    task_type: str
    target_column: str
    evaluation_mode: str
    models: list[str]
    seed: int
    config_hash: str
    rerun_of_id: uuid.UUID | None
    created_at: datetime
    completed_at: datetime | None
    headline: dict[str, Any] | None
    audit: AuditBriefOut | None
    job: JobOut | None

    @classmethod
    def build(
        cls,
        r: TrainingRun,
        job: Job | None,
        audit: AnalysisRun | None = None,
        evals: list[ModelEvaluation] | None = None,
    ) -> "TrainingRunSummaryOut":
        headline = headline_of(r, job, evals)
        return cls(
            id=r.id,
            dataset_id=r.dataset_id,
            version_id=r.dataset_version_id,
            evaluation_version_id=r.evaluation_version_id,
            name=r.name,
            task_type=r.task_type,
            target_column=r.target_column,
            evaluation_mode=r.evaluation_mode,
            models=r.models,
            seed=r.seed,
            config_hash=r.config_hash.hex(),
            rerun_of_id=r.rerun_of_id,
            created_at=r.created_at,
            completed_at=r.completed_at,
            headline=headline,
            audit=AuditBriefOut.build(audit) if audit is not None else None,
            job=JobOut.from_job(job) if job else None,
        )


class ModelEvaluationOut(BaseModel):
    model: str
    status: str
    comparable: bool
    comparability_reason: str | None
    metrics: dict[str, Any]
    details: dict[str, Any]
    fold_results: list[dict[str, Any]]
    selection: dict[str, Any]
    importance: dict[str, Any]
    failures: list[dict[str, Any]]

    @classmethod
    def build(cls, e: ModelEvaluation) -> "ModelEvaluationOut":
        return cls(
            model=e.model_key,
            status=e.status,
            comparable=e.comparable,
            comparability_reason=e.comparability_reason,
            metrics=e.metrics,
            details=e.details,
            fold_results=e.fold_results,
            selection=e.selection,
            importance=e.importance,
            failures=e.failures,
        )


class TrainingRunOut(TrainingRunSummaryOut):
    config: dict[str, Any]
    eligibility: dict[str, Any]
    replay: dict[str, Any] | None
    split_config: dict[str, Any]
    split: SplitOut | None
    tuning: dict[str, Any]
    preprocess_steps: list[dict[str, Any]]
    spec_hash: str
    source_content_hash: str
    summary: dict[str, Any] | None
    library_versions: dict[str, Any] | None
    engine_version: str
    oof_artifact_sha256: str | None
    evaluations: list[ModelEvaluationOut]
    # Sensitive-column protection applied when serving (training.privacy).
    privacy: dict[str, Any]


class TrainingListOut(BaseModel):
    items: list[TrainingRunSummaryOut]


class TrainingAcceptedOut(BaseModel):
    training_run: TrainingRunSummaryOut
    job: JobOut


class AuditOut(BaseModel):
    training_run_id: uuid.UUID
    evaluation_mode: str
    run: dict[str, Any] | None
    findings: list[FindingOut]
    stale: bool
    job: JobOut | None

    @classmethod
    def build(
        cls,
        run: TrainingRun,
        audit: AnalysisRun | None,
        findings: list[FindingRow],
        job: Job | None,
        stale: bool,
    ) -> "AuditOut":
        return cls(
            training_run_id=run.id,
            evaluation_mode=run.evaluation_mode,
            run=None
            if audit is None
            else {
                "id": str(audit.id),
                "created_at": audit.created_at.isoformat(),
                "summary": audit.summary,
                "config": audit.config,
                "engine_version": audit.engine_version,
                "library_versions": audit.library_versions,
            },
            findings=[FindingOut.build(f) for f in findings],
            stale=stale,
            job=JobOut.from_job(job) if job else None,
        )


class AuditAcceptedOut(BaseModel):
    job: JobOut

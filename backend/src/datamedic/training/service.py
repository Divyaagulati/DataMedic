"""Supervised training runs and reliability audits: configuration, eligibility and queueing
(API side; standard library and SQLAlchemy only, the API never loads pandas).

The API resolves the task and target from the confirmed target role, checks feature
eligibility, walks the version lineage to decide the evaluation mode (direct, replay from the
eligible ancestor, or blocked), resolves the frozen split from the roles, validates the
in-fold preprocessing specification, estimates the fit budget and stores the complete
reproducible configuration. The worker re-checks everything before reading data.
Training never writes dataset versions. Every lookup is scoped to the owner (and RLS).
"""

import hashlib
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from datamedic.analyses.semantics import semantics_hash
from datamedic.analyses.service import current_semantics_hash
from datamedic.config import Settings
from datamedic.db.models import (
    AnalysisRun,
    Dataset,
    DatasetVersion,
    FindingRow,
    Job,
    ModelEvaluation,
    SplitDefinition,
    TrainingRun,
    TransformationRun,
)
from datamedic.engine import ENGINE_VERSION, ROW_ID
from datamedic.engine.ml import audit_spec
from datamedic.engine.ml import spec as ml
from datamedic.engine.ml.eligibility import (
    BLOCKED,
    DIRECT,
    EXPLORATORY,
    REPLAY,
    Eligibility,
    Lineage,
    LineageRun,
    assess,
)
from datamedic.engine.preprocess.catalog import get_op
from datamedic.engine.preprocess.spec import Step, canonical_json
from datamedic.engine.preprocess.state import ValidationContext
from datamedic.engine.preprocess.validate import ValidationResult, validate_steps
from datamedic.errors import (
    DatasetNotReady,
    DomainError,
    EvaluationBlocked,
    NotFound,
    QuotaExceeded,
    ValidationFailed,
)
from datamedic.experiments.service import resolve_split
from datamedic.jobs.service import create_job
from datamedic.preprocessing.context import role_specs, roles_snapshot, validation_context
from datamedic.training import privacy
from datamedic.versions.service import get_owned_version

MAX_CHAIN = 1_000
NUMERIC_DTYPES = ("integer", "float", "boolean")


def _fail(message: str, pointer: str, code: str) -> DomainError:
    return ValidationFailed(
        message, errors=[{"pointer": pointer, "code": code, "message": message}]
    )


# ---- Lineage ------------------------------------------------------------------------------------


def lineage_of(db: Session, version: DatasetVersion) -> tuple[Lineage, list[dict[str, Any]]]:
    """Lineage facts for eligibility, and the committed steps walked (for the audit)."""
    vid = str(version.id)
    if not version.fitted_on_full_data:
        return (
            Lineage(vid, False, vid, version.schema, int(version.n_rows), int(version.n_rows)),
            [],
        )
    runs: list[LineageRun] = []
    walked: list[dict[str, Any]] = []
    current: DatasetVersion | None = version
    ancestor: DatasetVersion | None = None
    problem = None
    for _ in range(MAX_CHAIN):
        assert current is not None
        if current.kind == "revert":
            current = db.get(DatasetVersion, current.restored_from_version_id)
            if current is None:
                problem = "A reverted version's target is missing."
                break
            if not current.fitted_on_full_data:
                ancestor = current
                break
            continue
        if current.kind == "transform":
            run = db.get(TransformationRun, current.transformation_run_id)
            source = db.get(DatasetVersion, run.source_version_id) if run else None
            if run is None or source is None:
                problem = "A committed transformation or its source version is missing."
                break
            snap = run.spec_snapshot
            runs.append(
                LineageRun(
                    run_id=str(run.id),
                    result_version_id=str(current.id),
                    source_version_id=str(source.id),
                    source_schema=source.schema,
                    source_n_rows=int(source.n_rows),
                    data_steps=snap.get("data_steps", []),
                    data_step_hashes=snap.get("data_step_hashes", []),
                    commit_roles=snap.get("roles", {"roles": {}}),
                    seed=int((run.seeds or {}).get("seed", 0)),
                )
            )
            for s in snap.get("data_steps", []):
                spec = get_op(str(s["op"]))
                walked.append(
                    {
                        "op": s["op"],
                        "needs_target": bool(spec and spec.needs_target),
                        "kind": spec.kind if spec else None,
                    }
                )
            current = source
            if not current.fitted_on_full_data:
                ancestor = current
                break
            continue
        problem = "The lineage reaches a version that cannot be replayed."
        break
    else:
        problem = "The version lineage is too long."
    runs.reverse()
    if ancestor is None:
        return Lineage(vid, True, None, problem=problem or "No eligible ancestor."), walked
    return (
        Lineage(
            vid,
            True,
            str(ancestor.id),
            ancestor.schema,
            int(ancestor.n_rows),
            int(version.n_rows),
            runs,
        ),
        walked,
    )


# ---- Resolution ---------------------------------------------------------------------------------


def _task_errors(ctx: ValidationContext, version: DatasetVersion) -> list[dict[str, str]]:
    errors: list[dict[str, str]] = []
    if ctx.target is None or ctx.task_type not in ml.TASKS:
        return [
            {
                "pointer": "/target",
                "code": "TARGET_REQUIRED",
                "message": "Declare exactly one target column with a confirmed task type "
                "(binary, multiclass or regression) in the roles.",
            }
        ]
    column = next((c for c in version.schema if c["name"] == ctx.target), None)
    if column is None:
        return [
            {
                "pointer": "/target",
                "code": "TARGET_REQUIRED",
                "message": "The target column is missing.",
            }
        ]
    n_unique = column.get("n_unique")
    dtype = str(column.get("dtype", ""))
    task = ctx.task_type
    mismatch = None
    if n_unique is not None:
        if task == "binary" and n_unique != 2:
            mismatch = f"A binary target needs exactly 2 classes; this one has {n_unique}."
        elif task == "multiclass" and not 3 <= n_unique <= ml.MAX_CLASSES:
            mismatch = (
                f"A multiclass target needs 3 to {ml.MAX_CLASSES} classes; this one has {n_unique}."
            )
        elif task == "regression" and n_unique < ml.MIN_REGRESSION_DISTINCT:
            mismatch = (
                f"A regression target needs at least {ml.MIN_REGRESSION_DISTINCT} distinct "
                f"values; this one has {n_unique}."
            )
    if task == "regression" and dtype not in ("integer", "float"):
        mismatch = "A regression target must be numeric."
    if mismatch:
        errors.append({"pointer": "/target", "code": "TASK_TARGET_MISMATCH", "message": mismatch})
    n_rows = int(version.n_rows) - int(column.get("n_missing") or 0)
    if n_rows < ml.MIN_COHORT_ROWS:
        errors.append(
            {
                "pointer": "/target",
                "code": "TOO_FEW_ROWS",
                "message": f"At least {ml.MIN_COHORT_ROWS} rows with a target value are needed; "
                f"there are {n_rows}.",
            }
        )
    return errors


def features_of(ctx: ValidationContext) -> list[str]:
    return [
        c.name
        for c in ctx.columns
        if c.name != ROW_ID and c.name not in ctx.protected and c.name != ctx.target
    ]


def _steps(raw: list[dict[str, Any]]) -> list[Step]:
    return [
        Step(str(s["op"]), tuple(s.get("columns") or ()), dict(s.get("params") or {})) for s in raw
    ]


def validate_preprocess(raw: list[dict[str, Any]], ctx: ValidationContext) -> ValidationResult:
    return validate_steps(_steps(raw), ctx)


def resolve(
    db: Session,
    settings: Settings,
    version: DatasetVersion,
    body: dict[str, Any],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    """Everything needed to queue (or to explain why not), without raising for configuration
    problems: ``errors`` lists them with pointers."""
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    ctx = validation_context(version, snapshot)
    errors += _task_errors(ctx, version)
    task = ctx.task_type
    features = features_of(ctx)
    if not features:
        errors.append(
            {
                "pointer": "/features",
                "code": "NO_FEATURES",
                "message": "No column is available as a feature.",
            }
        )
    if len(features) > settings.training_max_features:
        errors.append(
            {
                "pointer": "/features",
                "code": "LIMIT_EXCEEDED",
                "message": f"At most {settings.training_max_features} feature columns.",
            }
        )
    excluded = sorted(
        [{"column": c, "roles": list(r)} for c, r in ctx.protected.items()],
        key=lambda d: str(d["column"]),
    )
    models = list(dict.fromkeys(body.get("models") or ml.MODEL_KEYS))
    bad = [m for m in models if m not in ml.MODELS]
    if bad:
        errors.append(
            {
                "pointer": "/models",
                "code": "MODEL_INVALID",
                "message": f"Models: {', '.join(ml.MODEL_KEYS)}.",
            }
        )
        models = [m for m in models if m in ml.MODELS]
    tuning = body.get("tuning") or {}
    mode = tuning.get("mode") or "nested"
    if mode not in ml.TUNING_MODES:
        errors.append(
            {
                "pointer": "/tuning/mode",
                "code": "TUNING_INVALID",
                "message": "Use 'nested' or 'defaults'.",
            }
        )
        mode = "defaults"
    seed = int(body.get("seed", 0))

    lineage, walked = lineage_of(db, version)
    eligibility = assess(lineage, snapshot, role_specs())
    allow = bool(body.get("allow_exploratory"))
    if eligibility.mode == BLOCKED:
        evaluation_mode = EXPLORATORY if allow else BLOCKED
        evaluation_version = version
    elif eligibility.mode == REPLAY:
        evaluation_mode = REPLAY
        found = db.get(DatasetVersion, uuid.UUID(eligibility.evaluation_version_id))
        assert found is not None
        evaluation_version = found
    else:
        evaluation_mode = DIRECT
        evaluation_version = version
    if allow and eligibility.mode != BLOCKED:
        warnings.append(
            {
                "code": "EXPLORATORY_NOT_NEEDED",
                "message": "Leakage-controlled evaluation is possible, so the run is not "
                "exploratory.",
            }
        )

    split: dict[str, Any] | None = None
    if task in ml.TASKS:
        eval_ctx = validation_context(evaluation_version, snapshot)
        try:
            split = resolve_split(body.get("split") or {}, snapshot, eval_ctx, task)
        except ValidationFailed as exc:
            errors += exc.errors or []
        if split is not None and not 2 <= split["n_splits"] <= 10:
            errors.append(
                {
                    "pointer": "/split/n_splits",
                    "code": "SPLIT_INVALID",
                    "message": "Use 2 to 10 folds.",
                }
            )
        if split is not None:
            warnings += split.get("warnings", [])

    raw_steps = body.get("preprocess_steps")
    source = "request"
    if raw_steps is None:
        raw_steps = [
            {"op": s["op"], "columns": s.get("columns", []), "params": s.get("params", {})}
            for s in (version.model_pipeline_spec or [])
        ]
        source = "version_model_pipeline_spec"
    pre = validate_preprocess(raw_steps, ctx)
    for e in pre.errors:
        pointer = "/preprocess_steps" + (f"/{e.step}" if e.step is not None else "")
        errors.append({"pointer": pointer, "code": e.code, "message": e.message})
    steps_out = []
    if pre.valid:
        for v in pre.steps:
            steps_out.append(
                {
                    **v.step.as_dict(),
                    "op_version": v.spec.version,
                    "step_hash": v.step_hash,
                    "stage": v.spec.stage,
                    "kind": v.spec.kind,
                    "risk": v.spec.risk,
                    "needs_target": v.spec.needs_target,
                }
            )

    n_splits = split["n_splits"] if split else 5
    planned = mode
    fallback = None
    fits = ml.estimate_fits(n_splits, models, task or "binary", mode)
    if mode == "nested" and fits > settings.training_max_fits:
        planned = "defaults"
        fallback = (
            f"Nested selection needs about {fits} model fits, above the limit of "
            f"{settings.training_max_fits}; default settings will be used."
        )
        fits = ml.estimate_fits(n_splits, models, task or "binary", planned)
    if fits > settings.training_max_fits:
        errors.append(
            {
                "pointer": "/models",
                "code": "QUOTA_EXCEEDED",
                "message": f"This run needs {fits} model fits; the limit is "
                f"{settings.training_max_fits}. Use fewer folds or models.",
            }
        )
    cohort_rows = int(evaluation_version.n_rows)
    if (
        cohort_rows > settings.training_max_rows
        or cohort_rows * max(int(evaluation_version.n_cols), 1) > settings.training_max_cells
    ):
        errors.append(
            {
                "pointer": "/version",
                "code": "QUOTA_EXCEEDED",
                "message": f"Training is limited to {settings.training_max_rows:,} rows and "
                f"{settings.training_max_cells:,} cells.",
            }
        )
    if fallback:
        warnings.append({"code": "TUNING_FALLBACK", "message": fallback})
    if task in ml.TASKS and cohort_rows < ml.MIN_ROWS_FOR_NESTED and planned == "nested":
        warnings.append(
            {
                "code": "TUNING_FALLBACK",
                "message": f"Fewer than {ml.MIN_ROWS_FOR_NESTED} rows: default settings will be "
                "used instead of nested selection.",
            }
        )
    datetime_columns = [
        c.name for c in ctx.columns if c.dtype.startswith("datetime") and c.name in features
    ]
    # Sensitivity: the snapshot's sensitive columns, columns sensitive now on any version of the
    # dataset, and columns committed steps derived from either (training.privacy).
    sensitive = set(ctx.sensitive) | privacy.currently_sensitive(
        db, version.owner_id, version.dataset_id
    )
    sensitive |= privacy.derived_from(db, version.owner_id, version.dataset_id, sensitive)
    from_protected = sorted(
        privacy.derived_from(db, version.owner_id, version.dataset_id, ctx.protected)
        & set(features)
    )
    if from_protected:
        warnings.append(
            {
                "code": "DERIVED_FROM_PROTECTED",
                "message": f"{len(from_protected)} feature column(s) were derived by committed "
                "steps from a column that now has a non-feature role (identifier, group, time "
                "or excluded): " + ", ".join(from_protected) + ". They are model inputs; give "
                "them the 'exclude' role if they should not be.",
            }
        )
    if datetime_columns:
        warnings.append(
            {
                "code": "DATETIME_FEATURES_IGNORED",
                "message": "Date and time columns are not model inputs.",
            }
        )
    return {
        "valid": not errors and evaluation_mode != BLOCKED,
        "errors": errors,
        "warnings": warnings,
        "task": task,
        "target": ctx.target,
        "features": features,
        "excluded_columns": excluded,
        "datetime_columns": datetime_columns,
        "sensitive_columns": sorted(sensitive & {c.name for c in ctx.columns}),
        "models": models,
        "tuning": {
            "requested": mode,
            "planned": planned,
            "fallback_reason": fallback,
            "inner_folds": ml.INNER_FOLDS,
            "grids": {m: list(ml.configs(m, task or "binary", "nested")) for m in models},
        },
        "estimated_fits": fits,
        "fit_limit": settings.training_max_fits,
        "seed": seed,
        "split": split,
        "eligibility": eligibility.as_dict(),
        "evaluation_mode": evaluation_mode,
        "evaluation_version_id": str(evaluation_version.id),
        "evaluation_version_number": evaluation_version.version_number,
        "preprocess_steps": steps_out,
        "preprocess_source": source,
        "spec_hash": pre.spec_hash,
        "lineage_steps": walked,
        "cohort_rows_upper_bound": cohort_rows,
        "allow_exploratory": allow,
    }


def plan(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    version_id: uuid.UUID,
    body: dict[str, Any],
) -> dict[str, Any]:
    """Advisory: the same resolution as ``create``, with errors listed rather than raised.
    The worker re-validates everything before reading data."""
    version, _ = get_owned_version(db, owner_id, version_id)
    snapshot = roles_snapshot(db, owner_id, version.id)
    return resolve(db, settings, version, body, snapshot)


# ---- Create, rerun ------------------------------------------------------------------------------


def config_hash(config: dict[str, Any]) -> bytes:
    return hashlib.sha256(canonical_json(config).encode()).digest()


def create(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    version_id: uuid.UUID,
    body: dict[str, Any],
    idempotency_key: str | None,
    rerun_of: TrainingRun | None = None,
) -> tuple[TrainingRun, Job, bool]:
    version, ds = get_owned_version(db, owner_id, version_id)
    if idempotency_key:
        prior = db.scalar(
            select(Job).where(Job.owner_id == owner_id, Job.idempotency_key == idempotency_key)
        )
        if prior is not None:
            run = db.scalar(
                select(TrainingRun).where(
                    TrainingRun.job_id == prior.id, TrainingRun.owner_id == owner_id
                )
            )
            if run is None:
                raise _fail(
                    "This Idempotency-Key was used for another request.",
                    "/headers/Idempotency-Key",
                    "IDEMPOTENCY_KEY_REUSED",
                )
            return run, prior, False
    if ds.status != "ready":
        raise DatasetNotReady("The dataset is not ready.")
    db.execute(select(Dataset.id).where(Dataset.id == ds.id).with_for_update())
    count = db.scalar(
        select(func.count()).select_from(TrainingRun).where(TrainingRun.dataset_id == ds.id)
    )
    if int(count or 0) >= settings.training_max_runs_per_dataset:
        raise QuotaExceeded(
            f"A dataset can have at most {settings.training_max_runs_per_dataset} training runs."
        )
    snapshot = rerun_of.config["roles"] if rerun_of else roles_snapshot(db, owner_id, version.id)
    resolved = resolve(db, settings, version, body, snapshot)
    if resolved["errors"]:
        quota = [e for e in resolved["errors"] if e["code"] == "QUOTA_EXCEEDED"]
        if quota and len(quota) == len(resolved["errors"]):
            raise QuotaExceeded(quota[0]["message"])
        raise ValidationFailed(
            "The training configuration is not valid.", errors=resolved["errors"]
        )
    if resolved["evaluation_mode"] == BLOCKED:
        reasons = resolved["eligibility"]["reasons"]
        raise EvaluationBlocked(
            "Leakage-controlled evaluation of this version is not possible: "
            + " ".join(r["message"] for r in reasons)
            + " Choose an exploratory run explicitly to evaluate it as stored.",
            errors=[
                {"pointer": "/version", "code": r["code"], "message": r["message"]} for r in reasons
            ],
        )
    assert resolved["split"] is not None and resolved["spec_hash"] is not None
    evaluation_version = db.get(DatasetVersion, uuid.UUID(resolved["evaluation_version_id"]))
    assert evaluation_version is not None
    split = {k: v for k, v in resolved["split"].items() if k != "warnings"}
    config = {
        "version_id": str(version.id),
        "content_hash": version.content_hash.hex(),
        "evaluation_version_id": str(evaluation_version.id),
        "evaluation_content_hash": evaluation_version.content_hash.hex(),
        "evaluation_mode": resolved["evaluation_mode"],
        "eligibility": resolved["eligibility"],
        "task": resolved["task"],
        "target": resolved["target"],
        "features": resolved["features"],
        "datetime_columns": resolved["datetime_columns"],
        "sensitive_columns": resolved["sensitive_columns"],
        "models": resolved["models"],
        "tuning": resolved["tuning"],
        "seed": resolved["seed"],
        "split": split,
        "split_warnings": resolved["split"].get("warnings", []),
        "roles": snapshot,
        "preprocess_steps": resolved["preprocess_steps"],
        "preprocess_source": resolved["preprocess_source"],
        "spec_hash": resolved["spec_hash"],
        "lineage_steps": resolved["lineage_steps"],
        "limits": {
            "permutation_rows_per_fold": settings.permutation_max_rows_per_fold,
            "permutation_columns": settings.permutation_max_columns,
            "permutation_repeats": settings.permutation_repeats,
            "bootstrap_resamples": settings.bootstrap_resamples,
            "audit_max_rows": settings.audit_max_rows,
            "audit_single_feature_max_columns": settings.audit_single_feature_max_columns,
            "memory_budget_mb": settings.training_memory_budget_mb,
        },
        "engine_version": ENGINE_VERSION,
        "ml_engine_version": ml.ENGINE_ML_VERSION,
    }
    job = create_job(db, settings, owner_id, "train", ds.id, {"version_id": str(version.id)})
    job.idempotency_key = idempotency_key
    name = str(body.get("name") or "").strip() or f"Training on version {version.version_number}"
    run = TrainingRun(
        owner_id=owner_id,
        dataset_id=ds.id,
        dataset_version_id=version.id,
        evaluation_version_id=evaluation_version.id,
        job_id=job.id,
        name=name[:200],
        task_type=resolved["task"],
        target_column=resolved["target"],
        evaluation_mode=resolved["evaluation_mode"],
        eligibility=resolved["eligibility"],
        split_config=split,
        models=resolved["models"],
        tuning=resolved["tuning"],
        preprocess_steps=resolved["preprocess_steps"],
        spec_hash=resolved["spec_hash"],
        config=config,
        config_hash=config_hash(config),
        source_content_hash=evaluation_version.content_hash,
        seed=resolved["seed"],
        rerun_of_id=rerun_of.id if rerun_of else None,
        engine_version=ENGINE_VERSION,
        created_by=owner_id,
    )
    db.add(run)
    db.flush()
    job.params = {"version_id": str(version.id), "training_run_id": str(run.id)}
    db.commit()
    return run, job, True


def rerun(
    db: Session, settings: Settings, owner_id: uuid.UUID, run_id: uuid.UUID, key: str | None
) -> tuple[TrainingRun, Job, bool]:
    """A new run with exactly the stored configuration (roles snapshot, split, seeds, models,
    tuning and preprocessing). An exploratory run stays exploratory."""
    original = get_owned(db, owner_id, run_id)
    cfg = original.config
    body = {
        "name": f"Re-run of {original.name}"[:200],
        "models": cfg["models"],
        "tuning": {"mode": cfg["tuning"]["requested"]},
        "seed": cfg["seed"],
        "split": {k: cfg["split"][k] for k in ("method", "n_splits", "seed")},
        "preprocess_steps": [
            {"op": s["op"], "columns": s["columns"], "params": s["params"]}
            for s in cfg["preprocess_steps"]
        ],
        "allow_exploratory": cfg["evaluation_mode"] == EXPLORATORY,
    }
    return create(db, settings, owner_id, original.dataset_version_id, body, key, rerun_of=original)


# ---- Reads --------------------------------------------------------------------------------------


def get_owned(db: Session, owner_id: uuid.UUID, run_id: uuid.UUID) -> TrainingRun:
    run = db.scalar(
        select(TrainingRun).where(TrainingRun.id == run_id, TrainingRun.owner_id == owner_id)
    )
    if run is None:
        raise NotFound()
    get_owned_version(db, owner_id, run.dataset_version_id)  # dataset not deleted
    return run


def detail(
    db: Session, owner_id: uuid.UUID, run_id: uuid.UUID
) -> tuple[TrainingRun, Job | None, list[ModelEvaluation], SplitDefinition | None]:
    run = get_owned(db, owner_id, run_id)
    job = db.scalar(select(Job).where(Job.id == run.job_id, Job.owner_id == owner_id))
    evals = list(
        db.scalars(
            select(ModelEvaluation)
            .where(ModelEvaluation.owner_id == owner_id, ModelEvaluation.training_run_id == run.id)
            .order_by(ModelEvaluation.created_at, ModelEvaluation.model_key)
        )
    )
    order = {m: i for i, m in enumerate(run.models)}
    evals.sort(key=lambda e: order.get(e.model_key, 99))
    split = (
        db.scalar(
            select(SplitDefinition).where(
                SplitDefinition.id == run.split_definition_id, SplitDefinition.owner_id == owner_id
            )
        )
        if run.split_definition_id
        else None
    )
    return run, job, evals, split


def list_for_dataset(
    db: Session, owner_id: uuid.UUID, dataset_id: uuid.UUID
) -> list[tuple[TrainingRun, Job | None, AnalysisRun | None, list[ModelEvaluation]]]:
    """Runs with their job, latest audit and model evaluations (for the headline)."""
    from datamedic.datasets.service import get_owned as get_dataset

    ds = get_dataset(db, owner_id, dataset_id)
    rows = db.execute(
        select(TrainingRun, Job)
        .join(Job, Job.id == TrainingRun.job_id, isouter=True)
        .where(TrainingRun.owner_id == owner_id, TrainingRun.dataset_id == ds.id)
        .order_by(TrainingRun.created_at.desc())
        .limit(100)
    ).all()
    run_ids = [run.id for run, _ in rows]
    by_run: dict[uuid.UUID, list[ModelEvaluation]] = {}
    if run_ids:
        for e in db.scalars(
            select(ModelEvaluation).where(
                ModelEvaluation.owner_id == owner_id,
                ModelEvaluation.training_run_id.in_(run_ids),
            )
        ):
            by_run.setdefault(e.training_run_id, []).append(e)
    out: list[tuple[TrainingRun, Job | None, AnalysisRun | None, list[ModelEvaluation]]] = []
    for run, job in rows:
        out.append((run, job, latest_audit_run(db, owner_id, run.id), by_run.get(run.id, [])))
    return out


def latest_audit_run(db: Session, owner_id: uuid.UUID, run_id: uuid.UUID) -> AnalysisRun | None:
    return db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.owner_id == owner_id,
            AnalysisRun.training_run_id == run_id,
            AnalysisRun.kind == "reliability_audit",
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )


def audit(
    db: Session, owner_id: uuid.UUID, run_id: uuid.UUID
) -> tuple[TrainingRun, AnalysisRun | None, list[FindingRow], Job | None, bool]:
    run = get_owned(db, owner_id, run_id)
    latest = latest_audit_run(db, owner_id, run.id)
    findings = (
        list(
            db.scalars(
                select(FindingRow)
                .where(FindingRow.owner_id == owner_id, FindingRow.analysis_run_id == latest.id)
                .order_by(FindingRow.position)
            )
        )
        if latest
        else []
    )
    job = db.scalar(
        select(Job)
        .where(
            Job.owner_id == owner_id,
            Job.type == "reliability_audit",
            Job.params["training_run_id"].astext == str(run.id),
        )
        .order_by(Job.queued_at.desc())
        .limit(1)
    )
    stale = False
    if latest is not None:
        ds = db.get(Dataset, run.dataset_id)
        assert ds is not None
        stale = latest.semantics_hash != current_semantics_hash(
            db, owner_id, run.dataset_version_id, ds.domain
        )
    return run, latest, findings, job, stale


def start_audit(db: Session, settings: Settings, owner_id: uuid.UUID, run_id: uuid.UUID) -> Job:
    run = get_owned(db, owner_id, run_id)
    if run.completed_at is None:
        raise _fail("The training run has not completed.", "/training_run", "RUN_NOT_COMPLETE")
    active = db.scalar(
        select(Job).where(
            Job.owner_id == owner_id,
            Job.type == "reliability_audit",
            Job.params["training_run_id"].astext == str(run.id),
            Job.status.in_(("queued", "running", "cancel_requested")),
        )
    )
    if active is not None:
        from datamedic.errors import Conflict

        raise Conflict("An audit of this run is already queued or running.")
    job = create_job(
        db,
        settings,
        owner_id,
        "reliability_audit",
        run.dataset_id,
        {"training_run_id": str(run.id)},
    )
    db.commit()
    return job


def audit_semantics_hash(domain: str, snapshot: dict[str, Any]) -> bytes:
    """Semantics hash in the diagnostics format, from a stored roles snapshot."""
    rows = []
    task_type = snapshot.get("task_type")
    for role, columns in snapshot.get("roles", {}).items():
        for c in columns:
            rows.append(
                {"column": c, "role": role, "task_type": task_type if role == "target" else None}
            )
    rows.sort(key=lambda r: (r["column"], r["role"]))
    return semantics_hash(domain, rows)


def catalogue() -> dict[str, Any]:
    return {
        "models": ml.catalogue(),
        "tasks": list(ml.TASKS),
        "tuning_modes": list(ml.TUNING_MODES),
        "inner_folds": ml.INNER_FOLDS,
        "metrics": ml.metric_info(),
        "task_metrics": {k: list(v) for k, v in ml.TASK_METRICS.items()},
        "primary_metric": ml.PRIMARY,
        "selection_metric": ml.SELECTION,
        "metric_conventions": ml.METRIC_CONVENTIONS,
        "audit_checks": audit_spec.catalogue(),
        "evaluation_status_rules": audit_spec.STATUS_RULES,
        "evaluation_status_limitation": audit_spec.STATUS_LIMITATION,
    }


def eligibility_from_dict(data: dict[str, Any]) -> Eligibility:
    return Eligibility(
        data["mode"],
        data["evaluation_version_id"],
        list(data.get("reasons", [])),
        data.get("replay"),
    )

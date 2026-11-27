"""Experiment Lab: configuration, validation and queueing (API side).

The API validates every candidate pipeline with the standard-library validator, derives
risk, stages and destructive flags from the catalogue, resolves the frozen split from the
declared roles, enforces resource limits, stores the complete reproducible configuration
(including a snapshot of the roles), and queues one ``experiment`` job. The worker runs it.
Experiments never write dataset versions. Every lookup is scoped to the owner (and RLS).
"""

import hashlib
import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from datamedic.config import Settings
from datamedic.db.models import (
    Dataset,
    DatasetVersion,
    Experiment,
    ExperimentCandidate,
    ExperimentResult,
    Job,
    SplitDefinition,
)
from datamedic.engine import ENGINE_VERSION, ROW_ID
from datamedic.engine.preprocess.recommend.facts import ENTITY_ROLES, TIME_ROLES
from datamedic.engine.preprocess.spec import Step, canonical_json
from datamedic.engine.preprocess.state import ValidationContext
from datamedic.engine.preprocess.validate import validate_steps
from datamedic.errors import (
    DatasetNotReady,
    DomainError,
    NotFound,
    QuotaExceeded,
    ValidationFailed,
)
from datamedic.jobs.service import create_job
from datamedic.preprocessing.context import roles_snapshot, validation_context
from datamedic.recommendations.service import owned_set
from datamedic.versions.service import get_owned_version

CLASSIFICATION = ("binary", "multiclass")
SUPERVISED_MODELS = ("dummy", "linear", "hgb")
CLUSTERING_MODELS = ("kmeans",)
BASELINE_KEY = "baseline"
BASELINE_LABEL = "Baseline: no preprocessing"
RISK_ORDER = ("low", "medium", "high")


def _fail(message: str, pointer: str, code: str) -> DomainError:
    return ValidationFailed(
        message, errors=[{"pointer": pointer, "code": code, "message": message}]
    )


def _first(roles: dict[str, list[str]], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if roles.get(key):
            return sorted(roles[key])[0]
    return None


def resolve_split(
    split: dict[str, Any], snapshot: dict[str, Any], ctx: ValidationContext, task: str
) -> dict[str, Any]:
    """The frozen-fold configuration from the request and the declared roles. Group and
    time columns always come from roles, never from the request."""
    roles = snapshot["roles"]
    entity, time = _first(roles, ENTITY_ROLES), _first(roles, TIME_ROLES)
    classification = task in CLASSIFICATION
    method = split.get("method")
    if method is None:
        if time:
            method = "time"
        elif entity:
            method = "stratified_group" if classification else "group"
        else:
            method = "stratified" if classification else "kfold"
    if method in ("group", "stratified_group") and entity is None:
        raise _fail(
            "Grouped folds need a column with a group or entity role (e.g. patient).",
            "/split/method",
            "SPLIT_ROLE_MISSING",
        )
    if method == "time" and time is None:
        raise _fail(
            "Time-ordered folds need a column with a time role.",
            "/split/method",
            "SPLIT_ROLE_MISSING",
        )
    if method in ("stratified", "stratified_group") and not classification:
        raise _fail(
            "Stratified folds need a classification target.", "/split/method", "SPLIT_ROLE_MISSING"
        )
    warnings = []
    if entity and method not in ("group", "stratified_group", "time"):
        warnings.append(
            {
                "code": "GROUP_ROLE_IGNORED",
                "message": "An entity role is declared "
                "but the folds do not keep entities together: rows of one entity can "
                "be in training and test rows, which usually inflates scores.",
            }
        )
    if time and method != "time":
        warnings.append(
            {
                "code": "TIME_ROLE_IGNORED",
                "message": "A time role is declared but "
                "the folds are not time-ordered: models may be tested on the past.",
            }
        )
    target = ctx.target if task != "clustering" else None
    features = [c.name for c in ctx.columns if c.name not in ctx.protected and c.name != ROW_ID]
    return {
        "method": method,
        "n_splits": int(split.get("n_splits", 5)),
        "seed": int(split.get("seed", 0)),
        "target": target if classification else None,
        "cohort_target": target,
        "group": entity if method in ("group", "stratified_group") else None,
        "time": time if method == "time" else None,
        "duplicate_columns": features + ([target] if target else []),
        "warnings": warnings,
    }


def split_config_hash(split: dict[str, Any], content_hash: bytes) -> bytes:
    payload = {
        k: split[k]
        for k in (
            "method",
            "n_splits",
            "seed",
            "target",
            "cohort_target",
            "group",
            "time",
            "duplicate_columns",
        )
    }
    payload["content_hash"] = content_hash.hex()
    return hashlib.sha256(canonical_json(payload).encode()).digest()


def _candidates(
    body: list[dict[str, Any]], ctx: ValidationContext, task: str
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [
        {
            "key": BASELINE_KEY,
            "label": BASELINE_LABEL,
            "steps": [],
            "spec_hash": validate_steps([], ctx).spec_hash,
            "max_risk": "low",
            "has_destructive": False,
            "stages": [],
            "recommendation_ids": [],
            "is_baseline": True,
        }
    ]
    errors: list[dict[str, str]] = []
    labels = {BASELINE_LABEL.lower()}
    for i, cand in enumerate(body):
        label = cand["label"].strip()
        if label.lower() in labels:
            errors.append(
                {
                    "pointer": f"/candidates/{i}/label",
                    "code": "DUPLICATE_LABEL",
                    "message": "Each candidate needs a distinct label.",
                }
            )
        labels.add(label.lower())
        if not cand["steps"]:
            errors.append(
                {
                    "pointer": f"/candidates/{i}/steps",
                    "code": "EMPTY_CANDIDATE",
                    "message": "A candidate needs at least one step (the baseline is "
                    "included automatically).",
                }
            )
            continue
        steps = [
            Step(s["op"], tuple(s.get("columns") or ()), dict(s.get("params") or {}))
            for s in cand["steps"]
        ]
        result = validate_steps(steps, ctx)
        for e in result.errors:
            pointer = f"/candidates/{i}/steps" + (f"/{e.step}" if e.step is not None else "")
            errors.append({"pointer": pointer, "code": e.code, "message": e.message})
        if not result.valid:
            continue
        if task == "clustering":
            for v in result.steps:
                if v.spec.needs_target or v.spec.kind == "training_only":
                    errors.append(
                        {
                            "pointer": f"/candidates/{i}/steps/{v.index}",
                            "code": "TARGET_REQUIRED",
                            "message": "This step needs a "
                            "target and cannot be used in a clustering experiment.",
                        }
                    )
        specs = [v.spec for v in result.steps]
        out.append(
            {
                "key": f"c{i + 1}",
                "label": label,
                "steps": [s.as_dict() for s in steps],
                "spec_hash": result.spec_hash,
                "max_risk": max((s.risk for s in specs), key=RISK_ORDER.index),
                "has_destructive": any(s.destructive for s in specs),
                "stages": sorted({s.stage for s in specs}),
                "recommendation_ids": [str(r) for r in cand.get("recommendation_ids") or []],
                "is_baseline": False,
            }
        )
    if errors:
        raise ValidationFailed("Some candidate pipelines are not valid.", errors=errors)
    return out


def _task_and_models(body: dict[str, Any], ctx: ValidationContext) -> tuple[str, list[str]]:
    if body.get("task") == "clustering":
        if body.get("n_clusters") is None:
            raise _fail(
                "Choose the number of clusters (2 to 20).", "/n_clusters", "N_CLUSTERS_REQUIRED"
            )
        allowed: tuple[str, ...] = CLUSTERING_MODELS
        task = "clustering"
    else:
        if ctx.target is None or ctx.task_type is None:
            raise _fail(
                "Declare a target column with a task type, or choose a clustering experiment.",
                "/task",
                "TARGET_REQUIRED",
            )
        allowed, task = SUPERVISED_MODELS, ctx.task_type
    models = list(dict.fromkeys(body.get("models") or allowed))
    bad = [m for m in models if m not in allowed]
    if bad:
        raise _fail(f"Models for this task: {', '.join(allowed)}.", "/models", "MODEL_INVALID")
    return task, models


def create(
    db: Session,
    settings: Settings,
    owner_id: uuid.UUID,
    version_id: uuid.UUID,
    body: dict[str, Any],
    idempotency_key: str | None = None,
    rerun_of: Experiment | None = None,
) -> tuple[Experiment, Job, bool]:
    """Returns (experiment, job, created). A repeated Idempotency-Key returns the original."""
    version, ds = get_owned_version(db, owner_id, version_id)
    if idempotency_key:
        prior = db.scalar(
            select(Job).where(Job.owner_id == owner_id, Job.idempotency_key == idempotency_key)
        )
        if prior is not None:
            exp = db.scalar(
                select(Experiment).where(
                    Experiment.job_id == prior.id, Experiment.owner_id == owner_id
                )
            )
            if exp is None:
                raise _fail(
                    "This Idempotency-Key was used for another request.",
                    "/headers/Idempotency-Key",
                    "IDEMPOTENCY_KEY_REUSED",
                )
            return exp, prior, False
    if ds.status != "ready":
        raise DatasetNotReady("The dataset is not ready.")
    db.execute(select(Dataset.id).where(Dataset.id == ds.id).with_for_update())
    snapshot = rerun_of.config["roles"] if rerun_of else roles_snapshot(db, owner_id, version.id)
    ctx = validation_context(version, snapshot)
    task, models = _task_and_models(body, ctx)
    if len(body["candidates"]) > settings.experiment_max_candidates:
        raise QuotaExceeded(
            f"At most {settings.experiment_max_candidates} candidates per "
            "experiment (plus the baseline)."
        )
    if (
        version.n_rows > settings.experiment_max_rows
        or version.n_rows * max(version.n_cols, 1) > settings.experiment_max_cells
    ):
        raise QuotaExceeded(
            f"Experiments are limited to {settings.experiment_max_rows:,} rows and "
            f"{settings.experiment_max_cells:,} cells."
        )
    split = resolve_split(body.get("split") or {}, snapshot, ctx, task)
    candidates = _candidates(body["candidates"], ctx, task)
    fits = len(candidates) * split["n_splits"] * len(models)
    if fits > settings.experiment_max_fits:
        raise QuotaExceeded(
            f"This experiment needs {fits} model fits; the limit is "
            f"{settings.experiment_max_fits}. Use fewer candidates, folds or "
            "models."
        )
    set_id = body.get("recommendation_set_id")
    if set_id is not None:
        owned_set(db, owner_id, uuid.UUID(str(set_id)), version.id)
    seed = int(body.get("seed", 0))
    config = {
        "version_id": str(version.id),
        "content_hash": version.content_hash.hex(),
        "task": task,
        "target": ctx.target if task != "clustering" else None,
        "models": models,
        "seed": seed,
        "n_clusters": body.get("n_clusters") if task == "clustering" else None,
        "split": split,
        "roles": snapshot,
        "candidates": [
            {k: c[k] for k in ("key", "label", "steps", "spec_hash")} for c in candidates
        ],
        "engine_version": ENGINE_VERSION,
    }
    job = create_job(db, settings, owner_id, "experiment", ds.id, {"version_id": str(version.id)})
    job.idempotency_key = idempotency_key
    exp = Experiment(
        owner_id=owner_id,
        dataset_id=ds.id,
        dataset_version_id=version.id,
        job_id=job.id,
        name=body["name"].strip(),
        task_type=task,
        target_column=config["target"],
        models=models,
        seed=seed,
        config=config,
        config_hash=hashlib.sha256(canonical_json(config).encode()).digest(),
        source_content_hash=version.content_hash,
        recommendation_set_id=uuid.UUID(str(set_id)) if set_id else None,
        rerun_of_id=rerun_of.id if rerun_of else None,
        engine_version=ENGINE_VERSION,
        created_by=owner_id,
    )
    db.add(exp)
    db.flush()
    job.params = {"version_id": str(version.id), "experiment_id": str(exp.id)}
    for position, c in enumerate(candidates):
        db.add(
            ExperimentCandidate(
                owner_id=owner_id,
                dataset_id=ds.id,
                experiment_id=exp.id,
                position=position,
                key=c["key"],
                label=c["label"],
                is_baseline=c["is_baseline"],
                steps=c["steps"],
                spec_hash=c["spec_hash"],
                max_risk=c["max_risk"],
                has_destructive=c["has_destructive"],
                stages=c["stages"],
                recommendation_ids=c["recommendation_ids"],
            )
        )
    db.commit()
    return exp, job, True


def rerun(
    db: Session, settings: Settings, owner_id: uuid.UUID, experiment_id: uuid.UUID
) -> tuple[Experiment, Job]:
    """A new experiment with exactly the stored configuration (same roles snapshot, split,
    seeds and candidates): its results must match the original."""
    original = get_owned(db, owner_id, experiment_id)
    cfg = original.config
    body = {
        "name": f"Re-run of {original.name}"[:200],
        "task": "clustering" if cfg["task"] == "clustering" else None,
        "n_clusters": cfg["n_clusters"],
        "models": cfg["models"],
        "seed": cfg["seed"],
        "split": {k: cfg["split"][k] for k in ("method", "n_splits", "seed")},
        "candidates": [
            {"label": c["label"], "steps": c["steps"]}
            for c in cfg["candidates"]
            if c["key"] != BASELINE_KEY
        ],
        "recommendation_set_id": original.recommendation_set_id,
    }
    exp, job, _ = create(
        db, settings, owner_id, original.dataset_version_id, body, rerun_of=original
    )
    return exp, job


def get_owned(db: Session, owner_id: uuid.UUID, experiment_id: uuid.UUID) -> Experiment:
    exp = db.scalar(
        select(Experiment).where(Experiment.id == experiment_id, Experiment.owner_id == owner_id)
    )
    if exp is None:
        raise NotFound()
    get_owned_version(db, owner_id, exp.dataset_version_id)  # dataset not deleted
    return exp


def detail(
    db: Session, owner_id: uuid.UUID, experiment_id: uuid.UUID
) -> tuple[
    Experiment,
    Job | None,
    list[ExperimentCandidate],
    dict[uuid.UUID, ExperimentResult],
    SplitDefinition | None,
]:
    exp = get_owned(db, owner_id, experiment_id)
    job = db.scalar(select(Job).where(Job.id == exp.job_id, Job.owner_id == owner_id))
    cands = list(
        db.scalars(
            select(ExperimentCandidate)
            .where(
                ExperimentCandidate.owner_id == owner_id,
                ExperimentCandidate.experiment_id == exp.id,
            )
            .order_by(ExperimentCandidate.position)
        )
    )
    results = {
        r.candidate_id: r
        for r in db.scalars(
            select(ExperimentResult).where(
                ExperimentResult.owner_id == owner_id, ExperimentResult.experiment_id == exp.id
            )
        )
    }
    split = (
        db.scalar(
            select(SplitDefinition).where(
                SplitDefinition.id == exp.split_definition_id, SplitDefinition.owner_id == owner_id
            )
        )
        if exp.split_definition_id
        else None
    )
    return exp, job, cands, results, split


def list_for_version(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID
) -> list[tuple[Experiment, Job | None]]:
    version, _ = get_owned_version(db, owner_id, version_id)
    rows = db.execute(
        select(Experiment, Job)
        .join(Job, Job.id == Experiment.job_id, isouter=True)
        .where(Experiment.owner_id == owner_id, Experiment.dataset_version_id == version.id)
        .order_by(Experiment.created_at.desc())
        .limit(100)
    ).all()
    return [(r[0], r[1]) for r in rows]


def version_of(db: Session, exp: Experiment) -> DatasetVersion:
    version = db.get(DatasetVersion, exp.dataset_version_id)
    assert version is not None
    return version

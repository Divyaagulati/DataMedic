"""Training and reliability-audit job handlers (worker, queue ``evaluate``).

``train``:
1. Load the run's stored configuration; verify the evaluated version's file hash and content
   hash, and the selected version's content hash.
2. Re-check everything the API checked: eligibility from the lineage (it must give the stored
   mode), the preprocessing specification (it must validate to the stored spec hash), the
   limits, and the data-dependent rules the API cannot see (classes in the cohort, cohort size,
   memory estimate).
3. For a replayed lineage, prove fidelity: refitting the committed steps on all ancestor rows
   must reproduce the selected version (exact hash, or the documented float tolerance).
4. Get or create the frozen split (shared with the Experiment Lab's split definitions).
5. Train (progress and cancellation through a separate session), write the out-of-fold
   predictions artifact, then write every result in the job's final transaction. A crash,
   failure or cancellation stores no results; an orphaned artifact object is reconciled.
6. Queue the reliability audit. Its failure never affects the training results.

``reliability_audit`` reads the completed run, its split and the evaluated cohort and stores
an ``analysis_runs`` row (kind ``reliability_audit``) with its findings. Nothing here writes a
dataset version, a role or a pipeline.
"""

import hashlib
import io
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from datamedic.analyses.service import current_semantics_hash
from datamedic.db.models import (
    AnalysisRun,
    Artifact,
    Dataset,
    DatasetVersion,
    FindingRow,
    ModelEvaluation,
    SplitDefinition,
    TrainingRun,
)
from datamedic.errors import INTEGRITY_FAILED, JOB_CANCELLED, LIMIT_EXCEEDED
from datamedic.experiments.jobs import _Progress, _split
from datamedic.jobs.service import create_job
from datamedic.preprocessing.context import role_specs, validation_context
from datamedic.storage import ObjectNotFound, artifact_object_key
from datamedic.training import service
from datamedic.workers.runner import HandlerContext, JobOutcome, register_handler

log = logging.getLogger(__name__)
BYTES_PER_CELL = 8
MEMORY_FACTOR = 12  # copies of the model matrix alive at once (preprocessing, folds, models)


def _fail(code: str, message: str) -> JobOutcome:
    return JobOutcome(False, error_code=code, error_message=message)


def _read(hctx: HandlerContext, version: DatasetVersion) -> bytes:
    with hctx.app.storage.open_reader(version.object_key) as fh:
        raw = fh.read()
    if hashlib.sha256(raw).digest() != version.file_sha256:
        raise ValueError("integrity")
    return raw


@register_handler("train")
def train(hctx: HandlerContext) -> JobOutcome:
    db, job = hctx.db, hctx.job
    run = db.scalar(select(TrainingRun).where(TrainingRun.job_id == job.id))
    if run is None:
        return _fail("NOT_FOUND", "The training run no longer exists.")
    if run.completed_at is not None:  # never re-run or overwrite a finished run
        return JobOutcome(True, result={"training_run_id": str(run.id), "skipped": "completed"})
    ds = db.get(Dataset, run.dataset_id)
    selected = db.get(DatasetVersion, run.dataset_version_id)
    source = db.get(DatasetVersion, run.evaluation_version_id)
    if ds is None or selected is None or source is None or ds.deleted_at is not None:
        return JobOutcome(True, result={"skipped": "dataset deleted"})
    cfg = run.config
    if service.config_hash(cfg) != run.config_hash:
        return _fail("CONFIG_CHANGED", "The stored configuration does not match its hash.")
    if (
        source.content_hash != run.source_content_hash
        or selected.content_hash.hex() != cfg["content_hash"]
    ):
        return _fail(INTEGRITY_FAILED, "A version does not match the run's configuration.")

    # -- Re-check eligibility and the specification (the plan endpoint is advisory) -----------
    from datamedic.engine.ml.eligibility import BLOCKED, EXPLORATORY, REPLAY, assess

    lineage, _ = service.lineage_of(db, selected)
    eligibility = assess(lineage, cfg["roles"], role_specs())
    stored = cfg["evaluation_mode"]
    expected = eligibility.mode if eligibility.mode != BLOCKED else EXPLORATORY
    if expected != stored or eligibility.evaluation_version_id != (
        cfg["evaluation_version_id"] if stored == REPLAY else str(selected.id)
    ):
        return _fail(
            "CONFIG_CHANGED", "The version's evaluation eligibility changed; re-create the run."
        )
    ctx = validation_context(selected, cfg["roles"])
    pre_validation = service.validate_preprocess(
        [
            {"op": s["op"], "columns": s["columns"], "params": s["params"]}
            for s in cfg["preprocess_steps"]
        ],
        ctx,
    )
    if not pre_validation.valid or pre_validation.spec_hash != cfg["spec_hash"]:
        return _fail(
            "CONFIG_CHANGED",
            "The preprocessing specification no longer validates to its stored hash (engine or "
            "catalogue changed); re-create the run.",
        )
    settings = hctx.app.settings
    if source.n_rows > settings.training_max_rows or source.n_rows * max(source.n_cols, 1) > (
        settings.training_max_cells
    ):
        return _fail(LIMIT_EXCEEDED, "The version exceeds the training limits.")

    try:
        raw = _read(hctx, source)
    except ObjectNotFound:
        return _fail("NOT_FOUND", "Version data missing.")
    except ValueError:
        return _fail(INTEGRITY_FAILED, "Stored version does not match its recorded hash.")

    import pandas as pd

    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.errors import EngineError
    from datamedic.engine.ml.replay import FoldPreprocessor, ReplayCommit, establish
    from datamedic.engine.ml.runner import Cancelled, SplitSpec, TrainingSpec, run_training
    from datamedic.engine.ml.spec import MAX_CLASSES, MIN_COHORT_ROWS
    from datamedic.engine.preprocess.experiment import cohort
    from datamedic.engine.preprocess.state import ValidationContext
    from datamedic.engine.versions import library_versions

    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    target = cfg["target"]
    data, excluded = cohort(frame, target)
    if len(data) < MIN_COHORT_ROWS:
        return _fail(
            "TOO_FEW_ROWS", f"At least {MIN_COHORT_ROWS} rows with a target value are needed."
        )
    task = cfg["task"]
    if task != "regression":
        n_classes = int(data[target].astype(object).nunique())
        if (task == "binary" and n_classes != 2) or (
            task == "multiclass" and not 3 <= n_classes <= MAX_CLASSES
        ):
            return _fail(
                "TASK_TARGET_MISMATCH",
                f"The {task} task does not match the target's {n_classes} classes among rows "
                "with a target value.",
            )
        smallest = int(data[target].astype(object).value_counts().min())
        if smallest < int(cfg["split"]["n_splits"]):
            return _fail(
                "CLASS_TOO_RARE",
                f"A class has fewer rows ({smallest}) than the number of folds; use fewer folds "
                "or merge rare classes in a pipeline.",
            )
    estimate_mb = len(data) * max(len(cfg["features"]), 1) * BYTES_PER_CELL * MEMORY_FACTOR / 2**20
    if estimate_mb > settings.training_memory_budget_mb:
        return _fail(
            LIMIT_EXCEEDED,
            f"The estimated memory ({estimate_mb:.0f} MB) exceeds the training budget "
            f"({settings.training_memory_budget_mb} MB); use fewer rows or features.",
        )

    commits: list[ReplayCommit] = []
    replay_result: dict[str, Any] | None = None
    if stored == REPLAY:
        for r in lineage.runs:
            rctx = ValidationContext.from_schema(
                r.source_schema,
                r.commit_roles.get("roles", {}),
                role_specs(),
                n_rows=r.source_n_rows,
                task_type=r.commit_roles.get("task_type"),
            )
            v = service.validate_preprocess([dict(x) for x in r.data_steps], rctx)
            if not v.valid or [s.step_hash for s in v.steps] != list(r.data_step_hashes):
                return _fail(
                    "REPLAY_NOT_ESTABLISHED", "A committed step no longer validates to its hash."
                )
            commits.append(ReplayCommit(v, r.seed))
        stored_frame = None
        try:
            stored_frame = pd.read_parquet(io.BytesIO(_read(hctx, selected)), engine="pyarrow")
            proof = establish(frame, commits, selected.content_hash, stored_frame)
        except ObjectNotFound:
            return _fail("NOT_FOUND", "Version data missing.")
        except ValueError:
            return _fail(INTEGRITY_FAILED, "Stored version does not match its recorded hash.")
        except EngineError as exc:
            return _fail(exc.code, exc.message)
        del stored_frame
        replay_result = {**(cfg["eligibility"].get("replay") or {}), "equivalence": proof}

    try:
        split_row, splits = _split(hctx, run, source, data, cfg["split"])  # type: ignore[arg-type]
    except EngineError as exc:
        code = "CLASS_TOO_RARE" if exc.code == "SPLIT_CLASS_TOO_SMALL" else exc.code
        return _fail(code, exc.message)

    roles_snapshot = cfg["roles"]
    sensitive = frozenset(cfg.get("sensitive_columns", []))
    split = cfg["split"]
    spec = TrainingSpec(
        task=task,
        target=target,
        models=tuple(cfg["models"]),
        mode=cfg["tuning"]["planned"],
        seed=int(cfg["seed"]),
        split=SplitSpec(
            method=split["method"],
            target=split["target"],
            group=split["group"],
            time=split["time"],
            duplicate_columns=tuple(split["duplicate_columns"]),
        ),
        sensitive=sensitive,
        sensitive_target=target in sensitive,
        permutation_rows=int(cfg["limits"]["permutation_rows_per_fold"]),
        permutation_columns=int(cfg["limits"]["permutation_columns"]),
        permutation_repeats=int(cfg["limits"]["permutation_repeats"]),
        bootstrap_resamples=int(cfg["limits"]["bootstrap_resamples"]),
    )
    del roles_snapshot
    progress = _Progress(hctx, job.id)
    try:
        out = run_training(
            data,
            splits,
            FoldPreprocessor(commits, pre_validation, int(cfg["seed"])),
            spec,
            excluded_missing_target=excluded,
            on_progress=progress.update,
            should_cancel=progress.cancelled,
        )
    except Cancelled:
        return JobOutcome(False, error_code=JOB_CANCELLED, error_message="Cancelled by user.")
    except EngineError as exc:
        return _fail(exc.code, exc.message)
    if progress.cancelled():  # a fit cannot be interrupted; never publish after a cancel
        return JobOutcome(False, error_code=JOB_CANCELLED, error_message="Cancelled by user.")

    # Out-of-fold predictions: platform-written Parquet, owner-only, never served raw.
    artifact_id = uuid.uuid4()
    buf = io.BytesIO()
    out.oof.to_parquet(buf, engine="pyarrow", index=False)
    payload = buf.getvalue()
    key = artifact_object_key(run.owner_id, run.dataset_id, artifact_id, "parquet")
    writer = hctx.app.storage.open_writer(key)
    try:
        writer.write(payload)
        stored_obj = writer.commit()
    except BaseException:
        writer.abort()
        raise

    db.add(
        Artifact(
            id=artifact_id,
            owner_id=run.owner_id,
            dataset_id=run.dataset_id,
            owner_type="training_run",
            owner_ref_id=run.id,
            kind="oof_predictions",
            object_key=key,
            sha256=stored_obj.sha256,
            size_bytes=stored_obj.size_bytes,
        )
    )
    db.flush()
    for model_key, res in out.models.items():
        db.add(
            ModelEvaluation(
                owner_id=run.owner_id,
                dataset_id=run.dataset_id,
                training_run_id=run.id,
                model_key=model_key,
                status=res["status"],
                comparable=res["comparable"],
                comparability_reason=res["comparability_reason"],
                metrics={
                    "folds": res["metrics"],
                    "reasons": res["metric_reasons"],
                    "pooled": res["pooled"],
                    "bootstrap": res["bootstrap"],
                },
                details={
                    k: res[k]
                    for k in (
                        "curves",
                        "calibration",
                        "residuals",
                        "confusion",
                        "per_class",
                        "error_slices",
                        "train_scores",
                        "timings",
                    )
                },
                fold_results=res["fold_results"],
                selection=res["selection"],
                importance=res["importance"],
                failures=res["failures"],
            )
        )
    summary = out.summary
    summary["split_warnings"] = list(split_row.warnings) + list(cfg.get("split_warnings", []))
    summary["evaluation_mode"] = stored
    summary["boundaries"] = out.evidence["boundaries"]
    run.split_definition_id = split_row.id
    run.replay = replay_result
    run.oof_artifact_id = artifact_id
    run.summary = summary
    run.library_versions = {**library_versions(), "engine": ENGINE_VERSION}
    run.completed_at = datetime.now(UTC)
    db.flush()
    follow = create_job(
        db,
        settings,
        run.owner_id,
        "reliability_audit",
        run.dataset_id,
        {"training_run_id": str(run.id)},
        enforce_limits=False,
    )
    return JobOutcome(
        True,
        result={"training_run_id": str(run.id), "split_definition_id": str(split_row.id)},
        follow_ups=[(follow.id, "reliability_audit")],
    )


@register_handler("reliability_audit")
def reliability_audit(hctx: HandlerContext) -> JobOutcome:
    db, job = hctx.db, hctx.job
    try:
        run_id = uuid.UUID(str(job.params.get("training_run_id")))
    except ValueError:
        return _fail("VALIDATION_FAILED", "Bad job params.")
    run = db.get(TrainingRun, run_id)
    if run is None or run.completed_at is None or run.summary is None:
        return _fail("NOT_FOUND", "The completed training run no longer exists.")
    ds = db.get(Dataset, run.dataset_id)
    source = db.get(DatasetVersion, run.evaluation_version_id)
    split_row = (
        db.get(SplitDefinition, run.split_definition_id) if run.split_definition_id else None
    )
    if ds is None or source is None or split_row is None or ds.deleted_at is not None:
        return JobOutcome(True, result={"skipped": "dataset deleted"})
    try:
        raw = _read(hctx, source)
        artifact = db.get(Artifact, split_row.fold_assignment_artifact_id)
        assert artifact is not None
        with hctx.app.storage.open_reader(artifact.object_key) as fh:
            fold_payload = fh.read()
    except ObjectNotFound:
        return _fail("NOT_FOUND", "Stored data missing.")
    except ValueError:
        return _fail(INTEGRITY_FAILED, "Stored version does not match its recorded hash.")

    import pandas as pd

    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.ml.audit import AuditContext, run_audit
    from datamedic.engine.ml.audit_spec import AUDIT_IDS, AUDIT_VERSION
    from datamedic.engine.preprocess.experiment import cohort, folds
    from datamedic.engine.versions import library_versions

    if folds.sha256(fold_payload) != split_row.assignment_sha256:
        return _fail(INTEGRITY_FAILED, "The stored fold assignment does not match its hash.")
    splits = folds.decode(fold_payload)
    cfg = run.config
    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    data, _ = cohort(frame, cfg["target"])
    evals = list(
        db.scalars(select(ModelEvaluation).where(ModelEvaluation.training_run_id == run.id))
    )
    models = {
        e.model_key: {
            "comparable": e.comparable,
            "metrics": e.metrics["folds"],
            "pooled": e.metrics["pooled"],
            "train_scores": e.details.get("train_scores", []),
        }
        for e in evals
    }
    diag = db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.dataset_version_id == run.dataset_version_id,
            AnalysisRun.kind == "diagnostics",
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )
    current = current_semantics_hash(db, run.owner_id, run.dataset_version_id, ds.domain)
    state = (
        "missing" if diag is None else ("current" if diag.semantics_hash == current else "stale")
    )
    features = [c for c in cfg["features"] if c in data.columns]
    ctx = AuditContext(
        task=cfg["task"],
        target=cfg["target"],
        data=data,
        features=features,
        splits=splits,
        summary=run.summary,
        models=models,
        evidence={"boundaries": run.summary.get("boundaries", [])},
        mode=run.evaluation_mode,
        eligibility={**run.eligibility, "replay": run.replay},
        roles=cfg["roles"].get("roles", {}),
        split_config=cfg["split"],
        preprocess_steps=cfg["preprocess_steps"],
        lineage_steps=cfg.get("lineage_steps", []),
        sensitive=frozenset(cfg.get("sensitive_columns", [])),
        datetime_columns=list(cfg.get("datetime_columns", [])),
        diagnostics={"state": state},
        seed=int(cfg["seed"]),
        max_rows=int(cfg["limits"]["audit_max_rows"]),
        max_single_features=int(cfg["limits"]["audit_single_feature_max_columns"]),
    )
    findings, summary = run_audit(ctx)
    audit_run = AnalysisRun(
        owner_id=run.owner_id,
        dataset_id=run.dataset_id,
        dataset_version_id=run.dataset_version_id,
        job_id=job.id,
        kind="reliability_audit",
        training_run_id=run.id,
        domain=ds.domain,
        check_set_hash=hashlib.sha256(
            "\n".join(f"{i}@{AUDIT_VERSION}" for i in sorted(AUDIT_IDS)).encode()
        ).digest(),
        semantics_hash=service.audit_semantics_hash(ds.domain, cfg["roles"]),
        roles_snapshot=[
            {"column": c, "role": role}
            for role, cols in sorted(cfg["roles"].get("roles", {}).items())
            for c in cols
        ],
        config={
            "audit_version": AUDIT_VERSION,
            "checks": list(AUDIT_IDS),
            "max_rows": ctx.max_rows,
            "max_single_features": ctx.max_single_features,
            "seed": ctx.seed,
            "diagnostics_state": state,
            "evaluation_mode": run.evaluation_mode,
        },
        summary=summary,
        engine_version=ENGINE_VERSION,
        library_versions=library_versions(),
    )
    db.add(audit_run)
    db.flush()
    for position, f in enumerate(findings):
        db.add(
            FindingRow(
                owner_id=run.owner_id,
                dataset_id=run.dataset_id,
                analysis_run_id=audit_run.id,
                position=position,
                **f.as_dict(),
            )
        )
    db.flush()
    return JobOutcome(
        True,
        result={
            "audit_run_id": str(audit_run.id),
            "evaluation_status": summary["evaluation_status"]["status"],
        },
    )

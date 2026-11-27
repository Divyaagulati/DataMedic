"""Experiment job handler (worker, queue ``evaluate``).

1. Load the experiment's stored configuration and the source version; verify the stored
   file hash and that the content hash equals the one the configuration was made for.
2. Rebuild every candidate from the stored steps and the stored *roles snapshot* (later role
   changes never alter an experiment) and verify each spec hash.
3. Get or create the frozen split: an existing split definition with the same configuration
   on the same version is reused from its stored assignment (hash verified); otherwise the
   splitter runs once and the assignment is stored as a ``fold_assignment`` artifact. This is
   committed in its own short transaction so concurrent experiments share it.
4. Run every candidate on those folds (progress and cancellation through a separate session,
   so the API sees them while the job runs).
5. Write all results and the summary in the job's final transaction: a crash, a failure or a
   cancellation writes nothing (no partial results). Nothing writes a dataset version.

Retries are safe: a job is claimed once (``claim_job``), and a completed experiment is never
re-run or overwritten.
"""

import hashlib
import io
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from datamedic.db.models import (
    Artifact,
    Dataset,
    DatasetVersion,
    Experiment,
    ExperimentCandidate,
    ExperimentResult,
    Job,
    SplitDefinition,
)
from datamedic.db.session import bind_user, session_scope
from datamedic.errors import INTEGRITY_FAILED, JOB_CANCELLED
from datamedic.experiments.service import split_config_hash
from datamedic.preprocessing.context import validation_context
from datamedic.storage import ObjectNotFound, artifact_object_key
from datamedic.workers.runner import HandlerContext, JobOutcome, register_handler

log = logging.getLogger(__name__)


def _fail(code: str, message: str) -> JobOutcome:
    return JobOutcome(False, error_code=code, error_message=message)


@register_handler("experiment")
def experiment(hctx: HandlerContext) -> JobOutcome:
    db, job, storage = hctx.db, hctx.job, hctx.app.storage
    exp = db.scalar(select(Experiment).where(Experiment.job_id == job.id))
    if exp is None:
        return _fail("NOT_FOUND", "The experiment no longer exists.")
    if exp.completed_at is not None:  # never re-run or overwrite a finished experiment
        return JobOutcome(True, result={"experiment_id": str(exp.id), "skipped": "completed"})
    version = db.get(DatasetVersion, exp.dataset_version_id)
    ds = db.get(Dataset, exp.dataset_id)
    if version is None or ds is None or ds.deleted_at is not None or version.deleted_at:
        return JobOutcome(True, result={"skipped": "version deleted"})
    if version.content_hash != exp.source_content_hash:
        return _fail(INTEGRITY_FAILED, "The source version does not match the experiment.")
    try:
        with storage.open_reader(version.object_key) as fh:
            raw = fh.read()
    except ObjectNotFound:
        return _fail("NOT_FOUND", "Version data missing.")
    if hashlib.sha256(raw).digest() != version.file_sha256:
        return _fail(INTEGRITY_FAILED, "Stored version does not match its recorded hash.")

    import pandas as pd

    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.errors import EngineError
    from datamedic.engine.preprocess.experiment import (
        Cancelled,
        Candidate,
        ExperimentSpec,
        cohort,
        run_experiment,
    )
    from datamedic.engine.preprocess.spec import Step
    from datamedic.engine.preprocess.validate import validate_steps
    from datamedic.engine.versions import library_versions

    cfg = exp.config
    ctx = validation_context(version, cfg["roles"])
    rows = list(
        db.scalars(
            select(ExperimentCandidate)
            .where(ExperimentCandidate.experiment_id == exp.id)
            .order_by(ExperimentCandidate.position)
        )
    )
    candidates = []
    for row in rows:
        steps = [Step(s["op"], tuple(s["columns"]), dict(s["params"])) for s in row.steps]
        result = validate_steps(steps, ctx)
        if not result.valid or result.spec_hash != row.spec_hash:
            return _fail(
                "CONFIG_CHANGED",
                "A candidate no longer validates to its stored "
                "specification (engine or catalogue changed); re-create the experiment.",
            )
        candidates.append(Candidate(row.key, row.label, result))

    frame = pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    split_cfg = cfg["split"]
    data, _ = cohort(frame, split_cfg["cohort_target"])
    try:
        split_row, splits = _split(hctx, exp, version, data, split_cfg)
    except EngineError as exc:
        return _fail(exc.code, exc.message)

    spec = ExperimentSpec(
        task=cfg["task"],
        target=cfg["target"],
        models=tuple(cfg["models"]),
        seed=int(cfg["seed"]),
        n_clusters=cfg.get("n_clusters"),
    )
    progress = _Progress(hctx, job.id)
    try:
        out = run_experiment(
            frame,
            splits,
            candidates,
            spec,
            on_progress=progress.update,
            should_cancel=progress.cancelled,
        )
    except Cancelled:
        return JobOutcome(False, error_code=JOB_CANCELLED, error_message="Cancelled by user.")

    by_key = {row.key: row for row in rows}
    for cand in out.pop("candidates"):
        row = by_key[cand["key"]]
        db.add(
            ExperimentResult(
                owner_id=exp.owner_id,
                dataset_id=exp.dataset_id,
                experiment_id=exp.id,
                candidate_id=row.id,
                status=cand["status"],
                comparable=cand["comparable"],
                comparability_reason=cand["comparability_reason"],
                metrics=cand["metrics"],
                fold_results=cand["folds"],
                data_quality=cand["data_quality"],
                selection_stability=cand["selection_stability"],
                details={
                    k: cand[k]
                    for k in (
                        "confusion_matrix",
                        "per_class",
                        "max_risk",
                        "has_destructive",
                        "stages",
                        "training_rows_change_per_fold",
                    )
                },
                failures=cand["failures"],
            )
        )
    out["split_warnings"] = split_row.warnings + split_cfg.get("warnings", [])
    if version.fitted_on_full_data:  # ML_ENGINE_DESIGN §5.5 (owner decision P4, Phase 5)
        out["warnings"].append(
            {
                "code": "FITTED_ON_FULL_DATA_SOURCE",
                "message": "This version was produced by committed steps that learned from all "
                "of its rows, including rows used here as test rows, so scores can be "
                "optimistic. For a leakage-controlled evaluation, use the Models workspace "
                "(which replays those steps inside each training fold) or experiment on an "
                "earlier version. Leakage control covers the implemented split and "
                "preprocessing safeguards; it does not guarantee external validity or the "
                "absence of other bias.",
            }
        )
    exp.split_definition_id = split_row.id
    exp.summary = out
    exp.library_versions = {**library_versions(), "engine": ENGINE_VERSION}
    exp.completed_at = datetime.now(UTC)
    db.flush()
    return JobOutcome(
        True, result={"experiment_id": str(exp.id), "split_definition_id": str(split_row.id)}
    )


class _Progress:
    """Progress, heartbeat and cancellation through a separate, short-lived session, so the
    API sees them while the job's own transaction is still open."""

    def __init__(self, hctx: HandlerContext, job_id: uuid.UUID) -> None:
        self.ctx, self.job_id, self.owner = hctx.app, job_id, hctx.job.owner_id

    def update(self, done: int, total: int, label: str) -> None:
        with session_scope(self.ctx.sessions) as db:
            bind_user(db, self.owner)
            db.execute(
                update(Job)
                .where(Job.id == self.job_id)
                .values(
                    progress={
                        "phase": "evaluating",
                        "pct": round(100 * done / max(total, 1)),
                        "done": done,
                        "total": total,
                    },
                    heartbeat_at=datetime.now(UTC),
                )
            )

    def cancelled(self) -> bool:
        with session_scope(self.ctx.sessions) as db:
            bind_user(db, self.owner)
            status = db.scalar(select(Job.status).where(Job.id == self.job_id))
        return status == "cancel_requested"


def _split(
    hctx: HandlerContext,
    exp: Experiment,
    version: DatasetVersion,
    data: Any,
    split_cfg: dict[str, Any],
) -> tuple[SplitDefinition, Any]:
    from datamedic.engine import ENGINE_VERSION
    from datamedic.engine.preprocess.experiment import folds
    from datamedic.engine.preprocess.split import make_splits
    from datamedic.engine.versions import library_versions

    digest = split_config_hash(split_cfg, version.content_hash)
    storage = hctx.app.storage

    def existing(db: Any) -> tuple[SplitDefinition, Any] | None:
        row = db.scalar(
            select(SplitDefinition).where(
                SplitDefinition.dataset_version_id == version.id,
                SplitDefinition.config_hash == digest,
            )
        )
        if row is None:
            return None
        artifact = db.get(Artifact, row.fold_assignment_artifact_id)
        with storage.open_reader(artifact.object_key) as fh:
            payload = fh.read()
        if folds.sha256(payload) != row.assignment_sha256:
            raise _integrity()
        db.expunge(row)
        return row, folds.decode(payload)

    with session_scope(hctx.app.sessions) as db:
        bind_user(db, exp.owner_id)
        found = existing(db)
    if found is not None:
        return found
    splits = make_splits(
        data,
        method=split_cfg["method"],
        n_splits=split_cfg["n_splits"],
        seed=split_cfg["seed"],
        target=split_cfg["target"],
        group=split_cfg["group"],
        time=split_cfg["time"],
        duplicate_columns=split_cfg["duplicate_columns"],
    )
    payload = folds.encode(splits)
    artifact_id, split_id = uuid.uuid4(), uuid.uuid4()
    key = artifact_object_key(exp.owner_id, exp.dataset_id, artifact_id, "json")
    writer = storage.open_writer(key)
    try:
        writer.write(payload)
        stored = writer.commit()
    except BaseException:
        writer.abort()
        raise
    stats = [
        {
            k: v
            for k, v in s.items()
            if k in ("fold", "n_train", "n_test", "train_class_counts", "test_class_counts")
        }
        for s in splits.fold_stats
    ]
    try:
        with session_scope(hctx.app.sessions) as db:
            bind_user(db, exp.owner_id)
            db.add(
                Artifact(
                    id=artifact_id,
                    owner_id=exp.owner_id,
                    dataset_id=exp.dataset_id,
                    owner_type="split_definition",
                    owner_ref_id=split_id,
                    kind="fold_assignment",
                    object_key=key,
                    sha256=stored.sha256,
                    size_bytes=stored.size_bytes,
                )
            )
            db.flush()
            row = SplitDefinition(
                id=split_id,
                owner_id=exp.owner_id,
                dataset_id=exp.dataset_id,
                dataset_version_id=version.id,
                config_hash=digest,
                method=splits.method,
                n_splits=splits.n_splits,
                seed=splits.seed,
                target_column=split_cfg["target"],
                group_column=split_cfg["group"],
                time_column=split_cfg["time"],
                duplicate_columns=split_cfg["duplicate_columns"],
                fold_assignment_artifact_id=artifact_id,
                assignment_sha256=folds.sha256(payload),
                cohort={"rows": len(data), "folds": splits.n_splits},
                fold_stats=stats,
                warnings=list(splits.warnings),
                engine_version=ENGINE_VERSION,
                library_versions=library_versions(),
                created_by=exp.owner_id,
            )
            db.add(row)
            db.flush()
            db.expunge(row)
    except IntegrityError:
        # Another experiment created the same split first. Storage is write-once with no
        # single-object delete; the unused object stays under the dataset's artifact prefix
        # and is removed with the dataset (purge deletes the whole prefix).
        with session_scope(hctx.app.sessions) as db:
            bind_user(db, exp.owner_id)
            found = existing(db)
        if found is None:
            raise
        return found
    return row, splits


def _integrity() -> Exception:
    from datamedic.engine.errors import EngineError

    return EngineError(INTEGRITY_FAILED, "The stored fold assignment does not match its hash.")

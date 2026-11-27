"""Biomedical workspace (API side; standard library and SQLAlchemy only).

Combines the stored descriptive summary (``analysis_runs`` kind ``biomedical``), the biomedical
findings of the latest diagnostics run and biomedical-specific model-reliability items derived
from stored training runs and audits. Nothing here recomputes data or writes versions."""

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from datamedic.analyses.service import current_semantics_hash, latest_analysis
from datamedic.config import Settings
from datamedic.db.models import AnalysisRun, FindingRow, Job, TrainingRun
from datamedic.errors import Conflict, DatasetNotReady
from datamedic.jobs.service import create_job
from datamedic.preprocessing.context import roles_snapshot
from datamedic.versions.service import get_owned_version

PATIENT_ROLES = ("bio.patient_id", "group")


def start(db: Session, settings: Settings, owner_id: uuid.UUID, version_id: uuid.UUID) -> Job:
    version, ds = get_owned_version(db, owner_id, version_id)
    if ds.status != "ready":
        raise DatasetNotReady("The dataset is not ready.")
    active = db.scalar(
        select(Job).where(
            Job.owner_id == owner_id,
            Job.type == "biomedical_summary",
            Job.params["version_id"].astext == str(version.id),
            Job.status.in_(("queued", "running", "cancel_requested")),
        )
    )
    if active is not None:
        raise Conflict("A biomedical summary of this version is already queued or running.")
    job = create_job(
        db, settings, owner_id, "biomedical_summary", ds.id, {"version_id": str(version.id)}
    )
    db.commit()
    return job


def _latest_job(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> Job | None:
    return db.scalar(
        select(Job)
        .where(
            Job.owner_id == owner_id,
            Job.type == "biomedical_summary",
            Job.params["version_id"].astext == str(version_id),
        )
        .order_by(Job.queued_at.desc())
        .limit(1)
    )


def latest_summary(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> AnalysisRun | None:
    return db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.owner_id == owner_id,
            AnalysisRun.dataset_version_id == version_id,
            AnalysisRun.kind == "biomedical",
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )


def _model_items(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID, patient: str | None
) -> list[dict[str, Any]]:
    """Biomedical reading of stored training runs and their latest audits."""
    items: list[dict[str, Any]] = []
    runs = list(
        db.scalars(
            select(TrainingRun)
            .where(
                TrainingRun.owner_id == owner_id,
                TrainingRun.dataset_version_id == version_id,
                TrainingRun.completed_at.is_not(None),
            )
            .order_by(TrainingRun.created_at.desc())
            .limit(5)
        )
    )
    if not runs:
        return [
            {
                "status": "not_assessed",
                "severity": "info",
                "title": "No completed training run on this version",
                "detail": "Train models in the Models workspace to assess model reliability.",
            }
        ]
    for run in runs:
        ref = {"training_run_id": str(run.id), "run_name": run.name}
        group = (run.split_config or {}).get("group")
        if run.evaluation_mode == "exploratory":
            items.append(
                {
                    **ref,
                    "status": "detected",
                    "severity": "critical",
                    "title": "Exploratory evaluation",
                    "detail": "Evaluated on a version fitted on all rows: scores are likely "
                    "optimistic and not leakage-controlled.",
                }
            )
        if patient is None:
            items.append(
                {
                    **ref,
                    "status": "not_assessed",
                    "severity": "info",
                    "title": "Patient-level separation not assessed",
                    "detail": "No patient role is declared, so it is unknown whether one "
                    "patient's rows were in both training and test folds.",
                }
            )
        elif group != patient:
            items.append(
                {
                    **ref,
                    "status": "detected",
                    "severity": "high",
                    "title": "Split not grouped by patient",
                    "detail": f"The run's folds were not grouped by '{patient}' (the patient "
                    "role was absent or different at training time). Rows of one patient may "
                    "appear in training and test folds, which inflates scores.",
                }
            )
        else:
            items.append(
                {
                    **ref,
                    "status": "not_detected",
                    "severity": "info",
                    "title": "Folds grouped by patient",
                    "detail": f"No patient appears in both training and test folds ('{patient}').",
                }
            )
        audit = db.scalar(
            select(AnalysisRun)
            .where(
                AnalysisRun.owner_id == owner_id,
                AnalysisRun.training_run_id == run.id,
                AnalysisRun.kind == "reliability_audit",
            )
            .order_by(AnalysisRun.created_at.desc())
            .limit(1)
        )
        if audit is None:
            continue
        for f in db.scalars(
            select(FindingRow).where(
                FindingRow.owner_id == owner_id,
                FindingRow.analysis_run_id == audit.id,
                FindingRow.status == "detected",
                FindingRow.severity.in_(("critical", "high", "medium")),
            )
        ):
            if f.check_id == "leakage.pipeline.fit_on_full_data":
                continue  # already reported as exploratory
            items.append(
                {
                    **ref,
                    "status": "detected",
                    "severity": f.severity,
                    "title": f.title,
                    "detail": f.explanation,
                    "basis": (f.evidence or {}).get("basis"),
                    "check_id": f.check_id,
                }
            )
    return items


def workspace(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> dict[str, Any]:
    version, ds = get_owned_version(db, owner_id, version_id)
    run = latest_summary(db, owner_id, version.id)
    job = _latest_job(db, owner_id, version.id)
    diag, findings, _, _, _ = latest_analysis(db, owner_id, version.id)
    snapshot = roles_snapshot(db, owner_id, version.id)
    roles = snapshot.get("roles", {})
    patient = next((c for k in PATIENT_ROLES for c in roles.get(k, [])), None)
    stale = run is not None and run.semantics_hash != current_semantics_hash(
        db, owner_id, version.id, ds.domain
    )
    return {
        "version_id": str(version.id),
        "domain": ds.domain,
        "summary": run.summary if run else None,
        "summary_created_at": run.created_at.isoformat() if run else None,
        "stale": stale,
        "job": job,
        "diagnostics": {
            "available": diag is not None,
            "findings": [f for f in findings if f.domain == "biomedical"],
        },
        "model_reliability": _model_items(db, owner_id, version.id, patient),
    }

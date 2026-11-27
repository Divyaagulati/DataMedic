"""Data Passport: one consolidated, structured report of a dataset version (API side; standard
library and SQLAlchemy only).

Built only from stored results; nothing is recomputed and nothing is invented. Every section
has ``status`` ``available`` or ``not_available`` (with a reason). Findings keep their own status
(detected / not detected / not assessed / check error) and basis, so observed facts, statistical
suspicions, unassessed checks and exploratory evaluations stay distinguishable. There is no
universal reliability score. Training results pass through the same sensitive-column redaction
as the API (``training.privacy``)."""

import uuid
from collections import Counter
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from datamedic.analyses.service import current_semantics_hash
from datamedic.db.models import (
    AnalysisRun,
    Approval,
    DatasetVersion,
    Experiment,
    ExperimentCandidate,
    FindingRow,
    Job,
    ModelEvaluation,
    Recommendation,
    RecommendationSet,
    TrainingRun,
    TransformationRun,
    VersionEvent,
    VersionProfile,
)
from datamedic.engine import ENGINE_VERSION
from datamedic.preprocessing.context import roles_snapshot
from datamedic.training import privacy
from datamedic.training.schemas import headline_of
from datamedic.versions.service import get_owned_version

PASSPORT_VERSION = "1.0.0"
MODE_LABELS = {
    "unbiased_direct": "Leakage-controlled evaluation",
    "unbiased_replay": "Leakage-controlled evaluation (replayed)",
    "exploratory": "Exploratory only",
}
LIMITATIONS = [
    "This passport summarises stored analyses of this version; it is not a certification of "
    "data quality, model validity or clinical suitability.",
    "A check that was not assessed is not a pass. Absence of a detected issue is not proof that "
    "no issue exists.",
    "Statistical suspicions are evidence to investigate, not confirmed problems.",
    "Model scores are estimated on folds of this dataset only; no independent external "
    "validation exists. Choosing the highest of several scores is optimistic.",
    "Exploratory evaluations are likely optimistic and are not leakage-controlled.",
    "Leakage-controlled evaluation covers the implemented split and preprocessing safeguards; it "
    "does not guarantee external validity, the absence of bias or clinical reliability.",
]


def _na(reason: str) -> dict[str, Any]:
    return {"status": "not_available", "reason": reason}


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat() if dt else None


def _finding(f: FindingRow) -> dict[str, Any]:
    return {
        "check_id": f.check_id,
        "domain": f.domain,
        "category": f.category,
        "status": f.status,
        "severity": f.severity,
        "basis": (f.evidence or {}).get("basis"),
        "title": f.title,
        "explanation": f.explanation,
        "columns": list(f.columns or []),
        "affected_pct": f.affected_pct,
        "recommendation": f.recommendation,
        "not_assessed_reason": f.not_assessed_reason,
    }


def _grouped(findings: list[FindingRow]) -> dict[str, Any]:
    items = [_finding(f) for f in findings]
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    detected = sorted(
        [i for i in items if i["status"] == "detected"], key=lambda i: order.get(i["severity"], 9)
    )
    return {
        "counts": dict(Counter(i["status"] for i in items)),
        "detected_by_severity": dict(Counter(i["severity"] for i in detected)),
        "detected": detected,
        "not_assessed": [i for i in items if i["status"] == "not_assessed"],
        "check_errors": [i for i in items if i["status"] == "check_error"],
        "not_detected_count": sum(1 for i in items if i["status"] == "not_detected"),
    }


def _latest_run(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID, kind: str
) -> AnalysisRun | None:
    return db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.owner_id == owner_id,
            AnalysisRun.dataset_version_id == version_id,
            AnalysisRun.kind == kind,
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )


def _findings(db: Session, owner_id: uuid.UUID, run_id: uuid.UUID) -> list[FindingRow]:
    return list(
        db.scalars(
            select(FindingRow)
            .where(FindingRow.owner_id == owner_id, FindingRow.analysis_run_id == run_id)
            .order_by(FindingRow.position)
        )
    )


def _quality(db: Session, owner_id: uuid.UUID, version: DatasetVersion) -> dict[str, Any]:
    p = db.scalar(
        select(VersionProfile)
        .where(VersionProfile.owner_id == owner_id, VersionProfile.dataset_version_id == version.id)
        .order_by(VersionProfile.created_at.desc())
        .limit(1)
    )
    if p is None:
        return _na("No profile of this version is stored.")
    columns = []
    for c in p.columns or []:
        num = c.get("numeric") or {}
        columns.append(
            {
                "name": c.get("name"),
                "inferred_type": c.get("inferred_type"),
                "missing": c.get("n_missing"),
                "missing_pct": c.get("missing_pct"),
                "n_unique": c.get("n_unique"),
                "is_constant": c.get("is_constant"),
                "median": num.get("median"),
                "iqr": num.get("iqr"),
            }
        )
    return {
        "status": "available",
        "created_at": _iso(p.created_at),
        "summary": p.summary,
        "columns": columns,
        "sampling": p.sampling,
        "limitations": p.limitations,
    }


def _history(db: Session, owner_id: uuid.UUID, version: DatasetVersion) -> dict[str, Any]:
    versions = list(
        db.scalars(
            select(DatasetVersion)
            .where(
                DatasetVersion.owner_id == owner_id,
                DatasetVersion.dataset_id == version.dataset_id,
                DatasetVersion.version_number <= version.version_number,
            )
            .order_by(DatasetVersion.version_number)
        )
    )
    runs = {
        r.id: r
        for r in db.scalars(
            select(TransformationRun).where(
                TransformationRun.owner_id == owner_id,
                TransformationRun.dataset_id == version.dataset_id,
            )
        )
    }
    approvals = Counter(
        a.pipeline_id
        for a in db.scalars(
            select(Approval).where(
                Approval.owner_id == owner_id, Approval.dataset_id == version.dataset_id
            )
        )
    )
    items = []
    for v in versions:
        run = runs.get(v.transformation_run_id) if v.transformation_run_id else None
        items.append(
            {
                "version_number": v.version_number,
                "kind": v.kind,
                "created_at": _iso(v.created_at),
                "rows": v.n_rows,
                "columns": v.n_cols,
                "content_hash": v.content_hash.hex(),
                "file_sha256": v.file_sha256.hex(),
                "fitted_on_full_data": v.fitted_on_full_data,
                "transformation": None
                if run is None
                else {
                    "spec_hash": run.spec_hash,
                    "steps": [
                        {"op": s.get("op"), "columns": s.get("columns", [])}
                        for s in (run.spec_snapshot or {}).get("data_steps", [])
                    ],
                    "model_stage_steps": len(run.model_pipeline_spec or []),
                    "approvals": len(run.approval_ids or []) or approvals.get(run.pipeline_id, 0),
                    "roles_lost": len((run.roles_report or {}).get("lost", [])),
                },
            }
        )
    events = [
        {"event": e.event, "created_at": _iso(e.created_at)}
        for e in db.scalars(
            select(VersionEvent)
            .where(VersionEvent.owner_id == owner_id, VersionEvent.dataset_id == version.dataset_id)
            .order_by(VersionEvent.created_at)
            .limit(200)
        )
    ]
    return {"status": "available", "versions": items, "events": events}


def _recommendations(db: Session, owner_id: uuid.UUID, version: DatasetVersion) -> dict[str, Any]:
    rs = db.scalar(
        select(RecommendationSet)
        .where(
            RecommendationSet.owner_id == owner_id,
            RecommendationSet.dataset_version_id == version.id,
        )
        .order_by(RecommendationSet.created_at.desc())
        .limit(1)
    )
    if rs is None:
        return _na("No preprocessing recommendations were generated for this version.")
    recs = list(
        db.scalars(
            select(Recommendation)
            .where(Recommendation.owner_id == owner_id, Recommendation.set_id == rs.id)
            .order_by(Recommendation.position)
        )
    )
    return {
        "status": "available",
        "created_at": _iso(rs.created_at),
        "items": [
            {
                "op": r.op,
                "stage": r.stage,
                "risk": r.risk,
                "destructive": r.destructive,
                "columns": list(r.columns or []),
                "status": r.status,
                "title": (r.payload or {}).get("title") or (r.payload or {}).get("op_title"),
                "rationale": (r.payload or {}).get("reason"),
                "expected_benefit": (r.payload or {}).get("expected_benefit"),
            }
            for r in recs
        ],
    }


def _experiments(db: Session, owner_id: uuid.UUID, version: DatasetVersion) -> dict[str, Any]:
    exps = list(
        db.scalars(
            select(Experiment)
            .where(Experiment.owner_id == owner_id, Experiment.dataset_version_id == version.id)
            .order_by(Experiment.created_at.desc())
            .limit(10)
        )
    )
    if not exps:
        return _na("No Experiment Lab comparisons were run on this version.")
    out = []
    for e in exps:
        labels = {
            str(c.id): c.label
            for c in db.scalars(
                select(ExperimentCandidate).where(
                    ExperimentCandidate.owner_id == owner_id,
                    ExperimentCandidate.experiment_id == e.id,
                )
            )
        }
        summary = e.summary or {}
        out.append(
            {
                "name": e.name,
                "task_type": e.task_type,
                "completed": e.completed_at is not None,
                "created_at": _iso(e.created_at),
                "candidates": list(labels.values()),
                "primary_metric": summary.get("primary_metric"),
                "comparisons": [
                    {
                        k: c.get(k)
                        for k in (
                            "candidate",
                            "baseline",
                            "model",
                            "metric",
                            "mean_difference",
                            "verdict",
                            "interpretation",
                        )
                    }
                    for c in summary.get("comparisons", [])
                    if c.get("metric") == summary.get("primary_metric")
                ][:30],
                "warnings": [w.get("code") for w in summary.get("warnings", [])],
            }
        )
    return {"status": "available", "items": out}


def _models(db: Session, owner_id: uuid.UUID, version: DatasetVersion) -> dict[str, Any]:
    runs = list(
        db.scalars(
            select(TrainingRun)
            .where(TrainingRun.owner_id == owner_id, TrainingRun.dataset_version_id == version.id)
            .order_by(TrainingRun.created_at.desc())
            .limit(10)
        )
    )
    if not runs:
        return _na("No models were trained on this version.")
    items = []
    for run in runs:
        job = db.scalar(select(Job).where(Job.id == run.job_id, Job.owner_id == owner_id))
        evals = list(
            db.scalars(
                select(ModelEvaluation).where(
                    ModelEvaluation.owner_id == owner_id, ModelEvaluation.training_run_id == run.id
                )
            )
        )
        protected = privacy.for_run(db, run)
        headline = headline_of(run, job, evals)
        primary = (run.summary or {}).get("primary_metric")
        models = []
        for e in evals:
            stats = (e.metrics.get("folds") or {}).get(primary) or {}
            perm = (e.importance or {}).get("permutation") or []
            models.append(
                {
                    "model": e.model_key,
                    "status": e.status,
                    "comparable": e.comparable,
                    "primary_mean": stats.get("mean"),
                    "primary_sd": stats.get("sd"),
                    "top_importance": [
                        {"column": p["column"], "mean": p["mean"]}
                        for p in perm[:5]
                        if not str(p["column"]).startswith("unresolved:")
                    ],
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
        items.append(
            {
                "name": run.name,
                "training_run_id": str(run.id),
                "task_type": run.task_type,
                "target": run.target_column,
                "evaluation_mode": run.evaluation_mode,
                "evaluation_label": MODE_LABELS.get(run.evaluation_mode, run.evaluation_mode),
                "job_status": job.status if job else None,
                "headline": headline,
                "primary_metric": primary,
                "models": models,
                "split": run.split_config,
                "seed": run.seed,
                "config_hash": run.config_hash.hex(),
                "library_versions": run.library_versions,
                "protected_columns": sorted(protected),
                "audit": None
                if audit is None
                else {
                    "status": (audit.summary.get("evaluation_status") or {}).get("status"),
                    "limitation": (audit.summary.get("evaluation_status") or {}).get("limitation"),
                    **_grouped(_findings(db, owner_id, audit.id)),
                },
            }
        )
    return {"status": "available", "items": items}


def _analytics(db: Session, owner_id: uuid.UUID, version: DatasetVersion) -> dict[str, Any]:
    """Saved dashboard configurations and stored insights (no query is executed here)."""
    from datamedic.analytics.service import summaries_for_version

    run = _latest_run(db, owner_id, version.id, "insights")
    insights: dict[str, Any] = (
        _na("Automated insights have not been generated for this version.")
        if run is None
        else {
            "status": "available",
            "created_at": _iso(run.created_at),
            "counts": run.summary.get("counts"),
            "unit": run.summary.get("unit"),
            "sampled": run.summary.get("sampled"),
            "top": [
                {
                    k: i.get(k)
                    for k in ("title", "basis", "category", "interpretation", "n", "method")
                }
                for i in run.summary.get("insights", [])[:10]
            ],
            "not_assessed": run.summary.get("not_assessed", []),
        }
    )
    return {
        "dashboards": summaries_for_version(db, owner_id, version.id),
        "insights": insights,
        "note": "Dashboards are saved configurations; their charts are computed only when opened.",
    }


def build(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> dict[str, Any]:
    version, ds = get_owned_version(db, owner_id, version_id)
    snapshot = roles_snapshot(db, owner_id, version.id)
    diag = _latest_run(db, owner_id, version.id, "diagnostics")
    bio = _latest_run(db, owner_id, version.id, "biomedical")
    current = current_semantics_hash(db, owner_id, version.id, ds.domain)
    diagnostics: dict[str, Any] = (
        _na("Diagnostics have not been run on this version.")
        if diag is None
        else {
            "status": "available",
            "created_at": _iso(diag.created_at),
            "domain": diag.domain,
            "stale": diag.semantics_hash != current,
            **_grouped(_findings(db, owner_id, diag.id)),
        }
    )
    domain: dict[str, Any]
    if diagnostics.get("status") == "available":
        domain_items = [
            i
            for i in diagnostics["detected"] + diagnostics["not_assessed"]
            if i["domain"] not in ("universal", None)
        ]
        domain = {"status": "available", "domain": ds.domain, "findings": domain_items}
    else:
        domain = _na("Diagnostics have not been run on this version.")
    if bio is not None:
        domain["biomedical_summary"] = bio.summary
    sensitive = sorted(privacy.protected_columns(db, owner_id, ds.id))
    return {
        "passport_version": PASSPORT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "engine_version": ENGINE_VERSION,
        "dataset": {
            "id": str(ds.id),
            "name": ds.name,
            "domain": ds.domain,
            "created_at": _iso(ds.created_at),
        },
        "version": {
            "id": str(version.id),
            "number": version.version_number,
            "kind": version.kind,
            "rows": version.n_rows,
            "columns": version.n_cols,
            "content_hash": version.content_hash.hex(),
            "file_sha256": version.file_sha256.hex(),
            "fitted_on_full_data": version.fitted_on_full_data,
            "created_at": _iso(version.created_at),
            "model_pipeline_spec": version.model_pipeline_spec or [],
        },
        "roles": snapshot,
        "protected_columns": sensitive,
        "quality": _quality(db, owner_id, version),
        "diagnostics": diagnostics,
        "domain_findings": domain,
        "recommendations": _recommendations(db, owner_id, version),
        "history": _history(db, owner_id, version),
        "experiments": _experiments(db, owner_id, version),
        "models": _models(db, owner_id, version),
        "analytics": _analytics(db, owner_id, version),
        "reproducibility": {
            "content_hash": version.content_hash.hex(),
            "code_export": f"/api/v1/versions/{version.id}/code",
            "note": "The generated pipeline.py reproduces this version from its original file. "
            "Training runs are reproduced by re-running their stored configuration with the "
            "recorded library versions.",
        },
        "limitations": LIMITATIONS,
    }

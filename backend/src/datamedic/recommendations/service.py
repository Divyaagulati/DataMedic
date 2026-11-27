"""Recommendations: generated synchronously in the API from the version's latest current
diagnostics, latest profile, declared roles and domain (deterministic standard-library
rules), stored for traceability, and regenerated only when one of those inputs changes.
Every lookup is scoped to the owner (and RLS)."""

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from datamedic.analyses.service import current_semantics_hash
from datamedic.db.models import (
    AnalysisRun,
    Dataset,
    DatasetVersion,
    FindingRow,
    Recommendation,
    RecommendationFinding,
    RecommendationSet,
    VersionProfile,
)
from datamedic.engine import ENGINE_VERSION
from datamedic.engine.preprocess.recommend import (
    RULES_VERSION,
    FindingRef,
    RecommendInput,
    recommend,
)
from datamedic.engine.preprocess.spec import canonical_json
from datamedic.errors import Conflict, NotFound
from datamedic.preprocessing.context import role_specs, roles_snapshot
from datamedic.versions.service import get_owned_version

STATUSES = ("open", "used", "dismissed")


class _Inputs:
    def __init__(
        self,
        run: AnalysisRun,
        findings: list[FindingRow],
        profile: VersionProfile | None,
        semantics: bytes,
        snapshot: dict[str, Any],
        domain: str,
    ) -> None:
        self.run, self.findings, self.profile = run, findings, profile
        self.semantics, self.snapshot, self.domain = semantics, snapshot, domain

    @property
    def input_hash(self) -> bytes:
        payload = {
            "analysis_run_id": str(self.run.id),
            "profile_id": str(self.profile.id) if self.profile else None,
            "semantics_hash": self.semantics.hex(),
            "roles": self.snapshot,
            "domain": self.domain,
            "rules_version": RULES_VERSION,
            "engine_version": ENGINE_VERSION,
        }
        return hashlib.sha256(canonical_json(payload).encode()).digest()


def _latest_run(db: Session, owner_id: uuid.UUID, version_id: uuid.UUID) -> AnalysisRun | None:
    return db.scalar(
        select(AnalysisRun)
        .where(
            AnalysisRun.owner_id == owner_id,
            AnalysisRun.dataset_version_id == version_id,
            AnalysisRun.kind == "diagnostics",
        )
        .order_by(AnalysisRun.created_at.desc())
        .limit(1)
    )


def _gather(db: Session, owner_id: uuid.UUID, version: DatasetVersion, ds: Dataset) -> _Inputs:
    run = _latest_run(db, owner_id, version.id)
    if run is None:
        raise Conflict("Run diagnostics for this version before generating recommendations.")
    semantics = current_semantics_hash(db, owner_id, version.id, ds.domain)
    if run.semantics_hash != semantics:
        raise Conflict(
            "Diagnostics are out of date: the domain or roles changed after they ran. Re-run "
            "diagnostics, then generate recommendations."
        )
    findings = list(
        db.scalars(
            select(FindingRow)
            .where(FindingRow.owner_id == owner_id, FindingRow.analysis_run_id == run.id)
            .order_by(FindingRow.position)
        )
    )
    profile = db.scalar(
        select(VersionProfile)
        .where(VersionProfile.owner_id == owner_id, VersionProfile.dataset_version_id == version.id)
        .order_by(VersionProfile.created_at.desc())
        .limit(1)
    )
    return _Inputs(
        run, findings, profile, semantics, roles_snapshot(db, owner_id, version.id), ds.domain
    )


def _engine_input(version: DatasetVersion, inputs: _Inputs) -> RecommendInput:
    profile = inputs.profile
    return RecommendInput(
        schema=version.schema,
        roles=inputs.snapshot["roles"],
        role_specs=role_specs(),
        domain=inputs.domain,
        task_type=inputs.snapshot["task_type"],
        n_rows=int(version.n_rows),
        profile_columns=profile.columns if profile else None,
        profile_correlations=profile.correlations if profile else None,
        findings=[
            FindingRef(
                id=str(f.id),
                check_id=f.check_id,
                status=f.status,
                severity=f.severity,
                columns=tuple(f.columns),
                evidence=f.evidence,
                title=f.title,
            )
            for f in inputs.findings
        ],
    )


def generate(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID
) -> tuple[RecommendationSet, list[Recommendation], bool]:
    """Return the set for the current inputs, creating it if needed. ``created`` is False when
    an identical set already existed (same inputs give the same recommendations)."""
    version, ds = get_owned_version(db, owner_id, version_id)
    db.execute(select(Dataset.id).where(Dataset.id == ds.id).with_for_update())
    inputs = _gather(db, owner_id, version, ds)
    digest = inputs.input_hash
    existing = db.scalar(
        select(RecommendationSet).where(
            RecommendationSet.owner_id == owner_id,
            RecommendationSet.dataset_version_id == version.id,
            RecommendationSet.input_hash == digest,
        )
    )
    if existing is not None:
        db.commit()
        return existing, _items(db, owner_id, existing.id), False
    content = recommend(_engine_input(version, inputs))
    recs = content.pop("recommendations")
    rec_set = RecommendationSet(
        owner_id=owner_id,
        dataset_id=ds.id,
        dataset_version_id=version.id,
        analysis_run_id=inputs.run.id,
        profile_id=inputs.profile.id if inputs.profile else None,
        semantics_hash=inputs.semantics,
        input_hash=digest,
        rules_version=RULES_VERSION,
        engine_version=ENGINE_VERSION,
        domain=ds.domain,
        task_type=inputs.snapshot["task_type"],
        content=content,
        created_by=owner_id,
    )
    db.add(rec_set)
    db.flush()
    finding_ids = {str(f.id) for f in inputs.findings}
    for rec in recs:
        row = Recommendation(
            owner_id=owner_id,
            dataset_id=ds.id,
            set_id=rec_set.id,
            rec_key=rec["id"],
            position=rec["rank"],
            rule_id=rec["rule_id"],
            rule_version=rec["rule_version"],
            op=rec["op"],
            op_version=rec["op_version"],
            stage=rec["stage"],
            risk=rec["risk"],
            destructive=rec["destructive"],
            columns=rec["columns"],
            params=rec["params"],
            payload=rec,
        )
        db.add(row)
        db.flush()
        for fid in rec["finding_ids"]:
            if fid in finding_ids:
                db.add(
                    RecommendationFinding(
                        recommendation_id=row.id,
                        finding_id=uuid.UUID(fid),
                        owner_id=owner_id,
                        dataset_id=ds.id,
                    )
                )
    db.commit()
    return rec_set, _items(db, owner_id, rec_set.id), True


def _items(db: Session, owner_id: uuid.UUID, set_id: uuid.UUID) -> list[Recommendation]:
    return list(
        db.scalars(
            select(Recommendation)
            .where(Recommendation.owner_id == owner_id, Recommendation.set_id == set_id)
            .order_by(Recommendation.position)
        )
    )


def latest(
    db: Session, owner_id: uuid.UUID, version_id: uuid.UUID
) -> tuple[RecommendationSet | None, list[Recommendation], bool]:
    """The newest set and whether it is stale (its inputs are no longer current)."""
    version, ds = get_owned_version(db, owner_id, version_id)
    rec_set = db.scalar(
        select(RecommendationSet)
        .where(
            RecommendationSet.owner_id == owner_id,
            RecommendationSet.dataset_version_id == version.id,
        )
        .order_by(RecommendationSet.created_at.desc())
        .limit(1)
    )
    if rec_set is None:
        return None, [], False
    try:
        stale = _gather(db, owner_id, version, ds).input_hash != rec_set.input_hash
    except Conflict:
        stale = True
    return rec_set, _items(db, owner_id, rec_set.id), stale


def set_status(
    db: Session, owner_id: uuid.UUID, recommendation_id: uuid.UUID, status: str
) -> Recommendation:
    row = db.scalar(
        select(Recommendation)
        .where(Recommendation.id == recommendation_id, Recommendation.owner_id == owner_id)
        .with_for_update()
    )
    if row is None:
        raise NotFound()
    get_owned_version(db, owner_id, _set_version(db, owner_id, row.set_id))
    row.status = status
    row.updated_at = datetime.now(UTC)
    db.commit()
    return row


def _set_version(db: Session, owner_id: uuid.UUID, set_id: uuid.UUID) -> uuid.UUID:
    version_id = db.scalar(
        select(RecommendationSet.dataset_version_id).where(
            RecommendationSet.id == set_id, RecommendationSet.owner_id == owner_id
        )
    )
    if version_id is None:
        raise NotFound()
    return version_id


def owned_set(
    db: Session, owner_id: uuid.UUID, set_id: uuid.UUID, version_id: uuid.UUID
) -> RecommendationSet:
    rec_set = db.scalar(
        select(RecommendationSet).where(
            RecommendationSet.id == set_id,
            RecommendationSet.owner_id == owner_id,
            RecommendationSet.dataset_version_id == version_id,
        )
    )
    if rec_set is None:
        raise NotFound()
    return rec_set

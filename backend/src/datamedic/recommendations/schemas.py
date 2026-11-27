import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from datamedic.db.models import Recommendation, RecommendationSet


class RecommendationOut(BaseModel):
    """``id`` identifies the stored row (for status changes); ``key`` is the deterministic
    engine id (identical for identical inputs). ``detail`` holds the full recommendation:
    reason, evidence, expected benefit, risks, information loss, domain and task
    compatibility, required decisions, evidence strength, limitations and ranking factors."""

    id: uuid.UUID
    key: str
    rank: int
    status: str
    op: str
    stage: str
    risk: str
    destructive: bool
    columns: list[str]
    params: dict[str, Any]
    detail: dict[str, Any]

    @classmethod
    def build(cls, r: Recommendation) -> "RecommendationOut":
        return cls(
            id=r.id,
            key=r.rec_key,
            rank=r.position,
            status=r.status,
            op=r.op,
            stage=r.stage,
            risk=r.risk,
            destructive=r.destructive,
            columns=r.columns,
            params=r.params,
            detail=r.payload,
        )


class RecommendationSetOut(BaseModel):
    id: uuid.UUID
    version_id: uuid.UUID
    analysis_run_id: uuid.UUID
    profile_id: uuid.UUID | None
    rules_version: str
    engine_version: str
    domain: str
    task_type: str | None
    created_at: datetime
    stale: bool
    recommendations: list[RecommendationOut]
    conflicts: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    split_plan: dict[str, Any]
    not_assessed: list[dict[str, Any]]
    not_recommended: list[dict[str, Any]]
    advisories: list[dict[str, Any]]
    inputs: dict[str, Any]
    limitations: list[str]

    @classmethod
    def build(
        cls, s: RecommendationSet, items: list[Recommendation], stale: bool
    ) -> "RecommendationSetOut":
        c = s.content
        return cls(
            id=s.id,
            version_id=s.dataset_version_id,
            analysis_run_id=s.analysis_run_id,
            profile_id=s.profile_id,
            rules_version=s.rules_version,
            engine_version=s.engine_version,
            domain=s.domain,
            task_type=s.task_type,
            created_at=s.created_at,
            stale=stale,
            recommendations=[RecommendationOut.build(r) for r in items],
            conflicts=c["conflicts"],
            candidates=c["candidates"],
            split_plan=c["split_plan"],
            not_assessed=c["not_assessed"],
            not_recommended=c["not_recommended"],
            advisories=c["advisories"],
            inputs=c["inputs"],
            limitations=c["limitations"],
        )


class RecommendationStateOut(BaseModel):
    set: RecommendationSetOut | None


class RecommendationStatusIn(BaseModel):
    status: Literal["open", "used", "dismissed"]

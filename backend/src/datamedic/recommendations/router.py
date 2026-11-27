import uuid

from fastapi import APIRouter, Response, status

from datamedic.api.deps import DbDep, PrincipalDep
from datamedic.recommendations import service
from datamedic.recommendations.schemas import (
    RecommendationOut,
    RecommendationSetOut,
    RecommendationStateOut,
    RecommendationStatusIn,
)

router = APIRouter(prefix="/api/v1", tags=["recommendations"])


@router.post("/versions/{version_id}/recommendations")
def generate_recommendations(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, response: Response
) -> RecommendationSetOut:
    """Generate recommendations from the current diagnostics, profile and roles (proposal
    only: nothing here changes data). Returns the existing set when the inputs are unchanged
    (200) and a new set otherwise (201). 409 when diagnostics are missing or out of date."""
    rec_set, items, created = service.generate(db, principal.user.id, version_id)
    response.status_code = status.HTTP_201_CREATED if created else status.HTTP_200_OK
    return RecommendationSetOut.build(rec_set, items, stale=False)


@router.get("/versions/{version_id}/recommendations")
def latest_recommendations(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> RecommendationStateOut:
    rec_set, items, stale = service.latest(db, principal.user.id, version_id)
    return RecommendationStateOut(
        set=RecommendationSetOut.build(rec_set, items, stale) if rec_set else None
    )


@router.patch("/recommendations/{recommendation_id}")
def update_recommendation(
    recommendation_id: uuid.UUID,
    body: RecommendationStatusIn,
    principal: PrincipalDep,
    db: DbDep,
) -> RecommendationOut:
    """Mark a recommendation open, used or dismissed. Changes nothing else."""
    return RecommendationOut.build(
        service.set_status(db, principal.user.id, recommendation_id, body.status)
    )

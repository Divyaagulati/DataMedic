import uuid
from typing import Any

from fastapi import APIRouter, status
from pydantic import BaseModel

from datamedic.analyses.schemas import FindingOut
from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.biomedical import service
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut

router = APIRouter(prefix="/api/v1", tags=["biomedical"])


class BiomedicalAcceptedOut(BaseModel):
    job: JobOut


class BiomedicalWorkspaceOut(BaseModel):
    version_id: uuid.UUID
    domain: str
    summary: dict[str, Any] | None
    summary_created_at: str | None
    stale: bool
    job: JobOut | None
    diagnostics_available: bool
    biomedical_findings: list[FindingOut]
    model_reliability: list[dict[str, Any]]


@router.get("/versions/{version_id}/biomedical")
def get_biomedical(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep
) -> BiomedicalWorkspaceOut:
    """Biomedical workspace: the stored descriptive summary, biomedical diagnostics findings and
    biomedical model-reliability items (research use only; nothing is recomputed)."""
    w = service.workspace(db, principal.user.id, version_id)
    return BiomedicalWorkspaceOut(
        version_id=version_id,
        domain=w["domain"],
        summary=w["summary"],
        summary_created_at=w["summary_created_at"],
        stale=w["stale"],
        job=JobOut.from_job(w["job"]) if w["job"] else None,
        diagnostics_available=w["diagnostics"]["available"],
        biomedical_findings=[FindingOut.build(f) for f in w["diagnostics"]["findings"]],
        model_reliability=w["model_reliability"],
    )


@router.post("/versions/{version_id}/biomedical", status_code=status.HTTP_202_ACCEPTED)
def start_biomedical(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> BiomedicalAcceptedOut:
    """Queue a descriptive biomedical summary of the version (never writes versions)."""
    job = service.start(db, ctx.settings, principal.user.id, version_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return BiomedicalAcceptedOut(job=JobOut.from_job(job))

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from datamedic.analyses import service
from datamedic.analyses.schemas import (
    AnalysisAcceptedOut,
    AnalysisRunOut,
    AnalysisStateOut,
    CheckListOut,
    CheckOut,
    DomainListOut,
    RoleOut,
    RolesIn,
    RolesOut,
    RoleSpecOut,
)
from datamedic.api.deps import CtxDep, DbDep, PrincipalDep
from datamedic.domains.bootstrap import build_registry
from datamedic.jobs.publisher import publish_committed
from datamedic.jobs.schemas import JobOut
from datamedic.versions.service import get_owned_version

router = APIRouter(prefix="/api/v1", tags=["diagnostics"])


@router.get("/domains")
def list_domains(principal: PrincipalDep) -> DomainListOut:
    registry = build_registry()
    return DomainListOut.build(
        registry.universal, "General tabular", [registry.modules[k] for k in registry.modules]
    )


@router.get("/checks")
def list_checks(
    principal: PrincipalDep, domain: Annotated[str | None, Query(max_length=64)] = None
) -> CheckListOut:
    registry = build_registry()
    specs = registry.checks_for(domain) if domain else registry.all_checks()
    return CheckListOut(items=[CheckOut.build(c) for c in specs])


def _roles_out(db: DbDep, owner_id: uuid.UUID, version_id: uuid.UUID) -> RolesOut:
    version, ds = get_owned_version(db, owner_id, version_id)
    rows = service.current_roles(db, owner_id, version.id)
    return RolesOut(
        version_id=version.id,
        domain=ds.domain,
        roles=[RoleOut(column=r.column_name, role=r.role, task_type=r.task_type) for r in rows],
        available_roles=[RoleSpecOut.build(r) for r in build_registry().roles_for(ds.domain)],
        semantics_hash=service.current_semantics_hash(db, owner_id, version.id, ds.domain).hex(),
    )


@router.get("/versions/{version_id}/roles")
def get_roles(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> RolesOut:
    return _roles_out(db, principal.user.id, version_id)


@router.put("/versions/{version_id}/roles")
def put_roles(version_id: uuid.UUID, body: RolesIn, principal: PrincipalDep, db: DbDep) -> RolesOut:
    """Replace the version's roles. Existing diagnostics become stale (not deleted)."""
    service.replace_roles(db, principal.user.id, version_id, [r.model_dump() for r in body.roles])
    return _roles_out(db, principal.user.id, version_id)


@router.post("/versions/{version_id}/analyses", status_code=status.HTTP_202_ACCEPTED)
def request_analysis(
    version_id: uuid.UUID, principal: PrincipalDep, db: DbDep, ctx: CtxDep
) -> AnalysisAcceptedOut:
    job = service.request_analysis(db, ctx.settings, principal.user.id, version_id)
    publish_committed(db, ctx.require_publisher(), job.id, job.type)
    db.refresh(job)
    return AnalysisAcceptedOut(job=JobOut.from_job(job))


@router.get("/versions/{version_id}/analyses/latest")
def latest_analysis(version_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> AnalysisStateOut:
    run, findings, job, current, _ = service.latest_analysis(db, principal.user.id, version_id)
    return AnalysisStateOut(
        run=AnalysisRunOut.build(run, findings, current) if run else None,
        job=JobOut.from_job(job) if job else None,
        current_semantics_hash=current.hex(),
    )


@router.get("/analyses/{analysis_id}")
def get_analysis(analysis_id: uuid.UUID, principal: PrincipalDep, db: DbDep) -> AnalysisRunOut:
    run, findings, current = service.get_analysis(db, principal.user.id, analysis_id)
    return AnalysisRunOut.build(run, findings, current)

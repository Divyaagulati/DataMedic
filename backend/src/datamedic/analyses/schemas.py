import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from datamedic.db.models import AnalysisRun, FindingRow
from datamedic.engine.checks.spec import CheckSpec, DomainSpec, RoleSpec
from datamedic.jobs.schemas import JobOut

TaskType = Literal["binary", "multiclass", "regression"]
Status = Literal["detected", "not_detected", "not_assessed", "error"]
Severity = Literal["critical", "high", "medium", "low", "info"]


class RoleSpecOut(BaseModel):
    key: str
    name: str
    description: str
    domain: str
    allowed_types: list[str]
    max_per_version: int | None
    is_feature: bool
    is_sensitive: bool

    @classmethod
    def build(cls, r: RoleSpec) -> "RoleSpecOut":
        return cls(
            key=r.key,
            name=r.name,
            description=r.description,
            domain=r.domain,
            allowed_types=sorted(r.allowed_types),
            max_per_version=r.max_per_version,
            is_feature=r.is_feature,
            is_sensitive=r.is_sensitive,
        )


class DomainOut(BaseModel):
    key: str
    name: str
    version: str
    description: str
    roles: list[RoleSpecOut]
    check_count: int


class DomainListOut(BaseModel):
    items: list[DomainOut]

    @classmethod
    def build(
        cls, universal: DomainSpec, generic_name: str, modules: list[DomainSpec]
    ) -> "DomainListOut":
        items = [
            DomainOut(
                key="generic",
                name=generic_name,
                version=universal.version,
                description="Universal checks only.",
                roles=[],
                check_count=0,
            )
        ]
        items += [
            DomainOut(
                key=m.key,
                name=m.name,
                version=m.version,
                description=m.description,
                roles=[RoleSpecOut.build(r) for r in m.roles],
                check_count=len(m.checks),
            )
            for m in modules
        ]
        return cls(items=items)


class RequirementsOut(BaseModel):
    roles: list[str]
    roles_any: list[str]
    task_types: list[str] | None
    min_rows: int


class CheckOut(BaseModel):
    id: str
    version: str
    domain: str
    category: str
    title: str
    description: str
    requirements: RequirementsOut
    limitations: list[str]
    expensive: bool

    @classmethod
    def build(cls, c: CheckSpec) -> "CheckOut":
        req = c.requirements
        return cls(
            id=c.id,
            version=c.version,
            domain=c.domain,
            category=c.category,
            title=c.title,
            description=c.description,
            requirements=RequirementsOut(
                roles=sorted(req.roles),
                roles_any=sorted(req.roles_any),
                task_types=sorted(req.task_types) if req.task_types is not None else None,
                min_rows=req.min_rows,
            ),
            limitations=list(c.limitations),
            expensive=c.expensive,
        )


class CheckListOut(BaseModel):
    items: list[CheckOut]


class RoleIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    column: str = Field(min_length=1, max_length=255)
    role: str = Field(min_length=1, max_length=64)
    task_type: TaskType | None = None


class RolesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roles: list[RoleIn] = Field(max_length=5000)


class RoleOut(BaseModel):
    column: str
    role: str
    task_type: TaskType | None


class RolesOut(BaseModel):
    version_id: uuid.UUID
    domain: str
    roles: list[RoleOut]
    available_roles: list[RoleSpecOut]
    semantics_hash: str


class FindingOut(BaseModel):
    id: uuid.UUID
    check_id: str
    check_version: str
    domain: str
    category: str
    status: Status
    severity: Severity
    title: str
    explanation: str
    evidence: dict[str, Any]
    columns: list[str]
    affected_rows: int | None
    affected_pct: float | None
    recommendation: str | None
    limitations: list[str]
    not_assessed_reason: str | None
    error_code: str | None
    sampled: bool
    sample_info: dict[str, Any] | None

    @classmethod
    def build(cls, f: FindingRow) -> "FindingOut":
        return cls.model_validate(f, from_attributes=True)


class AnalysisSummaryOut(BaseModel):
    total: int
    by_status: dict[str, int]
    detected_by_severity: dict[str, int]
    detected_by_domain: dict[str, int]
    detected_by_category: dict[str, int]
    check_errors: int


class AnalysisRunOut(BaseModel):
    id: uuid.UUID
    version_id: uuid.UUID
    job_id: uuid.UUID
    domain: str
    created_at: datetime
    stale: bool
    roles_snapshot: list[RoleOut]
    summary: AnalysisSummaryOut
    config: dict[str, Any]
    engine_version: str
    library_versions: dict[str, Any]
    findings: list[FindingOut]

    @classmethod
    def build(
        cls, run: AnalysisRun, findings: list[FindingRow], current_hash: bytes
    ) -> "AnalysisRunOut":
        return cls(
            id=run.id,
            version_id=run.dataset_version_id,
            job_id=run.job_id,
            domain=run.domain,
            created_at=run.created_at,
            stale=run.semantics_hash != current_hash,
            roles_snapshot=[RoleOut.model_validate(r) for r in run.roles_snapshot],
            summary=AnalysisSummaryOut.model_validate(run.summary),
            config=run.config,
            engine_version=run.engine_version,
            library_versions=run.library_versions,
            findings=[FindingOut.build(f) for f in findings],
        )


class AnalysisStateOut(BaseModel):
    """The latest stored run of the version (if any) and the latest analysis job."""

    run: AnalysisRunOut | None
    job: JobOut | None
    current_semantics_hash: str


class AnalysisAcceptedOut(BaseModel):
    job: JobOut

"""Running checks: context, honest ``not_assessed``, error isolation, finding validation.

Rules (DEVELOPMENT_GUIDE.md invariant 8):
- an unmet requirement yields ``not_assessed`` with the reason, never "no issue found";
- every finding carries evidence, severity and limitations; sampling is disclosed;
- an exception inside a check becomes an ``error`` finding (``CHECK_ERROR``) whose text
  never includes the exception message (it could quote data); the run continues;
- findings describe; they never modify data and never claim an anomaly is an error.
"""

import importlib
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.checks.spec import (
    SEVERITIES,
    STATUSES,
    CheckSpec,
    Requirements,
    RoleSpec,
)

MAX_EXAMPLES = 5
MAX_EXAMPLE_CHARS = 64


@dataclass
class Finding:
    check_id: str
    check_version: str
    domain: str
    category: str
    status: str
    severity: str
    title: str
    explanation: str
    evidence: dict[str, Any]
    columns: list[str] = field(default_factory=list)
    affected_rows: int | None = None
    affected_pct: float | None = None
    recommendation: str | None = None
    limitations: list[str] = field(default_factory=list)
    not_assessed_reason: str | None = None
    error_code: str | None = None
    sampled: bool = False
    sample_info: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class Budget:
    """Row caps for costly checks (seeded uniform samples, always disclosed)."""

    expensive_rows: int = 20_000
    moderate_rows: int = 200_000


@dataclass
class RoleMap:
    """Declared roles: role key -> columns, plus the target's task type."""

    columns: dict[str, list[str]] = field(default_factory=dict)
    task_type: str | None = None

    def has(self, role: str) -> bool:
        return bool(self.columns.get(role))

    def first(self, role: str) -> str | None:
        cols = self.columns.get(role)
        return cols[0] if cols else None

    def all(self, role: str) -> list[str]:
        return list(self.columns.get(role, []))

    def roles_of(self, column: str) -> set[str]:
        return {role for role, cols in self.columns.items() if column in cols}


@dataclass
class CheckContext:
    frame: pd.DataFrame  # canonical frame including ROW_ID; never mutated
    schema: list[dict[str, Any]]
    roles: RoleMap
    domain: str
    role_specs: dict[str, RoleSpec]
    budget: Budget = field(default_factory=Budget)
    seed: int = 0
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def data(self) -> pd.DataFrame:
        if "data" not in self.cache:
            self.cache["data"] = self.frame.drop(columns=[ROW_ID])
        frame: pd.DataFrame = self.cache["data"]
        return frame

    @property
    def n_rows(self) -> int:
        return len(self.frame)

    def type_of(self, column: str) -> str:
        return next((c["inferred_type"] for c in self.schema if c["name"] == column), "text")

    def is_feature(self, column: str) -> bool:
        for role in self.roles.roles_of(column):
            spec = self.role_specs.get(role)
            if spec is not None and not spec.is_feature:
                return False
        return True

    def is_sensitive(self, column: str) -> bool:
        return any(
            (spec := self.role_specs.get(role)) is not None and spec.is_sensitive
            for role in self.roles.roles_of(column)
        )

    def features(self, *types: str) -> list[str]:
        """Feature columns (roles permitting), optionally of the given semantic types."""
        return [
            c["name"]
            for c in self.schema
            if self.is_feature(c["name"]) and (not types or c["inferred_type"] in types)
        ]

    def numeric_features(self) -> list[str]:
        """Numeric features with at least two distinct values (identifiers excluded)."""
        return [c for c in self.features("numeric") if self.data[c].nunique(dropna=True) > 1]

    def target(self) -> str | None:
        return self.roles.first("target")

    def sample(self, columns: list[str], limit: int) -> tuple[pd.DataFrame, dict[str, Any] | None]:
        sub = self.data[columns]
        if len(sub) <= limit:
            return sub, None
        return sub.sample(n=limit, random_state=self.seed), {
            "n": limit,
            "of": len(sub),
            "seed": self.seed,
            "strategy": "uniform random rows",
        }

    def examples(self, column: str, values: Any) -> list[Any] | None:
        """At most 5 example values, each cut to 64 characters; none for sensitive columns."""
        if self.is_sensitive(column):
            return None
        out: list[Any] = []
        for value in list(values)[:MAX_EXAMPLES]:
            if isinstance(value, float) and not math.isfinite(value):
                out.append(str(value))
            elif isinstance(value, bool | int | float):
                out.append(value)
            elif isinstance(value, np.integer | np.floating | np.bool_):
                out.append(value.item())
            else:
                text = str(value)
                out.append(text if len(text) <= MAX_EXAMPLE_CHARS else text[:63] + "…")
        return out


# ---- Builders -------------------------------------------------------------------------------


def pct(part: float, whole: float) -> float:
    return round(100.0 * part / whole, 4) if whole else 0.0


def detected(
    spec: CheckSpec,
    ctx: CheckContext,
    severity: str,
    title: str,
    explanation: str,
    evidence: dict[str, Any],
    *,
    columns: list[str] | None = None,
    affected_rows: int | None = None,
    recommendation: str,
    limitations: list[str] | None = None,
    sample_info: dict[str, Any] | None = None,
) -> Finding:
    return Finding(
        check_id=spec.id,
        check_version=spec.version,
        domain=spec.domain,
        category=spec.category,
        status="detected",
        severity=severity,
        title=title,
        explanation=explanation,
        evidence=evidence,
        columns=columns or [],
        affected_rows=affected_rows,
        affected_pct=pct(affected_rows, ctx.n_rows) if affected_rows is not None else None,
        recommendation=recommendation,
        limitations=[*spec.limitations, *(limitations or [])],
        sampled=sample_info is not None,
        sample_info=sample_info,
    )


def not_detected(
    spec: CheckSpec,
    explanation: str,
    evidence: dict[str, Any],
    *,
    columns: list[str] | None = None,
    limitations: list[str] | None = None,
    sample_info: dict[str, Any] | None = None,
) -> Finding:
    return Finding(
        check_id=spec.id,
        check_version=spec.version,
        domain=spec.domain,
        category=spec.category,
        status="not_detected",
        severity="info",
        title=spec.title,
        explanation=explanation,
        evidence=evidence,
        columns=columns or [],
        limitations=[*spec.limitations, *(limitations or [])],
        sampled=sample_info is not None,
        sample_info=sample_info,
    )


def not_assessed(spec: CheckSpec, reason: str) -> Finding:
    return Finding(
        check_id=spec.id,
        check_version=spec.version,
        domain=spec.domain,
        category=spec.category,
        status="not_assessed",
        severity="info",
        title=spec.title,
        explanation=f"Not assessed: {reason}",
        evidence={},
        limitations=list(spec.limitations),
        not_assessed_reason=reason,
    )


def _error(spec: CheckSpec, exc: Exception) -> Finding:
    return Finding(
        check_id=spec.id,
        check_version=spec.version,
        domain=spec.domain,
        category=spec.category,
        status="error",
        severity="info",
        title=spec.title,
        explanation="The check failed unexpectedly and produced no result. Other checks ran.",
        evidence={"exception_type": type(exc).__name__},
        limitations=list(spec.limitations),
        error_code="CHECK_ERROR",
    )


# ---- Running --------------------------------------------------------------------------------


def unmet_requirement(req: Requirements, ctx: CheckContext) -> str | None:
    for role in sorted(req.roles):
        if not ctx.roles.has(role):
            label = ctx.role_specs[role].name if role in ctx.role_specs else role
            return f"no column has the role '{label}' ({role}). Declare it to run this check."
    if req.roles_any and not any(ctx.roles.has(r) for r in req.roles_any):
        names = ", ".join(sorted(req.roles_any))
        return f"none of the roles {names} is declared."
    if req.task_types is not None and ctx.roles.task_type not in req.task_types:
        allowed = " or ".join(sorted(req.task_types))
        return f"the target's task type must be {allowed} (it is {ctx.roles.task_type or 'unset'})."
    if ctx.n_rows < req.min_rows:
        return f"at least {req.min_rows} rows are needed (the dataset has {ctx.n_rows})."
    return None


CheckImpl = Callable[[CheckContext, CheckSpec], list[Finding]]


def resolve(spec: CheckSpec) -> CheckImpl:
    module_name, func_name = spec.impl.split(":")
    impl: CheckImpl = getattr(importlib.import_module(module_name), func_name)
    return impl


def validate(finding: Finding) -> None:
    if finding.status not in STATUSES or finding.severity not in SEVERITIES:
        raise ValueError(f"invalid status/severity from {finding.check_id}")
    if finding.status == "detected" and not finding.evidence:
        raise ValueError(f"detected finding without evidence from {finding.check_id}")
    if finding.status == "detected" and not finding.recommendation:
        raise ValueError(f"detected finding without a suggested action from {finding.check_id}")
    if not finding.limitations:
        raise ValueError(f"finding without limitations from {finding.check_id}")


def run_checks(specs: list[CheckSpec], ctx: CheckContext) -> list[Finding]:
    findings: list[Finding] = []
    for spec in specs:
        reason = unmet_requirement(spec.requirements, ctx)
        if reason is not None:
            findings.append(not_assessed(spec, reason))
            continue
        try:
            produced = resolve(spec)(ctx, spec)
            if not produced:  # e.g. every declared column was empty
                produced = [not_assessed(spec, "the declared columns have no values to check.")]
            for finding in produced:
                validate(finding)
            findings.extend(produced)
        except MemoryError:
            raise  # the job's memory cap: fail the job as LIMIT_EXCEEDED
        except Exception as exc:  # noqa: BLE001  (isolated per check by design)
            findings.append(_error(spec, exc))
    return findings


SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITIES)}


def summarise(findings: list[Finding]) -> dict[str, Any]:
    by_status = dict.fromkeys(STATUSES, 0)
    by_severity = dict.fromkeys(SEVERITIES, 0)
    by_domain: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for f in findings:
        by_status[f.status] += 1
        if f.status == "detected":
            by_severity[f.severity] += 1
            by_domain[f.domain] = by_domain.get(f.domain, 0) + 1
            by_category[f.category] = by_category.get(f.category, 0) + 1
    return {
        "total": len(findings),
        "by_status": by_status,
        "detected_by_severity": by_severity,
        "detected_by_domain": by_domain,
        "detected_by_category": by_category,
        "check_errors": by_status["error"],
    }

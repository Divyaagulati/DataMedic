"""Declarations for checks, roles and domain modules. Standard library only: the API imports
these to list checks and validate roles without loading data libraries."""

from dataclasses import dataclass, field

SEVERITIES = ("critical", "high", "medium", "low", "info")
STATUSES = ("detected", "not_detected", "not_assessed", "error")
TASK_TYPES = ("binary", "multiclass", "regression")

UNIVERSAL = "universal"
GENERIC = "generic"  # dataset domain with universal checks only

NUMERIC_TYPES = frozenset({"numeric"})
TIME_TYPES = frozenset({"datetime"})
ANY_PRESENT = frozenset({"numeric", "boolean", "datetime", "categorical", "text", "identifier"})


@dataclass(frozen=True)
class RoleSpec:
    key: str
    name: str
    description: str
    domain: str  # "universal" or a module key
    allowed_types: frozenset[str] = ANY_PRESENT
    max_per_version: int | None = None
    is_feature: bool = True  # False: excluded from features (targets, identifiers, times)
    is_sensitive: bool = False  # True: never quote example values in evidence


@dataclass(frozen=True)
class Requirements:
    """Evaluated before a check runs; the first unmet one yields ``not_assessed``."""

    roles: frozenset[str] = frozenset()  # all of these roles must be declared
    roles_any: frozenset[str] = frozenset()  # at least one of these
    task_types: frozenset[str] | None = None  # target task type must be one of these
    min_rows: int = 1


@dataclass(frozen=True)
class CheckSpec:
    id: str
    version: str
    domain: str
    category: str
    title: str
    description: str
    impl: str  # "package.module:function"
    requirements: Requirements = field(default_factory=Requirements)
    limitations: tuple[str, ...] = ()
    expensive: bool = False  # sampled to the analysis budget


@dataclass(frozen=True)
class DomainSpec:
    key: str
    name: str
    version: str
    description: str
    roles: tuple[RoleSpec, ...] = ()
    checks: tuple[CheckSpec, ...] = ()

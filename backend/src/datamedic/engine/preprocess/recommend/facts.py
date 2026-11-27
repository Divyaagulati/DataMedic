"""Facts about a version, assembled from its schema, profile, declared roles and diagnostic
findings (standard library only).

Rules read these facts, never column names: a recommendation is triggered by measured
evidence (missing share, skewness, extreme-value counts, class counts, correlations) and by
declared roles, so renaming a column never changes what is recommended for it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from datamedic.engine.checks.spec import RoleSpec
from datamedic.engine.preprocess.state import ValidationContext

# Roles whose rows belong together (several rows per entity): splits keep them on one side.
ENTITY_ROLES = ("group", "bio.patient_id", "fin.account_id", "ind.asset_id", "biz.customer_id")
# Roles that order rows in time: splits train on the past and test on later rows.
TIME_ROLES = ("time", "fin.timestamp", "ind.timestamp")
CLASSIFICATION = ("binary", "multiclass")
FEATURE_TYPES = ("numeric", "boolean", "categorical", "datetime", "text")


@dataclass(frozen=True)
class FindingRef:
    """A stored diagnostic finding, as the rules need it."""

    id: str
    check_id: str
    status: str  # detected | not_detected | not_assessed | error
    severity: str
    columns: tuple[str, ...]
    evidence: Mapping[str, Any]
    title: str = ""


@dataclass(frozen=True)
class RecommendInput:
    schema: Sequence[Mapping[str, Any]]
    roles: Mapping[str, Sequence[str]]  # role -> columns
    role_specs: Mapping[str, RoleSpec]
    domain: str
    task_type: str | None
    n_rows: int
    profile_columns: Sequence[Mapping[str, Any]] | None = None  # None: no profile yet
    profile_correlations: Mapping[str, Any] | None = None
    findings: Sequence[FindingRef] | None = None  # None: diagnostics have not run


@dataclass(frozen=True)
class Column:
    name: str
    inferred_type: str
    dtype: str
    roles: tuple[str, ...]
    protected: bool  # target, identifier, group, time, weight, excluded...
    domain_roles: tuple[str, ...]  # feature roles of a domain module (e.g. bio.age)
    sensitive: bool
    n_missing: int
    missing_pct: float
    n_present: int
    n_unique: int | None
    numeric: Mapping[str, Any] | None  # profile numeric statistics
    frequencies: Mapping[str, Any] | None  # profile frequency table (capped)

    @property
    def feature(self) -> bool:
        return not self.protected and self.inferred_type in FEATURE_TYPES

    @property
    def unique_pct(self) -> float | None:
        if self.n_unique is None or self.n_present == 0:
            return None
        return 100.0 * self.n_unique / self.n_present

    def stat(self, name: str) -> Any:
        return None if self.numeric is None else self.numeric.get(name)


class Facts:
    def __init__(self, inp: RecommendInput) -> None:
        self.inp = inp
        roles = {role: list(columns) for role, columns in inp.roles.items()}
        self.context = ValidationContext.from_schema(
            inp.schema, roles, inp.role_specs, n_rows=inp.n_rows, task_type=inp.task_type
        )
        self.target = self.context.target
        self.task = self.context.task_type
        self.n_rows = int(inp.n_rows)
        by_role: dict[str, list[str]] = {}
        roles_of: dict[str, list[str]] = {}
        for role, columns in inp.roles.items():
            for column in columns:
                by_role.setdefault(role, []).append(column)
                roles_of.setdefault(column, []).append(role)
        self.by_role = by_role
        profile = {str(c["name"]): c for c in inp.profile_columns or ()}
        self.columns: dict[str, Column] = {}
        for entry in inp.schema:
            name = str(entry["name"])
            prof = profile.get(name, {})
            n_missing = int(entry.get("n_missing") or prof.get("n_missing") or 0)
            n_present = max(self.n_rows - n_missing, 0)
            n_unique = entry.get("n_unique", prof.get("n_unique"))
            self.columns[name] = Column(
                name=name,
                inferred_type=str(entry.get("inferred_type", "")),
                dtype=str(entry.get("dtype", "")),
                roles=tuple(sorted(roles_of.get(name, []))),
                protected=name in self.context.protected,
                domain_roles=tuple(self.context.domain_roles.get(name, ())),
                sensitive=name in self.context.sensitive,
                n_missing=n_missing,
                missing_pct=round(100.0 * n_missing / self.n_rows, 3) if self.n_rows else 0.0,
                n_present=n_present,
                n_unique=int(n_unique) if n_unique is not None else None,
                numeric=prof.get("numeric"),
                frequencies=prof.get("frequencies"),
            )
        self._findings: dict[str, list[FindingRef]] = {}
        for f in inp.findings or ():
            self._findings.setdefault(f.check_id, []).append(f)

    # -- Columns ---------------------------------------------------------------------------

    def features(self, *types: str) -> list[Column]:
        return [
            c
            for c in self.columns.values()
            if c.feature and (not types or c.inferred_type in types)
        ]

    def role_columns(self, roles: Sequence[str]) -> list[str]:
        out: list[str] = []
        for role in roles:
            out.extend(c for c in self.by_role.get(role, []) if c not in out)
        return out

    @property
    def classification(self) -> bool:
        return self.target is not None and self.task in CLASSIFICATION

    @property
    def supervised(self) -> bool:
        return self.target is not None and self.task is not None

    @property
    def has_profile(self) -> bool:
        return self.inp.profile_columns is not None

    @property
    def has_diagnostics(self) -> bool:
        return self.inp.findings is not None

    # -- Findings --------------------------------------------------------------------------

    def detected(self, check_id: str, column: str | None = None) -> list[FindingRef]:
        return [
            f
            for f in self._findings.get(check_id, [])
            if f.status == "detected" and (column is None or column in f.columns)
        ]

    def findings_of(self, check_id: str) -> list[FindingRef]:
        return list(self._findings.get(check_id, []))

    def check_ran(self, check_id: str) -> bool:
        """True when the check produced a definite answer (detected or not detected)."""
        return any(
            f.status in ("detected", "not_detected") for f in self._findings.get(check_id, [])
        )

    # -- Correlations (profile) ------------------------------------------------------------

    def strongest_correlation(self, column: str, among: Sequence[str]) -> float | None:
        """Largest absolute Pearson correlation between ``column`` and another column in
        ``among`` (profile matrix; pairwise complete rows). None when not available."""
        corr = self.inp.profile_correlations or {}
        if corr.get("status") != "ok":
            return None
        names = [str(n) for n in corr.get("columns", [])]
        matrix = corr.get("pearson") or []
        if column not in names:
            return None
        i = names.index(column)
        best: float | None = None
        for other in among:
            if other == column or other not in names:
                continue
            value = matrix[i][names.index(other)] if i < len(matrix) else None
            if value is None:
                continue
            best = abs(float(value)) if best is None else max(best, abs(float(value)))
        return best

    @property
    def correlations_sampled(self) -> bool:
        return bool((self.inp.profile_correlations or {}).get("sampled"))

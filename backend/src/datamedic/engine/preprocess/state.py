"""Validation state types (standard library only): the schema as seen by validation, the
validation context built from a version's schema and roles, and validation issues."""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from datamedic.engine.checks.spec import RoleSpec


@dataclass(frozen=True)
class ColumnState:
    name: str
    inferred_type: str
    dtype: str  # dtype label: integer | float | boolean | string | datetime | datetime_utc
    n_missing: int | None = None  # known for source columns; None once a step may change it


TARGET_ROLE = "target"
UNIVERSAL_DOMAIN = "universal"


@dataclass(frozen=True)
class ValidationContext:
    columns: tuple[ColumnState, ...]
    protected: Mapping[str, tuple[str, ...]] = field(default_factory=dict)  # column -> roles
    sensitive: frozenset[str] = frozenset()
    n_rows: int | None = None
    target: str | None = None  # the column with the target role, if declared
    task_type: str | None = None  # binary | multiclass | regression (declared with the target)
    # Feature columns with a domain role (e.g. a biomarker, age, sex): column -> roles. Feature
    # selection never removes them automatically.
    domain_roles: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @classmethod
    def from_schema(
        cls,
        schema: Sequence[Mapping[str, Any]],
        roles: Mapping[str, Iterable[str]],
        role_specs: Mapping[str, RoleSpec],
        *,
        n_rows: int | None = None,
        task_type: str | None = None,
    ) -> "ValidationContext":
        """Build from a version schema and declared roles (role -> columns).

        A column is protected when any of its roles is not a feature role (target, id, group,
        time, weight, exclude, domain identifiers...). Unknown role keys are treated as
        protected (fail safe)."""
        protected: dict[str, list[str]] = {}
        sensitive: set[str] = set()
        domain: dict[str, list[str]] = {}
        for role, columns in roles.items():
            spec = role_specs.get(role)
            for column in columns:
                if spec is None or not spec.is_feature:
                    protected.setdefault(column, []).append(role)
                elif spec.domain != UNIVERSAL_DOMAIN:
                    domain.setdefault(column, []).append(role)
                if spec is not None and spec.is_sensitive:
                    sensitive.add(column)
        targets = list(dict.fromkeys(roles.get(TARGET_ROLE, ())))
        return cls(
            columns=tuple(
                ColumnState(
                    name=str(c["name"]),
                    inferred_type=str(c["inferred_type"]),
                    dtype=str(c.get("dtype", "")),
                    n_missing=c.get("n_missing"),
                )
                for c in schema
            ),
            protected={k: tuple(sorted(v)) for k, v in protected.items()},
            sensitive=frozenset(sensitive),
            n_rows=n_rows,
            target=targets[0] if len(targets) == 1 else None,
            task_type=task_type if len(targets) == 1 else None,
            domain_roles={k: tuple(sorted(v)) for k, v in domain.items()},
        )


@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    step: int | None = None  # 0-based step index
    column: str | None = None
    param: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "step": self.step,
            "column": self.column,
            "param": self.param,
        }


def is_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )

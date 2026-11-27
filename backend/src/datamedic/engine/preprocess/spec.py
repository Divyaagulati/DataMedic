"""Declarations for preprocessing operations and pipeline steps.

Standard library only: the API imports these to list the catalogue and validate steps
without loading pandas or scikit-learn (implementations are referenced as
``"module:Class"`` and imported only where data is processed).

Terms (ML_ENGINE_DESIGN §3.1, PHASE4_PLAN §2.1):
- **risk**: low | medium | high. Fitted value-introducing ops are medium; removing rows or
  columns or overwriting observed values is high.
- **destructive**: removes rows or columns, or overwrites observed values.
- **kind**: how an op behaves around a train/test split. ``fitted`` ops learn from the rows
  passed to ``fit`` only; ``stateless_*`` ops learn nothing; ``training_only`` ops (resampling,
  class weights) change only the training table built by ``fit_resample`` and leave every
  other row set unchanged.
- **stage**: ``data`` (may be committed into a version), ``model`` (only inside a model's
  training-fold pipeline; never committed), or ``either``.
"""

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

RISKS = ("low", "medium", "high")
KINDS = (
    "stateless_row",
    "stateless_column",
    "fitted",
    "row_filter_rule",
    "row_filter_fitted",
    "training_only",
)
STAGES = ("data", "model", "either")
FAMILIES = (
    "missing",
    "outliers",
    "encoding",
    "scaling",
    "transform",
    "selection",
    "imbalance",
)

PARAM_TYPES = ("int", "float", "bool", "choice", "scalar", "columns", "text", "text_list")
"""``scalar`` = a JSON number, string or boolean whose type must match the column;
``columns`` = a list of column names; ``text`` = a short string (e.g. an output column name);
``text_list`` = an ordered list of distinct short strings (e.g. an explicit category order)."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    description: str
    default: Any = None
    required: bool = False
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.type not in PARAM_TYPES:
            raise ValueError(f"unknown param type {self.type!r}")


@dataclass(frozen=True)
class OpSpec:
    op: str
    version: str
    family: str
    title: str
    description: str
    when_to_use: str
    downsides: tuple[str, ...]
    limitations: tuple[str, ...]
    stage: str
    kind: str
    risk: str
    destructive: bool
    column_types: frozenset[str]  # semantic types the op accepts in ``columns``
    impl: str  # "package.module:Class" (a DMTransformer subclass)
    params: tuple[ParamSpec, ...] = ()
    adds_columns: bool = False
    max_columns: int = 500
    # False: the op works on whole rows (resampling, class weights); the request selects no
    # columns and validation resolves the target and the columns passed through unchanged.
    uses_columns: bool = True
    needs_target: bool = False  # reads the target column (fitting rows only)

    def __post_init__(self) -> None:
        if self.risk not in RISKS or self.kind not in KINDS or self.stage not in STAGES:
            raise ValueError(f"invalid classification for {self.op}")
        if self.kind == "training_only" and self.stage != "model":
            raise ValueError(f"training-only op {self.op} must be model stage")
        if self.family not in FAMILIES:
            raise ValueError(f"unknown family for {self.op}")
        if self.destructive and self.risk != "high":
            raise ValueError(f"destructive op {self.op} must be high risk")

    def param(self, name: str) -> ParamSpec | None:
        return next((p for p in self.params if p.name == name), None)


@dataclass(frozen=True)
class Step:
    """One requested transformation. ``params`` are JSON primitives (or lists of strings for
    ``columns`` params). After validation, params are complete (defaults filled)."""

    op: str
    columns: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"op": self.op, "columns": list(self.columns), "params": dict(self.params)}


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN/Infinity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def step_hash(step: Step, op_version: str) -> str:
    """SHA-256 (hex) of the canonical step: op, op version, columns (ordered) and params."""
    payload = {
        "op": step.op,
        "op_version": op_version,
        "columns": list(step.columns),
        "params": step.params,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def spec_hash(step_hashes: list[str]) -> str:
    """SHA-256 (hex) of the ordered step hashes. Any edit, insert, delete or reorder changes it."""
    return hashlib.sha256(canonical_json(step_hashes).encode()).hexdigest()

"""Validate preprocessing steps against a version's schema and roles (standard library only).

Validation is the only way to obtain executable steps: it fills defaults, resolves
predictor columns, tracks the schema through the steps (added columns, dtype changes) and
computes step and spec hashes. Rules enforced here (PHASE4_PLAN §9.1):
- unknown ops, unknown or ill-typed params, missing required params are errors;
- columns must exist at that point of the pipeline, match the op's column types, and never
  be ``_dm_row_id`` or a protected (non-feature) role column such as the target;
- protected columns are never used as predictors;
- op-specific checks and schema effects live in ``rules.py``;
- issue messages never contain data values (column names are carried in ``column``).
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from datamedic.engine import ROW_ID
from datamedic.engine.preprocess.catalog import Limits, get_op
from datamedic.engine.preprocess.rules import (
    FIT_ROW_CAPS,
    RESERVED_PREFIX,
    RULES,
    RuleContext,
)
from datamedic.engine.preprocess.spec import OpSpec, ParamSpec, Step, spec_hash, step_hash
from datamedic.engine.preprocess.state import (
    ColumnState,
    Issue,
    ValidationContext,
    is_number,
)

__all__ = [
    "ColumnState",
    "Issue",
    "ValidatedStep",
    "ValidationContext",
    "ValidationResult",
    "validate_steps",
]


@dataclass(frozen=True)
class ValidatedStep:
    index: int  # position in the submitted steps
    step: Step  # normalised: defaults filled, predictors resolved
    spec: OpSpec
    step_hash: str


@dataclass
class ValidationResult:
    steps: list[ValidatedStep]
    errors: list[Issue]
    warnings: list[Issue]
    output_columns: tuple[ColumnState, ...]
    sensitive: frozenset[str]
    protected: frozenset[str] = frozenset()  # non-feature role columns (never changed)
    target: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def spec_hash(self) -> str | None:
        return spec_hash([s.step_hash for s in self.steps]) if self.valid else None


# ---- Parameter checks -------------------------------------------------------------------------


def _check_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return "must be text"
    if not value.strip():
        return "must not be empty or blank"
    if len(value) > Limits.MAX_TEXT:
        return f"must be at most {Limits.MAX_TEXT} characters"
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        return "must not contain control characters"
    return None


def _check_param(p: ParamSpec, value: Any) -> str | None:
    """Return an error message or None."""
    if p.type == "int":
        if not isinstance(value, int) or isinstance(value, bool):
            return "must be a whole number"
    elif p.type == "float":
        if not is_number(value):
            return "must be a finite number"
    elif p.type == "bool":
        if not isinstance(value, bool):
            return "must be true or false"
    elif p.type == "choice":
        if value not in (p.choices or ()):
            return f"must be one of: {', '.join(p.choices or ())}"
    elif p.type == "scalar":
        if not (isinstance(value, str | bool) or is_number(value)):
            return "must be a finite number, text or true/false"
    elif p.type == "columns":
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            return "must be a list of column names"
        if len(set(value)) != len(value):
            return "lists a column more than once"
    elif p.type == "text":
        return _check_text(value)
    elif p.type == "text_list":
        if not isinstance(value, list) or not value:
            return "must be a non-empty list of text values"
        if len(value) > Limits.MAX_TEXT_LIST:
            return f"must have at most {Limits.MAX_TEXT_LIST} items"
        for item in value:
            problem = _check_text(item)
            if problem:
                return f"has an item that {problem}"
        if len(set(value)) != len(value):
            return "lists a value more than once"
    if p.type in ("int", "float") and is_number(value):
        if p.minimum is not None and value < p.minimum:
            return f"must be at least {p.minimum:g}"
        if p.maximum is not None and value > p.maximum:
            return f"must be at most {p.maximum:g}"
    return None


# ---- Validation -------------------------------------------------------------------------------


def validate_steps(raw_steps: Sequence[Step], ctx: ValidationContext) -> ValidationResult:
    errors: list[Issue] = []
    warnings: list[Issue] = []
    validated: list[ValidatedStep] = []
    columns: dict[str, ColumnState] = {c.name: c for c in ctx.columns}  # insertion-ordered
    reserved_prefixes: set[str] = set()
    maybe_removed: dict[str, ColumnState] = {}

    def result(steps: list[ValidatedStep]) -> ValidationResult:
        return ValidationResult(
            steps,
            errors,
            warnings,
            tuple(columns.values()),
            ctx.sensitive,
            frozenset(ctx.protected),
            ctx.target,
        )

    if len(raw_steps) > Limits.MAX_STEPS:
        errors.append(Issue("TOO_MANY_STEPS", f"A pipeline has at most {Limits.MAX_STEPS} steps."))
        return result([])

    for index, raw in enumerate(raw_steps):
        step_errors: list[Issue] = []

        def err(
            code: str,
            message: str,
            *,
            column: str | None = None,
            param: str | None = None,
            _into: list[Issue] = step_errors,
            _step: int = index,
        ) -> None:
            _into.append(Issue(code, message, _step, column, param))

        spec = get_op(raw.op)
        if spec is None:
            err("UNKNOWN_OP", "This operation does not exist.")
            errors.extend(step_errors)
            continue

        # Parameters: unknown keys, required, types; defaults filled.
        params: dict[str, Any] = {}
        known = {p.name for p in spec.params}
        for key in raw.params:
            if key not in known:
                err("UNKNOWN_PARAM", "This operation has no such parameter.", param=key)
        for p in spec.params:
            if p.name in raw.params and raw.params[p.name] is not None:
                value = raw.params[p.name]
                problem = _check_param(p, value)
                if problem:
                    err("PARAM_INVALID", f"'{p.name}' {problem}.", param=p.name)
                params[p.name] = value
            elif p.required:
                err("PARAM_REQUIRED", f"'{p.name}' is required.", param=p.name)
            else:
                params[p.name] = p.default

        # Columns. Row-level ops (resampling, class weights) select none.
        selected = list(raw.columns)
        if not spec.uses_columns:
            if selected:
                err(
                    "COLUMNS_NOT_APPLICABLE",
                    "This operation works on whole training rows; select no columns.",
                )
            selected = []
        elif not selected:
            err("COLUMNS_REQUIRED", "Select at least one column.")
        if len(selected) > spec.max_columns:
            err("TOO_MANY_COLUMNS", f"Select at most {spec.max_columns} column(s).")
        if len(set(selected)) != len(selected):
            err("DUPLICATE_COLUMN", "A column is selected more than once.")
        for name in dict.fromkeys(selected):
            state = columns.get(name)
            if name == ROW_ID or name.startswith(RESERVED_PREFIX):
                err(
                    "RESERVED_COLUMN",
                    "Internal columns (the row identifier and other reserved columns) cannot "
                    "be transformed.",
                    column=name,
                )
            elif state is None:
                err(
                    "COLUMN_NOT_FOUND",
                    "Column not found at this point of the pipeline.",
                    column=name,
                )
            elif name in ctx.protected:
                err(
                    "COLUMN_PROTECTED",
                    "This column has a role that is not a feature ("
                    + ", ".join(ctx.protected[name])
                    + ") and is never transformed.",
                    column=name,
                )
            elif state.inferred_type not in spec.column_types:
                err(
                    "COLUMN_TYPE_UNSUPPORTED",
                    f"This operation does not support {state.inferred_type} columns.",
                    column=name,
                )

        # Op-specific checks (rules.py).
        check, effect = RULES.get(spec.op, (None, None))
        rc = RuleContext(
            index=index,
            spec=spec,
            selected=selected,
            params=params,
            explicit=frozenset(k for k, v in raw.params.items() if v is not None),
            columns=columns,
            reserved_prefixes=reserved_prefixes,
            vctx=ctx,
            maybe_removed=maybe_removed,
        )
        if check is not None:
            check(rc)
        step_errors.extend(rc.errors)
        warnings.extend(rc.warnings)

        if step_errors:
            errors.extend(step_errors)
            continue

        step = Step(op=spec.op, columns=tuple(selected), params=rc.params)
        validated.append(ValidatedStep(index, step, spec, step_hash(step, spec.version)))
        if effect is not None:
            effect(rc)  # track the schema through the step

    # Training-only ops (resampling, class weights) change the training table that the model
    # is fitted on, so nothing may be fitted after them: at most one, and it must be last.
    training_only = [i for i, raw in enumerate(raw_steps) if _kind(raw.op) == "training_only"]
    if len(training_only) > 1:
        errors.append(
            Issue(
                "TOO_MANY_TRAINING_ONLY",
                "A pipeline has at most one resampling or class-weight step.",
                training_only[1],
            )
        )
    elif training_only and training_only[0] != len(raw_steps) - 1:
        errors.append(
            Issue(
                "TRAINING_ONLY_NOT_LAST",
                "A resampling or class-weight step must be the last step of the pipeline.",
                training_only[0],
            )
        )

    if ctx.n_rows is not None:
        for v in validated:
            row_cap = FIT_ROW_CAPS.get(v.step.op)
            if row_cap is not None and ctx.n_rows > row_cap:
                warnings.append(
                    Issue(
                        "MAY_EXCEED_LIMIT",
                        f"This operation fits on at most {row_cap:,} rows; fitting on all rows of "
                        "this version will fail. Training folds may be small enough.",
                        v.index,
                    )
                )

    return result(validated)


def _kind(op: str) -> str | None:
    spec = get_op(op)
    return None if spec is None else spec.kind

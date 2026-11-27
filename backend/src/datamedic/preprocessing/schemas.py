from typing import Any, Literal

from pydantic import BaseModel

from datamedic.engine.preprocess.spec import OpSpec, ParamSpec

Risk = Literal["low", "medium", "high"]
Kind = Literal[
    "stateless_row",
    "stateless_column",
    "fitted",
    "row_filter_rule",
    "row_filter_fitted",
    "training_only",
]
Stage = Literal["data", "model", "either"]


class OpParamOut(BaseModel):
    name: str
    type: Literal["int", "float", "bool", "choice", "scalar", "columns", "text", "text_list"]
    description: str
    default: Any = None
    required: bool
    minimum: float | None
    maximum: float | None
    choices: list[str] | None

    @classmethod
    def build(cls, p: ParamSpec) -> "OpParamOut":
        return cls(
            name=p.name,
            type=p.type,
            description=p.description,
            default=p.default,
            required=p.required,
            minimum=p.minimum,
            maximum=p.maximum,
            choices=list(p.choices) if p.choices is not None else None,
        )


class OpOut(BaseModel):
    op: str
    version: str
    family: str
    title: str
    description: str
    when_to_use: str
    downsides: list[str]
    limitations: list[str]
    stage: Stage
    kind: Kind
    risk: Risk
    destructive: bool
    column_types: list[str]
    adds_columns: bool
    max_columns: int
    uses_columns: bool  # False: works on whole training rows (resampling, class weights)
    needs_target: bool  # reads the target column of the training rows
    params: list[OpParamOut]

    @classmethod
    def build(cls, s: OpSpec) -> "OpOut":
        return cls(
            op=s.op,
            version=s.version,
            family=s.family,
            title=s.title,
            description=s.description,
            when_to_use=s.when_to_use,
            downsides=list(s.downsides),
            limitations=list(s.limitations),
            stage=s.stage,
            kind=s.kind,
            risk=s.risk,
            destructive=s.destructive,
            column_types=sorted(s.column_types),
            adds_columns=s.adds_columns,
            max_columns=s.max_columns,
            uses_columns=s.uses_columns,
            needs_target=s.needs_target,
            params=[OpParamOut.build(p) for p in s.params],
        )


class OpListOut(BaseModel):
    items: list[OpOut]

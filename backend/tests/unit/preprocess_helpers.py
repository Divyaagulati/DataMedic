"""Shared helpers for preprocessing tests (synthetic data only)."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from datamedic.domains.bootstrap import build_registry
from datamedic.engine import ROW_ID
from datamedic.engine.preprocess.executor import PreprocessExecutor, TransformResult
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.table_schema import schema_of
from datamedic.engine.preprocess.validate import (
    ValidationContext,
    ValidationResult,
    validate_steps,
)

ROLE_SPECS = {r.key: r for r in build_registry().all_roles()}


def frame(**columns: Any) -> pd.DataFrame:
    """Canonical frame: ``_dm_row_id`` first, then the given columns."""
    data = pd.DataFrame(columns)
    data.insert(0, ROW_ID, np.arange(len(data), dtype="int64"))
    return data


def mixed_frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """Numeric (int + float), categorical, boolean, a target and a protected id, with gaps."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50, 10, n)
    x2 = x1 * 0.8 + rng.normal(0, 2, n)
    age = pd.array(rng.integers(18, 90, n), dtype="Int64")
    city = pd.Series(rng.choice(["north", "south", "east"], n), dtype=object)
    flag = pd.array(rng.random(n) > 0.4, dtype="boolean")
    target = (x1 > 50).astype("int64")
    df = frame(
        patient=[f"p{i:03d}" for i in range(n)],
        age=age,
        x1=x1,
        x2=x2,
        city=city,
        flag=flag,
        outcome=target,
    )
    for column, share in (("age", 0.2), ("x1", 0.15), ("x2", 0.25), ("city", 0.2), ("flag", 0.2)):
        mask = rng.random(n) < share
        mask[0] = False  # keep at least one observed value per column
        df.loc[mask, column] = None
    df["city"] = df["city"].astype(object).where(df["city"].notna(), None)
    return df


MIXED_ROLES: dict[str, list[str]] = {"target": ["outcome"], "id": ["patient"]}


def context(
    df: pd.DataFrame,
    roles: Mapping[str, Sequence[str]] | None = None,
    types: Mapping[str, str] | None = None,
    task: str | None = None,
) -> ValidationContext:
    schema = schema_of(df, types or {})
    return ValidationContext.from_schema(
        schema,
        {k: list(v) for k, v in (roles or {}).items()},
        ROLE_SPECS,
        n_rows=len(df),
        task_type=task,
    )


def validated(
    df: pd.DataFrame,
    steps: Sequence[Step],
    roles: Mapping[str, Sequence[str]] | None = None,
    types: Mapping[str, str] | None = None,
    task: str | None = None,
) -> ValidationResult:
    result = validate_steps(list(steps), context(df, roles, types, task))
    assert result.valid, [e.as_dict() for e in result.errors]
    return result


def run(
    df: pd.DataFrame,
    steps: Sequence[Step],
    *,
    fit_rows: Any = None,
    roles: Mapping[str, Sequence[str]] | None = None,
    seed: int = 0,
    task: str | None = None,
) -> tuple[TransformResult, PreprocessExecutor]:
    result = validated(df, steps, roles, task=task)
    executor = PreprocessExecutor(result, seed=seed)
    executor.fit(df if fit_rows is None else df.loc[fit_rows])
    return executor.transform(df), executor

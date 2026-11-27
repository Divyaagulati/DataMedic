"""Helpers for diagnostics tests: build a realistic CheckContext from a DataFrame.

Columns go through the real canonicalisation (text -> type inference), so schemas, dtypes and
semantic types match what the worker sees after parsing.
"""

from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from datamedic.domains.bootstrap import build_registry
from datamedic.engine.checks.framework import CheckContext, Finding, RoleMap, run_checks
from datamedic.engine.table import RawColumns, build_canonical, to_text

NOW = datetime(2026, 11, 9, tzinfo=UTC)


def make_ctx(
    frame: pd.DataFrame,
    roles: dict[str, list[str]] | None = None,
    task_type: str | None = None,
    domain: str = "generic",
) -> CheckContext:
    raw = RawColumns(
        names=list(frame.columns),
        columns=[frame[c].map(to_text).astype(object) for c in frame.columns],
    )
    table, _ = build_canonical(raw)
    registry = build_registry()
    return CheckContext(
        frame=table.frame,
        schema=[c.as_dict() for c in table.columns],
        roles=RoleMap(dict(roles or {}), task_type),
        domain=domain,
        role_specs={r.key: r for r in registry.all_roles()},
        now=NOW,
    )


def run_domain(ctx: CheckContext) -> list[Finding]:
    return run_checks(build_registry().checks_for(ctx.domain), ctx)


def by_check(findings: list[Finding], check_id: str) -> list[Finding]:
    return [f for f in findings if f.check_id == check_id]


def detected_ids(findings: list[Finding]) -> set[str]:
    return {f.check_id for f in findings if f.status == "detected"}


def clean_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A well-behaved dataset: Gaussian features, balanced target, no missing values."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(50, 10, n).round(3)
    x2 = rng.normal(0, 1, n).round(4)
    x3 = rng.normal(120, 15, n).round(2)
    group = rng.choice(["north", "south", "east", "west"], n)
    flag = rng.choice(["yes", "no"], n)
    logit = 0.04 * (x1 - 50) + 0.5 * x2
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "region": group, "flag": flag, "y": y})


def summary(findings: list[Finding]) -> list[tuple[str, str, str]]:
    return [(f.check_id, f.status, f.severity) for f in findings]


def jsonable(findings: list[Finding]) -> list[dict[str, Any]]:
    import json

    out = [f.as_dict() for f in findings]
    json.dumps(out, allow_nan=False)  # must be storable as JSONB
    return out

"""Build recommendation inputs from a synthetic frame with the real profiler and checks."""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from datamedic.domains.bootstrap import build_registry
from datamedic.engine.adapters.builtin_profiler import BuiltinProfiler
from datamedic.engine.checks.framework import CheckContext, RoleMap, run_checks
from datamedic.engine.checks.spec import GENERIC
from datamedic.engine.preprocess.recommend import FindingRef, RecommendInput, recommend
from datamedic.engine.preprocess.table_schema import schema_of
from tests.unit.preprocess_helpers import frame

REGISTRY = build_registry()
ROLE_SPECS = {r.key: r for r in REGISTRY.all_roles()}


def make_input(
    df: pd.DataFrame,
    roles: Mapping[str, Sequence[str]] | None = None,
    *,
    task: str | None = None,
    domain: str = GENERIC,
    types: Mapping[str, str] | None = None,
    diagnostics: bool = True,
    profile: bool = True,
) -> RecommendInput:
    roles = {k: list(v) for k, v in (roles or {}).items()}
    schema = schema_of(df, types or {})
    prof = BuiltinProfiler().profile(df, schema, seed=0) if profile else None
    findings = None
    if diagnostics:
        role_map = RoleMap(columns={k: list(v) for k, v in roles.items()}, task_type=task)
        key = domain if domain in REGISTRY.domain_keys() else GENERIC
        ctx = CheckContext(
            frame=df, schema=schema, roles=role_map, domain=key, role_specs=ROLE_SPECS, seed=0
        )
        findings = [
            FindingRef(
                id=f"f{i}",
                check_id=f.check_id,
                status=f.status,
                severity=f.severity,
                columns=tuple(f.columns),
                evidence=f.evidence,
                title=f.title,
            )
            for i, f in enumerate(run_checks(REGISTRY.checks_for(key), ctx))
        ]
    return RecommendInput(
        schema=schema,
        roles=roles,
        role_specs=ROLE_SPECS,
        domain=domain,
        task_type=task,
        n_rows=len(df),
        profile_columns=prof["columns"] if prof else None,
        profile_correlations=prof["correlations"] if prof else None,
        findings=findings,
    )


def recs_for(
    df: pd.DataFrame, roles: Mapping[str, Sequence[str]] | None = None, **kwargs: Any
) -> dict[str, Any]:
    return recommend(make_input(df, roles, **kwargs))


def clinical_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Skewed positive lab, correlated pair with gaps, an outlier, an imbalanced target."""
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, n)
    lab = np.exp(rng.normal(1, 0.8, n))  # right-skewed, positive
    b = a * 0.9 + rng.normal(0, 0.3, n)  # correlated with a
    noise = rng.normal(0, 1, n)
    y = (rng.random(n) < 0.08).astype(int)
    df = frame(a=a, b=b, lab=lab, noise=noise, y=y)
    df.loc[rng.random(n) < 0.25, "b"] = np.nan
    df.loc[5, "noise"] = 60.0
    df.loc[6, "noise"] = -55.0
    return df

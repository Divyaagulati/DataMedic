"""Check framework: requirement honesty, error isolation, validation, registry extensibility,
and outlier-detector properties."""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

from datamedic.domains.bootstrap import build_registry
from datamedic.engine.adapters.outliers import (
    COPODDetector,
    ECODDetector,
    IsolationForestDetector,
)
from datamedic.engine.checks.framework import (
    Finding,
    detected,
    run_checks,
    unmet_requirement,
)
from datamedic.engine.checks.spec import (
    CheckSpec,
    DomainSpec,
    Requirements,
    RoleSpec,
)
from datamedic.engine.checks.universal import MODULE as UNIVERSAL
from datamedic.engine.registry import Registry
from tests.unit.diag_helpers import make_ctx

REGISTRY = build_registry()
ALL_CHECKS = REGISTRY.all_checks()


def full_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 200
    return pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "y": rng.choice([0, 1], n),
            "t": pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )


def test_registry_lists_every_check_once() -> None:
    ids = [c.id for c in ALL_CHECKS]
    assert len(ids) == len(set(ids)) >= 36
    assert REGISTRY.domain_keys() == [
        "generic",
        "biomedical",
        "financial",
        "business",
        "industrial",
    ]
    for check in ALL_CHECKS:
        assert check.limitations, f"{check.id} declares no limitations"
        assert check.description and check.title
        module_name, func = check.impl.split(":")
        assert hasattr(__import__(module_name, fromlist=[func]), func), check.impl


@pytest.mark.parametrize(
    "check",
    [c for c in ALL_CHECKS if c.requirements.roles or c.requirements.roles_any],
    ids=lambda c: c.id,
)
def test_missing_required_role_is_not_assessed(check: CheckSpec) -> None:
    """Registry-wide honesty: with any required role absent, the result is not_assessed with
    a reason, never 'not detected'."""
    ctx = make_ctx(
        full_frame(), roles={}, domain=check.domain if check.domain != "universal" else "generic"
    )
    (finding,) = run_checks([check], ctx)
    assert finding.status == "not_assessed" and finding.not_assessed_reason
    assert finding.evidence == {}


@pytest.mark.parametrize(
    "check", [c for c in ALL_CHECKS if c.requirements.task_types], ids=lambda c: c.id
)
def test_wrong_task_type_is_not_assessed(check: CheckSpec) -> None:
    allowed = check.requirements.task_types or frozenset()
    wrong = next((t for t in ("binary", "multiclass", "regression") if t not in allowed), None)
    if wrong is None:
        pytest.skip("accepts every task type")
    roles = {r: ["a"] for r in check.requirements.roles}
    ctx = make_ctx(full_frame(), roles=roles, task_type=wrong)
    assert unmet_requirement(check.requirements, ctx) is not None


def test_min_rows_requirement() -> None:
    req = Requirements(min_rows=500)
    assert "500 rows" in (unmet_requirement(req, make_ctx(full_frame())) or "")


def _install_fake_check(name: str, func) -> CheckSpec:  # type: ignore[no-untyped-def]
    module = types.ModuleType(f"fake_checks_{name}")
    module.run = func  # type: ignore[attr-defined]
    sys.modules[module.__name__] = module
    return CheckSpec(
        id=f"test.{name}",
        version="1",
        domain="test",
        category="test",
        title="Fake",
        description="",
        impl=f"{module.__name__}:run",
        limitations=("test",),
    )


def test_exceptions_become_error_findings_without_their_message() -> None:
    def boom(ctx, spec):  # type: ignore[no-untyped-def]
        raise ValueError("could not convert 'SECRET-VALUE'")

    def fine(ctx, spec):  # type: ignore[no-untyped-def]
        return [detected(spec, ctx, "low", "ok", "ok", {"n": 1}, recommendation="none")]

    findings = run_checks(
        [_install_fake_check("boom", boom), _install_fake_check("fine", fine)],
        make_ctx(full_frame()),
    )
    assert [f.status for f in findings] == ["error", "detected"]  # the run continued
    error = findings[0]
    assert error.error_code == "CHECK_ERROR" and error.evidence == {"exception_type": "ValueError"}
    assert "SECRET-VALUE" not in json.dumps(error.as_dict())


def test_invalid_findings_are_rejected() -> None:
    def no_action(ctx, spec):  # type: ignore[no-untyped-def]
        return [
            Finding(
                spec.id,
                "1",
                "test",
                "test",
                "detected",
                "low",
                "t",
                "e",
                {"n": 1},
                limitations=["x"],
            )
        ]

    (finding,) = run_checks([_install_fake_check("no_action", no_action)], make_ctx(full_frame()))
    assert finding.status == "error"  # a detected finding without a suggested action


def test_empty_result_is_not_assessed() -> None:
    (finding,) = run_checks(
        [_install_fake_check("empty", lambda ctx, spec: [])], make_ctx(full_frame())
    )
    assert finding.status == "not_assessed"


def test_memory_errors_propagate_to_the_job() -> None:
    def oom(ctx, spec):  # type: ignore[no-untyped-def]
        raise MemoryError

    with pytest.raises(MemoryError):
        run_checks([_install_fake_check("oom", oom)], make_ctx(full_frame()))


def test_stub_domain_registers_without_core_changes() -> None:
    """DOMAIN_MODULES §2 extensibility proof: a new module needs only a spec + registration."""
    check = _install_fake_check("stub", lambda ctx, spec: [])
    stub = DomainSpec(
        key="stub",
        name="Stub",
        version="0.1",
        description="test module",
        roles=(RoleSpec("stub.thing", "Thing", "", "stub"),),
        checks=(CheckSpec(**{**check.__dict__, "domain": "stub", "id": "stub.check"}),),
    )
    registry = Registry(universal=UNIVERSAL)
    registry.register(stub)
    assert "stub" in registry.domain_keys()
    assert [c.id for c in registry.checks_for("stub")][-1] == "stub.check"
    assert registry.role("stub.thing") is not None
    with pytest.raises(ValueError):
        registry.register(stub)  # duplicate keys are refused


def test_core_never_imports_domain_modules() -> None:
    import ast
    from pathlib import Path

    engine = Path(__file__).resolve().parents[2] / "src" / "datamedic" / "engine"
    for path in engine.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith("datamedic.domains"), path


# ---- Outlier detectors ----------------------------------------------------------------------


DETECTORS = [IsolationForestDetector(), ECODDetector(), COPODDetector()]


@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
def test_detectors_rank_planted_outliers_first(detector) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(500, 4))
    matrix[:5] += 6
    scores = detector.score(matrix, 0)
    assert scores.shape == (500,)
    assert set(np.argsort(-scores)[:5]) == set(range(5))


@pytest.mark.parametrize("detector", [ECODDetector(), COPODDetector()], ids=lambda d: d.name)
def test_ecdf_detectors_are_invariant_to_unit_changes(detector) -> None:  # type: ignore[no-untyped-def]
    """Positive affine rescaling (a unit or offset change) leaves ranks and skewness signs
    unchanged, so scores are identical. Nonlinear transforms (log, exp) can flip a feature's
    skewness and therefore which tail is used: that is by design, not invariance."""
    rng = np.random.default_rng(1)
    matrix = rng.normal(size=(300, 3))
    rescaled = matrix * np.array([100.0, 0.01, 7.5]) + np.array([5.0, -3.0, 1000.0])
    np.testing.assert_allclose(detector.score(matrix, 0), detector.score(rescaled, 0))


@pytest.mark.parametrize("detector", DETECTORS, ids=lambda d: d.name)
def test_detectors_are_deterministic(detector) -> None:  # type: ignore[no-untyped-def]
    matrix = np.random.default_rng(2).normal(size=(200, 3))
    np.testing.assert_array_equal(detector.score(matrix, 7), detector.score(matrix, 7))


def test_ecod_matches_hand_calculation() -> None:
    # One feature [1, 2, 3, 4, 10] (right-skewed): F_left = rank/n, F_right = (n-rank+1)/n.
    # Row 4 (value 10): left = 5/5 -> 0; right = 1/5 -> -log(0.2) = 1.6094; skew > 0 -> right.
    matrix = np.array([[1.0], [2.0], [3.0], [4.0], [10.0]])
    scores = ECODDetector().score(matrix, 0)
    assert scores[4] == pytest.approx(-np.log(0.2))
    assert scores[0] == pytest.approx(-np.log(0.2))  # the left tail also counts (max of sums)
    assert scores[2] == pytest.approx(-np.log(0.6))

"""Part E engine: stage planning, materialisation, impact preview, version comparison and the
generated pipeline.py (verified in an isolated ``python -I`` subprocess). Synthetic data."""

import dataclasses
import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.hashing import content_hash
from datamedic.engine.io.parse import parse_original, write_frame
from datamedic.engine.preprocess import commit as cm
from datamedic.engine.preprocess.compare import compare
from datamedic.engine.preprocess.pipeline_code import Commit, embedded_sources, render_pipeline
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.state import ValidationContext
from datamedic.engine.preprocess.table_schema import schema_of
from datamedic.engine.preprocess.validate import validate_steps
from tests.unit.preprocess_helpers import ROLE_SPECS

ROLES = {"target": ["y"], "sensitive": ["code"]}


def csv_bytes(n: int = 300, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": rng.normal(10, 2, n).round(3),
            "b": rng.normal(size=n).round(3),
            "lab": np.exp(rng.normal(1, 0.7, n)).round(3),
            "code": rng.choice(["k1", "k2", "k3"], n),
            "y": rng.integers(0, 2, n),
        }
    )
    df.loc[rng.random(n) < 0.3, "b"] = None
    df.loc[[3, 7], "a"] = [80.0, -40.0]
    return df.to_csv(index=False).encode()


@pytest.fixture
def original(tmp_path: Path) -> Path:
    path = tmp_path / "original.csv"
    path.write_bytes(csv_bytes())
    return path


def stored(frame: pd.DataFrame) -> pd.DataFrame:
    """A frame as the platform reads a stored version (through Parquet)."""
    buf = io.BytesIO()
    write_frame(frame, buf)
    return pd.read_parquet(io.BytesIO(buf.getvalue()), engine="pyarrow")


def context(schema: list[dict], roles: dict = ROLES) -> ValidationContext:  # type: ignore[type-arg]
    return ValidationContext.from_schema(schema, roles, ROLE_SPECS, task_type="binary")


def planned(*items: tuple[Step, str]) -> list[cm.PlannedStep]:
    return [cm.PlannedStep(s, stage) for s, stage in items]


FIRST = planned(
    (Step("missing.indicator", ("b",)), "data"),
    (Step("impute.median", ("b",)), "data"),
    (Step("outliers.cap", ("a",), {"method": "iqr", "iqr_multiplier": 3.0}), "data"),
    (Step("transform.log", ("lab",)), "data"),
    (Step("scale.standard", ("a", "b")), "model"),
    (Step("imbalance.class_weight", ()), "model"),
)


# ---- Stage planning ---------------------------------------------------------------------------


def test_stages_are_enforced_and_never_moved(original: Path) -> None:
    parsed = parse_original(original, "csv", ".csv")
    ctx = context(parsed.schema)
    plan = cm.plan(FIRST, ctx)
    assert plan.valid and [v.spec.op for v in plan.data.steps] == [
        "missing.indicator",
        "impute.median",
        "outliers.cap",
        "transform.log",
    ]
    assert [s["op"] for s in plan.model_spec()] == ["scale.standard", "imbalance.class_weight"]
    bad = cm.plan(planned((Step("scale.standard", ("a",)), "data")), ctx)
    assert [e.code for e in bad.errors] == ["STAGE_NOT_ALLOWED"]
    order = cm.plan(
        planned((Step("scale.standard", ("a",)), "model"), (Step("impute.median", ("b",)), "data")),
        ctx,
    )
    assert "STAGE_ORDER" in [e.code for e in order.errors]
    either = cm.plan(planned((Step("impute.median", ("b",)), "model")), ctx)
    assert either.valid and not either.data.steps  # 'either' keeps the chosen stage
    assert cm.default_stage("impute.median") == "data"
    assert cm.default_stage("scale.standard") == "model"
    a, b = cm.plan(FIRST, ctx), cm.plan(FIRST[:-1], ctx)
    assert a.spec_hash and a.spec_hash != b.spec_hash
    moved = cm.plan(planned((Step("impute.median", ("b",)), "model")), ctx)
    same = cm.plan(planned((Step("impute.median", ("b",)), "data")), ctx)
    assert moved.step_hashes != same.step_hashes  # the stage is part of the step hash


# ---- Materialisation and preview --------------------------------------------------------------


def test_materialise_matches_preview_and_keeps_rows(original: Path) -> None:
    parsed = parse_original(original, "csv", ".csv")
    source = stored(parsed.table.frame)
    plan = cm.plan(FIRST, context(parsed.schema))
    before = source.copy(deep=True)
    result = cm.materialise(source, plan.data)
    pd.testing.assert_frame_equal(source, before)  # the source is never modified
    assert result.frame[ROW_ID].tolist() == source[ROW_ID].tolist()
    assert result.fitted_on_full_data
    assert "b__missing" in result.frame.columns and result.frame["b"].notna().all()
    assert "_dm_sample_weight" not in result.frame.columns  # model-stage step not applied
    assert result.frame["a"].std() < source["a"].std()  # data-stage cap applied
    np.testing.assert_allclose(result.frame["b"].mean(), result.frame["b"].mean())
    summary, rows = cm.preview(source, plan, sensitive=frozenset({"code"}))
    assert summary["expected_content_hash"] == result.content_hash.hex()
    assert summary["spec_hash"] == plan.spec_hash
    steps = {s["op"]: s for s in summary["steps"]}
    impute = steps["impute.median"]
    assert impute["rows_affected"] == int(source["b"].isna().sum())
    assert impute["columns"][0]["estimates_filled"] == impute["rows_affected"]
    assert "Missing values replaced by estimates." in impute["information_loss"]
    cap = steps["outliers.cap"]
    assert cap["destructive"] and cap["columns"][0]["observed_overwritten"] >= 2
    assert steps["transform.log"]["columns"][0]["observed_re_expressed"] > 0
    assert [m["op"] for m in summary["model_steps"]] == ["scale.standard", "imbalance.class_weight"]
    listed = rows[0]
    assert len(listed["rows"]) == steps["missing.indicator"]["rows_listed"]
    assert summary["totals"]["rows"] == len(source)
    json.dumps(summary, allow_nan=False)
    json.dumps(rows, allow_nan=False)


def test_preview_rows_are_bounded_deterministic_and_redacted(original: Path) -> None:
    parsed = parse_original(original, "csv", ".csv")
    source = stored(parsed.table.frame)
    steps = planned(
        (Step("impute.mode", ("code",)), "data"),
        (Step("missing.indicator", ("b",)), "data"),
        (Step("impute.median", ("b",)), "data"),
    )
    source.loc[source.index[:40], "code"] = None
    plan = cm.plan(steps, context(parse_original(original, "csv", ".csv").schema))
    summary, rows = cm.preview(source, plan, sensitive=frozenset({"code"}), max_rows=10)
    mode = summary["steps"][0]
    assert mode["rows_affected"] == 40 and mode["rows_listed"] == 10 and mode["rows_truncated"]
    assert mode["columns"][0]["before"] == {"redacted": True}
    assert rows[0]["redacted_columns"] == ["code"]
    assert all(r["after"]["code"] == "[redacted]" for r in rows[0]["rows"])
    ids = [r["row_id"] for r in rows[0]["rows"]]
    assert ids == sorted(ids) and ids == sorted(source.loc[source["code"].isna(), ROW_ID])[:10]
    again, rows_again = cm.preview(source, plan, sensitive=frozenset({"code"}), max_rows=10)
    assert again == summary and rows_again == rows


def test_materialise_refuses_invalid_plans_and_changed_rows(original: Path) -> None:
    parsed = parse_original(original, "csv", ".csv")
    bad = cm.plan(planned((Step("scale.standard", ("a",)), "data")), context(parsed.schema))
    with pytest.raises(EngineError):
        cm.preview(stored(parsed.table.frame), bad, sensitive=frozenset())
    with pytest.raises(EngineError):
        cm._check_rows(parsed.table.frame, parsed.table.frame.iloc[:-1])


# ---- Comparison -------------------------------------------------------------------------------


def test_compare_identical_changed_and_unassessable(original: Path) -> None:
    parsed = parse_original(original, "csv", ".csv")
    v0 = stored(parsed.table.frame)
    same = compare(v0, v0)
    assert same["rows_affected"] == 0 and not same["columns_added"]
    assert all(c["values_changed"] == 0 for c in same["columns"])
    ks = next(c for c in same["columns"] if c["column"] == "a")["test"]
    assert ks["status"] == "assessed" and ks["statistic"] == 0
    v1 = cm.materialise(v0, cm.plan(FIRST, context(parsed.schema)).data).frame
    diff = compare(v0, v1)
    assert diff["columns_added"] == ["b__missing"]
    cols = {c["column"]: c for c in diff["columns"]}
    assert cols["b"]["filled"] == int(v0["b"].isna().sum())
    assert cols["lab"]["values_changed"] > 0 and cols["lab"]["test"]["p_value"] < 0.05
    assert cols["code"]["test"]["test"] == "chi_square"
    assert diff["rows_affected"] > 0 and diff["information_loss"]
    const = v0.assign(a=1.0)
    assert compare(const, const)["columns"][0]["test"]["status"] == "not_assessed"
    tiny = compare(v0.head(5), v0.head(5))
    assert all(c["test"]["status"] == "not_assessed" for c in tiny["columns"])
    retyped = v1.assign(code=1.0)
    t = {c["column"]: c for c in compare(v1, retyped.assign(code=v1["code"]))["columns"]}
    assert t["code"]["test"]["status"] == "assessed"
    changed = {c["column"]: c for c in compare(v1, retyped)["columns"]}
    assert changed["code"]["test"]["status"] == "not_assessed"
    assert any("independent" in text for text in diff["limitations"])


# ---- Generated pipeline.py --------------------------------------------------------------------


def run_pipeline(tmp_path: Path, source: str, original: Path) -> tuple[str, pd.DataFrame]:
    script = tmp_path / "pipeline.py"
    script.write_text(source, encoding="utf-8")
    out = tmp_path / "out.parquet"
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(script), str(original), str(out)],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return proc.stdout.strip(), pd.read_parquet(out)


def test_generated_pipeline_reproduces_a_two_commit_lineage(original: Path, tmp_path: Path) -> None:
    parsed = parse_original(original, "csv", ".csv")
    v0 = stored(parsed.table.frame)
    plan1 = cm.plan(FIRST, context(parsed.schema))
    v1 = cm.materialise(v0, plan1.data)
    known = {c["name"]: c["inferred_type"] for c in parsed.schema}
    schema1 = schema_of(v1.frame, known)
    second = planned(
        (Step("outliers.flag", ("lab",)), "data"), (Step("transform.yeo_johnson", ("b",)), "data")
    )
    plan2 = cm.plan(second, context(schema1))
    v2 = cm.materialise(v1.frame, plan2.data)
    model = validate_steps(
        [s.step for s in plan1.model_steps], context(schema_of(v2.frame, known))
    ).steps
    source = render_pipeline(
        version_number=2,
        fmt="csv",
        declared_extension=".csv",
        parse_options=parsed.parse_options,
        commits=[Commit(1, plan1.data.steps), Commit(2, plan2.data.steps)],
        model_steps=model,
        seed=0,
        libraries={"pandas": pd.__version__},
    )
    digest, frame = run_pipeline(tmp_path, source, original)
    assert digest == v2.content_hash.hex()  # exact content-hash equivalence
    assert content_hash(frame) == v2.content_hash
    assert "_dm_sample_weight" not in frame.columns  # the model spec is not applied
    assert "pickle" not in source and "joblib" not in source


def test_generated_pipeline_for_version_zero_and_changed_options(
    original: Path, tmp_path: Path
) -> None:
    parsed = parse_original(original, "csv", ".csv")
    source = render_pipeline(
        version_number=0,
        fmt="csv",
        declared_extension=".csv",
        parse_options=parsed.parse_options,
        commits=[],
        model_steps=[],
        seed=0,
        libraries={},
    )
    digest, _ = run_pipeline(tmp_path, source, original)
    assert digest == parsed.content_hash.hex()
    wrong = render_pipeline(
        version_number=0,
        fmt="csv",
        declared_extension=".csv",
        parse_options={**parsed.parse_options, "delimiter": ";"},
        commits=[],
        model_steps=[],
        seed=0,
        libraries={},
    )
    with pytest.raises(subprocess.CalledProcessError):
        run_pipeline(tmp_path, wrong, original)


def test_embedded_sources_have_no_internal_imports_or_conflicts() -> None:
    source = embedded_sources()
    assert "from datamedic" not in source and "import datamedic" not in source
    compile(source, "embedded", "exec")


def test_hostile_column_names_stay_literals(tmp_path: Path) -> None:
    name = 'x"); __import__("os").system("echo pwned"); ("'
    df = pd.DataFrame({name: [1.0, None, 3.0] * 10, "y": [0, 1, 0] * 10})
    path = tmp_path / "h.csv"
    path.write_bytes(df.to_csv(index=False).encode())
    parsed = parse_original(path, "csv", ".csv")
    col = parsed.schema[0]["name"]
    plan = cm.plan(
        planned((Step("impute.median", (col,)), "data")),
        dataclasses.replace(context(parsed.schema), target=None, task_type=None),
    )
    v1 = cm.materialise(stored(parsed.table.frame), plan.data)
    source = render_pipeline(
        version_number=1,
        fmt="csv",
        declared_extension=".csv",
        parse_options=parsed.parse_options,
        commits=[Commit(1, plan.data.steps)],
        model_steps=[],
        seed=0,
        libraries={},
    )
    digest, _ = run_pipeline(tmp_path, source, path)
    assert digest == v1.content_hash.hex()  # ran without executing the injected text
    text = (tmp_path / "pipeline.py").read_text(encoding="utf-8")
    # Every occurrence of the name is inside a string literal (fit and apply lines).
    assert text.count("pwned") == text.count(repr([col])) >= 1

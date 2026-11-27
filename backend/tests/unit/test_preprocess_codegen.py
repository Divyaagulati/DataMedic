"""Generated code reproduces the platform's results, standalone and safely.

The generated module is run in an isolated subprocess (``python -I``, temporary directory)
that cannot import DataMedic, against Parquet copies of the inputs.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.preprocess.codegen import CodegenError, literal, render_steps
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from datamedic.engine.preprocess.validate import ValidatedStep
from tests.unit.preprocess_helpers import MIXED_ROLES, frame, mixed_frame, validated

DRIVER = """

if __name__ == "__main__":
    import sys
    import pandas as pd
    fit_df = pd.read_parquet(sys.argv[1])
    df = pd.read_parquet(sys.argv[2])
    preprocess(fit_df, df).to_parquet(sys.argv[3])
"""


def run_generated(
    tmp_path: Path,
    steps: list[ValidatedStep],
    fit_df: pd.DataFrame,
    df: pd.DataFrame,
    seed: int = 0,
) -> pd.DataFrame:
    source = render_steps(steps, seed=seed)
    script = tmp_path / "generated_steps.py"
    script.write_text(source + DRIVER, encoding="utf-8")
    fit_df.to_parquet(tmp_path / "fit.parquet")
    df.to_parquet(tmp_path / "all.parquet")
    subprocess.run(  # noqa: S603
        [sys.executable, "-I", str(script), "fit.parquet", "all.parquet", "out.parquet"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=120,
    )
    return pd.read_parquet(tmp_path / "out.parquet")


def assert_equivalent(platform: pd.DataFrame, generated: pd.DataFrame) -> None:
    assert list(platform.columns) == list(generated.columns)
    assert platform[ROW_ID].tolist() == generated[ROW_ID].tolist()
    for column in platform.columns:
        a, b = platform[column], generated[column]
        assert str(a.dtype) == str(b.dtype), column
        if pd.api.types.is_float_dtype(a):
            assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), column
        else:
            assert a.astype(object).tolist() == b.astype(object).tolist(), column


CASES = {
    "mean": [Step("impute.mean", ("x1",))],
    "median": [Step("impute.median", ("age",))],
    "mode": [Step("impute.mode", ("city", "flag"))],
    "constant": [
        Step("impute.constant", ("city",), {"fill_value": "Unknown"}),
        Step("impute.constant", ("age",), {"fill_value": 0}),
        Step("impute.constant", ("flag",), {"fill_value": True}),
    ],
    "knn": [Step("impute.knn", ("x1", "x2"), {"n_neighbors": 4, "weights": "distance"})],
    "iterative": [Step("impute.iterative", ("x1", "x2"), {"max_iter": 5})],
    "indicator": [Step("missing.indicator", ("age", "city"))],
    "combined": [
        Step("missing.indicator", ("x1", "city")),
        Step("impute.median", ("age",)),
        Step("impute.knn", ("x1",)),
        Step("impute.iterative", ("x2",)),
        Step("impute.mode", ("city", "flag")),
    ],
    # Part B
    "outlier_flag": [Step("outliers.flag", ("x1", "x2"), {"method": "mad"})],
    "outlier_cap_integer": [Step("outliers.cap", ("age", "x1"), {"method": "iqr"})],
    "outlier_cap_percentile": [Step("outliers.cap", ("x2",))],
    "signed_log": [Step("outliers.signed_log", ("x1", "x2"))],
    "isolation_forest": [
        Step("outliers.isolation_forest_flag", ("x1", "x2"), {"contamination": 0.1})
    ],
    "onehot": [Step("encode.onehot", ("city", "flag"), {"max_categories": 2})],
    "ordinal": [Step("encode.ordinal", ("city",), {"order": ["east", "north", "south"]})],
    "frequency": [Step("encode.frequency", ("city",))],
    "scalers": [
        Step("scale.standard", ("x1",)),
        Step("scale.minmax", ("age",), {"clip": True}),
        Step("scale.robust", ("x2",)),
        Step("scale.none", ("x1",)),
    ],
    "combined_b": [
        Step("missing.indicator", ("x1",)),
        Step("impute.median", ("x1", "x2", "age")),
        Step("outliers.flag", ("x1",)),
        Step("outliers.cap", ("x2",), {"method": "zscore", "z_threshold": 2.5}),
        Step("impute.mode", ("city",)),
        Step("encode.onehot", ("city",)),
        Step("scale.robust", ("x1", "x2")),
    ],
}


@pytest.mark.parametrize("name", list(CASES))
def test_generated_code_matches_platform_in_fold(name: str, tmp_path: Path) -> None:
    df = mixed_frame(n=60, seed=4)
    fit_df = df[df[ROW_ID] % 4 != 0]  # training rows only
    result = validated(df, CASES[name], MIXED_ROLES)
    platform = PreprocessExecutor(result, seed=7).fit(fit_df).transform(df).frame
    generated = run_generated(tmp_path, result.steps, fit_df, df, seed=7)
    assert_equivalent(platform, generated)


def test_generated_code_reproduces_a_full_data_fit(tmp_path: Path) -> None:
    df = mixed_frame(n=50, seed=9)
    result = validated(df, CASES["combined"], MIXED_ROLES)
    platform = PreprocessExecutor(result).fit_transform(df).frame
    assert_equivalent(platform, run_generated(tmp_path, result.steps, df, df))


def test_hostile_column_names_are_literals(tmp_path: Path) -> None:
    names = [
        "a'b",
        'quote"s',
        "new\nline",
        "__import__('pathlib').Path('pwned').touch()",
        '"""; raise SystemExit(3); """',
        "back\\slash",
        "ünïcode ✓",
    ]
    df = frame(**{name: [1.0, None, 3.0, float(i)] for i, name in enumerate(names)})
    steps = [Step("missing.indicator", tuple(names)), Step("impute.median", tuple(names))]
    result = validated(df, steps)
    source = render_steps(result.steps)
    ast.parse(source)  # syntactically valid
    for name in names:
        assert repr(name) in source
    assert "Path('pwned').touch()" not in source.replace(repr(names[3]), "")
    platform = PreprocessExecutor(result).fit_transform(df).frame
    assert_equivalent(platform, run_generated(tmp_path, result.steps, df, df))
    assert not (tmp_path / "pwned").exists()


def test_hostile_constant_value_is_a_literal(tmp_path: Path) -> None:
    value = "x'); import os; os.remove('important'); ('"
    df = frame(c=pd.Series([None, "a"], dtype=object))
    result = validated(
        df, [Step("impute.constant", ("c",), {"fill_value": value})], types={"c": "categorical"}
    )
    source = render_steps(result.steps)
    assert repr(value) in source
    out = run_generated(tmp_path, result.steps, df, df)
    assert out["c"].tolist() == [value, "a"]


def test_literal_refuses_non_json_values() -> None:
    assert literal([1, 2.5, "a", True, None]) == "[1, 2.5, 'a', True, None]"
    assert literal({"iqr_multiplier": 1.5}) == "{'iqr_multiplier': 1.5}"  # outlier rule params
    for bad in (float("nan"), float("inf"), {1: "a"}, {"a": object()}, object(), b"bytes"):
        with pytest.raises(CodegenError):
            literal(bad)


def test_generated_module_has_no_datamedic_import_and_no_comments_with_names() -> None:
    df = frame(**{"secret column": [1.0, None]})
    result = validated(df, [Step("impute.mean", ("secret column",))])
    source = render_steps(result.steps)
    tree = ast.parse(source)
    imported = {
        (n.module or "") if isinstance(n, ast.ImportFrom) else a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.Import | ast.ImportFrom)
        for a in (n.names if isinstance(n, ast.Import) else [n])
    }
    assert not any(str(m).startswith("datamedic") for m in imported)
    comments = [line for line in source.splitlines() if line.strip().startswith("#")]
    assert all("secret" not in line for line in comments)


def test_empty_pipeline_is_identity(tmp_path: Path) -> None:
    df = frame(a=[1.0, None])
    out = run_generated(tmp_path, [], df, df)
    assert_equivalent(df, out)


def test_render_is_deterministic() -> None:
    df = mixed_frame()
    result = validated(df, CASES["combined"], MIXED_ROLES)
    assert render_steps(result.steps) == render_steps(result.steps)
    assert json.dumps([s.step.as_dict() for s in result.steps])  # steps are JSON


def test_generated_code_preserves_nullable_dtypes(tmp_path: Path) -> None:
    df = frame(
        f=pd.Series(pd.array([1.5, None, 2.5, pd.NA, 4.0], dtype="Float64")),
        n=pd.Series(pd.array([1, pd.NA, 3, 3, None], dtype="Int64")),
        s=pd.Series(["a", None, "b", pd.NA, "a"], dtype="string"),
        k=pd.Series(["x", None, "y", "x", None], dtype="category"),
        o=pd.Series(["u", np.nan, None, "u", pd.NA], dtype=object),
    )
    types = {
        "f": "numeric",
        "n": "numeric",
        "s": "categorical",
        "k": "categorical",
        "o": "categorical",
    }
    steps = [
        Step("missing.indicator", ("f", "s", "k", "o")),
        Step("impute.mode", ("f", "n", "s")),
        Step("impute.constant", ("k", "o"), {"fill_value": "Unknown"}),
    ]
    result = validated(df, steps, types=types)
    platform = PreprocessExecutor(result).fit(df.iloc[:3]).transform(df).frame
    generated = run_generated(tmp_path, result.steps, df.iloc[:3], df)
    assert_equivalent(platform, generated)
    assert str(generated["f"].dtype) == "Float64" and str(generated["s"].dtype) == "string"
    assert str(generated["k"].dtype) == "category"


def test_hostile_category_values_never_become_code(tmp_path: Path) -> None:
    values = ["a'b", 'q"uote', "__import__('pathlib').Path('pwned').touch()", "x=y", "\u00fc"]
    df = frame(c=pd.Series(values * 4, dtype=object))
    steps = [Step("encode.frequency", ("c",))]
    result = validated(df, steps, types={"c": "categorical"})
    source = render_steps(result.steps)
    assert "pwned" not in source  # categories are learned at run time, never embedded
    onehot = validated(df, [Step("encode.onehot", ("c",))], types={"c": "categorical"})
    assert "pwned" not in render_steps(onehot.steps)
    for case in (result, onehot):
        platform = PreprocessExecutor(case).fit_transform(df).frame
        assert_equivalent(platform, run_generated(tmp_path, case.steps, df, df))
    assert not (tmp_path / "pwned").exists()


def test_hostile_ordinal_order_and_output_name_are_literals(tmp_path: Path) -> None:
    order = ["lo'w", "__import__('os')", 'hi"gh']
    df = frame(c=pd.Series(order * 3, dtype=object), a=list(range(9)), b=list(range(9, 0, -1)))
    df = pd.concat([df] * 3, ignore_index=True)
    df["_dm_row_id"] = range(len(df))
    steps = [
        Step("encode.ordinal", ("c",), {"order": order}),
        Step("outliers.isolation_forest_flag", ("a", "b"), {"output_column": "odd'); x = ('"}),
    ]
    result = validated(df, steps, types={"c": "categorical", "a": "numeric", "b": "numeric"})
    source = render_steps(result.steps)
    ast.parse(source)
    assert all(repr(v) in source for v in order)
    platform = PreprocessExecutor(result).fit_transform(df).frame
    assert_equivalent(platform, run_generated(tmp_path, result.steps, df, df))

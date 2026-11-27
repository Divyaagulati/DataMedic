"""Generated code for Part C ops reproduces the platform in an isolated subprocess.

Two outputs are compared per case: ``preprocess(fit_df, df)`` (applying the fitted steps to
all rows, never resampling) against ``PreprocessExecutor.transform``, and
``prepare_training(fit_df)`` (the training table, resampled or weighted) against
``PreprocessExecutor.fit_resample``. The subprocess runs ``python -I`` from a temporary
directory, so it cannot import DataMedic; learned parameters are re-learned, never embedded.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.preprocess.codegen import render_steps
from datamedic.engine.preprocess.executor import PreprocessExecutor
from datamedic.engine.preprocess.spec import Step
from tests.unit.preprocess_helpers import frame, validated
from tests.unit.test_preprocess_codegen import assert_equivalent

DRIVER = """

if __name__ == "__main__":
    import sys
    import pandas as pd
    fit_df = pd.read_parquet(sys.argv[1])
    df = pd.read_parquet(sys.argv[2])
    preprocess(fit_df, df).to_parquet(sys.argv[3])
    prepare_training(fit_df).to_parquet(sys.argv[4])
"""

ROLES = {"target": ["y"], "id": ["pid"]}

ACK = {"acknowledge_dependent_rows": True}  # roles declare ids/groups (see audit)


def data(n: int = 90, seed: int = 4, labels: tuple = (0, 1)) -> pd.DataFrame:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    y = np.array([labels[1] if i % 4 == 0 else labels[0] for i in range(n)], dtype=object)
    x1 = rng.lognormal(0, 1, n)
    return frame(
        pid=[f"p{i}" for i in range(n)],
        x1=x1,
        x2=x1 * 2 + rng.normal(0, 0.1, n),
        x3=rng.normal(0, 1, n),
        n4=np.arange(n, dtype="int64"),
        cat=pd.Series(rng.choice(["u", "v", "w"], n), dtype=object),
        y=pd.Series(list(y)).astype("int64" if isinstance(labels[0], int) else object),
    )


CASES: dict[str, tuple[list[Step], str]] = {
    "log": ([Step("transform.log", ("x1",), {"base": "10"})], "binary"),
    "log1p": ([Step("transform.log1p", ("x1", "n4"))], "binary"),
    "yeo_johnson": ([Step("transform.yeo_johnson", ("x1", "x3"))], "binary"),
    "quantile": (
        [Step("transform.quantile", ("x1",), {"output_distribution": "normal", "n_quantiles": 40})],
        "binary",
    ),
    "variance": (
        [Step("select.variance", ("x1", "x3"), {"threshold": 1.5, "action": "drop"})],
        "binary",
    ),
    "correlation": (
        [Step("select.correlation", ("x1", "x2", "x3"), {"method": "spearman"})],
        "binary",
    ),
    "mutual_information": (
        [Step("select.mutual_information", ("x1", "x3", "cat"), {"k": 1, "action": "drop"})],
        "binary",
    ),
    "univariate": (
        [Step("select.univariate", ("x1", "x3"), {"test": "anova_f", "k": 1, "action": "drop"})],
        "binary",
    ),
    "drop_columns": ([Step("select.drop_columns", ("cat", "x2"))], "binary"),
    "class_weight": ([Step("imbalance.class_weight", ())], "binary"),
    "oversample": ([Step("imbalance.random_oversample", ())], "binary"),
    "undersample": ([Step("imbalance.random_undersample", ())], "binary"),
    "smote": (
        [
            Step("select.drop_columns", ("cat",)),
            Step("imbalance.smote", (), {**ACK, "k_neighbors": 3}),
        ],
        "binary",
    ),
    "combined_abc": (
        [
            Step("missing.indicator", ("x3",)),
            Step("impute.median", ("x3",)),
            Step("outliers.cap", ("x1",), {"method": "iqr"}),
            Step("transform.yeo_johnson", ("x1", "x2")),
            Step("encode.onehot", ("cat",)),
            Step("scale.standard", ("x3",)),
            Step("select.mutual_information", ("x1", "x2", "x3"), {"k": 2, "action": "drop"}),
            Step("imbalance.random_oversample", ()),
        ],
        "binary",
    ),
    "combined_smote": (
        [
            Step("impute.knn", ("x3",), {"predictors": ["x1"]}),
            Step("transform.quantile", ("x1",)),
            Step("select.drop_columns", ("cat",)),
            Step("scale.robust", ("x2", "n4")),
            Step("imbalance.smote", (), {**ACK, "k_neighbors": 2, "sampling_strategy": "minority"}),
        ],
        "binary",
    ),
}


def run_generated(
    tmp_path: Path, source: str, fit_df: pd.DataFrame, df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    script = tmp_path / "generated_steps.py"
    script.write_text(source + DRIVER, encoding="utf-8")
    fit_df.to_parquet(tmp_path / "fit.parquet")
    df.to_parquet(tmp_path / "all.parquet")
    subprocess.run(  # noqa: S603
        [
            sys.executable,
            "-I",
            str(script),
            "fit.parquet",
            "all.parquet",
            "out.parquet",
            "train.parquet",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        timeout=180,
    )
    return pd.read_parquet(tmp_path / "out.parquet"), pd.read_parquet(tmp_path / "train.parquet")


@pytest.mark.parametrize("name", list(CASES))
def test_generated_code_matches_platform_for_test_rows_and_training_table(
    name: str, tmp_path: Path
) -> None:
    steps, task = CASES[name]
    df = data()
    if name.startswith("combined"):
        df.loc[df.index[::9], "x3"] = np.nan
    fit_df = df[df[ROW_ID] % 3 != 0]
    validation = validated(df, steps, ROLES, task=task)
    ex = PreprocessExecutor(validation, seed=11)
    training = ex.fit_resample(fit_df).frame
    platform = ex.transform(df).frame
    generated, generated_training = run_generated(
        tmp_path, render_steps(validation.steps, seed=11), fit_df, df
    )
    assert_equivalent(platform.reset_index(drop=True), generated.reset_index(drop=True))
    assert_equivalent(training.reset_index(drop=True), generated_training.reset_index(drop=True))
    # Test rows are never resampled by generated code either.
    assert len(generated) == len(df)


def test_generated_code_embeds_no_learned_values() -> None:
    df = data()
    steps = [
        Step("transform.yeo_johnson", ("x1",)),
        Step("transform.quantile", ("x3",)),
        Step("select.mutual_information", ("x1", "x3"), {"k": 1, "action": "drop"}),
        Step("imbalance.class_weight", ()),
    ]
    validation = validated(df, steps, ROLES, task="binary")
    ex = PreprocessExecutor(validation, seed=0)
    training = ex.fit_resample(df)
    source = render_steps(validation.steps, seed=0)
    generated_part = source.split("SEED = 0", 1)[1]
    lam = training.reports[0].learned["columns"]["x1"]["lambda"]
    weights = training.reports[3].learned["columns"]["y"]["weights"]
    for value in [lam, *weights]:
        assert repr(value) not in generated_part and f"{value:.6g}" not in generated_part
    assert "fit_yeo_johnson(fit_df" in generated_part and "fit_quantile(fit_df" in generated_part


def test_generated_module_without_resampling_has_no_training_step() -> None:
    validation = validated(data(), [Step("transform.log", ("x1",))], ROLES, task="binary")
    source = render_steps(validation.steps)
    assert "TRAINING_STEP = None" in source and "def training_step" not in source

"""Catalogue integrity, hashing and the API-safety of the spec layer."""

import importlib
import subprocess
import sys

import pytest

from datamedic.engine.preprocess.base import DMTransformer
from datamedic.engine.preprocess.catalog import all_ops, get_op
from datamedic.engine.preprocess.spec import OpSpec, Step, canonical_json, spec_hash, step_hash

PART_A_OPS = {
    "impute.mean",
    "impute.median",
    "impute.mode",
    "impute.constant",
    "impute.knn",
    "impute.iterative",
    "missing.indicator",
}


PART_B_OPS = {
    "outliers.flag",
    "outliers.cap",
    "outliers.signed_log",
    "outliers.isolation_forest_flag",
    "encode.onehot",
    "encode.ordinal",
    "encode.frequency",
    "scale.standard",
    "scale.minmax",
    "scale.robust",
    "scale.none",
}


PART_C_OPS = {
    "transform.log",
    "transform.log1p",
    "transform.yeo_johnson",
    "transform.quantile",
    "select.variance",
    "select.correlation",
    "select.mutual_information",
    "select.univariate",
    "select.drop_columns",
    "imbalance.class_weight",
    "imbalance.random_oversample",
    "imbalance.random_undersample",
    "imbalance.smote",
}


def test_catalogue_is_complete() -> None:
    assert {s.op for s in all_ops()} == PART_A_OPS | PART_B_OPS | PART_C_OPS


def test_classification_of_part_c_ops() -> None:
    def spec(op: str) -> OpSpec:
        found = get_op(op)
        assert found is not None
        return found

    # Feature selection and imbalance handling are model stage only (owner decision D1):
    # never committed into a dataset version.
    for op in PART_C_OPS:
        if op.startswith(("select.", "imbalance.")):
            assert spec(op).stage == "model", op
    # Resampling and class weights act on the training table only and select no columns.
    for op in PART_C_OPS:
        if op.startswith("imbalance."):
            s = spec(op)
            assert s.kind == "training_only" and not s.uses_columns and s.needs_target, op
    # Removing rows or columns is destructive and high risk; SMOTE invents rows: high risk.
    for op in (
        "select.variance",
        "select.mutual_information",
        "select.univariate",
        "select.drop_columns",
        "imbalance.random_undersample",
        "transform.quantile",
    ):
        assert spec(op).destructive and spec(op).risk == "high", op
    assert spec("imbalance.smote").risk == "high"
    # Correlation only proposes removals; it never removes.
    assert not spec("select.correlation").destructive
    assert spec("select.correlation").risk == "low"
    # Invertible re-expressions are not destructive and may be data or model stage.
    for op in ("transform.log", "transform.log1p", "transform.yeo_johnson"):
        assert not spec(op).destructive and spec(op).stage == "either", op
    # Supervised selectors read the target (fitting rows only); the others never do.
    assert {op for op in PART_C_OPS if spec(op).needs_target} == {
        "select.mutual_information",
        "select.univariate",
        "imbalance.class_weight",
        "imbalance.random_oversample",
        "imbalance.random_undersample",
        "imbalance.smote",
    }


def test_training_only_ops_must_be_model_stage() -> None:
    import dataclasses

    base = get_op("imbalance.smote")
    assert base is not None
    with pytest.raises(ValueError, match="model stage"):
        dataclasses.replace(base, stage="either")


@pytest.mark.parametrize("spec", all_ops(), ids=lambda s: s.op)
def test_every_op_is_documented_and_loadable(spec: OpSpec) -> None:
    assert spec.title and spec.description and spec.when_to_use
    assert spec.downsides and spec.limitations, "every op states downsides and limitations"
    module, _, name = spec.impl.partition(":")
    cls = getattr(importlib.import_module(module), name)
    assert issubclass(cls, DMTransformer)
    assert cls.op == spec.op
    if spec.destructive:
        assert spec.risk == "high"


def test_classification_of_part_a_ops() -> None:
    # Imputation introduces values -> medium; an added flag column is lossless -> low.
    for op in PART_A_OPS - {"missing.indicator"}:
        spec = get_op(op)
        assert spec is not None and spec.risk == "medium" and not spec.destructive
    indicator = get_op("missing.indicator")
    assert indicator is not None and indicator.risk == "low" and indicator.adds_columns
    assert get_op("impute.constant").kind == "stateless_column"  # type: ignore[union-attr]
    assert get_op("impute.median").kind == "fitted"  # type: ignore[union-attr]
    # No Part A op is restricted to the model stage.
    assert all(get_op(op).stage == "either" for op in PART_A_OPS)  # type: ignore[union-attr]


def test_classification_of_part_b_ops() -> None:
    def spec(op: str) -> OpSpec:
        found = get_op(op)
        assert found is not None
        return found

    # Encoding and scaling run only inside model training (owner decision D1).
    for op in PART_B_OPS:
        if op.startswith(("encode.", "scale.")):
            assert spec(op).stage == "model", op
    # Capping overwrites observed values irreversibly: destructive and high risk.
    assert spec("outliers.cap").destructive and spec("outliers.cap").risk == "high"
    # Flags add information without changing values.
    for op in ("outliers.flag", "outliers.isolation_forest_flag"):
        assert spec(op).risk == "low" and not spec(op).destructive and spec(op).adds_columns
    # Invertible or representation-only changes are not destructive.
    for op in (
        "outliers.signed_log",
        "encode.onehot",
        "encode.ordinal",
        "encode.frequency",
        "scale.standard",
        "scale.minmax",
        "scale.robust",
        "scale.none",
    ):
        assert not spec(op).destructive, op
    # Ordinal encoding never invents an order: the order parameter is required.
    order = spec("encode.ordinal").param("order")
    assert order is not None and order.required
    # No catalogue op removes rows.
    assert not any(s.kind.startswith("row_filter") for s in all_ops())


def test_invalid_specs_are_refused() -> None:
    base = get_op("impute.mean")
    assert base is not None
    with pytest.raises(ValueError):
        OpSpec(**{**base.__dict__, "risk": "medium", "destructive": True})
    with pytest.raises(ValueError):
        OpSpec(**{**base.__dict__, "stage": "anywhere"})


def test_step_hash_is_deterministic_and_sensitive() -> None:
    a = Step("impute.knn", ("x",), {"n_neighbors": 5, "weights": "uniform", "predictors": []})
    same = Step("impute.knn", ("x",), {"predictors": [], "weights": "uniform", "n_neighbors": 5})
    assert step_hash(a, "1.0.0") == step_hash(same, "1.0.0")  # key order irrelevant
    variants = [
        Step("impute.knn", ("x",), {**a.params, "n_neighbors": 6}),
        Step("impute.knn", ("y",), a.params),
        Step("impute.mean", ("x",), a.params),
    ]
    assert all(step_hash(v, "1.0.0") != step_hash(a, "1.0.0") for v in variants)
    assert step_hash(a, "1.0.1") != step_hash(a, "1.0.0")  # op version is bound


def test_spec_hash_changes_with_order() -> None:
    assert spec_hash(["a", "b"]) != spec_hash(["b", "a"])
    assert spec_hash([]) == spec_hash([])


def test_canonical_json_refuses_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"v": float("nan")})


def test_spec_layer_does_not_load_data_libraries() -> None:
    """The API imports the catalogue and validation; they must not pull in pandas/sklearn."""
    code = (
        "import sys\n"
        "import datamedic.engine.preprocess.catalog, datamedic.engine.preprocess.validate\n"
        "import datamedic.preprocessing.router\n"
        "print(','.join(m for m in ('pandas', 'numpy', 'sklearn') if m in sys.modules))\n"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == ""

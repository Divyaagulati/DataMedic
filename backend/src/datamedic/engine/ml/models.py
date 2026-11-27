"""Estimators for supervised training, built from ``engine.ml.spec`` (worker only).

Reuses the Experiment Lab's feature handling (``feature_kinds``, ``model_matrix`` and the
one-hot preprocessing of the linear model) so both evaluate the same documented minimal
in-fold preprocessing. Every estimator is single-threaded and seeded.
"""

from typing import Any

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from datamedic.engine.ml.spec import CLASSIFICATION, MODELS
from datamedic.engine.preprocess.experiment.models import (
    FeatureKinds,
    _onehot_pre,
)

RF_TREES = 100
RF_MAX_DEPTH = 12


def _ordinal_pre(kinds: FeatureKinds, impute: bool) -> ColumnTransformer:
    parts: list[tuple[str, Any, list[str]]] = []
    if kinds.numeric:
        parts.append(
            (
                "num",
                SimpleImputer(strategy="median", keep_empty_features=True)
                if impute
                else "passthrough",
                kinds.numeric,
            )
        )
    if kinds.categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
            encoded_missing_value=np.nan,
        )
        cat: Any = (
            Pipeline(
                [
                    ("encode", encoder),
                    ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                ]
            )
            if impute
            else encoder
        )
        parts.append(("cat", cat, kinds.categorical))
    return ColumnTransformer(parts, remainder="drop")


def build_estimator(
    model: str, task: str, kinds: FeatureKinds, params: dict[str, Any], seed: int
) -> Pipeline:
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    classification = task in CLASSIFICATION
    est: Any
    if model == "dummy":
        est = (
            DummyClassifier(strategy="prior")
            if classification
            else DummyRegressor(strategy="median")
        )
        return Pipeline([("model", est)])
    if model == "linear":
        est = (
            LogisticRegression(C=float(params["C"]), max_iter=2000, random_state=seed)
            if classification
            else Ridge(alpha=float(params["alpha"]))
        )
        return Pipeline([("pre", _onehot_pre(kinds)), ("model", est)])
    if model == "random_forest":
        depth = min(int(params["max_depth"]), RF_MAX_DEPTH)
        common: dict[str, Any] = {
            "n_estimators": RF_TREES,
            "max_depth": depth,
            "min_samples_leaf": int(params["min_samples_leaf"]),
            "n_jobs": 1,
            "random_state": seed,
        }
        est = (
            RandomForestClassifier(**common) if classification else RandomForestRegressor(**common)
        )
        return Pipeline([("pre", _ordinal_pre(kinds, impute=True)), ("model", est)])
    hgb_params: dict[str, Any] = {
        "max_leaf_nodes": int(params["max_leaf_nodes"]),
        "learning_rate": float(params["learning_rate"]),
        "early_stopping": False,
        "random_state": seed,
    }
    est = (
        HistGradientBoostingClassifier(**hgb_params)
        if classification
        else HistGradientBoostingRegressor(**hgb_params)
    )
    return Pipeline([("pre", _ordinal_pre(kinds, impute=False)), ("model", est)])


def output_groups(pipeline: Pipeline, kinds: FeatureKinds) -> list[str] | None:
    """For each model input column after the estimator's own preprocessing, the transformed
    table column it came from. ``None`` when the mapping is not available."""
    if "pre" not in pipeline.named_steps:
        return None
    pre = pipeline.named_steps["pre"]
    out: list[str] = []
    for name, transformer, columns in pre.transformers_:
        if name == "remainder":
            continue
        if name == "num":
            out.extend(columns)
        elif name == "cat":
            encoder = transformer
            if isinstance(transformer, Pipeline):
                encoder = transformer.named_steps.get("onehot") or transformer.named_steps.get(
                    "encode"
                )
            if hasattr(encoder, "categories_") and encoder.__class__.__name__ == "OneHotEncoder":
                names = encoder.get_feature_names_out(columns)
                # get_feature_names_out prefixes each output with its input column name.
                for out_name in names:
                    source = next(
                        (
                            c
                            for c in sorted(columns, key=len, reverse=True)
                            if str(out_name).startswith(f"{c}_")
                        ),
                        None,
                    )
                    if source is None:
                        return None
                    out.append(source)
            else:
                out.extend(columns)
    return out

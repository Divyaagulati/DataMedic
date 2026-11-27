"""Baseline models for the Experiment Lab (owner decision D3: no tuning, library defaults).

Every model gets the same minimal preprocessing for every candidate, fitted on the training
rows of each fold, so "no preprocessing" stays honest and candidates differ only in their
own steps:

- ``dummy`` (classification / regression): prior class / median; ignores features.
- ``linear``: logistic regression / ridge, after median imputation and standardisation of
  numeric columns and most-frequent imputation plus one-hot encoding (20 most frequent
  categories) of categorical columns.
- ``hgb``: histogram gradient boosting; categories as ordinal codes, missing values handled
  natively. Internal early stopping is disabled (``early_stopping=False``, the library's
  ``max_iter=100``): with the library default ``'auto'`` it switches on above 10,000 training
  rows and holds out a random, possibly stratified, 10% of the training rows that ignores
  declared groups, time order and duplicate clusters (Phase 5 closure correction).
- ``kmeans`` (clustering): k-means with the user's k, on the same inputs as ``linear``.

Datetime columns are not model inputs (reported). Categories are compared as text.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

SUPERVISED_MODELS = ("dummy", "linear", "hgb")
CLUSTERING_MODELS = ("kmeans",)
MAX_ONEHOT_CATEGORIES = 20

MODEL_NOTES = {
    "dummy": "Predicts the training-fold class prior (classification) or median (regression). "
    "Any useful model should beat it.",
    "linear": "Logistic regression (max_iter=1000) or ridge regression (alpha=1), with in-fold "
    "median imputation, standardisation and one-hot encoding of the 20 most frequent "
    "categories.",
    "hgb": "Histogram gradient boosting with library defaults except internal early stopping, "
    "which is off (a fixed 100 boosting iterations): its random validation split would ignore "
    "groups, time order and duplicate clusters. Missing values handled natively; categories "
    "as ordinal codes.",
    "kmeans": "k-means (n_init=10) on standardised, imputed, one-hot encoded inputs.",
}


@dataclass(frozen=True)
class FeatureKinds:
    numeric: list[str]
    categorical: list[str]
    ignored: list[str]  # datetime columns: not model inputs


def feature_kinds(frame: pd.DataFrame, columns: list[str]) -> FeatureKinds:
    numeric, categorical, ignored = [], [], []
    for c in columns:
        s = frame[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            ignored.append(c)
        elif pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
            numeric.append(c)
        else:
            categorical.append(c)
    return FeatureKinds(numeric, categorical, ignored)


def model_matrix(frame: pd.DataFrame, kinds: FeatureKinds) -> pd.DataFrame:
    """Plain NumPy-backed columns: numbers as float (missing -> NaN), categories as text."""
    data: dict[str, Any] = {}
    for c in kinds.numeric:
        data[c] = frame[c].to_numpy(dtype=float, na_value=np.nan)
    for c in kinds.categorical:
        s = frame[c]
        data[c] = np.where(s.notna().to_numpy(), s.astype(str).to_numpy(dtype=object), np.nan)
    out = pd.DataFrame(data, index=frame.index)
    if out.shape[1] == 0:  # the dummy model needs a column to exist
        out["__constant__"] = 0.0
    return out


def _onehot_pre(kinds: FeatureKinds) -> ColumnTransformer:
    parts: list[tuple[str, Any, list[str]]] = []
    if kinds.numeric:
        parts.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scale", StandardScaler()),
                    ]
                ),
                kinds.numeric,
            )
        )
    if kinds.categorical:
        parts.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "impute",
                            SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                max_categories=MAX_ONEHOT_CATEGORIES,
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                kinds.categorical,
            )
        )
    return ColumnTransformer(parts, remainder="drop")


def build_model(
    name: str, task: str, kinds: FeatureKinds, seed: int, n_clusters: int | None = None
) -> Pipeline:
    classification = task in ("binary", "multiclass")
    if name == "dummy":
        est: Any = (
            DummyClassifier(strategy="prior")
            if classification
            else DummyRegressor(strategy="median")
        )
        return Pipeline([("model", est)])
    if name == "linear":
        est = (
            LogisticRegression(max_iter=1000, random_state=seed)
            if classification
            else Ridge(alpha=1.0)
        )
        return Pipeline([("pre", _onehot_pre(kinds)), ("model", est)])
    if name == "hgb":
        parts: list[tuple[str, Any, list[str]]] = []
        if kinds.numeric:
            parts.append(("num", "passthrough", kinds.numeric))
        if kinds.categorical:
            parts.append(
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=np.nan,
                        encoded_missing_value=np.nan,
                    ),
                    kinds.categorical,
                )
            )
        # No internal early stopping: its random validation split cannot respect groups,
        # time order or duplicate clusters (see the module docstring).
        est = (
            HistGradientBoostingClassifier(early_stopping=False, random_state=seed)
            if classification
            else HistGradientBoostingRegressor(early_stopping=False, random_state=seed)
        )
        return Pipeline([("pre", ColumnTransformer(parts, remainder="drop")), ("model", est)])
    if name == "kmeans":
        if n_clusters is None:
            raise ValueError("k-means needs a number of clusters")
        return Pipeline(
            [
                ("pre", _onehot_pre(kinds)),
                ("model", KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)),
            ]
        )
    raise ValueError(f"unknown model {name!r}")

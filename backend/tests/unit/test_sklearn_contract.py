"""Contract tests for the scikit-learn behaviour the preprocessing ops rely on.

They pin facts verified against the locked version (scikit-learn 1.9.1). If an upgrade
changes one, this file fails first and the op must be reviewed (CLAUDE.md: never assume
library APIs).
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


def test_pinned_version() -> None:
    assert sklearn.__version__ == "1.9.1"


def test_imputer_parameters_exist() -> None:
    assert {"strategy", "fill_value", "missing_values"} <= set(
        inspect.signature(SimpleImputer).parameters
    )
    assert {"n_neighbors", "weights"} <= set(inspect.signature(KNNImputer).parameters)
    assert {"max_iter", "random_state", "sample_posterior", "min_value", "max_value"} <= set(
        inspect.signature(IterativeImputer).parameters
    )


def test_simple_imputer_ignores_none_in_object_arrays() -> None:
    """Why ``runtime.object_values`` converts every missing marker to NaN first."""
    out = SimpleImputer(strategy="most_frequent").fit_transform(
        pd.DataFrame({"c": ["x", None, "y", "x"]})
    )
    assert out[1, 0] is None  # left unfilled, silently


def test_most_frequent_ties_choose_the_smallest_value() -> None:
    values = np.array([["b"], ["a"], ["b"], ["a"], [np.nan]], dtype=object)
    assert SimpleImputer(strategy="most_frequent").fit(values).statistics_[0] == "a"
    numbers = np.array([[3.0], [1.0], [3.0], [1.0]])
    assert SimpleImputer(strategy="most_frequent").fit(numbers).statistics_[0] == 1.0


def test_column_transformer_reorders_columns() -> None:
    """Why ops select their own columns instead of using ColumnTransformer."""
    df = pd.DataFrame({"_dm_row_id": [0, 1], "a": [1.0, np.nan], "b": [1, 2]})
    ct = ColumnTransformer(
        [("i", SimpleImputer(), ["a"])], remainder="passthrough", verbose_feature_names_out=False
    ).set_output(transform="pandas")
    assert list(ct.fit_transform(df).columns) == ["a", "_dm_row_id", "b"]


def test_knn_uses_fewer_donors_when_needed() -> None:
    fit = np.array([[1.0, 1.0], [2.0, 2.0]])
    out = KNNImputer(n_neighbors=5).fit(fit).transform(np.array([[1.5, np.nan]]))
    assert out[0, 1] == 1.5


def test_iterative_is_deterministic_with_a_seed() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 3))
    data[rng.random((50, 3)) < 0.2] = np.nan
    a = IterativeImputer(random_state=0).fit_transform(data)
    b = IterativeImputer(random_state=0).fit_transform(data)
    assert np.array_equal(a, b)


def test_iterative_requires_strictly_increasing_bounds() -> None:
    """Why ``runtime.fit_iterative`` widens the bound of a constant column."""
    data = np.array([[1.0, 2.0], [1.0, np.nan], [1.0, 4.0]])
    try:
        IterativeImputer(min_value=np.array([1.0, 2.0]), max_value=np.array([1.0, 4.0])).fit(data)
    except ValueError:
        pass
    else:  # pragma: no cover - would mean the workaround is no longer needed
        raise AssertionError("IterativeImputer accepted min_value == max_value")


def test_iterative_warns_with_convergence_warning() -> None:
    rng = np.random.default_rng(3)
    data = rng.normal(size=(60, 4))
    data[rng.random((60, 4)) < 0.4] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        IterativeImputer(max_iter=1, random_state=0).fit(data)
    assert any(issubclass(w.category, ConvergenceWarning) for w in caught)


def test_working_memory_can_be_bounded_for_knn() -> None:
    """``runtime.apply_knn`` caps scikit-learn's distance chunk memory with config_context."""
    assert "working_memory" in inspect.signature(sklearn.config_context).parameters
    with sklearn.config_context(working_memory=256):
        assert sklearn.get_config()["working_memory"] == 256


# ---- Part B ----------------------------------------------------------------------------------


def test_part_b_parameters_exist() -> None:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    assert {"n_estimators", "contamination", "random_state"} <= set(
        inspect.signature(IsolationForest).parameters
    )
    assert "clip" in inspect.signature(MinMaxScaler).parameters
    assert inspect.signature(RobustScaler).parameters["quantile_range"].default == (25.0, 75.0)


def test_isolation_forest_is_deterministic_and_has_the_auto_offset() -> None:
    from sklearn.ensemble import IsolationForest

    data = np.random.default_rng(0).normal(size=(200, 3))
    a = IsolationForest(random_state=3).fit(data)
    b = IsolationForest(random_state=3).fit(data)
    assert np.array_equal(a.predict(data), b.predict(data))
    assert a.offset_ == -0.5  # contamination="auto"


def test_isolation_forest_accepts_nan_but_the_op_does_not_rely_on_it() -> None:
    """Fact pinned for review: 1.9.1 accepts NaN. The op still fits and scores complete rows
    only, because tree handling of missing values is not a documented anomaly semantics."""
    from sklearn.ensemble import IsolationForest

    data = np.random.default_rng(0).normal(size=(50, 2))
    data[0, 0] = np.nan
    IsolationForest(random_state=0).fit(data)


def test_scalers_ignore_nan_in_fit_and_keep_it_in_transform() -> None:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

    fit = np.array([[5.0, 1.0], [5.0, np.nan], [5.0, 3.0]])
    for scaler in (StandardScaler(), MinMaxScaler(), RobustScaler()):
        out = scaler.fit(fit).transform(np.array([[7.0, np.nan]]))
        assert out[0, 0] == 2.0  # constant column: scale 1, only shifted
        assert np.isnan(out[0, 1])

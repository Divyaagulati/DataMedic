"""Library contract for Part C: the pinned imbalanced-learn, scikit-learn and SciPy behaviours
the ops rely on. A failure here after an upgrade means an op's documented behaviour changed.
"""

from importlib.metadata import version

import numpy as np
import pytest
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.versions import library_versions


def test_pinned_versions_are_tracked() -> None:
    assert version("imbalanced-learn") == "0.14.2"
    assert version("scikit-learn") == "1.9.1"
    assert library_versions()["imbalanced-learn"] == "0.14.2"


def xy(n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    return rng.normal(size=(n, 3)), np.array([0] * (n - n // 5) + [1] * (n // 5))


def test_random_oversampler_keeps_originals_first_and_exposes_indices() -> None:
    x, y = xy()
    sampler = RandomOverSampler(random_state=0)
    xr, yr = sampler.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
    indices = sampler.sample_indices_
    assert np.array_equal(indices[: len(y)], np.arange(len(y)))
    assert np.array_equal(xr[:, 0], indices) and np.bincount(yr).tolist() == [48, 48]


def test_random_undersampler_exposes_indices_of_kept_rows() -> None:
    _, y = xy()
    sampler = RandomUnderSampler(random_state=0)
    _, yr = sampler.fit_resample(np.arange(len(y)).reshape(-1, 1), y)
    assert len(set(sampler.sample_indices_.tolist())) == len(yr) == 24


def test_smote_keeps_originals_first_and_is_seeded() -> None:
    x, y = xy()
    a, ya = SMOTE(k_neighbors=3, random_state=1).fit_resample(x, y)
    b, _ = SMOTE(k_neighbors=3, random_state=1).fit_resample(x, y)
    assert np.array_equal(a[: len(x)], x) and np.array_equal(ya[: len(y)], y)
    assert np.array_equal(a, b)


def test_smote_rejects_k_not_below_the_minority_size() -> None:
    x, y = xy(20)  # 4 minority rows
    with pytest.raises(ValueError):
        SMOTE(k_neighbors=4, random_state=0).fit_resample(x, y)


def test_sampling_strategy_strings_we_map_to_exist() -> None:
    x, y = xy()
    for ours, theirs in rt.IMBLEARN_STRATEGY.items():
        cls = RandomUnderSampler if ours in ("not_minority", "majority") else RandomOverSampler
        cls(sampling_strategy=theirs, random_state=0).fit_resample(x, y)


def test_power_transformer_constant_column_and_formula() -> None:
    assert PowerTransformer(standardize=False).fit(np.full((10, 1), 3.0)).lambdas_[0] == 1.0
    rng = np.random.default_rng(2)
    x = rng.normal(0, 2, 200)
    for lam in (-1.5, 0.0, 0.7, 1.0, 2.0, 3.1):
        model = PowerTransformer(standardize=False).fit(x.reshape(-1, 1))
        model.lambdas_ = np.array([lam])
        expected = model.transform(x.reshape(-1, 1))[:, 0]
        assert np.allclose(rt.yeo_johnson(x, lam), expected, rtol=1e-12, atol=1e-12), lam


def test_quantile_transformer_behaviours_we_document() -> None:
    x = np.arange(10, dtype="float64").reshape(-1, 1)
    model = QuantileTransformer(n_quantiles=10).fit(x)
    assert model.transform(np.array([[-100.0], [100.0]]))[:, 0].tolist() == [0.0, 1.0]
    constant = QuantileTransformer(n_quantiles=5).fit(np.full((5, 1), 2.0))
    assert constant.transform(np.full((1, 1), 2.0))[0, 0] == 0.0  # why we special-case it
    normal = QuantileTransformer(n_quantiles=10, output_distribution="normal").fit(x)
    low, high = rt.normal_output_bounds()
    assert normal.transform(np.array([[-100.0], [100.0]]))[:, 0].tolist() == [low, high]
    with pytest.raises(ValueError):
        QuantileTransformer(n_quantiles=50, subsample=10).fit(np.arange(100.0).reshape(-1, 1))

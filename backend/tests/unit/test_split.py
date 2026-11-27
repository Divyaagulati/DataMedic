"""Frozen fold splitter (Part C): stratified, group, time and duplicate-aware folds."""

import numpy as np
import pandas as pd
import pytest

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.split import make_splits


def data(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            ROW_ID: rng.permutation(np.arange(1000, 1000 + n)),
            "y": (rng.random(n) < 0.3).astype(int),
            "patient": rng.integers(0, 30, n),
            "day": rng.integers(0, 60, n),
            "x": rng.integers(0, 4, n),
        }
    )
    return df


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "kfold"},
        {"method": "stratified", "target": "y"},
        {"method": "group", "group": "patient"},
        {"method": "stratified_group", "target": "y", "group": "patient"},
    ],
)
def test_every_row_is_tested_once_and_never_in_both_sides(kwargs: dict) -> None:  # type: ignore[type-arg]
    df = data()
    splits = make_splits(df, n_splits=4, seed=3, **kwargs)
    tested = np.concatenate([f.test_ids for f in splits.folds])
    assert sorted(tested.tolist()) == sorted(df[ROW_ID].tolist())
    for fold in splits.folds:
        assert not set(fold.train_ids) & set(fold.test_ids)
        assert len(fold.train_ids) + len(fold.test_ids) == len(df)
    assignment = splits.assignment()
    assert set(assignment) == set(df[ROW_ID]) and -1 not in assignment.values()


def test_splits_are_frozen_by_seed() -> None:
    df = data()
    a = make_splits(df, method="stratified", target="y", seed=1).assignment()
    b = make_splits(df, method="stratified", target="y", seed=1).assignment()
    c = make_splits(df, method="stratified", target="y", seed=2).assignment()
    assert a == b and a != c


def test_stratified_folds_preserve_class_shares() -> None:
    df = data(n=400)
    share = df["y"].mean()
    for fold in make_splits(df, method="stratified", target="y", n_splits=5).folds:
        test = df[df[ROW_ID].isin(fold.test_ids)]
        assert abs(test["y"].mean() - share) < 0.03


@pytest.mark.parametrize("method", ["group", "stratified_group"])
def test_groups_never_cross_train_and_test(method: str) -> None:
    df = data()
    splits = make_splits(df, method=method, target="y", group="patient", n_splits=4)
    for fold in splits.folds:
        train, test = splits.frames(df, fold)
        assert not set(train["patient"]) & set(test["patient"])


def test_duplicates_are_kept_on_one_side_and_merged_with_groups() -> None:
    df = data()
    splits = make_splits(
        df, method="stratified", target="y", duplicate_columns=["x", "y"], n_splits=3
    )
    assert "DUPLICATES_KEPT_TOGETHER" in [w["code"] for w in splits.warnings]
    for fold in splits.folds:
        train, test = splits.frames(df, fold)
        keys = lambda d: set(map(tuple, d[["x", "y"]].to_numpy().tolist()))  # noqa: E731
        assert not keys(train) & keys(test)
    df2 = data()
    df2["k"] = df2["patient"] // 3 * 10 + df2["x"] % 2  # links rows of different patients
    grouped = make_splits(df2, method="group", group="patient", duplicate_columns=["k"], n_splits=2)
    for fold in grouped.folds:
        train, test = grouped.frames(df2, fold)
        assert not set(train["patient"]) & set(test["patient"])
        assert not set(train["k"]) & set(test["k"])


def test_time_folds_always_train_on_the_past() -> None:
    df = data()
    splits = make_splits(df, method="time", time="day", n_splits=4, duplicate_columns=["x"])
    for fold in splits.folds:
        train, test = splits.frames(df, fold)
        assert train["day"].max() < test["day"].min()  # ties never straddle
    codes = [w["code"] for w in splits.warnings]
    assert "EARLIEST_BLOCK_NOT_TESTED" in codes and "DUPLICATES_CROSS_TIME_SPLIT" in codes
    assert -1 in splits.assignment().values()


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"method": "magic"}, "SPLIT_INVALID"),
        ({"method": "kfold", "n_splits": 1}, "SPLIT_INVALID"),
        ({"method": "stratified"}, "SPLIT_INVALID"),
        ({"method": "group"}, "SPLIT_INVALID"),
        ({"method": "time", "time": "day", "group": "patient"}, "SPLIT_INVALID"),
        ({"method": "group", "group": "patient", "n_splits": 20}, "SPLIT_TOO_FEW_GROUPS"),
    ],
)
def test_invalid_split_requests_fail_clearly(kwargs: dict, code: str) -> None:  # type: ignore[type-arg]
    df = data(n=60)
    df["patient"] = df["patient"] % 10
    with pytest.raises(EngineError) as exc:
        make_splits(df, **kwargs)
    assert exc.value.code == code


def test_too_small_classes_missing_labels_groups_and_times_are_refused() -> None:
    df = data(n=60)
    df.loc[df.index[:58], "y"] = 0
    with pytest.raises(EngineError) as exc:
        make_splits(df, method="stratified", target="y", n_splits=5)
    assert exc.value.code == "SPLIT_CLASS_TOO_SMALL"
    for column, kwargs, code in (
        ("y", {"method": "stratified", "target": "y"}, "SPLIT_TARGET_MISSING"),
        ("patient", {"method": "group", "group": "patient"}, "SPLIT_GROUP_MISSING"),
        ("day", {"method": "time", "time": "day"}, "SPLIT_TIME_MISSING"),
    ):
        broken = data(n=60).astype({column: "float64"})
        broken.loc[broken.index[0], column] = np.nan
        with pytest.raises(EngineError) as exc:
            make_splits(broken, **kwargs)
        assert exc.value.code == code

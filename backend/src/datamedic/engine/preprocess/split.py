"""Frozen fold assignments for in-fold preprocessing and evaluation (Part C).

Folds are defined on ``_dm_row_id``, so the same assignment can be stored and replayed:
every candidate pipeline is fitted on exactly the same training rows (owner decision D3).

Methods:
- ``kfold``: shuffled K-fold (regression or unlabelled tasks).
- ``stratified``: shuffled stratified K-fold on the target classes.
- ``group``: whole groups (e.g. patients) are kept on one side (shuffled GroupKFold).
- ``stratified_group``: groups kept together, class balance approximately preserved.
- ``time``: forward-chaining folds on a time column. Training rows always come before test
  rows, rows with the same time are never split, and the earliest block is never tested.

Duplicate awareness: rows identical in ``duplicate_columns`` are kept on the same side
(merged with groups when both are given), so an exact duplicate never sits in training and
test at once. With the ``time`` method, duplicates cannot be forced together; the number of
duplicate keys crossing a train/test boundary is reported instead.

Random splits are not assumed to be appropriate: grouped, repeated-measures and time-ordered
data need ``group`` / ``stratified_group`` / ``time``. Choosing the method is the caller's
decision (Part D recommends one).
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedGroupKFold, StratifiedKFold

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError

SPLIT_METHODS = ("kfold", "stratified", "group", "stratified_group", "time")
MIN_SPLITS, MAX_SPLITS = 2, 20
SPLIT_INVALID = "SPLIT_INVALID"


@dataclass(frozen=True)
class Fold:
    index: int
    train_ids: np.ndarray  # sorted _dm_row_id values
    test_ids: np.ndarray


@dataclass
class Splits:
    method: str
    n_splits: int
    seed: int
    folds: list[Fold]
    warnings: list[dict[str, Any]] = field(default_factory=list)
    # Class labels (stable order) and, per fold, row counts and class counts of the training
    # and test rows (aligned with ``classes``). Labels are data values: redact them for a
    # sensitive target before showing them.
    classes: list[Any] = field(default_factory=list)
    fold_stats: list[dict[str, Any]] = field(default_factory=list)

    def assignment(self) -> dict[int, int]:
        """Row id -> the fold in which it is a test row (-1: never tested, time method)."""
        out: dict[int, int] = {}
        for fold in self.folds:
            for row_id in fold.test_ids.tolist():
                out[int(row_id)] = fold.index
        for fold in self.folds:
            for row_id in fold.train_ids.tolist():
                out.setdefault(int(row_id), -1)
        return out

    def frames(self, df: pd.DataFrame, fold: Fold) -> tuple[pd.DataFrame, pd.DataFrame]:
        """(training rows, test rows) of ``df`` for one fold, in the original order."""
        ids = df[ROW_ID]
        return df[ids.isin(fold.train_ids)], df[ids.isin(fold.test_ids)]


def _fail(code: str, message: str, **details: Any) -> EngineError:
    return EngineError(code, message, details)


def _blocks(df: pd.DataFrame, group: str | None, duplicate_columns: Sequence[str]) -> np.ndarray:
    """Block label per row: rows sharing a group or identical in the duplicate columns end up
    in the same block (union of both relations)."""
    n = len(df)
    parent = np.arange(n)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = int(parent[i])
        return i

    def union_by(keys: np.ndarray) -> None:
        first: dict[Any, int] = {}
        for i, key in enumerate(keys.tolist()):
            if key in first:
                a, b = find(i), find(first[key])
                if a != b:
                    parent[max(a, b)] = min(a, b)
            else:
                first[key] = i

    if group is not None:
        union_by(pd.factorize(df[group].astype(object))[0])
    if duplicate_columns:
        hashes = pd.util.hash_pandas_object(df[list(duplicate_columns)], index=False)
        union_by(hashes.to_numpy())
    return np.array([find(i) for i in range(n)])


UNEVEN_RATIO = 3  # largest / smallest test fold above which folds are reported as uneven


def make_splits(
    df: pd.DataFrame,
    *,
    method: str,
    n_splits: int = 5,
    seed: int = 0,
    target: str | None = None,
    group: str | None = None,
    time: str | None = None,
    duplicate_columns: Sequence[str] = (),
) -> Splits:
    """Build the folds, then report per-fold sizes and class counts and warn about folds
    that lack a class or are very uneven. Unexpected library errors become ``SPLIT_FAILED``
    carrying only the error type (library messages can quote values)."""
    for name, value in (("n_splits", n_splits), ("seed", seed)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise _fail(SPLIT_INVALID, f"'{name}' must be a whole number.")
    try:
        splits = _make_splits(
            df,
            method=method,
            n_splits=n_splits,
            seed=seed,
            target=target,
            group=group,
            time=time,
            duplicate_columns=duplicate_columns,
        )
    except EngineError:
        raise
    except MemoryError:
        raise
    except Exception as exc:
        raise _fail(
            "SPLIT_FAILED",
            "The split failed unexpectedly. Details are withheld because they could contain "
            "data values.",
            error_type=type(exc).__name__,
        ) from exc
    _describe(splits, df, target)
    return splits


def _describe(splits: Splits, df: pd.DataFrame, target: str | None) -> None:
    ids = df[ROW_ID]
    labels = None
    if target is not None and df[target].notna().all():
        values = df[target].astype(object)
        try:
            splits.classes = sorted(pd.unique(values).tolist())
        except TypeError:
            splits.classes = sorted(
                pd.unique(values).tolist(), key=lambda v: (type(v).__name__, str(v))
            )
        if len(splits.classes) <= max(50, len(df) // 2):
            labels = values
        else:
            splits.classes = []
    missing_class = False
    for fold in splits.folds:
        train, test = ids.isin(fold.train_ids), ids.isin(fold.test_ids)
        stats: dict[str, Any] = {
            "fold": fold.index,
            "n_train": int(train.sum()),
            "n_test": int(test.sum()),
        }
        if labels is not None:
            stats["train_class_counts"] = [int((labels[train] == c).sum()) for c in splits.classes]
            stats["test_class_counts"] = [int((labels[test] == c).sum()) for c in splits.classes]
            if 0 in stats["train_class_counts"] or 0 in stats["test_class_counts"]:
                missing_class = True
        splits.fold_stats.append(stats)
    if missing_class:
        splits.warnings.append(
            {
                "code": "CLASS_MISSING_IN_FOLD",
                "message": "In some folds a class has no training or no test rows (groups or "
                "time order prevent balancing). Metrics for that class cannot be computed in "
                "those folds.",
            }
        )
    sizes = [s["n_test"] for s in splits.fold_stats]
    if sizes and min(sizes) > 0 and max(sizes) > UNEVEN_RATIO * min(sizes):
        splits.warnings.append(
            {
                "code": "UNEVEN_FOLDS",
                "message": f"The largest test fold is more than {UNEVEN_RATIO} times the "
                "smallest (unequal group or time-block sizes); fold results are not equally "
                "reliable.",
            }
        )


def _make_splits(
    df: pd.DataFrame,
    *,
    method: str,
    n_splits: int,
    seed: int,
    target: str | None,
    group: str | None,
    time: str | None,
    duplicate_columns: Sequence[str],
) -> Splits:
    if method not in SPLIT_METHODS:
        raise _fail(SPLIT_INVALID, "Unknown split method.")
    if not MIN_SPLITS <= n_splits <= MAX_SPLITS:
        raise _fail(SPLIT_INVALID, f"Use between {MIN_SPLITS} and {MAX_SPLITS} folds.")
    if ROW_ID not in df.columns or not df[ROW_ID].is_unique:
        raise _fail(SPLIT_INVALID, "Splits need unique row identifiers.")
    for name in [target, group, time, *duplicate_columns]:
        if name is not None and name not in df.columns:
            raise _fail(SPLIT_INVALID, "A column needed for the split is missing.", column=name)
    needs = {
        "stratified": target,
        "stratified_group": target,
        "group": group,
        "time": time,
    }
    if method in needs and needs[method] is None:
        raise _fail(SPLIT_INVALID, f"The '{method}' method needs its column.")
    if method in ("stratified_group",) and group is None:
        raise _fail(SPLIT_INVALID, "The 'stratified_group' method needs a group column.")
    if method == "time" and group is not None:
        raise _fail(SPLIT_INVALID, "Time-ordered folds with groups are not supported; choose one.")
    if group is not None and df[group].isna().any():
        raise _fail("SPLIT_GROUP_MISSING", "Some rows have no group value.", column=group)
    ids = df[ROW_ID].to_numpy()
    warnings: list[dict[str, Any]] = []

    if method == "time":
        return _time_splits(df, ids, time, n_splits, seed, duplicate_columns)  # type: ignore[arg-type]

    blocks = _blocks(df, group, duplicate_columns)
    grouped = len(np.unique(blocks)) < len(df)
    if len(np.unique(blocks)) < n_splits:
        raise _fail(
            "SPLIT_TOO_FEW_GROUPS",
            f"Fewer than {n_splits} independent groups of rows; use fewer folds.",
        )
    labels = None
    if method in ("stratified", "stratified_group"):
        y = df[target]
        if y.isna().any():
            raise _fail("SPLIT_TARGET_MISSING", "Stratification needs a label in every row.")
        labels = pd.factorize(y.astype(object))[0]
        fractional = pd.api.types.is_float_dtype(y) and bool((y % 1 != 0).any())
        if fractional or labels.max() + 1 > max(2, len(df) // 2):
            raise _fail(
                SPLIT_INVALID,
                "Stratification needs a class target; this target has too many distinct values "
                "(numeric target?). Use 'kfold' or 'group'.",
                column=target,
            )
        if np.bincount(labels).min() < n_splits:
            raise _fail(
                "SPLIT_CLASS_TOO_SMALL",
                f"A class has fewer rows than the {n_splits} folds; use fewer folds.",
                column=target,
            )
    position = np.arange(len(df)).reshape(-1, 1)
    splitter: Any
    if method == "kfold" and not grouped:
        splits = KFold(n_splits, shuffle=True, random_state=seed).split(position)
    elif method == "stratified" and not grouped:
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(position, labels)
    elif method in ("stratified", "stratified_group"):
        splitter = StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(position, labels, blocks)
    else:
        splitter = GroupKFold(n_splits, shuffle=True, random_state=seed)
        splits = splitter.split(position, groups=blocks)
    folds = [
        Fold(i, np.sort(ids[train]), np.sort(ids[test])) for i, (train, test) in enumerate(splits)
    ]
    if grouped and method in ("kfold", "stratified"):
        warnings.append(
            {
                "code": "DUPLICATES_KEPT_TOGETHER",
                "message": "Identical rows were kept on the same side of every split.",
            }
        )
    _check(folds, len(df))
    return Splits(method, n_splits, seed, folds, warnings)


def _time_splits(
    df: pd.DataFrame,
    ids: np.ndarray,
    time: str,
    n_splits: int,
    seed: int,
    duplicate_columns: Sequence[str],
) -> Splits:
    values = df[time]
    if values.isna().any():
        raise _fail("SPLIT_TIME_MISSING", "Some rows have no time value.", column=time)
    try:
        codes, uniques = pd.factorize(values, sort=True)
    except TypeError:
        raise _fail(
            "SPLIT_TIME_INVALID",
            "The time values cannot be ordered (mixed types, or time zones mixed with "
            "naive times).",
            column=time,
        ) from None
    if len(uniques) < n_splits + 1:
        raise _fail(
            "SPLIT_TOO_FEW_GROUPS",
            f"Fewer than {n_splits + 1} distinct times; use fewer folds.",
            column=time,
        )
    edges = np.linspace(0, len(uniques), n_splits + 2).round().astype(int)
    folds = []
    for i in range(n_splits):
        train = codes < edges[i + 1]
        test = (codes >= edges[i + 1]) & (codes < edges[i + 2])
        folds.append(Fold(i, np.sort(ids[train]), np.sort(ids[test])))
    warnings: list[dict[str, Any]] = [
        {
            "code": "EARLIEST_BLOCK_NOT_TESTED",
            "message": "Forward-chaining folds never test the earliest rows.",
        }
    ]
    if duplicate_columns:
        keys = pd.util.hash_pandas_object(df[list(duplicate_columns)], index=False).to_numpy()
        crossing = 0
        for fold in folds:
            train_keys = set(keys[np.isin(ids, fold.train_ids)].tolist())
            test_keys = set(keys[np.isin(ids, fold.test_ids)].tolist())
            crossing = max(crossing, len(train_keys & test_keys))
        if crossing:
            warnings.append(
                {
                    "code": "DUPLICATES_CROSS_TIME_SPLIT",
                    "message": f"Up to {crossing} duplicated rows appear on both sides of a "
                    "time split; time order is kept rather than grouping them.",
                }
            )
    for fold in folds:
        if np.intersect1d(fold.train_ids, fold.test_ids).size:
            raise _fail("POSTCONDITION_FAILED", "A row is in training and test rows.")
    return Splits("time", n_splits, seed, folds, warnings)


def _check(folds: list[Fold], n_rows: int) -> None:
    tested = np.concatenate([f.test_ids for f in folds])
    if len(tested) != n_rows or len(np.unique(tested)) != n_rows:
        raise _fail("POSTCONDITION_FAILED", "Every row must be tested exactly once.")
    for fold in folds:
        if np.intersect1d(fold.train_ids, fold.test_ids).size:
            raise _fail("POSTCONDITION_FAILED", "A row is in training and test rows.")

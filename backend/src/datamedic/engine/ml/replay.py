"""In-fold preprocessing for training: replayed committed steps plus the model-stage spec.

``FoldPreprocessor`` reproduces, inside a fold, what the platform did when it committed the
evaluated version, but with every learned parameter fitted on the fold's training rows only:

1. for each replayed commit (oldest first): fit its data-stage steps on the training rows,
   apply them to training *and* test rows (applying learns nothing), and pass the combined
   table through Parquet exactly like the platform did between versions;
2. then the model-stage specification: ``fit_resample`` on the training rows (resampling and
   class weights change only the training table) and ``transform`` on the test rows.

``prove_equivalence`` fits step 1 on *all* rows of the eligible ancestor and compares the
result's content hash with the evaluated version's: equality proves that the replayed
specification is the one that produced the version (fidelity). Safety is decided separately by
``engine.ml.eligibility``.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.commit import finalise
from datamedic.engine.preprocess.executor import PreprocessExecutor, TransformResult
from datamedic.engine.preprocess.validate import ValidationResult

REPLAY_MISMATCH = "REPLAY_NOT_ESTABLISHED"
RTOL, ATOL = 1e-9, 1e-12


@dataclass(frozen=True)
class ReplayCommit:
    validation: ValidationResult  # the commit's data-stage steps, re-validated
    seed: int


@dataclass
class Prepared:
    training: TransformResult  # model training table (may contain resampled rows)
    held_out: TransformResult  # test rows, transformed only
    executor: PreprocessExecutor  # fitted model-stage executor (feature columns, lineage)
    lineage: list[dict[str, Any]] = field(default_factory=list)  # replayed + model lineage
    fit_rows: list[int] = field(default_factory=list)  # rows each learned step was fitted on


class FoldPreprocessor:
    def __init__(self, commits: list[ReplayCommit], model: ValidationResult, seed: int) -> None:
        self.commits = commits
        self.model = model
        self.seed = seed

    def _replay(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], list[int]]:
        lineage: list[dict[str, Any]] = []
        fit_rows: list[int] = []
        if not self.commits:
            return train, test, lineage, fit_rows
        train_ids = train[ROW_ID].to_numpy()
        combined = pd.concat([train, test], axis=0)
        for commit in self.commits:
            executor = PreprocessExecutor(commit.validation, seed=commit.seed)
            current_train = combined[combined[ROW_ID].isin(train_ids)]
            executor.fit(current_train)
            result = executor.transform(combined)
            fit_rows.extend(r.n_fit_rows for r in result.reports)
            lineage.extend(result.lineage)
            _, combined, _ = finalise(result.frame)
        is_train = combined[ROW_ID].isin(train_ids).to_numpy()
        return combined[is_train], combined[~is_train], lineage, fit_rows

    def prepare(self, train: pd.DataFrame, test: pd.DataFrame) -> Prepared:
        """Fit everything on ``train`` only; transform ``test`` without learning."""
        r_train, r_test, lineage, fit_rows = self._replay(train, test)
        executor = PreprocessExecutor(self.model, seed=self.seed)
        training = executor.fit_resample(r_train)
        held_out = executor.transform(r_test)
        fit_rows.extend(r.n_fit_rows for r in training.reports)
        return Prepared(training, held_out, executor, lineage + training.lineage, fit_rows)

    def transform_only(self, train: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
        """Convenience for tests: the replayed table for ``rows`` with steps fitted on
        ``train``."""
        _, out, _, _ = self._replay(train, rows)
        return out


def prove_equivalence(
    ancestor: pd.DataFrame, commits: list[ReplayCommit], expected_hash: bytes
) -> dict[str, Any]:
    """Refit the replayed commits on every ancestor row and compare with the evaluated
    version. Returns ``{status, ...}``: ``exact`` (content hash equal),
    ``equivalent_within_tolerance`` (same rows, columns and non-float values; floats within
    ``allclose(rtol=1e-9, atol=1e-12)``) or raises ``REPLAY_NOT_ESTABLISHED``."""
    frame = ancestor
    digest = b""
    for commit in commits:
        executor = PreprocessExecutor(commit.validation, seed=commit.seed)
        result = executor.commit_transform(frame)
        _, frame, digest = finalise(result.frame)
    if digest == expected_hash:
        return {"status": "exact", "content_hash": digest.hex()}
    return {"status": "mismatch", "content_hash": digest.hex(), "frame": frame}


def compare_within_tolerance(replayed: pd.DataFrame, stored: pd.DataFrame) -> bool:
    """Same columns, rows and non-float values; float columns within the documented
    tolerance. Used only when the content hashes differ (e.g. another platform)."""
    if list(replayed.columns) != list(stored.columns) or len(replayed) != len(stored):
        return False
    a = replayed.sort_values(ROW_ID).reset_index(drop=True)
    b = stored.sort_values(ROW_ID).reset_index(drop=True)
    for column in a.columns:
        x, y = a[column], b[column]
        if pd.api.types.is_float_dtype(x) and pd.api.types.is_float_dtype(y):
            xv = x.to_numpy(dtype=float, na_value=np.nan)
            yv = y.to_numpy(dtype=float, na_value=np.nan)
            if not np.allclose(xv, yv, rtol=RTOL, atol=ATOL, equal_nan=True):
                return False
        elif not x.astype(object).equals(y.astype(object)):
            return False
    return True


def establish(
    ancestor: pd.DataFrame,
    commits: list[ReplayCommit],
    expected_hash: bytes,
    stored: pd.DataFrame,
) -> dict[str, Any]:
    """The fidelity proof the worker records before training on a replayed lineage."""
    proof = prove_equivalence(ancestor, commits, expected_hash)
    if proof["status"] == "exact":
        return proof
    if compare_within_tolerance(proof["frame"], stored):
        return {
            "status": "equivalent_within_tolerance",
            "content_hash": proof["content_hash"],
            "tolerance": {"rtol": RTOL, "atol": ATOL},
        }
    raise EngineError(
        REPLAY_MISMATCH,
        "Refitting the committed steps on the ancestor's rows does not reproduce the "
        "evaluated version, so the replay is not faithful.",
    )

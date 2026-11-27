"""Class-imbalance ops (Part C; model stage, training rows only).

These ops are ``training_only``: ``transform`` returns every row set unchanged (validation
and test rows, previews, committed versions), and only ``resample`` — called by
``PreprocessExecutor.fit_resample`` on the exact rows the step was fitted on — builds the
training table. ``resample`` refuses any other rows, so test rows can never be resampled.

Class labels and counts are reported under ``columns[<target>]`` so they are redacted when
the target is a sensitive column. Algorithms live in ``preprocess.runtime``.
"""

from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.base import DMTransformer
from datamedic.engine.preprocess.catalog import SAMPLE_WEIGHT, Limits
from datamedic.engine.preprocess.ops.impute import _check_fit_rows

SMALL_CLASS = 10


class _ImbalanceOp(DMTransformer):
    fit_cap = Limits.RESAMPLE_MAX_FIT_ROWS

    @property
    def target(self) -> str:
        return str(self.param("target"))

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        _check_fit_rows(self.op, len(X), self.fit_cap)
        self.state_ = rt.fit_class_counts(X, self.target)
        self.fit_row_ids_ = X[ROW_ID].to_numpy(copy=True)
        self.after_counts_: list[int] | None = None
        self.rows_after_: int | None = None
        counts = self.state_["counts"]
        if min(counts) < SMALL_CLASS:
            self.warn(
                "SMALL_CLASS",
                f"A class has fewer than {SMALL_CLASS} training rows; balancing cannot create "
                "information that is not there, and results will be unstable.",
                self.target,
            )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return X.copy()  # other rows are never resampled or weighted

    def _resample(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        ids = X[ROW_ID].to_numpy()
        if ids.shape != self.fit_row_ids_.shape or not np.array_equal(ids, self.fit_row_ids_):
            raise rt.PreprocessError(
                "RESAMPLE_NOT_FIT_ROWS",
                "Resampling and class weights apply only to the rows the step was fitted on "
                "(the training rows).",
            )
        out = self._build(X)
        classes, codes = rt.class_labels(out, self.target)
        lookup = {c: j for j, c in enumerate(self.state_["classes"])}
        after = [0] * len(self.state_["classes"])
        for c, n in zip(classes, np.bincount(codes, minlength=len(classes)), strict=True):
            after[lookup[c]] = int(n)
        self.after_counts_ = after
        self.rows_after_ = len(out)
        self.effects_ = {
            "rows_before": len(X),
            "rows_after": len(out),
            "rows_added": int((out[ROW_ID] < 0).sum()),
            "rows_removed": int(len(X) - (out[ROW_ID] >= 0).sum()),
        }
        return out

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        raise NotImplementedError

    def _extra(self) -> dict[str, Any]:
        return {}

    def learned_params(self) -> dict[str, Any]:
        info = {
            "classes": list(self.state_["classes"]),
            "counts_before": list(self.state_["counts"]),
            "counts_after": None if self.after_counts_ is None else list(self.after_counts_),
            **self._extra(),
        }
        return {
            "strategy": self.param("sampling_strategy"),
            "seed": int(self.seed),
            "columns": {self.target: info},
        }


class ClassWeightOp(_ImbalanceOp):
    op = "imbalance.class_weight"

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.resample_class_weight(X, self.target, self.state_, SAMPLE_WEIGHT)

    def _extra(self) -> dict[str, Any]:
        return {"weights": rt.class_weights(self.state_["counts"])}

    def learned_params(self) -> dict[str, Any]:
        out = super().learned_params()
        out["strategy"] = "balanced"
        out["weight_column"] = SAMPLE_WEIGHT
        return out


class RandomOversampleOp(_ImbalanceOp):
    op = "imbalance.random_oversample"

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.resample_random(
            X,
            self.target,
            "oversample",
            str(self.param("sampling_strategy")),
            int(self.seed),
            Limits.RESAMPLE_MAX_OUTPUT_ROWS,
            Limits.RESAMPLE_MAX_OUTPUT_CELLS,
        )


class RandomUndersampleOp(_ImbalanceOp):
    op = "imbalance.random_undersample"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        super()._fit(X)
        removed = sum(self.state_["counts"]) - rt.expected_rows(
            self.state_["counts"], "undersample", str(self.param("sampling_strategy"))
        )
        if removed:
            self.warn(
                "ROWS_DISCARDED",
                f"{removed} real training rows will be discarded from the training table.",
            )

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.resample_random(
            X,
            self.target,
            "undersample",
            str(self.param("sampling_strategy")),
            int(self.seed),
            Limits.RESAMPLE_MAX_OUTPUT_ROWS,
            Limits.RESAMPLE_MAX_OUTPUT_CELLS,
        )


class SmoteOp(_ImbalanceOp):
    op = "imbalance.smote"
    fit_cap = Limits.SMOTE_MAX_FIT_ROWS

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        super()._fit(X)
        k = int(self.param("k_neighbors"))
        counts = self.state_["counts"]
        # Checked before any resampling so an impossible configuration fails at fit time.
        for j in rt.oversampled_classes(counts, str(self.param("sampling_strategy"))):
            if counts[j] <= k:
                raise rt.PreprocessError(
                    "SMOTE_TOO_FEW_SAMPLES",
                    f"A class to be oversampled has {counts[j]} training rows; SMOTE with "
                    f"k_neighbors={k} needs more than {k}. Lower k_neighbors or use random "
                    "oversampling.",
                    self.target,
                )
        self.warn(
            "SYNTHETIC_ROWS",
            "SMOTE adds synthetic rows that are not real observations. They exist only in the "
            "training table and are never saved into a dataset version.",
        )
        if self.param("dependent_rows"):
            self.warn(
                "GROUPED_OR_TEMPORAL_DATA",
                "Rows are not independent (patient, group, identifier or time columns): "
                "synthetic rows blend different patients, groups or times.",
            )

    def _build(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.resample_smote(
            X,
            self.target,
            list(self.param("passthrough") or []),
            str(self.param("sampling_strategy")),
            int(self.param("k_neighbors")),
            int(self.seed),
            Limits.SMOTE_MAX_FEATURES,
            Limits.RESAMPLE_MAX_OUTPUT_ROWS,
            Limits.RESAMPLE_MAX_OUTPUT_CELLS,
        )

"""Scaling ops (Part B; model stage only). scikit-learn scalers fitted on the fitting rows and
applied unchanged to any rows; values outside the fitting range are counted, not hidden."""

from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.base import DMTransformer


class _ScaleOp(DMTransformer):
    kind = ""

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self.state_ = rt.fit_scaler(X, self.cols, self.kind, bool(self.param("clip") or False))
        matrix = rt.numeric_matrix(X, self.cols)
        self.fit_range_ = (np.nanmin(matrix, axis=0), np.nanmax(matrix, axis=0))
        for column, zero in zip(self.cols, self.state_["zero_spread"], strict=True):
            if zero:
                self.warn(
                    "CONSTANT_COLUMN",
                    "The column has no spread in the fitting rows; it is only shifted (scale 1), "
                    "so other rows keep their distance from the fitting value.",
                    column,
                )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        matrix = rt.numeric_matrix(X, self.cols)
        low, high = self.fit_range_
        with np.errstate(invalid="ignore"):
            outside = (matrix < low) | (matrix > high)
        self.effects_ = {
            "outside_fit_range_rows": {c: int(outside[:, j].sum()) for j, c in enumerate(self.cols)}
        }
        return rt.apply_scaler(X, self.cols, self.state_)

    def _per_column(self, **arrays: Any) -> dict[str, Any]:
        return {
            "columns": {
                c: {key: float(values[j]) for key, values in arrays.items()}
                for j, c in enumerate(self.cols)
            }
        }


class StandardScaleOp(_ScaleOp):
    op = "scale.standard"
    kind = "standard"

    def learned_params(self) -> dict[str, Any]:
        s = self.state_["scaler"]
        return self._per_column(mean=s.mean_, scale=s.scale_)


class MinMaxScaleOp(_ScaleOp):
    op = "scale.minmax"
    kind = "minmax"

    def learned_params(self) -> dict[str, Any]:
        s = self.state_["scaler"]
        return {**self._per_column(min=s.data_min_, max=s.data_max_), "clip": bool(s.clip)}


class RobustScaleOp(_ScaleOp):
    op = "scale.robust"
    kind = "robust"

    def learned_params(self) -> dict[str, Any]:
        s = self.state_["scaler"]
        return self._per_column(median=s.center_, scale=s.scale_)


class NoScaleOp(DMTransformer):
    op = "scale.none"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_no_scaling(X)

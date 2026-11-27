"""Outlier ops (Part B). Detection rules (iqr, zscore, mad, percentile) are a parameter;
``outliers.flag`` and ``outliers.cap`` are the treatments. Algorithms live in
``preprocess.runtime``; these classes add warnings, limits, learned parameters and effect
counts (never values)."""

from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.base import DMTransformer
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.ops.impute import _check_columns, _check_fit_rows

SKEW_WARNING = 1.0
PERCENTILE_SMALL_SAMPLE = 100


class _BoundsOp(DMTransformer):
    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        method = str(self.param("method"))
        self.state_ = rt.fit_bounds(
            X, self.cols, method, dict(self.params or {}), Limits.OUTLIER_MIN_VALUES
        )
        for column, info in zip(self.cols, self.state_["stats"], strict=True):
            if info.get("zero_spread"):
                self.warn(
                    "ZERO_SPREAD",
                    "The fitting rows have no spread under this rule (e.g. a constant or mostly "
                    "identical column), so no bounds exist and the column is left unchanged.",
                    column,
                )
            skew = info.get("skewness")
            if method in ("iqr", "zscore") and skew is not None and abs(skew) >= SKEW_WARNING:
                self.warn(
                    "SKEWED_COLUMN",
                    "The column is skewed in the fitting rows; this symmetric rule marks many "
                    "ordinary values in the long tail. Consider the percentile rule or a "
                    "transformation.",
                    column,
                )
            if method == "percentile" and info["n_values"] < PERCENTILE_SMALL_SAMPLE:
                self.warn(
                    "SMALL_SAMPLE_PERCENTILE",
                    f"With fewer than {PERCENTILE_SMALL_SAMPLE} values, percentile bounds lie "
                    "close to the smallest and largest observed values.",
                    column,
                )

    def _bound_for(self, X: pd.DataFrame, column: str, bound: list[float] | None) -> Any:  # noqa: N803
        return bound

    def _count_effects(self, X: pd.DataFrame) -> None:  # noqa: N803
        per_column: dict[str, dict[str, Any]] = {}
        any_row = np.zeros(len(X), dtype=bool)
        for column, bound in zip(self.cols, self.state_["bounds"], strict=True):
            values = rt.numeric_matrix(X, [column])[:, 0]
            effective = self._bound_for(X, column, bound)
            if effective is None:
                below = above = np.zeros(len(X), dtype=bool)
            else:
                with np.errstate(invalid="ignore"):
                    below, above = values < effective[0], values > effective[1]
            any_row |= below | above
            affected = int((below | above).sum())
            per_column[column] = {
                "below": int(below.sum()),
                "above": int(above.sum()),
                "affected_rows": affected,
                "affected_pct": round(100 * affected / len(X), 4) if len(X) else 0.0,
            }
        self.effects_ = {"columns": per_column, "rows_affected": int(any_row.sum())}

    def learned_params(self) -> dict[str, Any]:
        columns = {}
        for column, bound, info in zip(
            self.cols, self.state_["bounds"], self.state_["stats"], strict=True
        ):
            columns[column] = {
                "lower": None if bound is None else bound[0],
                "upper": None if bound is None else bound[1],
                **info,
            }
        return {"method": self.param("method"), "columns": columns}


class OutlierFlagOp(_BoundsOp):
    op = "outliers.flag"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        self._count_effects(X)
        return rt.apply_outlier_flag(X, self.cols, self.state_)


class OutlierCapOp(_BoundsOp):
    op = "outliers.cap"

    def _bound_for(self, X: pd.DataFrame, column: str, bound: list[float] | None) -> Any:  # noqa: N803
        if bound is not None and pd.api.types.is_integer_dtype(X[column]):
            return list(rt.integer_bounds(bound))
        return bound

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        self._count_effects(X)
        return rt.apply_outlier_cap(X, self.cols, self.state_)


class SignedLogOp(DMTransformer):
    op = "outliers.signed_log"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_signed_log(X, self.cols)


class IsolationForestFlagOp(DMTransformer):
    op = "outliers.isolation_forest_flag"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        _check_fit_rows(self.op, len(X), Limits.IF_MAX_FIT_ROWS)
        _check_columns(self.op, self.cols, [], Limits.IF_MAX_COLUMNS)
        self.state_ = rt.fit_isolation_forest(
            X,
            self.cols,
            int(self.param("n_estimators")),
            float(self.param("contamination")),
            int(self.seed),
            Limits.IF_MIN_ROWS,
        )
        for column in self.cols:
            if column not in self.state_["used"]:
                self.warn(
                    "CONSTANT_COLUMN_DROPPED",
                    "The column is constant in the fitting rows, so it cannot separate rows "
                    "and was left out of the model.",
                    column,
                )
        if not self.state_["used"]:
            self.warn(
                "ZERO_SPREAD",
                "No selected column varies in the fitting rows; no model was fitted and no "
                "row is flagged.",
            )
        unused = len(X) - self.state_["n_complete"]
        if unused:
            self.warn(
                "ROWS_NOT_USED",
                f"{unused} fitting rows with a missing value in a selected column were not used.",
            )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out = rt.apply_isolation_forest(X, self.cols, str(self.param("output_column")), self.state_)
        flags = out[str(self.param("output_column"))]
        complete = X[self.cols].notna().all(axis=1)
        self.effects_ = {
            "rows_flagged": int(flags.sum()),
            "rows_not_scored": int((~complete).sum()),
        }
        return out

    def learned_params(self) -> dict[str, Any]:
        return {
            "n_estimators": int(self.param("n_estimators")),
            "contamination": float(self.param("contamination")),
            "columns_used": list(self.state_["used"]),
            "score_offset": self.state_["offset"],
            "n_complete_fit_rows": self.state_["n_complete"],
            "seed": int(self.seed),
        }

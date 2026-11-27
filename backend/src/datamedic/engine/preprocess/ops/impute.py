"""Missing-value ops (PHASE4_PLAN §9.1). Algorithms live in ``preprocess.runtime``; these
classes add the scikit-learn interface, cost limits, warnings and learned-parameter
reporting."""

from typing import Any

import pandas as pd

from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.base import OP_LIMIT_EXCEEDED, DMTransformer
from datamedic.engine.preprocess.catalog import Limits


class _SimpleNumeric(DMTransformer):
    strategy = ""

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self.state_ = rt.fit_simple_numeric(X, self.cols, self.strategy)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_simple_numeric(X, self.cols, self.state_)

    def learned_params(self) -> dict[str, Any]:
        return {
            "columns": {
                c: {"statistic": v}
                for c, v in zip(self.cols, self.state_["statistics"], strict=True)
            }
        }


class MeanImputer(_SimpleNumeric):
    op = "impute.mean"
    strategy = "mean"


class MedianImputer(_SimpleNumeric):
    op = "impute.median"
    strategy = "median"


class ModeImputer(DMTransformer):
    op = "impute.mode"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self.state_ = rt.fit_mode(X, self.cols)
        for column, tie in zip(self.cols, self.state_["ties"], strict=True):
            if tie:
                self.warn(
                    "MODE_TIE",
                    "Several values are equally most frequent; the smallest was chosen.",
                    column,
                )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_mode(X, self.cols, self.state_)

    def learned_params(self) -> dict[str, Any]:
        return {
            "columns": {
                c: {"value": v, "tie": t}
                for c, v, t in zip(
                    self.cols, self.state_["values"], self.state_["ties"], strict=True
                )
            }
        }


class ConstantImputer(DMTransformer):
    op = "impute.constant"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_constant(X, self.cols, self.param("fill_value"))

    def learned_params(self) -> dict[str, Any]:
        # Nothing is learned; the configured value is reported so the report is complete.
        return {"columns": {c: {"value": self.param("fill_value")} for c in self.cols}}


def _check_columns(op: str, columns: list[str], predictors: list[str], cap: int) -> None:
    """Re-checked at fit time (validation already enforces it) so the cap never depends on
    the caller having validated."""
    if len(dict.fromkeys([*columns, *predictors])) > cap:
        raise EngineError(
            OP_LIMIT_EXCEEDED,
            f"This operation uses at most {cap} columns in total.",
            {"op": op, "limit": cap},
        )


def _check_fit_rows(op: str, n_rows: int, cap: int) -> None:
    if n_rows > cap:
        raise EngineError(
            OP_LIMIT_EXCEEDED,
            f"This operation fits on at most {cap:,} rows; it was given {n_rows:,}.",
            {"op": op, "limit": cap},
        )


class _ModelBased(DMTransformer):
    def _record_predictors(self) -> None:
        for column in self.state_["dropped_predictors"]:
            self.warn(
                "PREDICTOR_EMPTY",
                "A predictor had no observed value in the fitting rows and was not used.",
                column,
            )
        if len(self.state_["names"]) == len(self.cols) and len(self.cols) == 1:
            self.warn(
                "NO_PREDICTORS",
                "No predictor was available; estimates use only this column's own values.",
                self.cols[0],
            )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return self.apply_fn(X, self.cols, self.state_)

    def apply_fn(self, X: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:  # noqa: N803
        raise NotImplementedError

    def _predictors_info(self) -> dict[str, Any]:
        return {
            "predictors_used": [n for n in self.state_["names"] if n not in self.cols],
            "dropped_predictors": list(self.state_["dropped_predictors"]),
        }


class KNNImputeOp(_ModelBased):
    op = "impute.knn"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        _check_fit_rows(self.op, len(X), Limits.KNN_MAX_FIT_ROWS)
        _check_columns(
            self.op, self.cols, list(self.param("predictors") or []), Limits.KNN_MAX_PREDICTORS
        )
        k = int(self.param("n_neighbors"))
        self.state_ = rt.fit_knn(
            X, self.cols, list(self.param("predictors") or []), k, str(self.param("weights"))
        )
        self._record_predictors()
        if len(X) < k:
            self.warn(
                "FEW_FIT_ROWS",
                f"Only {len(X)} fitting rows for {k} neighbours; fewer neighbours are used.",
            )

    def apply_fn(self, X: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:  # noqa: N803
        # Checked before any matrix is built: scikit-learn computes distances from every row
        # with a gap (in any used column) to every fitting row.
        needing = int(X[state["names"]].isna().any(axis=1).sum())
        pairs = needing * int(state["n_fit_rows"])
        if pairs > Limits.KNN_MAX_DISTANCE_PAIRS:
            raise EngineError(
                OP_LIMIT_EXCEEDED,
                f"Imputing {needing:,} rows against {state['n_fit_rows']:,} fitting rows exceeds "
                f"the limit of {Limits.KNN_MAX_DISTANCE_PAIRS:,} neighbour comparisons.",
                {"op": self.op, "limit": Limits.KNN_MAX_DISTANCE_PAIRS},
            )
        return rt.apply_knn(X, columns, state)

    def learned_params(self) -> dict[str, Any]:
        names = self.state_["names"]
        return {
            **self._predictors_info(),
            "n_neighbors": int(self.param("n_neighbors")),
            "weights": self.param("weights"),
            "columns": {
                c: {
                    "scale_mean": self.state_["mean"][names.index(c)],
                    "scale_std": self.state_["std"][names.index(c)],
                }
                for c in self.cols
            },
        }


class IterativeImputeOp(_ModelBased):
    op = "impute.iterative"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        _check_fit_rows(self.op, len(X), Limits.ITERATIVE_MAX_FIT_ROWS)
        _check_columns(
            self.op,
            self.cols,
            list(self.param("predictors") or []),
            Limits.ITERATIVE_MAX_PREDICTORS,
        )
        self.state_ = rt.fit_iterative(
            X,
            self.cols,
            list(self.param("predictors") or []),
            int(self.param("max_iter")),
            bool(self.param("clip_to_fit_range")),
            int(self.seed),
        )
        self._record_predictors()
        if not self.state_["converged"]:
            self.warn(
                "NOT_CONVERGED",
                "The estimates had not stabilised after the maximum number of rounds; consider "
                "more rounds or a simpler method.",
            )

    def apply_fn(self, X: pd.DataFrame, columns: list[str], state: dict[str, Any]) -> pd.DataFrame:  # noqa: N803
        return rt.apply_iterative(X, columns, state)

    def learned_params(self) -> dict[str, Any]:
        names = self.state_["names"]
        clip = bool(self.param("clip_to_fit_range"))
        return {
            **self._predictors_info(),
            "max_iter": int(self.param("max_iter")),
            "n_iter": self.state_["n_iter"],
            "converged": self.state_["converged"],
            "seed": int(self.seed),
            "columns": {
                c: (
                    {
                        "clip_min": self.state_["low"][names.index(c)],
                        "clip_max": self.state_["high"][names.index(c)],
                    }
                    if clip
                    else {}
                )
                for c in self.cols
            },
        }


class MissingIndicatorOp(DMTransformer):
    op = "missing.indicator"

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_indicator(X, self.cols)

    def learned_params(self) -> dict[str, Any]:
        return {"added_columns": [c + rt.INDICATOR_SUFFIX for c in self.cols]}

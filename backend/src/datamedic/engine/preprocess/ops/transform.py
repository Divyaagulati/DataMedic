"""Distribution transforms (Part C). Algorithms live in ``preprocess.runtime``; these classes
add limits, warnings, learned parameters and before/after distribution statistics.

The statistics describe the rows being transformed (skewness, excess kurtosis, a tail ratio
and the number of distinct values) so the report shows whether the transform helped, how
much it compressed the tails, and whether it merged distinct values (information loss). They
are never presented as an improvement by default: a transform that does not reduce skewness
is reported as such.
"""

from typing import Any

import numpy as np
import pandas as pd

from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.base import DMTransformer
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.ops.impute import _check_columns, _check_fit_rows


def _round(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def distribution_stats(values: np.ndarray) -> dict[str, Any]:
    """Shape statistics of the observed values: skewness, excess kurtosis, the tail ratio
    (q99 - q50) / (q50 - q01) (1 = symmetric tails; scale-free) and distinct values."""
    observed = values[~np.isnan(values)]
    n = int(observed.size)
    stats: dict[str, Any] = {
        "n": n,
        "skewness": None,
        "excess_kurtosis": None,
        "tail_ratio": None,
        "distinct": int(np.unique(observed).size),
    }
    if n >= 3:
        series = pd.Series(observed)
        with np.errstate(all="ignore"):
            stats["skewness"] = _round(series.skew())
            stats["excess_kurtosis"] = _round(series.kurt()) if n >= 4 else None
            q01, q50, q99 = np.quantile(observed, [0.01, 0.5, 0.99])
            if q50 - q01 > 0:
                stats["tail_ratio"] = _round((q99 - q50) / (q50 - q01))
    return stats


def outside_fit_range(
    op: DMTransformer,
    X: pd.DataFrame,
    fit_range: list[list[float]],
    what: str,  # noqa: N803
) -> dict[str, int]:
    """Rows per column outside the fitting range (counts only), with a warning."""
    outside = {}
    for column, (low, high) in zip(op.cols, fit_range, strict=True):
        x = rt.numeric_matrix(X, [column])[:, 0]
        with np.errstate(invalid="ignore"):
            outside[column] = int(((x < low) | (x > high)).sum())
    if any(outside.values()):
        op.transform_warnings_.append(
            {"code": "OUTSIDE_FIT_RANGE", "message": what, "column": None}
        )
    return outside


class _DistributionOp(DMTransformer):
    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        raise NotImplementedError

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        out = self._apply(X)
        distribution: dict[str, Any] = {}
        not_reduced: list[str] = []
        merged: list[str] = []
        for column in self.cols:
            before = distribution_stats(rt.numeric_matrix(X, [column])[:, 0])
            after = distribution_stats(rt.numeric_matrix(out, [column])[:, 0])
            distribution[column] = {
                "before": before,
                "after": after,
                "distinct_values_merged": before["distinct"] - after["distinct"],
            }
            sb, sa = before["skewness"], after["skewness"]
            if before["distinct"] > 1 and sb is not None and sa is not None and abs(sa) >= abs(sb):
                not_reduced.append(column)
            if after["distinct"] < before["distinct"]:
                merged.append(column)
        self.effects_ = {"distribution": distribution, **self._extra_effects(X)}
        for column in not_reduced:
            self.transform_warnings_.append(
                {
                    "code": "SKEW_NOT_REDUCED",
                    "message": "In these rows the transformation did not reduce the "
                    "column's skewness; it may not help.",
                    "column": column,
                }
            )
        for column in merged:
            self.transform_warnings_.append(
                {
                    "code": "VALUES_MERGED",
                    "message": "Distinct values became equal after the transformation "
                    "(information loss).",
                    "column": column,
                }
            )
        return out

    def _extra_effects(self, X: pd.DataFrame) -> dict[str, Any]:  # noqa: N803
        return {}


class LogOp(_DistributionOp):
    op = "transform.log"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        rt.apply_log(X, self.cols, str(self.param("base")))  # refuses zero/negative early

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_log(X, self.cols, str(self.param("base")))


class Log1pOp(_DistributionOp):
    op = "transform.log1p"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        rt.apply_log1p(X, self.cols)

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_log1p(X, self.cols)


class YeoJohnsonOp(_DistributionOp):
    op = "transform.yeo_johnson"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        _check_fit_rows(self.op, len(X), Limits.POWER_MAX_FIT_ROWS)
        _check_columns(self.op, self.cols, [], Limits.POWER_MAX_COLUMNS)
        self.state_ = rt.fit_yeo_johnson(X, self.cols, Limits.TRANSFORM_MIN_VALUES)
        for column, constant in zip(self.cols, self.state_["constant"], strict=True):
            if constant:
                self.warn(
                    "CONSTANT_COLUMN",
                    "The column is constant in the fitting rows; lambda is 1, so it is left "
                    "unchanged.",
                    column,
                )

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_yeo_johnson(X, self.cols, self.state_)

    def _extra_effects(self, X: pd.DataFrame) -> dict[str, Any]:  # noqa: N803
        outside = outside_fit_range(
            self,
            X,
            self.state_["fit_range"],
            "Some values lie outside the range of the fitting rows; the power transformation "
            "extrapolates there, and with lambda far from 1 the results can be extreme.",
        )
        return {"outside_fit_range_rows": outside}

    def learned_params(self) -> dict[str, Any]:
        return {
            "columns": {
                c: {"lambda": lam, "constant": const, "n_values": n}
                for c, lam, const, n in zip(
                    self.cols,
                    self.state_["lambdas"],
                    self.state_["constant"],
                    self.state_["n_values"],
                    strict=True,
                )
            }
        }


class QuantileOp(_DistributionOp):
    op = "transform.quantile"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        _check_fit_rows(self.op, len(X), Limits.QUANTILE_MAX_FIT_ROWS)
        requested = int(self.param("n_quantiles"))
        self.state_ = rt.fit_quantile(
            X,
            self.cols,
            requested,
            str(self.param("output_distribution")),
            int(self.seed),
            Limits.QUANTILE_SUBSAMPLE,
            Limits.TRANSFORM_MIN_VALUES,
        )
        for column, used, n in zip(
            self.cols, self.state_["n_quantiles"], self.state_["n_values"], strict=True
        ):
            if used == 0:
                self.warn(
                    "CONSTANT_COLUMN",
                    "The column is constant in the fitting rows: that value maps to the middle "
                    "of the output, smaller values to the lower end and larger values to the "
                    "upper end.",
                    column,
                )
            elif used < requested:
                self.warn(
                    "QUANTILES_REDUCED",
                    f"Only {n} observed fitting values: {used} quantiles were estimated instead "
                    f"of {requested}.",
                    column,
                )
            if n > Limits.QUANTILE_SUBSAMPLE:
                self.warn(
                    "QUANTILES_SUBSAMPLED",
                    f"Quantiles were estimated from a seeded sample of "
                    f"{Limits.QUANTILE_SUBSAMPLE:,} fitting values.",
                    column,
                )

    def _apply(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return rt.apply_quantile(X, self.cols, self.state_)

    def _extra_effects(self, X: pd.DataFrame) -> dict[str, Any]:  # noqa: N803
        outside = outside_fit_range(
            self,
            X,
            self.state_["fit_range"],
            "Some values lie outside the range of the fitting rows and were mapped to the ends "
            "of the output (their differences are lost).",
        )
        return {"outside_fit_range_rows": outside}

    def learned_params(self) -> dict[str, Any]:
        # The quantiles themselves are interpolated fitting values and are never reported.
        return {
            "output_distribution": self.state_["output_distribution"],
            "columns": {
                c: {"n_quantiles": q, "n_values": n, "constant": q == 0}
                for c, q, n in zip(
                    self.cols, self.state_["n_quantiles"], self.state_["n_values"], strict=True
                )
            },
        }

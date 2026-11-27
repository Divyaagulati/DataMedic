"""Feature selection (Part C; model stage only). Every selector is fitted on the fitting
rows (a training fold) and, with ``action='drop'``, removes only columns it was given and
that are not kept by the user or by a domain role. The default ``action='report'`` removes
nothing.

Learned parameters hold scores per column (redacted for sensitive columns); ``decisions``
holds only the decision per column (kept, would_remove, removed, kept_by_user,
kept_domain_role, not_assessed), so feature lineage stays visible for every column.
Supervised selectors read the target of the fitting rows only.
"""

from typing import Any

import pandas as pd

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.base import OP_INVALID, OP_LIMIT_EXCEEDED, DMTransformer
from datamedic.engine.preprocess.catalog import Limits
from datamedic.engine.preprocess.ops.impute import _check_columns, _check_fit_rows

SMALL_CLASS = 5  # fitting rows per class below which supervised scores are unreliable

DOUBT_TEXT = {
    "normality": "a class group is strongly skewed (ANOVA assumes roughly normal groups); "
    "consider 'kruskal'",
    "equal_variances": "class groups have very different variances (ANOVA assumes similar "
    "variances); consider 'kruskal'",
    "expected_counts": "more than 20% of expected counts are below 5, or one is below 1 "
    "(the chi-square approximation is unreliable); consider merging rare categories",
    "linearity_outliers": "the column or target is strongly skewed, so a linear test is "
    "dominated by extreme values; consider 'spearman'",
}


class _SelectOp(DMTransformer):
    """Shared: target guard, keep lists, removal at transform, decisions and effects."""

    def keep(self) -> list[str]:
        return list(self.param("keep") or [])

    def role_kept(self) -> list[str]:
        return list(self.param("role_kept") or [])

    def _guard_target(self) -> None:
        target = self.param("target")
        if target is not None and (target in self.cols or target == ROW_ID):
            raise EngineError(
                OP_INVALID,
                "The target can never be an input of feature selection.",
                {"op": self.op, "column": target},
            )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        drop = list(self.state_["drop"])
        self.effects_ = {
            "columns_removed": len(drop),
            "columns_kept": len(self.cols) - len(drop),
        }
        return rt.apply_drop_columns(X, drop)

    def decisions(self) -> dict[str, str]:
        return dict(self.state_["decisions"])

    def _warn_proposals(self) -> None:
        decisions = self.state_["decisions"]
        if any(d == "kept_domain_role" for d in decisions.values()):
            self.warn(
                "DOMAIN_ROLE_KEPT",
                "Columns with a domain role (e.g. a biomarker) failed the criterion but were "
                "kept: selection never removes them automatically.",
            )
        if decisions and all(d in ("removed", "removed_by_user") for d in decisions.values()):
            self.warn(
                "ALL_SELECTED_REMOVED",
                "Every selected column was removed from the model's input in this training "
                "fold. Check the threshold or k.",
            )
        if self.param("action") == "drop" and any(d == "removed" for d in decisions.values()):
            self.warn(
                "COLUMNS_REMOVED",
                "Columns were removed from the model's input in this training fold. Removal "
                "can discard meaningful information; compare with 'report' mode.",
            )


class VarianceSelectOp(_SelectOp):
    op = "select.variance"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self.state_ = rt.fit_variance(
            X,
            self.cols,
            float(self.param("threshold")),
            str(self.param("action")),
            self.keep(),
            self.role_kept(),
        )
        self._warn_proposals()

    def learned_params(self) -> dict[str, Any]:
        return {
            "threshold": float(self.param("threshold")),
            "action": self.param("action"),
            "decisions": self.decisions(),
            "columns": {
                c: {"variance": v, "n_values": n}
                for c, v, n in zip(
                    self.cols, self.state_["variances"], self.state_["n_values"], strict=True
                )
            },
        }


class CorrelationSelectOp(_SelectOp):
    op = "select.correlation"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        _check_fit_rows(self.op, len(X), Limits.CORR_MAX_FIT_ROWS)
        _check_columns(self.op, self.cols, [], Limits.CORR_MAX_COLUMNS)
        if len(X) * len(self.cols) > Limits.CORR_MAX_CELLS:
            raise EngineError(
                OP_LIMIT_EXCEEDED,
                f"This operation uses at most {Limits.CORR_MAX_CELLS:,} cells (fitting rows x "
                "columns). Select fewer columns.",
                {"op": self.op, "limit": Limits.CORR_MAX_CELLS},
            )
        self.state_ = rt.fit_correlation(
            X,
            self.cols,
            float(self.param("threshold")),
            str(self.param("method")),
            self.keep(),
            self.role_kept(),
            Limits.CORR_MIN_PAIRS,
        )
        proposed = set(self.state_["proposed"])
        protected = set(self.keep()) | set(self.role_kept())
        in_pairs = {c for pair in self.state_["pairs"] for c in pair["columns"]}
        decisions = {}
        for column in self.cols:
            if column in proposed:
                decisions[column] = "would_remove"
            elif column in in_pairs and column in self.role_kept():
                decisions[column] = "kept_domain_role"
            elif column in in_pairs and column in protected:
                decisions[column] = "kept_by_user"
            else:
                decisions[column] = "kept"
        self.state_["decisions"] = decisions
        self.state_["drop"] = []  # proposals only: removal is the user's decision
        if self.state_["pairs"]:
            self.warn(
                "CORRELATED_PAIRS",
                f"{len(self.state_['pairs'])} highly correlated pairs found. Nothing was "
                "removed: correlated columns can each carry meaningful information. Review the "
                "proposals and remove columns yourself if appropriate.",
            )
        if self.state_["pairs_not_assessed"]:
            self.warn(
                "PAIRS_NOT_ASSESSED",
                f"{self.state_['pairs_not_assessed']} pairs could not be assessed (constant "
                f"column or fewer than {Limits.CORR_MIN_PAIRS} rows with both values).",
            )
        self._warn_proposals()

    def learned_params(self) -> dict[str, Any]:
        return {
            "threshold": float(self.param("threshold")),
            "method": self.param("method"),
            "decisions": self.decisions(),
            "proposed_removals": list(self.state_["proposed"]),
            "pairs": [dict(pair) for pair in self.state_["pairs"]],
            "pairs_not_assessed": self.state_["pairs_not_assessed"],
            "columns": {
                c: {"max_abs_correlation": m}
                for c, m in zip(self.cols, self.state_["max_abs_correlation"], strict=True)
            },
        }


class _SupervisedSelectOp(_SelectOp):
    def _common_fit(self, X: pd.DataFrame, cap: int) -> None:  # noqa: N803
        self._guard_target()
        _check_fit_rows(self.op, len(X), cap)
        target = str(self.param("target"))
        if target not in X.columns:
            raise EngineError(OP_INVALID, "The target column is not in the table.", {"op": self.op})
        if self.param("task") != "regression":
            counts = X[target].dropna().astype(object).value_counts()
            if len(counts) < 2:
                raise EngineError(
                    "TARGET_SINGLE_CLASS",
                    "The fitting rows contain a single class, so no column can be ranked.",
                    {"op": self.op, "column": target},
                )
            if int(counts.min()) < SMALL_CLASS:
                self.warn(
                    "SMALL_CLASS_COUNT",
                    f"A class has fewer than {SMALL_CLASS} fitting rows; scores and test "
                    "results for it are unreliable and can change completely between folds.",
                    target,
                )

    def _warn_not_assessed(self) -> None:
        for column, reason in zip(self.cols, self.state_["reasons"], strict=True):
            if reason is not None:
                self.warn(
                    "NOT_ASSESSED",
                    f"The column could not be assessed ({reason.replace('_', ' ')}) and is kept.",
                    column,
                )


class MutualInfoSelectOp(_SupervisedSelectOp):
    op = "select.mutual_information"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self._common_fit(X, Limits.MI_MAX_FIT_ROWS)
        _check_columns(self.op, self.cols, [], Limits.MI_MAX_COLUMNS)
        self.state_ = rt.fit_mutual_information(
            X,
            self.cols,
            str(self.param("target")),
            str(self.param("task")),
            [bool(v) for v in self.param("discrete")],
            int(self.param("n_neighbors")),
            int(self.param("k")),
            str(self.param("action")),
            self.keep(),
            self.role_kept(),
            int(self.seed),
            Limits.SELECT_MIN_ROWS,
        )
        self._warn_not_assessed()
        for column, flag in zip(self.cols, self.state_["near_perfect"], strict=True):
            if flag:
                self.warn(
                    "POSSIBLE_TARGET_LEAKAGE",
                    "This column almost determines the target in the fitting rows. Check that "
                    "it is not a copy of, or recorded after, the outcome.",
                    column,
                )
        self._warn_proposals()

    def learned_params(self) -> dict[str, Any]:
        return {
            "task": self.param("task"),
            "k": int(self.param("k")),
            "action": self.param("action"),
            "seed": int(self.seed),
            "decisions": self.decisions(),
            "columns": {
                c: {"mutual_information": s, "rank": r, "n_rows": n, "not_assessed": why}
                for c, s, r, n, why in zip(
                    self.cols,
                    self.state_["scores"],
                    self.state_["rank"],
                    self.state_["n_rows"],
                    self.state_["reasons"],
                    strict=True,
                )
            },
        }


class UnivariateSelectOp(_SupervisedSelectOp):
    op = "select.univariate"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self._common_fit(X, Limits.UNIVARIATE_MAX_FIT_ROWS)
        self.state_ = rt.fit_univariate(
            X,
            self.cols,
            str(self.param("target")),
            str(self.param("task")),
            str(self.param("test")),
            int(self.param("k")),
            str(self.param("action")),
            self.keep(),
            self.role_kept(),
            Limits.SELECT_MIN_ROWS,
            Limits.CHI2_MAX_LEVELS,
        )
        self._warn_not_assessed()
        for column, doubts in zip(self.cols, self.state_["doubts"], strict=True):
            for doubt in doubts:
                self.warn(
                    "ASSUMPTION_DOUBTFUL",
                    f"Test assumption doubtful in the fitting rows: {DOUBT_TEXT[doubt]}.",
                    column,
                )
        self._warn_proposals()

    def learned_params(self) -> dict[str, Any]:
        return {
            "test": self.param("test"),
            "task": self.param("task"),
            "k": int(self.param("k")),
            "action": self.param("action"),
            "decisions": self.decisions(),
            "columns": {
                c: {
                    "statistic": s,
                    "p_value": p,
                    "adjusted_p_value": q,
                    "rank": r,
                    "n_rows": n,
                    "not_assessed": why,
                    "doubtful_assumptions": list(d),
                }
                for c, s, p, q, r, n, why, d in zip(
                    self.cols,
                    self.state_["statistics"],
                    self.state_["p_values"],
                    self.state_["adjusted_p_values"],
                    self.state_["rank"],
                    self.state_["n_rows"],
                    self.state_["reasons"],
                    self.state_["doubts"],
                    strict=True,
                )
            },
        }


class DropColumnsOp(_SelectOp):
    op = "select.drop_columns"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self.state_ = {
            "drop": list(self.cols),
            "decisions": {c: "removed_by_user" for c in self.cols},
        }

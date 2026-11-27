"""Categorical encoding ops (Part B; model stage only). Category mappings and frequencies are
learned from the fitting rows; unseen categories are handled explicitly and counted."""

from typing import Any

import pandas as pd

from datamedic.engine.preprocess import runtime as rt
from datamedic.engine.preprocess.base import DMTransformer


def _unseen(series: pd.Series, known: list[Any]) -> int:
    present = series.notna()
    return int((present & ~series.astype(object).isin(known)).sum())


class OneHotOp(DMTransformer):
    op = "encode.onehot"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self.state_ = rt.fit_onehot(
            X, self.cols, int(self.param("max_categories")), str(self.param("name_by"))
        )
        for column, fallback in zip(self.cols, self.state_["fallback"], strict=True):
            if fallback:
                self.warn(
                    "POSITIONAL_NAMES_FALLBACK",
                    "Some category labels cannot be used in column names (control "
                    "characters, blank, '__other__', duplicates as text, or too long), so this "
                    "column's outputs are named by position.",
                    column,
                )
        for column, seen, kept in zip(
            self.cols, self.state_["seen"], self.state_["categories"], strict=True
        ):
            if len(seen) > len(kept):
                self.warn(
                    "RARE_GROUPED",
                    f"{len(seen) - len(kept)} less frequent categories were grouped into "
                    "the '__other__' column.",
                    column,
                )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        unseen = {
            c: _unseen(X[c], seen) for c, seen in zip(self.cols, self.state_["seen"], strict=True)
        }
        out = rt.apply_onehot(X, self.cols, self.state_, str(self.param("handle_unknown")))
        self.effects_ = {
            "unseen_category_rows": unseen,
            "output_columns": {
                c: len(cats) + int(other)
                for c, cats, other in zip(
                    self.cols, self.state_["categories"], self.state_["other"], strict=True
                )
            },
        }
        if any(unseen.values()):
            handled = (
                "were placed in the '__other__' column where one exists, otherwise encoded "
                "as all zeros (the same as a missing value)"
            )
            self.transform_warnings_.append(
                {
                    "code": "UNSEEN_CATEGORIES",
                    "message": f"Rows with categories not seen in the fitting rows {handled}.",
                    "column": None,
                }
            )
        return out

    def learned_params(self) -> dict[str, Any]:
        columns = {}
        for column, kept, seen, other in zip(
            self.cols,
            self.state_["categories"],
            self.state_["seen"],
            self.state_["other"],
            strict=True,
        ):
            columns[column] = {
                "categories": [str(v) for v in kept],
                "n_categories_seen": len(seen),
                "grouped_as_other": len(seen) - len(kept),
                "other_column": bool(other),
            }
        for column, fallback in zip(self.cols, self.state_["fallback"], strict=True):
            columns[column]["positional_names"] = bool(
                fallback or self.param("name_by") == "position"
            )
        return {"name_by": self.param("name_by"), "columns": columns}


class OrdinalOp(DMTransformer):
    op = "encode.ordinal"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self.state_ = rt.fit_ordinal(X, self.cols[0], list(self.param("order")))

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        order = list(self.param("order"))
        unknown = _unseen(X[self.cols[0]], order)
        out = rt.apply_ordinal(X, self.cols[0], order, str(self.param("handle_unknown")))
        self.effects_ = {"unknown_to_missing_rows": unknown}
        return out

    def learned_params(self) -> dict[str, Any]:
        # Nothing is learned; the order is configuration and is not repeated here.
        return {"columns": {self.cols[0]: {"levels": len(self.param("order"))}}}


class FrequencyOp(DMTransformer):
    op = "encode.frequency"

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        self.state_ = rt.fit_frequency(X, self.cols)
        for column, tied in zip(self.cols, self.state_["tied"], strict=True):
            if tied:
                self.warn(
                    "FREQUENCY_TIES",
                    f"{tied} categories share a frequency with another category and become "
                    "indistinguishable.",
                    column,
                )

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        unseen = {
            c: _unseen(X[c], list(table))
            for c, table in zip(self.cols, self.state_["tables"], strict=True)
        }
        self.effects_ = {"unseen_category_rows": unseen}
        return rt.apply_frequency(X, self.cols, self.state_)

    def learned_params(self) -> dict[str, Any]:
        return {
            "columns": {
                c: {"n_categories": len(table), "tied_categories": tied}
                for c, table, tied in zip(
                    self.cols, self.state_["tables"], self.state_["tied"], strict=True
                )
            }
        }

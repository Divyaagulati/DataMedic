"""Base class for preprocessing ops: scikit-learn transformers over canonical DataFrames.

Contract (tested for every op):
- ``fit`` learns only from the rows it is given; ``transform`` never learns.
- Inputs are never mutated; ``_dm_row_id`` is preserved; observed values are never changed
  by imputation ops.
- Errors are ``EngineError`` with a stable code; messages contain no data values (the
  affected column is in ``details``). Unexpected exceptions become ``OP_FAILED`` carrying
  only the exception type, because library messages can quote cell values.
- Library warnings are captured during fit and transform and reported as
  ``LIBRARY_WARNING`` with their category only (their text could quote data and would
  otherwise reach worker logs through Celery's stderr redirection).
"""

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any, ClassVar

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from datamedic.engine import ROW_ID
from datamedic.engine.errors import EngineError
from datamedic.engine.preprocess.runtime import PreprocessError

OP_INVALID = "OP_INVALID"
OP_LIMIT_EXCEEDED = "OP_LIMIT_EXCEEDED"
OP_FAILED = "OP_FAILED"

# Warnings about code, not data: re-emitted unchanged so CI warning policies still apply.
_CODE_WARNINGS = (DeprecationWarning, PendingDeprecationWarning, FutureWarning)


class DMTransformer(TransformerMixin, BaseEstimator):  # type: ignore[misc]
    """Subclasses set ``op`` and implement ``_fit`` / ``_transform`` / ``learned_params``."""

    op: ClassVar[str] = ""

    def __init__(
        self,
        columns: tuple[str, ...] = (),
        params: dict[str, Any] | None = None,
        seed: int = 0,
    ) -> None:
        self.columns = columns
        self.params = params
        self.seed = seed

    # -- scikit-learn API ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: Any = None) -> "DMTransformer":  # noqa: N803
        self._check_frame(X)
        self.warnings_: list[dict[str, Any]] = []
        with _engine_errors(self.op), _library_warnings(self.warnings_):
            self._fit(X)
        self.n_fit_rows_ = len(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        check_is_fitted(self, "n_fit_rows_")
        self._check_frame(X)
        self.transform_warnings_: list[dict[str, Any]] = []
        # Counts describing what this transform did (no values), e.g. capped or unseen rows.
        self.effects_: dict[str, Any] = {}
        with _engine_errors(self.op), _library_warnings(self.transform_warnings_):
            out = self._transform(X)
        return out

    def resample(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        """Training-only ops: build the training table from the fitting rows (resampled or
        weighted). Every other op returns ``X`` unchanged. Never applied to other rows."""
        check_is_fitted(self, "n_fit_rows_")
        self._check_frame(X)
        self.resample_warnings_: list[dict[str, Any]] = []
        with _engine_errors(self.op), _library_warnings(self.resample_warnings_):
            out = self._resample(X)
        return out

    # -- Subclass hooks --------------------------------------------------------------------

    def _resample(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return X

    def _fit(self, X: pd.DataFrame) -> None:  # noqa: N803
        """Learn from ``X`` (default: nothing to learn)."""

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        raise NotImplementedError

    def learned_params(self) -> dict[str, Any]:
        """JSON-serialisable learned state, ``{"columns": {name: {...}}, ...}``. Values may be
        derived from data, so reports redact them for sensitive columns."""
        return {}

    # -- Helpers ---------------------------------------------------------------------------

    @property
    def cols(self) -> list[str]:
        return list(self.columns)

    def param(self, name: str) -> Any:
        return (self.params or {}).get(name)

    def warn(self, code: str, message: str, column: str | None = None) -> None:
        self.warnings_.append({"code": code, "message": message, "column": column})

    def _check_frame(self, X: Any) -> None:  # noqa: N803
        if not isinstance(X, pd.DataFrame) or ROW_ID not in X.columns:
            raise EngineError(OP_INVALID, "Steps run on canonical tables with a row identifier.")
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise EngineError(
                OP_INVALID,
                "A column required by this step is not in the table.",
                {"op": self.op, "column": missing[0]},
            )


@contextmanager
def _engine_errors(op: str) -> Iterator[None]:
    """Translate runtime errors into EngineError with structured details."""
    try:
        yield
    except PreprocessError as exc:
        raise EngineError(exc.code, exc.message, {"op": op, "column": exc.column}) from None
    except (EngineError, MemoryError):
        raise  # MemoryError: the worker reports LIMIT_EXCEEDED
    except Exception as exc:
        raise EngineError(
            OP_FAILED,
            "The step failed unexpectedly. Details are withheld because they could contain "
            "data values.",
            {"op": op, "error_type": type(exc).__name__},
        ) from exc


@contextmanager
def _library_warnings(into: list[dict[str, Any]]) -> Iterator[None]:
    """Record warnings raised inside a step by category only (never their text)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        yield
    categories: set[str] = set()
    for w in caught:
        if issubclass(w.category, _CODE_WARNINGS):
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        else:
            categories.add(w.category.__name__)
    for name in sorted(categories):
        into.append(
            {
                "code": "LIBRARY_WARNING",
                "message": f"A library warning ({name}) was raised; its text is not shown "
                "because it could quote data values.",
                "column": None,
            }
        )

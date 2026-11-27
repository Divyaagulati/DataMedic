"""Engine ports (ML_ENGINE_DESIGN §1.6). Callers depend on these interfaces, never on a
concrete library, so V2 adapters (e.g. fg-data-profiling) can be swapped in without changes.
"""

from typing import Any, Protocol

import numpy as np
import pandas as pd


class Profiler(Protocol):
    name: str
    version: str

    def profile(
        self, frame: pd.DataFrame, schema: list[dict[str, Any]], *, seed: int
    ) -> dict[str, Any]:
        """Describe a canonical frame (including ``_dm_row_id``, which is ignored).

        Returns a JSON-serialisable profile: ``summary``, ``columns``, ``correlations``,
        ``sampling`` and ``limitations``. Never mutates ``frame``.
        """
        ...


class OutlierDetector(Protocol):
    """Multivariate outlier scoring (ML_ENGINE_DESIGN §1.6). Higher score = more unusual.

    Input: a complete numeric matrix (rows x features, no missing values). Scores rank rows;
    they are not probabilities and do not say a row is wrong."""

    name: str
    version: str

    def score(self, matrix: np.ndarray, seed: int) -> np.ndarray: ...

"""Adapters behind the ``OutlierDetector`` port.

- ``IsolationForestDetector``: scikit-learn ``IsolationForest`` (pinned 1.9.1).
- ``ECODDetector`` and ``COPODDetector``: implemented in-house (owner decision, Phase 3:
  no PyOD in the MVP) from the published methods, using empirical CDFs:

  ECOD  (Li, Zhao, Hu, Botta, Ionescu, Chen 2022, "ECOD: Unsupervised Outlier Detection
        Using Empirical Cumulative Distribution Functions", IEEE TKDE), Algorithm 1:
        per row, O_left = -sum_j log F_left_j(x_j), O_right = -sum_j log F_right_j(x_j),
        O_auto = the same sum using the left tail where feature j is negatively skewed and
        the right tail otherwise; score = max(O_left, O_right, O_auto).

  COPOD (Li, Zhao, Botta, Ionescu, Hu 2020, "COPOD: Copula-Based Outlier Detection",
        IEEE ICDM): the same empirical copula tails, aggregated as
        score = max(sum_j skew-corrected tail, sum_j -log((F_left_j + F_right_j) / 2)).
        The two-sided term makes it less dominated by a single extreme feature than ECOD.

  Both are deterministic. They are validated by tests on synthetic data (planted outliers
  rank first; scores are unchanged by unit or offset changes; a hand-computed
  case), not by comparison with PyOD.
  A PyOD adapter can replace them in V2 without changing callers.
"""

import numpy as np
from scipy.stats import rankdata, skew
from sklearn.ensemble import IsolationForest

_EPS = 1e-12


def ecdf_tails(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column: F_left(x) = P(X <= x) and F_right(x) = P(X >= x), empirical."""
    n = matrix.shape[0]
    left = rankdata(matrix, method="max", axis=0) / n
    right = rankdata(-matrix, method="max", axis=0) / n
    return left, right


class IsolationForestDetector:
    name = "isolation_forest"
    version = "sklearn-1.9.1"

    def score(self, matrix: np.ndarray, seed: int) -> np.ndarray:
        model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
        model.fit(matrix)
        # score_samples is the negated anomaly score s(x, n) of Liu et al.: higher = normal.
        scores: np.ndarray = -model.score_samples(matrix)
        return scores


class ECODDetector:
    name = "ecod"
    version = "inhouse-1.0"

    def score(self, matrix: np.ndarray, seed: int) -> np.ndarray:
        left, right = ecdf_tails(matrix)
        neg_log_left = -np.log(np.maximum(left, _EPS))
        neg_log_right = -np.log(np.maximum(right, _EPS))
        skewed_left = skew(matrix, axis=0) < 0
        auto = np.where(skewed_left, neg_log_left, neg_log_right)
        scores: np.ndarray = np.maximum.reduce(
            [neg_log_left.sum(axis=1), neg_log_right.sum(axis=1), auto.sum(axis=1)]
        )
        return scores


class COPODDetector:
    name = "copod"
    version = "inhouse-1.0"

    def score(self, matrix: np.ndarray, seed: int) -> np.ndarray:
        left, right = ecdf_tails(matrix)
        neg_log_left = -np.log(np.maximum(left, _EPS))
        neg_log_right = -np.log(np.maximum(right, _EPS))
        skewed_left = skew(matrix, axis=0) < 0
        corrected = np.where(skewed_left, neg_log_left, neg_log_right)
        two_sided = -np.log(np.maximum((left + right) / 2, _EPS))
        scores: np.ndarray = np.maximum.reduce([corrected.sum(axis=1), two_sided.sum(axis=1)])
        return scores


DETECTORS = (IsolationForestDetector(), ECODDetector(), COPODDetector())

"""Similarity ordering for nominal features (Molnar/iml recipe, scipy-only).

Nominal levels have no natural order, but ALE needs one to accumulate.
`similarity_order` induces it from the *other* features: levels whose
conditional distributions look alike end up adjacent, so adjacent-level
differences stay meaningful. The accumulated curve's *shape* still depends on
the chosen order — the honest quantities are the adjacent differences
(docs/method_semantics.md, ALE-nominal caveat).
"""

import typing

import numpy as np
from scipy import stats

from effector import ingestion


def _level_distance_matrix(
    data: np.ndarray,
    feature: int,
    levels: np.ndarray,
    feature_types: typing.List[str],
) -> np.ndarray:
    """Pairwise level distances, summed over the other features: two-sample
    Kolmogorov-Smirnov distance for continuous features, total-variation
    distance of the level-frequency tables for discrete ones."""
    K = len(levels)
    dist = np.zeros((K, K))
    col = data[:, feature]
    members = [np.isclose(col, lev) for lev in levels]

    for j in range(data.shape[1]):
        if j == feature:
            continue
        other = data[:, j]
        discrete = ingestion.is_categorical(feature_types[j])
        for a in range(K):
            for b in range(a + 1, K):
                xa, xb = other[members[a]], other[members[b]]
                if len(xa) == 0 or len(xb) == 0:
                    continue
                if discrete:
                    values = np.unique(other)
                    pa = np.array([np.isclose(xa, v).mean() for v in values])
                    pb = np.array([np.isclose(xb, v).mean() for v in values])
                    d = 0.5 * np.abs(pa - pb).sum()
                else:
                    d = stats.ks_2samp(xa, xb).statistic
                dist[a, b] += d
                dist[b, a] += d
    return dist


def _seriate_1d(dist: np.ndarray) -> np.ndarray:
    """Order the levels along the first classical-MDS coordinate: double-center
    -0.5 * J D^2 J, take the leading eigenvector, sort. Deterministic sign:
    the level with the smaller original code comes first."""
    K = dist.shape[0]
    j_mat = np.eye(K) - np.ones((K, K)) / K
    b_mat = -0.5 * j_mat @ (dist**2) @ j_mat
    eigenvalues, eigenvectors = np.linalg.eigh(b_mat)
    coord = eigenvectors[:, np.argmax(eigenvalues)]
    order = np.argsort(coord, kind="stable")
    if order[0] > order[-1]:
        order = order[::-1]
    return order


def similarity_order(
    data: np.ndarray,
    feature: int,
    levels: np.ndarray,
    feature_types: typing.List[str],
) -> np.ndarray:
    """The induced level order as a permutation of ``arange(len(levels))``."""
    if len(levels) < 3:
        return np.arange(len(levels))
    dist = _level_distance_matrix(data, feature, levels, feature_types)
    return _seriate_1d(dist)

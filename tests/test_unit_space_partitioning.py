"""Unit layer for effector.space_partitioning (PLAN II §3.2).

Absorbs the old test_space_partitioning.py Gini toy (kept verbatim as the
validity check) and adds the pointed assertions the refactor needs:
split-position correctness, categorical ==/!= splits, the no-split threshold,
and Best vs BestLevelWise agreement on a clean toy.
"""

import numpy as np
import pytest

from effector.space_partitioning import Best, BestLevelWise

BIG = 10_000_000_000


def _make_toy():
    """Four groups defined by thresholds on feature 1 (1.5 / 3 / 5).

    Hand computation of the weighted-Gini optimum: class probabilities are
    0.15 / 0.15 / 0.20 / 0.50, so splitting at 5.0 gives 0.5 * 0.66 + 0.5 * 0
    = 0.33 while splitting at 3.0 gives 0.3 * 0.5 + 0.7 * 0.41 = 0.44 — the
    correct first split is feature 1 at 5.0."""
    np.random.seed(0)
    N, D = 1000, 3
    X = np.random.uniform(0, 10, size=(N, D))
    y = np.empty(N, dtype=int)
    for i in range(N):
        if X[i, 1] < 3:
            y[i] = 0 if X[i, 1] < 1.5 else 1
        else:
            y[i] = 2 if X[i, 1] < 5 else 3
    return X, y


def _gini(y):
    def heterogeneity(mask):
        indices = np.where(mask)[0]
        if len(indices) < 50:
            return BIG
        labels = y[indices]
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p**2)

    return heterogeneity


AXIS_LIMITS = np.array([[0, 10], [0, 10], [0, 10]]).T


def _compile(partitioner, X, heter, **kwargs):
    partitioner.compile(
        feature=0,
        data=X,
        heter_func=heter,
        axis_limits=AXIS_LIMITS,
        candidate_conditioning_features=[0, 1, 2],
        feature_names=["x1", "x2", "x3"],
        target_name="y",
        **kwargs,
    )
    return partitioner.fit()


def _parent_heter_lower(node, is_lower):
    if not is_lower:
        return False
    if node.parent_node is None:
        return is_lower
    return _parent_heter_lower(
        node.parent_node,
        node.info["weighted_heterogeneity"]
        <= node.parent_node.info["weighted_heterogeneity"],
    )


@pytest.mark.parametrize("cls", [Best, BestLevelWise])
def test_heterogeneity_decreases_along_the_tree(cls):
    X, y = _make_toy()
    tree = _compile(
        cls(
            min_heterogeneity_decrease_pcg=0.1,
            heter_small_enough=0.0,
            max_depth=2,
            min_samples_leaf=10,
            numerical_features_grid_size=20,
        ),
        X,
        _gini(y),
    )
    assert tree is not None
    assert all(_parent_heter_lower(n, True) for n in tree.nodes)


def _root_split(tree):
    root = tree.get_root()
    children = tree.get_children(root.name)
    assert len(children) == 2
    return children


def test_split_position_correctness():
    X, y = _make_toy()
    tree = _compile(Best(max_depth=1, heter_small_enough=0.0), X, _gini(y))
    children = _root_split(tree)
    for child in children:
        assert child.info["foc_index"] == 1
        assert abs(child.info["foc_split_position"] - 5.0) < 0.5


def test_best_and_best_level_wise_find_same_first_split():
    X, y = _make_toy()
    tree_a = _compile(Best(max_depth=1, heter_small_enough=0.0), X, _gini(y))
    tree_b = _compile(BestLevelWise(max_depth=1, heter_small_enough=0.0), X, _gini(y))
    child_a = _root_split(tree_a)[0]
    child_b = _root_split(tree_b)[0]
    assert child_a.info["foc_index"] == child_b.info["foc_index"]
    np.testing.assert_allclose(
        child_a.info["foc_split_position"], child_b.info["foc_split_position"]
    )


def test_categorical_conditioning_feature_splits_with_equality():
    np.random.seed(0)
    N = 600
    X = np.stack(
        [
            np.random.uniform(0, 10, N),
            np.random.uniform(0, 10, N),
            np.random.randint(0, 2, N).astype(float),
        ],
        axis=1,
    )
    y = X[:, 2].astype(int)  # heterogeneity fully explained by the binary x3

    tree = _compile(
        Best(max_depth=1, heter_small_enough=0.0),
        X,
        _gini(y),
        feature_types=["cont", "cont", "cat"],
    )
    children = _root_split(tree)
    assert {c.info["comparison"] for c in children} == {"==", "!="}
    for child in children:
        assert child.info["foc_index"] == 2


def test_no_split_when_threshold_huge():
    X, y = _make_toy()
    tree = _compile(
        Best(min_heterogeneity_decrease_pcg=1000.0, heter_small_enough=0.0), X, _gini(y)
    )
    assert len(tree.nodes) == 1
    assert tree.get_root() is not None

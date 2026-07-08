"""Unit layer for effector.space_partitioning (PLAN II §3.2).

Absorbs the old test_space_partitioning.py Gini toy (kept verbatim as the
validity check) and adds the pointed assertions the refactor needs:
split-position correctness, categorical one-vs-rest splits, the no-split
threshold, Best vs BestLevelWise agreement on a clean toy, and the proposer
seam (a custom k-way proposer flows through search and construction).
"""

import numpy as np
import pytest

from effector import proposers
from effector.proposers import CandidateSplit
from effector.rules import Condition, Interval, LevelSet
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


def _children_of(part, idx):
    return [r for r in part if r.parent_idx == idx]


def _split_position(region, foc):
    """The threshold a binary continuous split put on `foc` (whichever
    Interval bound is finite)."""
    interval = region.rule[foc]
    assert isinstance(interval, Interval)
    return interval.hi if np.isfinite(interval.hi) else interval.lo


@pytest.mark.parametrize("cls", [Best, BestLevelWise])
def test_heterogeneity_decreases_along_the_tree(cls):
    X, y = _make_toy()
    part = _compile(
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
    assert part is not None
    for r in part:
        if r.parent_idx is None:
            continue
        parent = part[r.parent_idx]
        assert r.heterogeneity * r.weight <= parent.heterogeneity * parent.weight


def _root_split(part):
    children = _children_of(part, 0)
    assert len(children) == 2
    return children


def test_split_position_correctness():
    X, y = _make_toy()
    part = _compile(Best(max_depth=1, heter_small_enough=0.0), X, _gini(y))
    for child in _root_split(part):
        assert child.rule.features == (1,)
        assert abs(_split_position(child, 1) - 5.0) < 0.5


def test_best_and_best_level_wise_find_same_first_split():
    X, y = _make_toy()
    part_a = _compile(Best(max_depth=1, heter_small_enough=0.0), X, _gini(y))
    part_b = _compile(BestLevelWise(max_depth=1, heter_small_enough=0.0), X, _gini(y))
    child_a = _root_split(part_a)[0]
    child_b = _root_split(part_b)[0]
    assert child_a.rule.features == child_b.rule.features
    np.testing.assert_allclose(_split_position(child_a, 1), _split_position(child_b, 1))


def test_categorical_conditioning_feature_splits_one_vs_rest():
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

    part = _compile(
        Best(max_depth=1, heter_small_enough=0.0),
        X,
        _gini(y),
        feature_types=["cont", "cont", "cat"],
    )
    children = _root_split(part)
    subsets = [c.rule[2] for c in children]
    assert all(isinstance(s, LevelSet) for s in subsets)
    # one level vs its explicit complement, jointly covering the universe
    assert {len(s.levels) for s in subsets} == {1}
    assert set().union(*(s.levels for s in subsets)) == {0.0, 1.0}
    for child in children:
        assert child.rule.features == (2,)


def test_no_split_when_threshold_huge():
    X, y = _make_toy()
    part = _compile(
        Best(min_heterogeneity_decrease_pcg=1000.0, heter_small_enough=0.0), X, _gini(y)
    )
    assert len(part) == 1
    assert part[0].idx == 0
    assert part[0].rule.is_root


class _ThreeWayOnX2:
    """A stub k-way proposer: one 3-way candidate on feature 1 (the toy's
    true group boundaries at 3 and 5), nothing for other features."""

    def propose(self, ctx, foc):
        if foc != 1:
            return []
        return [
            CandidateSplit(
                (
                    Condition(1, Interval(hi=3.0)),
                    Condition(1, Interval(lo=3.0, hi=5.0)),
                    Condition(1, Interval(lo=5.0)),
                )
            )
        ]


def test_kway_proposer_flows_through_search_and_construction():
    X, y = _make_toy()
    finder = Best(max_depth=1, heter_small_enough=0.0)
    finder.proposer_factory = lambda ftype: _ThreeWayOnX2()
    part = _compile(finder, X, _gini(y))

    children = _children_of(part, 0)
    assert len(children) == 3
    assert [c.rule[1] for c in children] == [
        Interval(hi=3.0),
        Interval(lo=3.0, hi=5.0),
        Interval(lo=5.0),
    ]
    # the children partition the data (also enforced by the Partition invariant)
    counts = np.sum([c.mask for c in children], axis=0)
    np.testing.assert_array_equal(counts, np.ones(len(X), dtype=int))
    assert sum(c.nof_instances for c in children) == len(X)


def _raw_gini(y):
    """Raw score_fn for the finder protocol: no min-points guard (the adapter
    owns that) — just the weighted-Gini of the masked labels."""

    def score(mask):
        labels = y[mask]
        _, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p**2)

    return score


def test_find_regions_returns_partition_and_leaves_caller_clean():
    from effector.partition import Partition

    X, y = _make_toy()
    finder = Best(max_depth=2, min_samples_leaf=50)
    part = finder.find_regions(
        feature=0,
        data=X,
        score_fn=_raw_gini(y),
        axis_limits=AXIS_LIMITS,
        feature_types=None,
        cat_limit=10,
        candidate_conditioning_features=[0, 1, 2],
        feature_names=["x1", "x2", "x3"],
        target_name="y",
    )
    assert isinstance(part, Partition)
    assert part[0].idx == 0
    assert part[0].weight == 1.0
    assert len(part) >= 3  # root + at least one accepted split on the structured toy
    assert part[1].mask.dtype == bool
    # the caller's finder instance must be untouched (compile ran on the deepcopy)
    assert finder.data is None
    assert finder.feature is None


def _make_cat_toy(nof_levels=3, with_refinement=False):
    """Classes fully explained by a `nof_levels`-way categorical x3 (and,
    with `with_refinement`, a further x1 >= 5 split inside every level)."""
    np.random.seed(0)
    N = 600
    X = np.stack(
        [
            np.random.uniform(0, 10, N),
            np.random.uniform(0, 10, N),
            np.random.randint(0, nof_levels, N).astype(float),
        ],
        axis=1,
    )
    y = X[:, 2].astype(int)
    if with_refinement:
        y = 2 * y + (X[:, 0] >= 5).astype(int)
    return X, y


def test_proposer_kwargs_defaults_match_current_and_junk_raises():
    factory = Best().proposer_factory
    assert isinstance(factory("cont"), proposers.ContinuousThreshold)
    assert isinstance(factory("cat"), proposers.CategoricalOneVsRest)
    with pytest.raises(ValueError, match="unknown categorical proposer"):
        Best(categorical_proposer="cart")
    with pytest.raises(ValueError, match="unknown continuous proposer"):
        BestLevelWise(continuous_proposer="chi2")


def test_multiway_kwarg_end_to_end_best():
    X, y = _make_cat_toy()
    part = _compile(
        Best(max_depth=1, heter_small_enough=0.0, categorical_proposer="multiway"),
        X,
        _gini(y),
        feature_types=["cont", "cont", "cat"],
    )
    children = _children_of(part, 0)
    assert [c.rule[2] for c in children] == [
        LevelSet({0.0}),
        LevelSet({1.0}),
        LevelSet({2.0}),
    ]
    counts = np.sum([c.mask for c in children], axis=0)
    np.testing.assert_array_equal(counts, np.ones(len(X), dtype=int))


def test_kway_through_best_level_wise_parent_major():
    X, y = _make_cat_toy(with_refinement=True)
    part = _compile(
        BestLevelWise(
            max_depth=2, heter_small_enough=0.0, categorical_proposer="multiway"
        ),
        X,
        _gini(y),
        feature_types=["cont", "cont", "cat"],
    )
    # level 1: the 3-way categorical candidate; level 2: one binary threshold
    # applied to every level-1 node -> 3 * 2 children, parent-major order
    assert len(part) == 10
    level1 = _children_of(part, 0)
    assert [r.idx for r in level1] == [1, 2, 3]
    assert [r.rule[2] for r in level1] == [
        LevelSet({0.0}),
        LevelSet({1.0}),
        LevelSet({2.0}),
    ]
    level2 = [r for r in part if r.level == 2]
    assert [r.parent_idx for r in level2] == [1, 1, 2, 2, 3, 3]
    for r in level2:
        assert abs(_split_position(r, 0) - 5.0) < 0.5
        parent = part[r.parent_idx]
        np.testing.assert_array_equal(r.mask & parent.mask, r.mask)


def test_proposer_instance_kwarg_end_to_end():
    X, y = _make_cat_toy()
    part = _compile(
        Best(
            max_depth=1,
            heter_small_enough=0.0,
            categorical_proposer=proposers.CategoricalOrdered(order=[2.0, 0.0, 1.0]),
        ),
        X,
        _gini(y),
        feature_types=["cont", "cont", "cat"],
    )
    children = _children_of(part, 0)
    # whichever cut wins, it must be a contiguous prefix/suffix of the
    # explicit order — {0, 1} vs {2} could never come out of [2, 0, 1]
    assert [c.rule[2] for c in children] in (
        [LevelSet({2.0}), LevelSet({0.0, 1.0})],
        [LevelSet({2.0, 0.0}), LevelSet({1.0})],
    )


def test_find_regions_rejects_min_points_below_two():
    X, y = _make_toy()
    finder = Best(min_samples_leaf=1)
    with pytest.raises(ValueError):
        finder.find_regions(
            feature=0,
            data=X,
            score_fn=_raw_gini(y),
            axis_limits=AXIS_LIMITS,
            feature_types=None,
            cat_limit=10,
            candidate_conditioning_features=[0, 1, 2],
            feature_names=["x1", "x2", "x3"],
            target_name="y",
        )

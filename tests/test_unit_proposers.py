"""Unit layer for effector.proposers (PR-C): candidate enumeration order,
exact Interval/LevelSet forms, mask semantics, and the k-way shape."""

import numpy as np
import pytest

from effector import ordering
from effector.proposers import (
    CandidateSplit,
    CategoricalMultiway,
    CategoricalOneVsRest,
    CategoricalOrdered,
    CategoricalSubsets,
    ContinuousThreshold,
    SearchContext,
    default_proposer,
)
from effector.rules import Condition, Interval, LevelSet


def _ctx(data, axis_limits, feature_types, grid=20):
    return SearchContext(
        data=data,
        axis_limits=axis_limits,
        feature_types=tuple(feature_types),
        numerical_grid_size=grid,
    )


@pytest.fixture
def cont_ctx():
    rng = np.random.default_rng(0)
    data = rng.uniform(0, 10, size=(100, 2))
    axis_limits = np.array([[0.0, 0.0], [10.0, 10.0]])
    return _ctx(data, axis_limits, ["cont", "cont"])


def test_continuous_positions_are_interior_linspace_ascending(cont_ctx):
    candidates = ContinuousThreshold().propose(cont_ctx, 1)
    # grid 20 on [0, 10] -> linspace(0, 10, 21)[1:-1] = 0.5, 1.0, ..., 9.5
    assert len(candidates) == 19
    thresholds = [c.conditions[0].subset.hi for c in candidates]
    np.testing.assert_allclose(thresholds, np.linspace(0, 10, 21)[1:-1])
    assert thresholds == sorted(thresholds)


def test_continuous_candidate_is_below_then_above(cont_ctx):
    cand = ContinuousThreshold().propose(cont_ctx, 1)[0]
    assert cand.feature == 1
    below, above = cand.conditions
    assert below.subset == Interval(hi=0.5)  # x < t
    assert above.subset == Interval(lo=0.5)  # x >= t
    col = cont_ctx.data[:, 1]
    np.testing.assert_array_equal(below.contains(cont_ctx.data), col < 0.5)
    np.testing.assert_array_equal(above.contains(cont_ctx.data), col >= 0.5)


def test_continuous_children_partition_the_axis(cont_ctx):
    for cand in ContinuousThreshold().propose(cont_ctx, 0):
        masks = [c.contains(cont_ctx.data) for c in cand.conditions]
        np.testing.assert_array_equal(
            np.sum(masks, axis=0), np.ones(len(cont_ctx.data), dtype=int)
        )


def test_categorical_one_vs_rest_over_observed_levels():
    data = np.stack([np.arange(9, dtype=float), np.array([2.0, 0.0, 1.0] * 3)], axis=1)
    ctx = _ctx(data, np.array([[0.0, 0.0], [8.0, 2.0]]), ["cont", "cat"])
    candidates = CategoricalOneVsRest().propose(ctx, 1)
    assert len(candidates) == 3  # one per observed level, ascending
    for cand, v in zip(candidates, [0.0, 1.0, 2.0]):
        eq, rest = cand.conditions
        assert eq.subset == LevelSet({v})
        assert rest.subset == LevelSet({0.0, 1.0, 2.0} - {v})
        np.testing.assert_array_equal(eq.contains(data), data[:, 1] == v)
        np.testing.assert_array_equal(rest.contains(data), data[:, 1] != v)


def test_categorical_single_level_universe_has_empty_complement():
    data = np.stack([np.arange(4, dtype=float), np.full(4, 7.0)], axis=1)
    ctx = _ctx(data, np.array([[0.0, 7.0], [3.0, 7.0]]), ["cont", "cat"])
    (cand,) = CategoricalOneVsRest().propose(ctx, 1)
    assert cand.conditions[0].subset == LevelSet({7.0})
    assert cand.conditions[1].subset.is_empty


def test_candidate_split_is_kway_capable():
    cand = CandidateSplit(
        (
            Condition(0, Interval(hi=1.0)),
            Condition(0, Interval(lo=1.0, hi=2.0)),
            Condition(0, Interval(lo=2.0)),
        )
    )
    assert cand.feature == 0
    assert len(cand.conditions) == 3


def test_candidate_split_rejects_degenerate_shapes():
    with pytest.raises(ValueError, match="at least 2"):
        CandidateSplit((Condition(0, Interval(hi=1.0)),))
    with pytest.raises(ValueError, match="one feature"):
        CandidateSplit((Condition(0, Interval(hi=1.0)), Condition(1, Interval(lo=1.0))))


@pytest.mark.parametrize(
    "ftype,cls",
    [
        ("cont", ContinuousThreshold),
        ("continuous", ContinuousThreshold),
        ("cat", CategoricalOneVsRest),
        ("ordinal", CategoricalOneVsRest),
        ("nominal", CategoricalOneVsRest),
    ],
)
def test_default_proposer_dispatch(ftype, cls):
    assert isinstance(default_proposer(ftype), cls)


# ---- the richer categorical proposers (PR-D) --------------------------------


def _cat_ctx(levels, ftype="cat"):
    """A (N, 2) dataset whose column 1 cycles through `levels`."""
    levels = np.asarray(levels, dtype=float)
    col = np.tile(levels, 3)
    data = np.stack([np.arange(len(col), dtype=float), col], axis=1)
    axis_limits = np.array([[0.0, col.min()], [len(col) - 1.0, col.max()]])
    return _ctx(data, axis_limits, ["cont", ftype])


def _single_level_ctx():
    data = np.stack([np.arange(4, dtype=float), np.full(4, 7.0)], axis=1)
    return _ctx(data, np.array([[0.0, 7.0], [3.0, 7.0]]), ["cont", "cat"])


def _assert_children_partition(cand, data):
    masks = [c.contains(data) for c in cand.conditions]
    np.testing.assert_array_equal(np.sum(masks, axis=0), np.ones(len(data), dtype=int))


def test_multiway_single_candidate_one_child_per_level_ascending():
    ctx = _cat_ctx([2.0, 0.0, 1.0, 3.0])
    (cand,) = CategoricalMultiway().propose(ctx, 1)
    assert cand.feature == 1
    assert [c.subset for c in cand.conditions] == [
        LevelSet({0.0}),
        LevelSet({1.0}),
        LevelSet({2.0}),
        LevelSet({3.0}),
    ]
    _assert_children_partition(cand, ctx.data)


def test_multiway_single_level_proposes_nothing():
    assert CategoricalMultiway().propose(_single_level_ctx(), 1) == []


def test_subsets_count_and_enumeration_order():
    ctx = _cat_ctx([2.0, 0.0, 1.0, 3.0])
    candidates = CategoricalSubsets().propose(ctx, 1)
    assert len(candidates) == 7  # 2^(4-1) - 1
    first_children = [cand.conditions[0].subset.levels for cand in candidates]
    assert first_children == [
        frozenset(s)
        for s in [
            {0.0},
            {0.0, 1.0},
            {0.0, 2.0},
            {0.0, 3.0},
            {0.0, 1.0, 2.0},
            {0.0, 1.0, 3.0},
            {0.0, 2.0, 3.0},
        ]
    ]
    for cand in candidates:
        first, second = (c.subset.levels for c in cand.conditions)
        assert second == frozenset({0.0, 1.0, 2.0, 3.0}) - first
        assert 0.0 in first  # smallest level pinned to the first child
        _assert_children_partition(cand, ctx.data)


def test_subsets_no_complement_duplicates():
    ctx = _cat_ctx([2.0, 0.0, 1.0, 3.0])
    seen = {
        frozenset((cand.conditions[0].subset.levels, cand.conditions[1].subset.levels))
        for cand in CategoricalSubsets().propose(ctx, 1)
    }
    assert len(seen) == 7  # each unordered {S, complement} pair exactly once


def test_subsets_two_levels_single_candidate():
    ctx = _cat_ctx([1.0, 0.0])
    (cand,) = CategoricalSubsets().propose(ctx, 1)
    assert cand.conditions[0].subset == LevelSet({0.0})
    assert cand.conditions[1].subset == LevelSet({1.0})


def test_subsets_single_level_proposes_nothing():
    assert CategoricalSubsets().propose(_single_level_ctx(), 1) == []


def test_subsets_cap_warns_and_degrades_to_one_vs_rest():
    ctx = _cat_ctx(np.arange(9.0))
    with pytest.warns(UserWarning, match="one-vs-rest"):
        capped = CategoricalSubsets(max_levels=4).propose(ctx, 1)
    assert capped == CategoricalOneVsRest().propose(ctx, 1)


def test_subsets_max_levels_below_two_raises():
    with pytest.raises(ValueError, match="max_levels"):
        CategoricalSubsets(max_levels=1)


def test_ordered_ordinal_auto_uses_natural_order():
    ctx = _cat_ctx([2.0, 0.0, 1.0, 3.0], ftype="ordinal")
    candidates = CategoricalOrdered().propose(ctx, 1)
    assert len(candidates) == 3  # K - 1 contiguous cuts, ascending
    prefixes = [cand.conditions[0].subset for cand in candidates]
    assert prefixes == [
        LevelSet({0.0}),
        LevelSet({0.0, 1.0}),
        LevelSet({0.0, 1.0, 2.0}),
    ]
    for cand in candidates:
        first, second = (c.subset.levels for c in cand.conditions)
        assert second == frozenset({0.0, 1.0, 2.0, 3.0}) - first
        _assert_children_partition(cand, ctx.data)


def test_ordered_nominal_auto_uses_similarity_order():
    # levels 0.0 and 2.0 share the companion-column distribution, level 1.0
    # sits far away -> seriation must keep 0.0 and 2.0 adjacent
    rng = np.random.default_rng(3)
    col = np.repeat([0.0, 1.0, 2.0], 20)
    companion = np.concatenate(
        [
            rng.normal(0.0, 0.1, 20),
            rng.normal(10.0, 0.1, 20),
            rng.normal(0.5, 0.1, 20),
        ]
    )
    data = np.stack([companion, col], axis=1)
    ctx = _ctx(
        data,
        np.array([[companion.min(), 0.0], [companion.max(), 2.0]]),
        ["cont", "nominal"],
    )
    candidates = CategoricalOrdered().propose(ctx, 1)
    universe = [0.0, 1.0, 2.0]
    perm = ordering.similarity_order(
        ctx.data, 1, np.array(universe), list(ctx.feature_types)
    )
    expected = [universe[i] for i in perm]
    assert expected != universe  # the constructed case is genuinely reordered
    assert [cand.conditions[0].subset for cand in candidates] == [
        LevelSet(expected[:i]) for i in range(1, 3)
    ]


def test_ordered_explicit_order():
    ctx = _cat_ctx([2.0, 0.0, 1.0])
    # unobserved entries (5.0) are dropped; order otherwise used as-is
    candidates = CategoricalOrdered(order=[2.0, 5.0, 0.0, 1.0]).propose(ctx, 1)
    assert [cand.conditions[0].subset for cand in candidates] == [
        LevelSet({2.0}),
        LevelSet({2.0, 0.0}),
    ]


def test_ordered_explicit_order_missing_observed_level_raises():
    ctx = _cat_ctx([2.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="missing the observed levels"):
        CategoricalOrdered(order=[2.0, 0.0]).propose(ctx, 1)


def test_ordered_junk_order_raises():
    with pytest.raises(ValueError, match="order must be one of"):
        CategoricalOrdered(order="alphabetical")


def test_ordered_two_levels_single_candidate():
    ctx = _cat_ctx([1.0, 0.0], ftype="nominal")
    (cand,) = CategoricalOrdered().propose(ctx, 1)
    assert cand.conditions[0].subset == LevelSet({0.0})
    assert cand.conditions[1].subset == LevelSet({1.0})


def test_ordered_single_level_proposes_nothing():
    assert CategoricalOrdered().propose(_single_level_ctx(), 1) == []

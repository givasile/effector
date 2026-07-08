"""Unit layer for effector.proposers (PR-C): candidate enumeration order,
exact Interval/LevelSet forms, mask semantics, and the k-way shape."""

import numpy as np
import pytest

from effector.proposers import (
    CandidateSplit,
    CategoricalOneVsRest,
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

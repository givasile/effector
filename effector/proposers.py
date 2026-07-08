"""Split proposers: a candidate split is *parent rule -> child conditions*.

A `CandidateSplit` is parent-independent — an ordered tuple of disjoint,
jointly-covering `Condition`s on ONE conditioning feature. Child ``i`` of a
parent region is ``parent_rule.refine(conditions[i])`` with mask
``parent_mask & conditions[i].contains(data)``. Parent-independence is what
lets a level-wise finder apply one candidate to every node of a level, and
it mirrors the finder's search space: positions come from the full data /
axis limits, never from a parent subset. Candidates are k-way by
construction (``len(conditions) >= 2``), so richer proposers (categorical
subsets, multiway, continuous change-point) plug in without a finder change.

This module is a **leaf**: numpy + stdlib + `effector.rules` (itself a leaf)
+ the `effector.ingestion` taxonomy predicate. It must NOT import
`partition`, `space_partitioning`, or `global_effect`.
"""

from dataclasses import dataclass

import numpy as np

from effector import ingestion
from effector.rules import Condition, Interval, LevelSet


@dataclass(frozen=True, eq=False)
class SearchContext:
    """Everything a proposer may read; built once per split search."""

    data: np.ndarray  # (N, D) — the full dataset
    axis_limits: np.ndarray  # (2, D)
    feature_types: tuple  # per-feature type strings
    numerical_grid_size: int  # candidate positions for continuous focs


@dataclass(frozen=True)
class CandidateSplit:
    """An ordered tuple of disjoint, jointly-covering conditions on one
    feature — the children of any parent split by this candidate."""

    conditions: tuple

    def __post_init__(self):
        object.__setattr__(self, "conditions", tuple(self.conditions))
        if len(self.conditions) < 2:
            raise ValueError(
                f"a CandidateSplit needs at least 2 conditions; "
                f"got {len(self.conditions)}"
            )
        features = {c.feature for c in self.conditions}
        if len(features) != 1:
            raise ValueError(
                f"a CandidateSplit must condition on exactly one feature; "
                f"got features {sorted(features)}"
            )

    @property
    def feature(self) -> int:
        return self.conditions[0].feature


class ContinuousThreshold:
    """Binary threshold split: interior linspace positions between the axis
    limits; candidate = ``(x < t, x >= t)``."""

    def propose(self, ctx: SearchContext, foc: int) -> list:
        lo, hi = ctx.axis_limits[0, foc], ctx.axis_limits[1, foc]
        positions = np.linspace(lo, hi, ctx.numerical_grid_size + 1)[1:-1]
        return [
            CandidateSplit(
                (
                    Condition(foc, Interval(hi=t)),  # mask: x < t
                    Condition(foc, Interval(lo=t)),  # mask: x >= t
                )
            )
            for t in positions
        ]


class CategoricalOneVsRest:
    """One-vs-rest over the observed levels: ``({v}, universe - {v})`` per
    level, ascending. Owns the ``!=`` semantics — the complement is
    materialized as an explicit `LevelSet` over the observed universe."""

    def propose(self, ctx: SearchContext, foc: int) -> list:
        universe = {float(v) for v in np.unique(ctx.data[:, foc])}
        return [
            CandidateSplit(
                (
                    Condition(foc, LevelSet({v})),  # mask: x == v
                    Condition(foc, LevelSet(universe - {v})),  # mask: x != v
                )
            )
            for v in sorted(universe)
        ]


def default_proposer(feature_type: str):
    """The finder default: one-vs-rest for categorical conditioning features,
    binary threshold for continuous ones."""
    if ingestion.is_categorical(feature_type):
        return CategoricalOneVsRest()
    return ContinuousThreshold()

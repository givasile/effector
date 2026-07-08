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

This module is a **leaf**: numpy + stdlib + `effector.rules` (itself a leaf),
the `effector.ingestion` taxonomy predicate, and `effector.ordering` (level
seriation). It must NOT import `partition`, `space_partitioning`, or
`global_effect`.

Proposers are **stateless**: the finder protocol deep-copies the finder but
not the proposer instances captured in its `proposer_factory`, so the same
instance may serve searches over different datasets — any instance cache
would silently go stale.
"""

import itertools
import math
import warnings
from dataclasses import dataclass

import numpy as np

from effector import ingestion, ordering
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


def _observed_levels(ctx: SearchContext, foc: int) -> list:
    """The observed level universe, ascending — the one float
    canonicalization every categorical proposer must share."""
    return sorted({float(v) for v in np.unique(ctx.data[:, foc])})


class CategoricalOneVsRest:
    """One-vs-rest over the observed levels: ``({v}, universe - {v})`` per
    level, ascending. Owns the ``!=`` semantics — the complement is
    materialized as an explicit `LevelSet` over the observed universe."""

    def propose(self, ctx: SearchContext, foc: int) -> list:
        universe = set(_observed_levels(ctx, foc))
        return [
            CandidateSplit(
                (
                    Condition(foc, LevelSet({v})),  # mask: x == v
                    Condition(foc, LevelSet(universe - {v})),  # mask: x != v
                )
            )
            for v in sorted(universe)
        ]


class CategoricalMultiway:
    """One k-way candidate: one ``LevelSet({v})`` child per observed level,
    ascending. Fewer than 2 observed levels proposes nothing (a 1-condition
    candidate is not a split)."""

    def propose(self, ctx: SearchContext, foc: int) -> list:
        universe = _observed_levels(ctx, foc)
        if len(universe) < 2:
            return []
        return [CandidateSplit(tuple(Condition(foc, LevelSet({v})) for v in universe))]


class CategoricalSubsets:
    """Every binary subset-vs-complement split over the observed levels, each
    unordered ``{S, complement}`` pair exactly once: the smallest level is
    pinned to the first child, subsets enumerate size-ascending (ties in the
    finder's argmin therefore prefer simpler splits), lexicographic within a
    size — so the first candidates are exactly the one-vs-rest singletons.

    ``2^(K-1) - 1`` candidates explode with the level count, so above
    ``max_levels`` the proposal degrades to the singleton slice (the
    one-vs-rest list) with a `UserWarning` — the feature stays searchable
    instead of silently vanishing from the candidate set.
    """

    def __init__(self, max_levels: int = 8):
        if max_levels < 2:
            raise ValueError(f"max_levels must be >= 2; got {max_levels}")
        self.max_levels = max_levels

    def propose(self, ctx: SearchContext, foc: int) -> list:
        universe = _observed_levels(ctx, foc)
        if len(universe) < 2:
            return []
        if len(universe) > self.max_levels:
            warnings.warn(
                f"feature {foc} has {len(universe)} observed levels "
                f"(> max_levels={self.max_levels}); proposing only the "
                f"one-vs-rest subsets"
            )
            return CategoricalOneVsRest().propose(ctx, foc)
        head, rest = universe[0], universe[1:]
        return [
            CandidateSplit(
                (
                    Condition(foc, LevelSet({head, *combo})),
                    Condition(foc, LevelSet(set(universe) - {head, *combo})),
                )
            )
            for r in range(len(rest))  # |S| - 1, ascending; complement nonempty
            for combo in itertools.combinations(rest, r)
        ]


class CategoricalOrdered:
    """``K - 1`` contiguous prefix/suffix binary splits after ordering the
    levels — both sides explicit `LevelSet`s.

    ``order`` decides the level order, resolved per feature at propose time:

    - ``"auto"`` (default): natural ascending for ordinal features,
      `effector.ordering.similarity_order` seriation for nominal ones;
    - ``"natural"`` / ``"similarity"``: force one of the two;
    - an explicit sequence of level values: used as-is, restricted to the
      observed levels (an observed level missing from it raises).

    The similarity order is recomputed on each propose call (per node or per
    level of the search) — deterministic, and cheap at ``cat_limit``-sized
    level counts; it must not be cached on the instance (see the module note
    on statelessness).
    """

    _ORDER_STRINGS = ("auto", "natural", "similarity")

    def __init__(self, order="auto"):
        if isinstance(order, str):
            if order not in self._ORDER_STRINGS:
                raise ValueError(
                    f"order must be one of {self._ORDER_STRINGS} or a "
                    f"sequence of level values; got {order!r}"
                )
            self.order = order
        else:
            levels = tuple(float(v) for v in order)
            if any(math.isnan(v) for v in levels):
                raise ValueError("an explicit order must not contain NaN")
            self.order = levels

    def _ordered_levels(self, ctx: SearchContext, foc: int, universe: list) -> list:
        if not isinstance(self.order, str):  # explicit level sequence
            observed = set(universe)
            ordered = [v for v in self.order if v in observed]
            missing = observed - set(ordered)
            if missing:
                raise ValueError(
                    f"explicit order for feature {foc} is missing the "
                    f"observed levels {sorted(missing)}"
                )
            return ordered
        by_similarity = self.order == "similarity" or (
            self.order == "auto"
            and ctx.feature_types[foc] in (ingestion.NOMINAL, "cat")
        )
        if by_similarity:
            perm = ordering.similarity_order(
                ctx.data, foc, np.array(universe), list(ctx.feature_types)
            )
            return [universe[i] for i in perm]
        return universe  # natural ascending

    def propose(self, ctx: SearchContext, foc: int) -> list:
        universe = _observed_levels(ctx, foc)
        if len(universe) < 2:
            return []
        levels = self._ordered_levels(ctx, foc, universe)
        return [
            CandidateSplit(
                (
                    Condition(foc, LevelSet(levels[:i])),
                    Condition(foc, LevelSet(levels[i:])),
                )
            )
            for i in range(1, len(levels))
        ]


class ContinuousQuantiles:
    """One k-way candidate per child count ``k`` in ``2..max_children``,
    ascending (ties in the finder's argmin therefore prefer fewer children):
    the children split the conditioning column at its marginal quantiles
    ``i/k`` — a jointly-covering chain of canonical half-open `Interval`s
    over ``(-inf, inf)``.

    Edges are data-driven (x-only): duplicate quantiles collapse, so a
    candidate may end up with fewer than ``k`` children; edges at the column
    minimum are dropped (they would bound an empty first child — a constant
    column therefore proposes nothing); identical edge chains produced by
    different ``k`` are proposed once. Both
    ``ctx.numerical_grid_size`` (the threshold-grid knob) and
    ``ctx.axis_limits`` are ignored — ``max_children`` is this proposer's own
    knob and the unbounded chain covers any axis.
    """

    def __init__(self, max_children: int = 5):
        if max_children < 2:
            raise ValueError(f"max_children must be >= 2; got {max_children}")
        self.max_children = max_children

    def propose(self, ctx: SearchContext, foc: int) -> list:
        col = ctx.data[:, foc]
        candidates, seen = [], set()
        for k in range(2, self.max_children + 1):
            edges = np.unique(np.quantile(col, np.arange(1, k) / k))
            edges = tuple(float(e) for e in edges if col.min() < e)
            if not edges or edges in seen:
                continue
            seen.add(edges)
            bounds = [-np.inf, *edges, np.inf]
            candidates.append(
                CandidateSplit(
                    tuple(
                        Condition(foc, Interval(lo=lo, hi=hi))
                        for lo, hi in zip(bounds[:-1], bounds[1:])
                    )
                )
            )
        return candidates


def default_proposer(feature_type: str):
    """The finder default: one-vs-rest for categorical conditioning features,
    binary threshold for continuous ones."""
    if ingestion.is_categorical(feature_type):
        return CategoricalOneVsRest()
    return ContinuousThreshold()


CATEGORICAL_PROPOSERS = {
    "one_vs_rest": CategoricalOneVsRest,
    "subsets": CategoricalSubsets,
    "ordered": CategoricalOrdered,
    "multiway": CategoricalMultiway,
}
CONTINUOUS_PROPOSERS = {
    "threshold": ContinuousThreshold,
    "quantiles": ContinuousQuantiles,
}


def resolve_proposer(spec, registry: dict, kind: str):
    """A registry name (default-constructed) or a proposer instance (anything
    exposing ``propose``) — everything else is a `ValueError`."""
    if isinstance(spec, str):
        try:
            return registry[spec]()
        except KeyError:
            raise ValueError(
                f"unknown {kind} proposer {spec!r}; expected one of "
                f"{sorted(registry)} or a proposer instance"
            ) from None
    if callable(getattr(spec, "propose", None)):
        return spec
    raise ValueError(
        f"a {kind} proposer must be a registry name or expose a "
        f"propose(ctx, foc) method; got {spec!r}"
    )


def make_proposer_factory(categorical="one_vs_rest", continuous="threshold"):
    """Build a finder's ``feature type -> proposer`` factory from one
    categorical and one continuous proposer spec (names or instances)."""
    categorical = resolve_proposer(categorical, CATEGORICAL_PROPOSERS, "categorical")
    continuous = resolve_proposer(continuous, CONTINUOUS_PROPOSERS, "continuous")

    def factory(feature_type: str):
        if ingestion.is_categorical(feature_type):
            return categorical
        return continuous

    return factory

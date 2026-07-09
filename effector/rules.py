"""The predicate algebra behind regions: subsets, conditions, and rules.

A `Rule` is a normalized conjunction of per-feature conditions — at most one
subset per feature (a hyperbox with categorical level sets). It is the single
source of truth for a region's identity: membership (`rule.contains(X)` is the
only place conditions become masks), display (`rule.format(...)`), and
serialization (`rule.to_dict()`) all derive from the same object, so they can
never drift apart.

Two subset types exist:

- `Interval` — a numeric interval with explicit closedness at both ends. The
  canonical finder form is half-open `[lo, hi)`, matching the split mask
  semantics `x < t` / `x >= t`; the flags exist so user-authored rules
  (`x <= t`) are representable faithfully.
- `LevelSet` — an explicit set of observed levels of a discrete feature. No
  level universe is stored: complements (`!=`) are materialized by the caller
  who knows the levels (the parser via `levels=`, the categorical proposer
  via data).

This module is a **leaf**: it imports only numpy + stdlib and
`effector.ingestion` (the categorical predicate). It must NOT import
`partition`, `space_partitioning`, or `global_effect`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Optional, Union

import numpy as np

from effector import ingestion


def _check_finite_number(value, what: str) -> float:
    """Bounds and levels must be real numbers; NaN never compares meaningfully."""
    value = float(value)
    if math.isnan(value):
        raise ValueError(f"{what} must not be NaN")
    return value


def _fmt_num(value: float, scale: Optional[dict]) -> str:
    """Render a numeric value at the format boundary: optional scale
    (``std * v + mean``), then two decimals (parity with the tree labels)."""
    if scale is not None:
        value = scale["std"] * value + scale["mean"]
    return f"{value:.2f}"


@dataclass(frozen=True)
class Interval:
    """A numeric interval with explicit closedness at both ends.

    Defaults produce the canonical finder form ``[lo, hi)``: a split at ``t``
    yields ``Interval(hi=t)`` (= ``x < t``) and ``Interval(lo=t)``
    (= ``x >= t``). `None` bounds normalize to ±inf; an infinite bound's flag
    is forced open so equal intervals hash equal.
    """

    lo: float = -np.inf
    hi: float = np.inf
    lo_closed: bool = True
    hi_closed: bool = False

    def __post_init__(self):
        lo = -np.inf if self.lo is None else _check_finite_number(self.lo, "lo")
        hi = np.inf if self.hi is None else _check_finite_number(self.hi, "hi")
        object.__setattr__(self, "lo", lo)
        object.__setattr__(self, "hi", hi)
        object.__setattr__(self, "lo_closed", bool(self.lo_closed and np.isfinite(lo)))
        object.__setattr__(self, "hi_closed", bool(self.hi_closed and np.isfinite(hi)))

    @property
    def is_empty(self) -> bool:
        if self.lo > self.hi:
            return True
        return self.lo == self.hi and not (self.lo_closed and self.hi_closed)

    def contains(self, column: np.ndarray) -> np.ndarray:
        column = np.asarray(column)
        lower = column >= self.lo if self.lo_closed else column > self.lo
        upper = column <= self.hi if self.hi_closed else column < self.hi
        return lower & upper

    def intersect(self, other) -> "Subset":
        if isinstance(other, LevelSet):
            return other.intersect(self)
        if self.lo > other.lo:
            lo, lo_closed = self.lo, self.lo_closed
        elif self.lo < other.lo:
            lo, lo_closed = other.lo, other.lo_closed
        else:
            lo, lo_closed = self.lo, self.lo_closed and other.lo_closed
        if self.hi < other.hi:
            hi, hi_closed = self.hi, self.hi_closed
        elif self.hi > other.hi:
            hi, hi_closed = other.hi, other.hi_closed
        else:
            hi, hi_closed = self.hi, self.hi_closed and other.hi_closed
        return Interval(lo, hi, lo_closed, hi_closed)

    def format(self, name: str, scale=None, level_names=None) -> str:
        if self.is_empty:
            return f"{name} ∈ ∅"
        lo_fin, hi_fin = np.isfinite(self.lo), np.isfinite(self.hi)
        if not lo_fin and not hi_fin:
            return f"{name} ∈ (-∞, ∞)"
        if self.lo == self.hi:  # both closed (else is_empty above)
            return f"{name} = {_fmt_num(self.lo, scale)}"
        if not lo_fin:
            glyph = "≤" if self.hi_closed else "<"
            return f"{name} {glyph} {_fmt_num(self.hi, scale)}"
        if not hi_fin:
            glyph = "≥" if self.lo_closed else ">"
            return f"{name} {glyph} {_fmt_num(self.lo, scale)}"
        lo_glyph = "≤" if self.lo_closed else "<"
        hi_glyph = "≤" if self.hi_closed else "<"
        return (
            f"{_fmt_num(self.lo, scale)} {lo_glyph} {name} "
            f"{hi_glyph} {_fmt_num(self.hi, scale)}"
        )

    def to_dict(self) -> dict:
        return {
            "kind": "interval",
            "lo": None if not np.isfinite(self.lo) else float(self.lo),
            "hi": None if not np.isfinite(self.hi) else float(self.hi),
            "lo_closed": self.lo_closed,
            "hi_closed": self.hi_closed,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Interval":
        return cls(d["lo"], d["hi"], d["lo_closed"], d["hi_closed"])


@dataclass(frozen=True)
class LevelSet:
    """An explicit set of levels of a discrete feature (floats, per the
    numpy-only contract). Membership is exact: rebinding to data with unseen
    levels excludes them — an explicit set is safer than a stored ``!=``."""

    levels: frozenset

    def __init__(self, levels: Iterable):
        object.__setattr__(
            self,
            "levels",
            frozenset(_check_finite_number(v, "level") for v in levels),
        )

    @property
    def is_empty(self) -> bool:
        return len(self.levels) == 0

    def contains(self, column: np.ndarray) -> np.ndarray:
        return np.isin(np.asarray(column), sorted(self.levels))

    def intersect(self, other) -> "LevelSet":
        if isinstance(other, Interval):
            return LevelSet(v for v in self.levels if other.contains(np.array([v]))[0])
        return LevelSet(self.levels & other.levels)

    def format(self, name: str, scale=None, level_names=None) -> str:
        def label(v):
            if level_names is not None and v in level_names:
                return level_names[v]
            return _fmt_num(v, scale)

        if self.is_empty:
            return f"{name} ∈ ∅"
        shown = sorted(self.levels)
        if len(shown) == 1:
            return f"{name} = {label(shown[0])}"
        if len(shown) > 4:
            head = ", ".join(label(v) for v in shown[:3])
            return f"{name} ∈ {{{head}, …}} ({len(shown)} levels)"
        return f"{name} ∈ {{{', '.join(label(v) for v in shown)}}}"

    def to_dict(self) -> dict:
        return {"kind": "levels", "levels": sorted(self.levels)}

    @classmethod
    def from_dict(cls, d: dict) -> "LevelSet":
        return cls(d["levels"])


Subset = Union[Interval, LevelSet]

_SUBSET_KINDS = {"interval": Interval, "levels": LevelSet}


def subset_from_dict(d: dict) -> Subset:
    try:
        cls = _SUBSET_KINDS[d["kind"]]
    except KeyError:
        raise ValueError(f"unknown subset kind {d.get('kind')!r}")
    return cls.from_dict(d)


@dataclass(frozen=True)
class Condition:
    """One feature, one domain subset — the atomic predicate."""

    feature: int
    subset: Subset

    def __post_init__(self):
        if not isinstance(self.feature, (int, np.integer)) or self.feature < 0:
            raise ValueError(
                f"Condition.feature must be a non-negative int; got {self.feature!r}"
            )
        object.__setattr__(self, "feature", int(self.feature))

    def contains(self, X: np.ndarray) -> np.ndarray:
        return self.subset.contains(np.asarray(X)[:, self.feature])

    def format(self, feature_names=None, scale_x_list=None, category_names=None) -> str:
        name = feature_names[self.feature] if feature_names else f"x_{self.feature}"
        scale = scale_x_list[self.feature] if scale_x_list else None
        level_names = category_names.get(self.feature) if category_names else None
        return self.subset.format(name, scale=scale, level_names=level_names)


class Rule:
    """A normalized conjunction of per-feature conditions — a region's identity.

    ```python
    rule = effector.Rule.parse("temp < 3 and season == 0", ...)
    mask = rule.contains(X)                  # (N,) bool
    pdp.plot("hr", rule=rule)                # effects accept rules directly
    ```

    At most one subset per feature (a hyperbox with categorical level sets);
    membership, display, and serialization all derive from this one object.
    Rules are immutable and hashable; equality is order-insensitive.
    ``Rule({})`` is the root rule — it contains everything and formats to
    ``""``. A full (unbounded) `Interval` constrains nothing and is dropped at
    construction, so ``Rule({0: Interval()}) == Rule({})``. Emptiness
    (`is_empty`) is representable and never raises; only `parse` rejects
    contradictions, as a courtesy to humans.
    """

    def __init__(self, conditions: Union[dict, Iterable[Condition]] = ()):
        merged: dict = {}
        if isinstance(conditions, dict):
            items = [Condition(f, s) for f, s in conditions.items()]
        else:
            items = list(conditions)
        for cond in items:
            if not isinstance(cond, Condition):
                raise ValueError(
                    f"Rule accepts Conditions or a {{feature: subset}} dict; "
                    f"got {cond!r}"
                )
            f, s = cond.feature, cond.subset
            merged[f] = merged[f].intersect(s) if f in merged else s
        # a full interval constrains nothing — drop it (canonical form)
        self._conditions = {
            f: s
            for f, s in merged.items()
            if not (isinstance(s, Interval) and s == Interval())
        }

    # -- introspection ---------------------------------------------------------
    @property
    def conditions(self):
        """Read-only {feature -> subset} view, in insertion (display) order."""
        return MappingProxyType(self._conditions)

    @property
    def features(self) -> tuple:
        """Indices of the constrained features, in display order."""
        return tuple(self._conditions)

    @property
    def is_root(self) -> bool:
        """`True` for ``Rule({})`` — no conditions, contains everything."""
        return not self._conditions

    @property
    def is_empty(self) -> bool:
        """`True` when any subset is empty — the rule selects nothing."""
        return any(s.is_empty for s in self._conditions.values())

    def __getitem__(self, feature: int) -> Subset:
        return self._conditions[feature]

    def get(self, feature: int, default=None):
        """The subset on `feature`, or `default` when unconstrained."""
        return self._conditions.get(feature, default)

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return NotImplemented
        return self._conditions == other._conditions

    def __hash__(self):
        return hash(frozenset(self._conditions.items()))

    def __repr__(self):
        return f"Rule({self.format() or '<root>'})"

    # -- semantics -------------------------------------------------------------
    def contains(self, X: np.ndarray) -> np.ndarray:
        """Which rows of `X` satisfy the rule — the only place rules become masks.

        Args:
            X: data matrix `(N, D)`.

        Returns:
            boolean mask `(N,)`; all-`True` for the root rule.

        Raises:
            ValueError: `X` is not 2D, or the rule references a feature index
                beyond `X.shape[1]`.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D (N, D) array; got ndim {X.ndim}")
        if self._conditions and max(self._conditions) >= X.shape[1]:
            raise ValueError(
                f"rule references feature {max(self._conditions)} but X has "
                f"only {X.shape[1]} columns"
            )
        mask = np.ones(X.shape[0], dtype=bool)
        for f, subset in self._conditions.items():
            mask &= subset.contains(X[:, f])
        return mask

    def intersect(self, other: "Rule") -> "Rule":
        """The conjunction of two rules — subsets on shared features intersect.

        Returns:
            a new `Rule`; may be empty (`is_empty`), never raises.
        """
        conditions = dict(self._conditions)
        for f, s in other._conditions.items():
            conditions[f] = conditions[f].intersect(s) if f in conditions else s
        return Rule(conditions)

    def refine(self, condition: Condition) -> "Rule":
        """A new rule with one more condition — sugar for `intersect(Rule([condition]))`."""
        return self.intersect(Rule([condition]))

    # -- format boundary ---------------------------------------------------------
    def format(self, feature_names=None, scale_x_list=None, category_names=None) -> str:
        """Human-readable conjunction, e.g. ``"temp < 3.00 and season = winter"``.

        Args:
            feature_names: display names, position = index; defaults to
                ``x_<i>``.
            scale_x_list: per-feature ``{"mean": ..., "std": ...}`` list to
                show numeric values in original units.
            category_names: ``{feature: {level: name}}`` map to show level
                names instead of numbers.

        Returns:
            the formatted string; the root rule formats to ``""``.
        """
        return " and ".join(
            Condition(f, s).format(feature_names, scale_x_list, category_names)
            for f, s in self._conditions.items()
        )

    # -- serialization -----------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize to a plain JSON-able dict — a list of per-feature conditions."""
        return {
            "conditions": [
                {"feature": f, **s.to_dict()} for f, s in self._conditions.items()
            ]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Rule":
        """Rebuild a `Rule` from `to_dict()` output."""
        return cls({c["feature"]: subset_from_dict(c) for c in d["conditions"]})

    # -- parser ------------------------------------------------------------------
    @classmethod
    def parse(
        cls,
        text: str,
        *,
        feature_names,
        feature_types=None,
        levels=None,
        category_names=None,
    ) -> "Rule":
        """Parse ``"temp < 3 and season != 2 and hr in {7, 8}"`` into a `Rule`.

        Grammar: clauses joined by ``and``; each clause
        ``<feature_name> <op> <value>`` with op in ``< <= > >= == != in``
        (``=`` is accepted as ``==``). Continuous features take the four
        inequalities; discrete features take ``==``, ``in {…}``, and ``!=``.
        Level *names* are accepted where `category_names` covers the feature.

        Args:
            text: the rule string.
            feature_names: feature display names, position = index.
            feature_types: per-feature types; required for the discrete-only
                ops ``==``/``!=``/``in``.
            levels: ``{feature_index: observed levels}`` map; required by
                ``!=`` to materialize the complement.
            category_names: ``{feature: {level: name}}`` map so values can be
                given by name.

        Returns:
            the parsed `Rule`.

        Raises:
            ValueError: unknown feature, unparsable clause, an op unsupported
                for the feature's type, or contradictory clauses.
        """
        name_to_idx = {str(n): i for i, n in enumerate(feature_names)}
        conditions: dict = {}

        for clause in re.split(r"(?i)\s+and\s+", text.strip()):
            feature, subset = cls._parse_clause(
                clause, name_to_idx, feature_types, levels, category_names
            )
            merged = (
                conditions[feature].intersect(subset)
                if feature in conditions
                else subset
            )
            if merged.is_empty:
                raise ValueError(
                    f"contradictory conditions on feature "
                    f"{feature_names[feature]!r} in {text!r}"
                )
            conditions[feature] = merged
        return cls(conditions)

    @staticmethod
    def _parse_clause(clause, name_to_idx, feature_types, levels, category_names):
        clause = clause.strip()
        m = re.match(r"(?is)^(.+?)\s+in\s+\{(.*)\}$", clause)
        if m:
            lhs, op, rhs = m.group(1), "in", m.group(2)
        else:
            m = re.match(r"^(.+?)\s*(<=|>=|==|!=|<|>|=)\s*(.+)$", clause)
            if m is None:
                raise ValueError(
                    f"cannot parse clause {clause!r}; expected "
                    f"'<feature> <op> <value>' with op in < <= > >= == != in"
                )
            lhs, op, rhs = m.group(1), m.group(2), m.group(3)
            if op == "=":
                op = "=="

        lhs = lhs.strip()
        if lhs not in name_to_idx:
            raise ValueError(
                f"unknown feature {lhs!r}; expected one of {sorted(name_to_idx)}"
            )
        feature = name_to_idx[lhs]

        level_names = category_names.get(feature) if category_names else None
        name_to_level = (
            {str(v): k for k, v in level_names.items()} if level_names else {}
        )

        def resolve_value(token):
            token = token.strip().strip("'\"")
            try:
                return _check_finite_number(token, "value")
            except ValueError:
                if token in name_to_level:
                    return name_to_level[token]
                raise ValueError(
                    f"cannot resolve value {token!r} for feature {lhs!r}"
                ) from None

        is_cat = feature_types is not None and ingestion.is_categorical(
            feature_types[feature]
        )

        if op in ("<", "<=", ">", ">="):
            if is_cat:
                raise ValueError(
                    f"{op!r} is not supported on the categorical feature {lhs!r}; "
                    f"use '==', '!=', or 'in {{…}}'"
                )
            t = resolve_value(rhs)
            subset = {
                "<": Interval(hi=t),
                "<=": Interval(hi=t, hi_closed=True),
                ">": Interval(lo=t, lo_closed=False),
                ">=": Interval(lo=t),
            }[op]
            return feature, subset

        # == / != / in need the discrete taxonomy
        if not is_cat:
            hint = (
                "feature types were not provided"
                if feature_types is None
                else "the feature is continuous (a point condition has measure zero)"
            )
            raise ValueError(f"{op!r} on feature {lhs!r} is not supported: {hint}")

        if op == "in":
            values = [resolve_value(tok) for tok in rhs.split(",") if tok.strip()]
            if not values:
                raise ValueError(f"empty level set in clause {clause!r}")
            return feature, LevelSet(values)
        if op == "==":
            return feature, LevelSet({resolve_value(rhs)})
        # op == "!=": complement needs the level universe
        if levels is None or feature not in levels:
            raise ValueError(
                f"'!=' on feature {lhs!r} needs the observed levels to "
                f"materialize the complement; pass levels={{...}} or use "
                f"'in {{…}}' instead"
            )
        universe = {float(v) for v in np.asarray(levels[feature]).ravel()}
        return feature, LevelSet(universe - {resolve_value(rhs)})

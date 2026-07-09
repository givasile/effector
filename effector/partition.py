"""Value objects returned by region finders.

`find_regions(feature)` returns a `Partition` — a *value*, not stored state
(design contract R12). A `Partition` holds an ordered list of `Region`s plus a
weak-ish binding to the effect that produced it (for the `plot`/`eval` sugar and
the report/interactive layers). Serialization crosses the boundary via
`to_dict()`; the bound effect is never serialized.

A `Region` is **rule-primary**: its identity is a `rules.Rule` (a normalized
conjunction of per-feature conditions). Membership, display, and serialization
all derive from that one object, so they cannot drift apart. The boolean mask
is a derived cache stamped against one dataset — `bind(effect)` recomputes it
from the rule and verifies it, which is what makes a deserialized partition
safely re-attachable.

This module is a **leaf**: it imports only numpy + stdlib, `effector.helpers`
(scale precedence), `effector.ingestion` (categorical predicate), and
`effector.rules` (itself a leaf). It must NOT import `space_partitioning` or
`global_effect` — the dependency flows one way (those import this).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from effector import helpers, ingestion
from effector.rules import Interval, Rule


@dataclass(frozen=True)
class Region:
    """One subregion of a feature's domain — a frozen value whose identity is its `Rule`.

    `idx == 0` is the root: full data, weight 1.0, the global effect itself.
    `heterogeneity` is `effect.heter_score(feature, mask)` within the region,
    stamped when the region was built. `mask` is a derived cache — the rule
    applied to one dataset — and is `None` on a deserialized region until
    `Partition.bind` recomputes it.

    Attributes:
        idx: position in the partition; 0 is the root.
        name: display name — the feature name, or ``"<feature> | <rule>"``.
        rule: the `Rule` that defines membership — the region's identity.
        heterogeneity: `heter_score(feature, mask)` within the region.
        nof_instances: how many instances the rule selects.
        weight: `nof_instances / N`.
        level: depth in the partition tree (root = 0).
        parent_idx: index of the parent region; `None` for the root.
        mask: boolean `(N,)` cache of `rule.contains(data)`; `None` until bound.
    """

    idx: int
    name: str
    rule: Rule
    heterogeneity: float
    nof_instances: int
    weight: float
    level: int = 0
    parent_idx: Optional[int] = None
    mask: Optional[np.ndarray] = field(default=None, compare=False, repr=False)


def _subset_sort_key(subset):
    if isinstance(subset, Interval):
        return (0, subset.lo, subset.hi)
    return (1, tuple(sorted(subset.levels)))


class Partition:
    """An ordered set of `Region`s covering one feature's domain — a value, not stored state.

    ```python
    part = pdp.find_regions("hr")     # a Partition, bound to the effect
    part.show()                       # tree + per-level stats
    part.plot(1)                      # the effect inside region 1
    ```

    `regions[0]` is always the root — full data, weight 1.0, the global effect
    itself — and the leaves partition it: pairwise disjoint, jointly covering
    (verified whenever masks are present). Each region's identity is its
    `Rule`; masks are derived caches, recomputed and re-verified by `bind`.

    !!! note "Heterogeneity = `heter_score(feature, mask)`"
        The heterogeneity a region reports is exactly the effect's
        `heter_score(feature, mask=region_mask)` — the same scalar the
        region finders minimize.
    """

    def __init__(
        self, regions, *, feature, feature_name, finder_name, feature_names=None
    ):
        if not regions:
            raise ValueError("Partition needs at least one region (the root).")
        for i, r in enumerate(regions):
            if r.idx != i:
                raise ValueError(
                    f"Region.idx must equal its position; region {i} has idx {r.idx}."
                )
        root = regions[0]
        if root.parent_idx is not None:
            raise ValueError("Root region (idx 0) must have parent_idx=None.")
        if root.weight != 1.0:
            raise ValueError("Root region (idx 0) must have weight == 1.0.")
        if not root.rule.is_root:
            raise ValueError("Root region (idx 0) must have the root rule (Rule({})).")

        self.regions = list(regions)
        self.feature = feature
        self.feature_name = feature_name
        self.finder_name = finder_name
        self.feature_names = list(feature_names) if feature_names is not None else None
        self._effect = None
        self._default_scale_x_list = None
        self._category_names = None
        self._check_leaves_partition()

    def _check_leaves_partition(self):
        """The partition invariant: leaves pairwise disjoint ∧ union == root.
        Runs only when every involved mask is present (post-`from_dict`
        regions carry no masks until `bind`)."""
        leaves = self.leaves
        root = self.regions[0]
        involved = [root] + leaves
        if any(r.mask is None for r in involved):
            return
        counts = np.zeros(root.mask.shape[0], dtype=int)
        for leaf in leaves:
            counts += leaf.mask.astype(int)
        overlap = int(np.sum(counts > root.mask.astype(int)))
        gap = int(np.sum((counts == 0) & root.mask))
        if overlap or gap:
            raise ValueError(
                f"Leaves do not partition the root region: {overlap} instance(s) "
                f"covered more than once (or outside the root), {gap} not covered."
            )

    # -- binding to an effect ----------------------------------------------------
    def bind(self, effect):
        """Attach an effect: recompute every region's mask from its rule and verify it.

        ```python
        part = effector.Partition.from_dict(d).bind(pdp)   # live again
        ```

        Masks are recomputed from the rules against `effect.data`, then checked
        against the stored evidence — the finder's mask when present (the
        `find_regions` path), else the serialized `nof_instances` (the
        `from_dict` path). This is what makes a deserialized partition safely
        re-attachable.

        Args:
            effect: a fitted global effect whose `data` the rules apply to.

        Returns:
            `self`, live — `eval`/`eval_heter`/`plot`/`mask` work afterwards.

        Raises:
            ValueError: a rule selects different instances than the stored
                evidence — the effect's data differs from the data the
                partition was built on (check `nof_instances` subsampling
                and `random_state`).
        """
        rebuilt = []
        for r in self.regions:
            m = r.rule.contains(effect.data)
            if r.mask is not None:
                if not (m.shape == r.mask.shape and np.array_equal(m, r.mask)):
                    raise ValueError(
                        f"Region {r.idx} ({r.name!r}): the rule selects different "
                        f"instances than the stored mask — the partition was built "
                        f"on different data."
                    )
                rebuilt.append(r)
            else:
                n = int(m.sum())
                if n != r.nof_instances:
                    raise ValueError(
                        f"Region {r.idx} ({r.name!r}): the rule selects {n} "
                        f"instances on this effect's data but the partition was "
                        f"built with {r.nof_instances}. The effect's data differs "
                        f"— check `nof_instances` subsampling and `random_state`."
                    )
                rebuilt.append(replace(r, mask=m))
        self.regions = rebuilt
        self._effect = effect
        self._default_scale_x_list = effect.scale_x_list
        self._category_names = effect.feature_metadata.category_names
        if self.feature_names is None:
            self.feature_names = list(effect.feature_names)
        self._check_leaves_partition()
        return self

    def _require_effect(self):
        if self._effect is None:
            raise RuntimeError(
                "This Partition is not bound to an effect (e.g. it was rebuilt "
                "from to_dict()); call bind(effect) to enable eval/plot."
            )
        return self._effect

    # -- user-authored partitions --------------------------------------------------
    @classmethod
    def from_rules(cls, rules, *, effect, feature, finder_name="user"):
        """Build a partition from your own rules — no search.

        ```python
        part = effector.Partition.from_rules(
            ["workingday == 0", "workingday == 1"], effect=pdp, feature=3
        )
        part.plot(1)
        ```

        Strings are parsed with the effect's metadata (`Rule.parse`); stats
        (heterogeneity, counts, weights) are stamped from the effect's cached
        local effects — zero model calls — and the result comes back bound.

        Args:
            rules: `Rule` objects or rule strings like `"temp < 3"`.
            effect: a fitted global effect providing data and metadata.
            feature: index of the feature of interest.
            finder_name: label recorded on the partition (default `"user"`).

        Returns:
            a bound `Partition`: the root plus one level-1 region per rule.

        Raises:
            ValueError: the rules do not partition the data — some instance
                is covered more than once or not at all.
        """
        parsed = []
        for r in rules:
            if isinstance(r, str):
                levels = {
                    j: np.unique(effect.data[:, j])
                    for j in range(effect.dim)
                    if ingestion.is_categorical(effect.feature_types[j])
                }
                r = Rule.parse(
                    r,
                    feature_names=effect.feature_names,
                    feature_types=effect.feature_types,
                    levels=levels,
                    category_names=effect.feature_metadata.category_names,
                )
            parsed.append(r)

        n_total = effect.data.shape[0]
        feature_name = effect.feature_names[feature]
        regions = [
            Region(
                idx=0,
                name=feature_name,
                rule=Rule({}),
                heterogeneity=float(effect.heter_score(feature)),
                nof_instances=n_total,
                weight=1.0,
                level=0,
                parent_idx=None,
                mask=np.ones(n_total, dtype=bool),
            )
        ]
        for i, rule in enumerate(parsed):
            mask = rule.contains(effect.data)
            n = int(mask.sum())
            regions.append(
                Region(
                    idx=i + 1,
                    name=f"{feature_name} | {rule.format(effect.feature_names)}",
                    rule=rule,
                    heterogeneity=float(effect.heter_score(feature, mask=mask)),
                    nof_instances=n,
                    weight=n / n_total,
                    level=1,
                    parent_idx=0,
                    mask=mask,
                )
            )
        partition = cls(
            regions,
            feature=feature,
            feature_name=feature_name,
            finder_name=finder_name,
            feature_names=effect.feature_names,
        )
        return partition.bind(effect)

    # -- container protocol ----------------------------------------------------
    def __len__(self):
        return len(self.regions)

    def __iter__(self):
        return iter(self.regions)

    def __getitem__(self, idx):
        try:
            return self.regions[idx]
        except IndexError:
            raise IndexError(
                f"Region index {idx} out of range (partition has {len(self)} regions)."
            )

    @property
    def leaves(self):
        """The regions that are no other region's parent — the finest partition.

        A one-region partition's only leaf is the root."""
        parents = {r.parent_idx for r in self.regions if r.parent_idx is not None}
        return [r for r in self.regions if r.idx not in parents]

    # -- masks & labels --------------------------------------------------------
    def mask(self, idx):
        """The boolean mask of region `idx` — a copy, safe to mutate.

        Args:
            idx: region index (0 = root).

        Returns:
            boolean array `(N,)` over the bound effect's data.

        Raises:
            RuntimeError: the region has no mask (unbound partition — rebuilt
                from `to_dict()`); call `bind(effect)` first.
        """
        region = self[idx]
        if region.mask is None:
            raise RuntimeError(
                f"Region {idx} has no mask (unbound partition — rebuilt from "
                f"to_dict()); call bind(effect) first."
            )
        return region.mask.copy()

    def _format_rule(self, rule, scale_x_list):
        return rule.format(self.feature_names, scale_x_list, self._category_names)

    def label(self, idx, scale_x_list=None):
        """Human-readable label for region `idx`.

        Root -> the feature name; otherwise ``"<feature> | <formatted rule>"``.

        Args:
            idx: region index.
            scale_x_list: optional per-feature ``{"mean": ..., "std": ...}``
                list to display values in original units; defaults to the
                bound effect's.

        Returns:
            the label string.
        """
        scale_x_list = helpers.resolve_scale(scale_x_list, self._default_scale_x_list)
        region = self[idx]
        if region.rule.is_root:
            return self.feature_name
        return f"{self.feature_name} | {self._format_rule(region.rule, scale_x_list)}"

    def _own_condition(self, region, scale_x_list=None):
        """The condition(s) that carve this region out of its parent (root ->
        feature name) — the per-node short label of the tree print."""
        if region.rule.is_root:
            return self.feature_name
        parent_rule = (
            self.regions[region.parent_idx].rule
            if region.parent_idx is not None
            else Rule({})
        )
        diff = {
            f: s for f, s in region.rule.conditions.items() if parent_rule.get(f) != s
        }
        return self._format_rule(Rule(diff), scale_x_list)

    # -- terminal summaries ------------------------------------------------------
    def show(self, scale_x_list=None):
        """Print the partition tree and per-level heterogeneity statistics.

        Each node shows its splitting condition plus a
        ``[id | heter | inst | w]`` chip; the summary shows the heterogeneity
        drop per level. Works on unbound partitions too.

        Args:
            scale_x_list: optional per-feature ``{"mean": ..., "std": ...}``
                list for display in original units; defaults to the bound
                effect's.
        """
        scale_x_list = helpers.resolve_scale(scale_x_list, self._default_scale_x_list)
        feature = self.feature

        # A flat producer yields non-root regions with parent_idx=None;
        # tree rendering does not apply there.
        is_flat = any(r.level > 0 and r.parent_idx is None for r in self.regions)

        print("\n")
        print("Feature {} - Full partition tree:".format(feature))
        if is_flat:
            for r in self.regions:
                print(self.label(r.idx, scale_x_list))
            print("-" * 50)
            print("Feature {} - Statistics per tree level:".format(feature))
            print("\n")
            return

        if len(self) == 1:
            print("No splits found for feature {}".format(feature))
        else:
            self._print_full_tree(scale_x_list)

        print("-" * 50)
        print("Feature {} - Statistics per tree level:".format(feature))
        if len(self) == 1:
            print("No splits found for feature {}".format(feature))
        else:
            self._print_level_stats()
        print("\n")

    def _print_full_tree(self, scale_x_list):
        print("🌳 Full Tree Structure:")
        print("─" * 23)
        for r in self.regions:
            indent = "    " * r.level
            print(
                f"{indent}{self._own_condition(r, scale_x_list)} 🔹 "
                f"[id: {r.idx} | heter: {r.heterogeneity:.2f} "
                f"| inst: {r.nof_instances:d} | w: {r.weight:.2f}]"
            )

    def _print_level_stats(self):
        print("🌳 Tree Summary:")
        print("─" * 17)
        max_level = max(r.level for r in self.regions)
        prev_heter = 0.0
        for lev in range(max_level + 1):
            hk = sum(r.heterogeneity * r.weight for r in self.regions if r.level == lev)
            if lev == 0:
                print(f"Level {lev}🔹heter: {hk:.2f}")
            else:
                indent = "    " * lev
                drop = prev_heter - hk
                perc = 100 * drop / prev_heter if prev_heter else 0
                print(
                    f"{indent}Level {lev}🔹heter: {hk:.2f} | 🔻{drop:.2f} ({perc:.2f}%)"
                )
            prev_heter = hk

    def _stat_chip(self, region, with_weight=True):
        chip = (
            f"[id: {region.idx} | heter: {region.heterogeneity:.2f} "
            f"| inst: {region.nof_instances:d}"
        )
        return chip + (f" | w: {region.weight:.2f}]" if with_weight else "]")

    def show_axes(self, scale_x_list=None):
        """Print the leaves as a partition of the conditioning axes.

        When the leaves differ on one conditioning feature, print them as a
        partition of that axis; on two, as a grid. Anything else falls back
        to the tree print (`show`).

        Args:
            scale_x_list: optional per-feature ``{"mean": ..., "std": ...}``
                list for display in original units; defaults to the bound
                effect's.
        """
        scale_x_list = helpers.resolve_scale(scale_x_list, self._default_scale_x_list)
        leaves = self.leaves
        feats = sorted({f for leaf in leaves for f in leaf.rule.features})

        def fmt(f, subset):
            return self._format_rule(Rule({f: subset}), scale_x_list)

        if not feats:
            return self.show(scale_x_list)

        if len(feats) == 1:
            f = feats[0]
            if any(leaf.rule.get(f) is None for leaf in leaves):
                return self.show(scale_x_list)
            print("\n")
            print(f"Feature {self.feature} - Partition along 1 axis:")
            for leaf in sorted(leaves, key=lambda r: _subset_sort_key(r.rule[f])):
                print(f"{fmt(f, leaf.rule[f])} 🔹 {self._stat_chip(leaf)}")
            print("\n")
            return

        if len(feats) == 2:
            f_row, f_col = feats
            cells = {}
            for leaf in leaves:
                sr, sc = leaf.rule.get(f_row), leaf.rule.get(f_col)
                if sr is None or sc is None or (sr, sc) in cells:
                    return self.show(scale_x_list)
                cells[(sr, sc)] = leaf
            rows = sorted({k[0] for k in cells}, key=_subset_sort_key)
            cols = sorted({k[1] for k in cells}, key=_subset_sort_key)
            if len(rows) * len(cols) != len(cells):
                return self.show(scale_x_list)

            row_labels = [fmt(f_row, s) for s in rows]
            col_labels = [fmt(f_col, s) for s in cols]
            cell_strs = [
                [self._stat_chip(cells[(sr, sc)], with_weight=False) for sc in cols]
                for sr in rows
            ]
            w0 = max(len(lab) for lab in row_labels)
            widths = [
                max(len(col_labels[j]), *(len(row[j]) for row in cell_strs))
                for j in range(len(cols))
            ]
            print("\n")
            print(f"Feature {self.feature} - Partition along 2 axes:")
            print(
                " " * w0
                + "   "
                + "   ".join(lab.ljust(widths[j]) for j, lab in enumerate(col_labels))
            )
            for i, row_lab in enumerate(row_labels):
                print(
                    row_lab.ljust(w0)
                    + "   "
                    + "   ".join(
                        cell_strs[i][j].ljust(widths[j]) for j in range(len(cols))
                    )
                )
            print("\n")
            return

        return self.show(scale_x_list)

    # -- effect-backed sugar ---------------------------------------------------
    def eval(self, idx, xs, **kwargs):
        """The mean effect within region `idx` at positions `xs`.

        Sugar for ``effect.eval(feature, xs, mask=part.mask(idx))`` —
        re-summarized from cached local effects, zero model calls.

        Args:
            idx: region index (0 = root = the global effect).
            xs: where to evaluate, `(T,)`.
            **kwargs: forwarded to `effect.eval` (e.g. `centering`).

        Returns:
            the mean effect at `xs`, shape `(T,)`.

        Raises:
            RuntimeError: the partition is unbound; call `bind(effect)` first.
        """
        return self._require_effect().eval(
            self.feature, xs, mask=self.mask(idx), **kwargs
        )

    def eval_heter(self, idx, xs):
        """The heterogeneity curve within region `idx` at positions `xs`.

        Sugar for ``effect.eval_heter(feature, xs, mask=part.mask(idx))`` —
        model-free.

        Args:
            idx: region index (0 = root = the global effect).
            xs: where to evaluate, `(T,)`.

        Returns:
            the heterogeneity curve at `xs`, shape `(T,)`, non-negative.

        Raises:
            RuntimeError: the partition is unbound; call `bind(effect)` first.
        """
        return self._require_effect().eval_heter(self.feature, xs, mask=self.mask(idx))

    def plot(self, idx, scale_x_list=None, **plot_kwargs):
        """Plot the effect within region `idx`, titled with the region's rule.

        Sugar for ``effect.plot(feature, mask=part.mask(idx), ...)`` with the
        region's `label` as the feature label.

        Args:
            idx: region index (0 = root = the global effect).
            scale_x_list: optional per-feature ``{"mean": ..., "std": ...}``
                list — scales both the axis and the rule in the label.
            **plot_kwargs: forwarded to `effect.plot` (e.g. `heterogeneity`,
                `centering`, `show_plot`).

        Raises:
            RuntimeError: the partition is unbound; call `bind(effect)` first.
        """
        effect = self._require_effect()
        scale_x = scale_x_list[self.feature] if isinstance(scale_x_list, list) else None
        return effect.plot(
            self.feature,
            mask=self.mask(idx),
            feature_label=self.label(idx, scale_x_list),
            scale_x=scale_x,
            **plot_kwargs,
        )

    # -- serialization boundary ------------------------------------------------
    @classmethod
    def from_dict(cls, d):
        """Rebuild a `Partition` from `to_dict()` output.

        ```python
        part = effector.Partition.from_dict(d)   # labels/stats work
        part.bind(pdp)                           # eval/plot work again
        ```

        !!! warning "The result is UNBOUND"
            No effect, no masks: `show`/`show_axes`/`label`/`leaves` work from
            the stored rules and stats, but `eval`/`eval_heter`/`plot`/`mask`
            raise `RuntimeError` until `bind(effect)` recomputes and verifies
            the masks.

        Args:
            d: a dict produced by `to_dict()` (schema version 2).

        Returns:
            an unbound `Partition`.

        Raises:
            ValueError: `d` is not a schema-version-2 (rule-based) dict.
        """
        if d.get("schema_version") != 2:
            raise ValueError(
                "Unsupported partition dict: expected schema_version 2 "
                "(rule-based); v1 (mask-based) dicts are not supported."
            )
        regions = [
            Region(
                idx=r["idx"],
                name=r["name"],
                rule=Rule.from_dict(r["rule"]),
                heterogeneity=float(r["heterogeneity"]),
                nof_instances=int(r["nof_instances"]),
                weight=float(r["weight"]),
                level=int(r["level"]),
                parent_idx=r["parent_idx"],
                mask=None,
            )
            for r in d["regions"]
        ]
        return cls(
            regions,
            feature=d["feature"],
            feature_name=d["feature_name"],
            finder_name=d["finder"],
            feature_names=d.get("feature_names"),
        )

    def to_dict(self):
        """Serialize to a plain JSON-able dict.

        !!! note "Rules + stats only"
            Each region's rule and stamped statistics are serialized — never
            masks, never the data, never the effect or the model.
            `from_dict(...)` + `bind(effect)` restore the rest.

        Returns:
            a dict with `schema_version` 2, feature metadata, and one entry
            per region (rule, heterogeneity, counts, weight, tree links).
        """
        return {
            "schema_version": 2,
            "feature": self.feature,
            "feature_name": self.feature_name,
            "feature_names": self.feature_names,
            "finder": self.finder_name,
            "regions": [
                {
                    "idx": r.idx,
                    "name": r.name,
                    "rule": r.rule.to_dict(),
                    "heterogeneity": float(r.heterogeneity),
                    "nof_instances": int(r.nof_instances),
                    "weight": float(r.weight),
                    "level": int(r.level),
                    "parent_idx": r.parent_idx,
                }
                for r in self.regions
            ],
        }

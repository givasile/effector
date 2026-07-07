"""Value objects returned by region finders.

`find_regions(feature)` returns a `Partition` — a *value*, not stored state
(design contract R12). A `Partition` holds an ordered list of `Region`s plus a
weak-ish binding to the effect that produced it (for the `plot`/`eval` sugar and
the future report/interactive layers). Serialization crosses the boundary via
`to_dict()`; the bound effect is never serialized.

This module is a **leaf**: it imports only numpy + stdlib + `effector.helpers`
(scale precedence) and `effector.ingestion` (categorical predicate). It must NOT
import `space_partitioning`, `tree`, or `global_effect` — the dependency flows
one way (those import this).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from effector import helpers

# Comparison glyphs — ported from tree.py:_comparison_str so we need not import
# tree (keeps this module a leaf). Keep in sync with tree.Tree._comparison_str.
_COMPARISON_GLYPH = {">=": "≥", "<=": "≤", "!=": "≠", "==": "="}


def _glyph(comparison: str) -> str:
    return _COMPARISON_GLYPH.get(comparison, comparison)


@dataclass(frozen=True)
class Region:
    """One subregion of a feature's domain, defined by a boolean mask over the data.

    `idx == 0` is the full-data region (root). Split metadata (`foc_*`,
    `comparison`) is None for the root and for flat finders that do not split.
    """

    idx: int
    name: str
    mask: np.ndarray
    heterogeneity: float
    nof_instances: int
    weight: float
    level: int = 0
    parent_idx: Optional[int] = None
    foc_index: Optional[int] = None
    foc_name: Optional[str] = None
    foc_type: Optional[str] = None
    foc_split_position: Optional[float] = None
    comparison: Optional[str] = None

    def _condition_string(self, scale_x_list=None) -> str:
        """The single condition that carves this region out of its parent, e.g.
        ``"temp ≤ 6.50"``. Mirrors tree.Tree._get_condition_string exactly."""
        pos = self.foc_split_position
        if scale_x_list:
            stats = scale_x_list[self.foc_index]
            if stats is not None:
                pos = stats["std"] * pos + stats["mean"]
        return f"{self.foc_name} {_glyph(self.comparison)} {pos:.2f}"


class Partition:
    """An ordered set of `Region`s covering a feature, produced by a finder.

    Constructed with keyword-only metadata. `regions[0]` is the root (full data,
    weight 1.0). A hierarchy is optional: `parent_idx` chains encode a tree when a
    tree finder produced it, but flat finders may leave `parent_idx is None`.
    """

    def __init__(self, regions, *, feature, feature_name, finder_name):
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

        self.regions = list(regions)
        self.feature = feature
        self.feature_name = feature_name
        self.finder_name = finder_name
        self._effect = None
        self._default_scale_x_list = None

    # -- binding to the producing effect (set by find_regions) ----------------
    def _bind(self, effect):
        self._effect = effect
        self._default_scale_x_list = effect.scale_x_list
        return self

    def _require_effect(self):
        if self._effect is None:
            raise RuntimeError(
                "This Partition is not bound to an effect (e.g. it was rebuilt "
                "from to_dict()); eval/plot are unavailable."
            )
        return self._effect

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
        """Regions that are no other region's parent. A one-region partition's
        only leaf is the root."""
        parents = {r.parent_idx for r in self.regions if r.parent_idx is not None}
        return [r for r in self.regions if r.idx not in parents]

    # -- masks & labels --------------------------------------------------------
    def mask(self, idx):
        """Boolean mask of region `idx` (a COPY — safe to mutate)."""
        return self[idx].mask.copy()

    def label(self, idx, scale_x_list=None):
        """Human-readable label for region `idx`. Root -> feature name; else
        ``"<feature> | cond and cond and ..."`` walking root->idx."""
        scale_x_list = helpers.resolve_scale(scale_x_list, self._default_scale_x_list)
        region = self[idx]
        if region.parent_idx is None and region.level == 0:
            return self.feature_name
        # walk from idx up to (but excluding) the root, collecting conditions
        chain = []
        cur = region
        while cur is not None and cur.parent_idx is not None:
            chain.append(cur)
            cur = self.regions[cur.parent_idx]
        chain.reverse()
        conds = [r._condition_string(scale_x_list) for r in chain]
        return f"{self.feature_name} | " + " and ".join(conds)

    def _short_label(self, region, scale_x_list=None):
        """The node's own single condition (root -> feature name)."""
        if region.parent_idx is None and region.level == 0:
            return self.feature_name
        return region._condition_string(scale_x_list)

    # -- terminal summary (byte-for-byte with old RegionalEffectBase.summary) ---
    def show(self, scale_x_list=None):
        scale_x_list = helpers.resolve_scale(scale_x_list, self._default_scale_x_list)
        feature = self.feature

        # A future flat finder may produce non-root regions with parent_idx=None;
        # tree rendering does not apply there.
        is_flat = any(
            r.level > 0 and r.parent_idx is None for r in self.regions
        )

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
                f"{indent}{self._short_label(r, scale_x_list)} 🔹 "
                f"[id: {r.idx} | heter: {r.heterogeneity:.2f} "
                f"| inst: {r.nof_instances:d} | w: {r.weight:.2f}]"
            )

    def _print_level_stats(self):
        print("🌳 Tree Summary:")
        print("─" * 17)
        max_level = max(r.level for r in self.regions)
        prev_heter = 0.0
        for lev in range(max_level + 1):
            hk = sum(
                r.heterogeneity * r.weight for r in self.regions if r.level == lev
            )
            if lev == 0:
                print(f"Level {lev}🔹heter: {hk:.2f}")
            else:
                indent = "    " * lev
                drop = prev_heter - hk
                perc = 100 * drop / prev_heter if prev_heter else 0
                print(f"{indent}Level {lev}🔹heter: {hk:.2f} | 🔻{drop:.2f} ({perc:.2f}%)")
            prev_heter = hk

    # -- effect-backed sugar ---------------------------------------------------
    def eval(self, idx, xs, **kwargs):
        return self._require_effect().eval(
            self.feature, xs, mask=self.mask(idx), **kwargs
        )

    def eval_heter(self, idx, xs):
        return self._require_effect().eval_heter(self.feature, xs, mask=self.mask(idx))

    def plot(self, idx, scale_x_list=None, **plot_kwargs):
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
    def to_dict(self):
        return {
            "feature": self.feature,
            "feature_name": self.feature_name,
            "finder": self.finder_name,
            "regions": [
                {
                    "idx": r.idx,
                    "name": r.name,
                    "mask": r.mask.astype(bool).tolist(),
                    "heterogeneity": float(r.heterogeneity),
                    "nof_instances": int(r.nof_instances),
                    "weight": float(r.weight),
                    "level": int(r.level),
                    "parent_idx": r.parent_idx,
                    "foc_index": r.foc_index,
                    "foc_name": r.foc_name,
                    "foc_type": r.foc_type,
                    "foc_split_position": r.foc_split_position,
                    "comparison": r.comparison,
                }
                for r in self.regions
            ],
        }


def partition_from_tree(tree, *, feature, feature_name, finder_name) -> Partition:
    """Build a `Partition` from a `space_partitioning` `Tree`. Region.idx equals
    the old node idx (insertion order), so downstream code that referenced
    node_idx keeps working."""
    regions = []
    for node in tree.nodes:
        regions.append(
            Region(
                idx=node.idx,
                name=node.name,
                mask=node.info["active_indices"].astype(bool),
                heterogeneity=float(node.info["heterogeneity"]),
                nof_instances=int(node.info["nof_instances"]),
                weight=float(node.info["weight"]),
                level=int(node.info["level"]),
                parent_idx=(
                    node.parent_node.idx if node.parent_node is not None else None
                ),
                foc_index=node.info.get("foc_index"),
                foc_name=node.info.get("foc_name"),
                foc_type=node.info.get("foc_type"),
                foc_split_position=node.info.get("foc_split_position"),
                comparison=node.info.get("comparison"),
            )
        )
    return Partition(
        regions, feature=feature, feature_name=feature_name, finder_name=finder_name
    )

"""CALM snapshots — the value objects returned by `select_regions`.

A **CALM** (Conditional Additive Local Model) is one snapshot of the feature
effect analysis: the global (GAM) read plus the regional partitions accepted
so far, with the surrogate R² against `f̂` stamped on it. `select_regions`
returns a `CalmSequence` — the chain `[GAM, calm1, calm2, ...]` produced by
greedy forward selection, each step one accepted split, ending when no
remaining split adds at least `min_gain` of `Var(f̂)`.

```python
chain = pdp.select_regions()
chain[0]            # the GAM snapshot (no partitions)
chain.final         # the last CALM — what the report renders as §2
chain.skipped       # rejected splits with reasons
chain.final.importances()   # per-feature, weighted-mean over subregions
```

Like `Partition`, a CALM is a *value*, not stored state (design contract
R12): its scalar summaries (per-feature importance/heterogeneity, R²) are
stamped at construction, so a deserialized CALM renders tables and triage
plots without the effect; `bind(effect)` re-attaches the partitions (with
mask verification) and re-enables the live verbs (`eval`/`plot` on the held
partitions, per-region importance).

An executable `predict(X)` is future work — the surrogate ŷ machinery
already exists in `explained_variance` (`_centered_contribution` +
`_fit_offsets`); a CALM holds everything needed to route new points through
the leaf rules and sum curve reads.

This module is a **leaf**: numpy + stdlib + `effector.partition` (itself a
leaf). It must NOT import `global_effect` — the dependency flows one way.
"""

from __future__ import annotations

import numpy as np

from effector.partition import Partition

_SCHEMA_VERSION = 1


class CALM:
    """One snapshot: global effects everywhere except the accepted split features.

    `index == 0` is the pure GAM (no partitions). Every later CALM adds one
    accepted split on top of the previous one; `stage` records that decision
    (feature, sequential `delta_r2`, running `cum_r2`, heterogeneity
    before/after the split).

    !!! note "Stamped scalars"
        `importance(f)`/`heter_score(f)` are computed at construction —
        weighted mean over subregions for split features (weights =
        instances per leaf), the global value otherwise — and stamped, so an
        unbound CALM still ranks and plots. When bound, the same calls can
        recompute live via the effect's masked verbs and must agree.
    """

    def __init__(
        self,
        *,
        index,
        r2,
        partitions,
        stage=None,
        feature_names,
        target_name,
        importances,
        heter_scores,
        baseline,
        scale_y=None,
    ):
        self.index = int(index)
        self.r2 = float(r2)
        self.partitions = dict(partitions)  # {feature_int: Partition}
        self.stage = stage
        self.feature_names = list(feature_names)
        self.target_name = target_name
        # the schema's y scaling ({"mean", "std"} or None): the stamped
        # scalars speak model-output units; display surfaces bridge by its std
        self.scale_y = scale_y
        self._importances = {int(k): float(v) for k, v in importances.items()}
        self._heter_scores = {int(k): float(v) for k, v in heter_scores.items()}
        # global (imp, het) start points of the split features' triage arrows
        self._baseline = {int(k): tuple(v) for k, v in baseline.items()}
        self._effect = None

    # -- construction ------------------------------------------------------------
    @classmethod
    def from_effect(cls, effect, partitions, *, r2, index=0, stage=None):
        """Stamp a snapshot off a fitted effect — used by `select_regions`.

        Args:
            effect: a fitted global effect (bound partitions apply to its data).
            partitions: `{feature_int: Partition}` — the accepted splits.
            r2: surrogate R² of this snapshot against `f̂` (see
                `explained_variance.surrogate_r2`).
            index: position in the chain; 0 is the GAM.
            stage: the decision that created this snapshot (`None` for the GAM).
        """
        partitions = {int(f): p for f, p in partitions.items()}
        imps, hets, base = {}, {}, {}
        for f in range(effect.dim):
            try:
                effect._check_feature_type_supported(f)
            except ValueError:
                imps[f] = np.nan
                hets[f] = np.nan
                continue
            if f in partitions:
                part = partitions[f]
                w = np.array([leaf.nof_instances for leaf in part.leaves], dtype=float)
                imps[f] = float(
                    np.average(
                        [
                            effect.importance(f, mask=part.mask(leaf.idx))
                            for leaf in part.leaves
                        ],
                        weights=w,
                    )
                )
                hets[f] = float(
                    np.average(
                        [float(leaf.heterogeneity) for leaf in part.leaves],
                        weights=w,
                    )
                )
                base[f] = (
                    float(effect.importance(f)),
                    float(effect.heter_score(f)),
                )
            else:
                imps[f] = float(effect.importance(f))
                hets[f] = float(effect.heter_score(f))
        from effector.report import _scale_payload

        calm = cls(
            index=index,
            r2=r2,
            partitions=partitions,
            stage=stage,
            feature_names=list(effect.feature_names),
            target_name=effect.target_name,
            importances=imps,
            heter_scores=hets,
            baseline=base,
            scale_y=_scale_payload(effect.scale_y),
        )
        calm._effect = effect
        return calm

    # -- binding -------------------------------------------------------------------
    def bind(self, effect):
        """Re-attach a fitted effect: binds every held partition (recomputing
        and verifying masks) and re-enables the live verbs.

        Returns:
            `self`, live.
        """
        for part in self.partitions.values():
            part.bind(effect)
        self._effect = effect
        return self

    def _require_effect(self):
        if self._effect is None:
            raise RuntimeError(
                "This CALM is not bound to an effect (e.g. it was rebuilt "
                "from to_dict()); call bind(effect) first."
            )
        return self._effect

    # -- feature resolution ----------------------------------------------------
    def _resolve(self, feature):
        if isinstance(feature, str):
            try:
                return self.feature_names.index(feature)
            except ValueError:
                raise ValueError(
                    f"Unknown feature {feature!r}; known names: {self.feature_names}"
                ) from None
        return int(feature)

    @property
    def features(self):
        """The split features of this snapshot, ascending."""
        return sorted(self.partitions)

    @property
    def is_gam(self):
        return not self.partitions

    # -- stamped scalar verbs ------------------------------------------------------
    def importance(self, feature, *, per_region=False):
        """This snapshot's importance of `feature` (output units).

        Split feature -> instance-weighted mean over its subregions; else the
        global value. With `per_region=True` (bound CALM, split feature only)
        returns the per-leaf list instead.
        """
        f = self._resolve(feature)
        if per_region:
            part = self._part_for(f)
            effect = self._require_effect()
            return [
                float(effect.importance(f, mask=part.mask(leaf.idx)))
                for leaf in part.leaves
            ]
        return self._importances[f]

    def heter_score(self, feature, *, per_region=False):
        """This snapshot's heterogeneity of `feature` (output units).

        Split feature -> instance-weighted mean over its subregions; else the
        global value. With `per_region=True` (split feature only) returns the
        stamped per-leaf list — available unbound.
        """
        f = self._resolve(feature)
        if per_region:
            part = self._part_for(f)
            return [float(leaf.heterogeneity) for leaf in part.leaves]
        return self._heter_scores[f]

    def _part_for(self, f):
        if f not in self.partitions:
            raise ValueError(
                f"Feature {self.feature_names[f]!r} is not split in this "
                f"CALM (split features: "
                f"{[self.feature_names[j] for j in self.features]})."
            )
        return self.partitions[f]

    def importances(self):
        """`(D,)` vector of this snapshot's importances; NaN = unsupported type."""
        return np.array([self._importances[f] for f in range(len(self.feature_names))])

    def heter_scores(self):
        """`(D,)` vector of this snapshot's heterogeneities; NaN = unsupported type."""
        return np.array([self._heter_scores[f] for f in range(len(self.feature_names))])

    # -- views -----------------------------------------------------------------
    def plot_triage(self, threshold=False, title=None, show_plot=True):
        """This snapshot's triage plane, from the stamped scalars (works unbound).

        One point per feature at the CALM's own (importance, heterogeneity);
        each split feature gets one arrow from its global point to its
        weighted-mean point — the movement the accepted split bought.

        Args:
            threshold: heterogeneity line — a float draws it (given in
                `heter_score` units; rescaled with the points when the schema
                declared a `scale_y`), `False` (default) nothing.
            title: figure title.
            show_plot: if `True`, show and return `None`; else `(fig, ax)`.
        """
        from effector.visualization import triage_scatter

        # stamped scalars speak model-output units; the axes claim the
        # target's units, so bridge by the schema's y-std (spreads: std only)
        sy = self.scale_y["std"] if self.scale_y else 1.0
        pts = [
            (
                self.feature_names[f],
                self._importances[f] * sy,
                self._heter_scores[f] * sy,
            )
            for f in range(len(self.feature_names))
            if np.isfinite(self._importances[f])
        ]
        arrows = {
            self.feature_names[f]: (
                tuple(v * sy for v in self._baseline[f]),
                (self._importances[f] * sy, self._heter_scores[f] * sy),
            )
            for f in self.features
            if f in self._baseline
        }
        if threshold is not None and threshold is not False:
            threshold = float(threshold) * sy
        default = (
            "Feature triage" if self.is_gam else f"Feature triage — CALM {self.index}"
        )
        return triage_scatter(
            pts,
            arrows=arrows,
            threshold=threshold,
            unit=f" ({self.target_name} units)",
            title=default if title is None else title,
            show_plot=show_plot,
        )

    def show(self):
        """Print the snapshot: R², the decision that created it, its partitions."""
        head = "GAM" if self.is_gam else f"CALM {self.index}"
        print(f"{head}: R2 = {self.r2:.1%}")
        if self.stage is not None:
            print(
                f"  + split {self.stage['name']} on {self.stage['on']} "
                f"({self.stage['n_regions']} regions, "
                f"{self.stage['delta_r2']:+.1%})"
            )
        for f in self.features:
            self.partitions[f].show()

    def __repr__(self):
        splits = ", ".join(self.feature_names[f] for f in self.features) or "none"
        return f"CALM(index={self.index}, r2={self.r2:.3f}, splits=[{splits}])"

    # -- serialization boundary --------------------------------------------------
    def to_dict(self):
        """Serialize to a plain JSON-able dict — rules and stamped stats only,
        never masks/data/effect. `from_dict(...)` + `bind(effect)` restore
        the rest."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "index": self.index,
            "r2": self.r2,
            "stage": self.stage,
            "feature_names": self.feature_names,
            "target_name": self.target_name,
            "partitions": {str(f): p.to_dict() for f, p in self.partitions.items()},
            "importances": {str(f): v for f, v in self._importances.items()},
            "heter_scores": {str(f): v for f, v in self._heter_scores.items()},
            "baseline": {str(f): list(v) for f, v in self._baseline.items()},
            "scale_y": self.scale_y,
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild an **unbound** CALM from `to_dict()` output — tables and
        triage work; partition `eval`/`plot` and per-region importance need
        `bind(effect)`."""
        if d.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CALM dict: expected schema_version "
                f"{_SCHEMA_VERSION}, got {d.get('schema_version')!r}."
            )
        return cls(
            index=d["index"],
            r2=d["r2"],
            partitions={
                int(f): Partition.from_dict(p) for f, p in d["partitions"].items()
            },
            stage=d["stage"],
            feature_names=d["feature_names"],
            target_name=d["target_name"],
            importances=d["importances"],
            heter_scores=d["heter_scores"],
            baseline=d["baseline"],
            scale_y=d.get("scale_y"),
        )


class CalmSequence:
    """The chain `select_regions` returns: `[GAM, calm1, ...]` plus the splits it rejected.

    List-like (`chain[0]`, `chain[-1]`, `len`, iteration) with the chain
    invariants: R² is non-decreasing along it and each stage's `delta_r2`
    is the *sequential* marginal, so the gains sum exactly to
    `regional_r2 - gam_r2`.
    """

    def __init__(self, calms, *, skipped=None, min_gain=0.01):
        if not calms:
            raise ValueError("CalmSequence needs at least the GAM snapshot.")
        self.calms = list(calms)
        self.skipped = list(skipped) if skipped else []
        self.min_gain = float(min_gain)

    # -- container protocol ----------------------------------------------------
    def __len__(self):
        return len(self.calms)

    def __iter__(self):
        return iter(self.calms)

    def __getitem__(self, idx):
        return self.calms[idx]

    @property
    def gam(self):
        """The chain's first snapshot — the pure GAM."""
        return self.calms[0]

    @property
    def final(self):
        """The chain's last snapshot — what the report renders as the regional analysis."""
        return self.calms[-1]

    @property
    def gam_r2(self):
        return self.gam.r2

    @property
    def regional_r2(self):
        return self.final.r2

    @property
    def stages(self):
        """The accepted decisions, in order — one per non-GAM snapshot."""
        return [c.stage for c in self.calms[1:]]

    def bind(self, effect):
        """Bind every snapshot (mask verification included). Returns `self`."""
        for c in self.calms:
            c.bind(effect)
        return self

    def show(self, ascii=False):
        """Print the decision sequence as the report's ledger tables:
        EXPLAINED VARIANCE (the GAM, each accepted split, the FINAL R²),
        then REJECTED SPLITS with the reason each one was refused.

        Args:
            ascii: draw with plain ASCII instead of box-drawing characters,
                for terminals and logs that mangle unicode.
        """
        from effector.report import _print_explained_variance

        sy = self.final.scale_y["std"] if self.final.scale_y else 1.0
        _print_explained_variance(
            {
                "gam_r2": self.gam_r2,
                "regional_r2": self.regional_r2,
                "min_gain": self.min_gain,
                "stages": [st for st in self.stages if st is not None],
                "skipped": self.skipped,
            },
            ascii=ascii,
            sy=sy,
        )

    def __repr__(self):
        return (
            f"CalmSequence({len(self.calms)} snapshots, "
            f"R2 {self.gam_r2:.3f} -> {self.regional_r2:.3f}, "
            f"{len(self.skipped)} skipped)"
        )

    # -- serialization boundary --------------------------------------------------
    def to_dict(self):
        """Serialize the chain — a superset of the report's explained-variance
        payload: the flat decision-sequence keys (`gam_r2`, `regional_r2`,
        `min_gain`, `stages`, `skipped`) plus the full `calms` list."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "gam_r2": self.gam_r2,
            "regional_r2": self.regional_r2,
            "min_gain": self.min_gain,
            "stages": self.stages,
            "skipped": self.skipped,
            "calms": [c.to_dict() for c in self.calms],
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild an **unbound** chain from `to_dict()` output."""
        if d.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported CalmSequence dict: expected schema_version "
                f"{_SCHEMA_VERSION}, got {d.get('schema_version')!r}."
            )
        return cls(
            [CALM.from_dict(c) for c in d["calms"]],
            skipped=d["skipped"],
            min_gain=d["min_gain"],
        )

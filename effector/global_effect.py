"""The global-effect engine: the two-block lifecycle (design contract R14).

Every effect object holds two caches and one config, all owned here (the
ALE family adds a third, cache (a′) — see below):

- **Cache (a) — local effects** (`self._local`): the model-touching material,
  one frame-carrying entry per feature. Recomputed iff absent or the stored
  frame differs from the frame derived from the current config; a frame change
  replaces the entry and bumps the feature's epoch.
- **Cache (b) — summaries** (`self._summaries`): everything derived from (a)
  in pure numpy — payloads and centering constants — memoized by
  `(feature, epoch, mask_key[, mode])`. Stale entries become unreachable when
  the epoch bumps (frame or config change); they are never served.
- **Config** (`self.fit_args`): the method settings `fit()` declares (binning,
  order, scope, default centering) — the kwargs `eval`/`plot` deliberately do
  not accept.
- **Cache (a′) — pairwise level effects** (`ALEBase._local_pairs` only): the
  all-pairs raw level differences feeding the order-free nominal scalars.
  Model-touching like (a), but framed by the ascending level set (immutable
  per object): an `order=` refit replaces (a) without touching (a′), and (a′)
  never bumps the epoch. Its summaries memoize into (b) under a `"pairs"` key.

Subclasses implement a frame declaration (`_frame_from_config`) and three pure
kernels, each split into a continuous and a categorical variant —
`_compute_local_cont|_cat` (the only kernels that may touch the model),
`_summarize_cont|_cat` (numpy in, payload dict out), `_eval_payload_cont|_cat`
(payload + xs in, numbers out) — and contain no cache, retrigger, or mask
logic. The base owns the dispatch (`_is_cat(feature)`, once per kernel); a
type-agnostic kernel is declared with a class-level alias
(`_compute_local_cat = _compute_local_cont`), never a fork in the body.
Methods whose `SUPPORTED_FEATURE_TYPES` exclude ordinal/nominal skip the
`_cat` variants entirely — the capability matrix makes them unreachable.
"""

import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union

import numpy as np

import effector.axis_partitioning as ap
from effector import helpers, ingestion, utils
from effector.rules import Rule


# the shared mask-key sentinel: `mask=None` and an all-ones mask are the same
# summary (rule M1) — that equivalence lives in `_mask_key` and nowhere else
_ALL = b"ALL"


def check_feature_type_supported(
    method_name: str, supported: frozenset, ftype: str, feature: int, feature_name: str
) -> None:
    """The capability matrix as an error (method_semantics.md). Module-level so
    the regional path can enforce the same contract as the global fit loop."""
    if ftype in supported:
        return
    raise ValueError(
        f"{method_name} does not support {ftype} features "
        f"(feature {feature} {feature_name!r} is {ftype})."
    )


def check_binning_scope(binning_scope: str) -> None:
    """Validate the `binning_scope` fit kwarg of the adaptive-binning methods
    (RHALE/ShapDP): the x-range handed to the binner when a mask restricts the
    data — `"global"` = the frozen global frame, `"effective"` = the masked
    column's own `[min, max]`."""
    if binning_scope not in ("global", "effective"):
        raise ValueError(
            f"binning_scope must be 'global' or 'effective'; got {binning_scope!r}"
        )


class GlobalEffectBase(ABC):
    # the class-level centering default (R3): each subclass declares it once
    DEFAULT_CENTERING: Union[bool, str] = False

    # capability contract per feature type (method_semantics.md): which of
    # continuous/ordinal/nominal the method supports, and the strategy code it
    # uses for discrete features (mirrored into the R5 registry)
    SUPPORTED_FEATURE_TYPES: frozenset = frozenset(
        {ingestion.CONTINUOUS, ingestion.ORDINAL, ingestion.NOMINAL}
    )
    CAT_STRATEGY: Optional[str] = None

    _SUMMARIES_MAX = 512

    def __init__(
        self,
        method_name: str,
        data,
        model: Callable,
        model_jac: Optional[Callable] = None,
        *,
        data_effect: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
    ) -> None:
        """Constructor: ingest and nothing else (R1) — no model calls happen here."""
        self.method_name = method_name.lower()
        self.random_state = random_state

        # the border crossing (R10): validate numpy `data`, resolve/auto-infer
        # metadata; `model`/`model_jac` pass through as given (numpy-only)
        ing = ingestion.ingest(data, model, model_jac, schema=schema)
        data = ing.data
        self.model = ing.model
        self.model_jac = ing.model_jac
        self.feature_metadata: ingestion.FeatureMetadata = ing.meta

        self.dim = data.shape[1]

        # shared preprocessing: filter to axis_limits (or infer them), then
        # subsample nof_instances (helpers.prep_data)
        data, data_effect, axis_limits, self.nof_instances, self.indices = (
            helpers.prep_data(
                data, axis_limits, nof_instances, data_effect, random_state
            )
        )
        self.axis_limits: np.ndarray = axis_limits

        # store the data
        self.data: np.ndarray = data
        self.data_effect: Optional[np.ndarray] = data_effect

        # flat mirrors of the resolved metadata
        self.feature_names: list = list(ing.meta.feature_names)
        self.feature_types: list = list(ing.meta.feature_types)
        self.cat_limit: int = ing.meta.cat_limit
        self.target_name: str = ing.meta.target_name
        self.scale_x_list: Optional[list] = ing.meta.scale_x_list
        self.scale_y: Optional[dict] = ing.meta.scale_y

        # state flag mirror (kept for introspection; the caches are the truth)
        self.is_fitted: np.ndarray = np.ones([self.dim]) < 0

        # the declared config (R14): set by fit(), read by the summary gate
        self.fit_args: dict = {}

        # cache (a) — local effects: {feature: {"frame": tuple, ...arrays}}
        self._local: dict = {}
        # per-feature epoch: bumped on frame replacement or config change
        self._epoch: dict = {}
        # cache (b) — summaries: payloads (f, epoch, mask_key) and centering
        # constants (f, epoch, mask_key, mode), LRU-bounded
        self._summaries: "OrderedDict" = OrderedDict()
        # model(data) predictions, computed once, first time an average output
        # is needed (global or masked)
        self._y_pred: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # the three kernel dispatchers + the frame declaration (R14). Each kernel
    # comes in a continuous and a categorical variant; the dispatch below is
    # the ONLY place a kernel forks on the feature type. The `_cat` defaults
    # raise: they are unreachable when `SUPPORTED_FEATURE_TYPES` excludes
    # ordinal/nominal (`_ensure_local` enforces the capability matrix first).
    # ------------------------------------------------------------------
    def _frame_from_config(self, feature: int) -> tuple:
        """The frame tuple this feature's local effects depend on, derived from
        the declared config. `()` means the local effects are instance-anchored
        (never invalidated; PDP's position store only grows)."""
        return ()

    def _compute_local(self, feature: int, frame: tuple) -> dict:
        if self._is_cat(feature):
            return self._compute_local_cat(feature, frame)
        return self._compute_local_cont(feature, frame)

    @abstractmethod
    def _compute_local_cont(self, feature: int, frame: tuple) -> dict:
        """The one model-touching kernel: compute the per-instance local
        effects of a continuous `feature` under `frame` and return the
        cache-(a) entry — a dict that includes `"frame": frame` plus
        instance-aligned arrays."""
        raise NotImplementedError

    def _compute_local_cat(self, feature: int, frame: tuple) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} declares no categorical local-effect "
            f"kernel — unreachable while SUPPORTED_FEATURE_TYPES excludes "
            f"ordinal/nominal"
        )

    def _summarize(
        self, feature: int, mask: Optional[np.ndarray] = None, **config
    ) -> dict:
        if self._is_cat(feature):
            return self._summarize_cat(feature, mask, **config)
        return self._summarize_cont(feature, mask, **config)

    @abstractmethod
    def _summarize_cont(
        self, feature: int, mask: Optional[np.ndarray] = None, **config
    ) -> dict:
        """Pure-numpy kernel: derive the payload of a continuous `feature`
        from the cached local effects restricted to `mask` (`None` = all
        instances), under the declared `config`. Must not touch the model."""
        raise NotImplementedError

    def _summarize_cat(
        self, feature: int, mask: Optional[np.ndarray] = None, **config
    ) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} declares no categorical summary kernel — "
            f"unreachable while SUPPORTED_FEATURE_TYPES excludes "
            f"ordinal/nominal"
        )

    def _eval_payload(
        self,
        feature: int,
        params: dict,
        x: np.ndarray,
        heterogeneity: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self._is_cat(feature):
            return self._eval_payload_cat(feature, params, x, heterogeneity)
        return self._eval_payload_cont(feature, params, x, heterogeneity)

    @abstractmethod
    def _eval_payload_cont(
        self,
        feature: int,
        params: dict,
        x: np.ndarray,
        heterogeneity: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Pure reader kernel: the *uncentered* mean effect at `x` read off the
        payload `params`, and — if `heterogeneity` — also the heterogeneity
        curve h(x). Same payload + same x → same answer; no model, no state."""
        raise NotImplementedError

    def _eval_payload_cat(
        self,
        feature: int,
        params: dict,
        x: np.ndarray,
        heterogeneity: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError(
            f"{type(self).__name__} declares no categorical reader kernel — "
            f"unreachable while SUPPORTED_FEATURE_TYPES excludes "
            f"ordinal/nominal"
        )

    # ------------------------------------------------------------------
    # the two gates (all queries go through these; nothing else checks caches)
    # ------------------------------------------------------------------
    def _config(self, feature: int) -> dict:
        """The declared method config for `feature` (fit_args minus centering)."""
        prev = self.fit_args.get("feature_" + str(feature), {})
        return {k: v for k, v in prev.items() if k != "centering"}

    def _bump_epoch(self, feature: int) -> None:
        self._epoch[feature] = self._epoch.get(feature, 0) + 1

    def _ensure_local(self, feature: int) -> dict:
        """Gate to cache (a). The one retrigger rule (R14): recompute iff the
        entry is absent or its stored frame differs from the frame derived from
        the current config. A replacement bumps the epoch. Every query passes
        through here, so the capability matrix is enforced once, for all of
        them."""
        self._check_feature_type_supported(feature)
        want = self._frame_from_config(feature)
        entry = self._local.get(feature)
        if entry is None or entry["frame"] != want:
            self._local[feature] = self._compute_local(feature, want)
            self._bump_epoch(feature)
        return self._local[feature]

    @staticmethod
    def _mask_key(mask: Optional[np.ndarray]) -> bytes:
        """`None` and all-ones normalize to one key (M1) — the single place
        that equivalence lives. `mask` must already be prepped."""
        if mask is None or mask.all():
            return _ALL
        return mask.tobytes()

    def _memo(self, key, builder):
        hit = self._summaries.get(key)
        if hit is not None:
            self._summaries.move_to_end(key)
            return hit
        val = builder()
        self._summaries[key] = val
        if len(self._summaries) > self._SUMMARIES_MAX:
            self._summaries.popitem(last=False)
        return val

    def _summary(self, feature: int, mask: Optional[np.ndarray] = None) -> dict:
        """Gate to cache (b): the payload of (feature, mask) under the current
        epoch — memoized, recomputed from (a) on a miss. Treat as read-only."""
        self._ensure_local(feature)
        key = (feature, self._epoch.get(feature, 0), self._mask_key(mask))
        return self._memo(
            key, lambda: self._summarize(feature, mask, **self._config(feature))
        )

    def _centering_const(self, feature: int, mask: Optional[np.ndarray], mode: str):
        """The centering constant of (feature, mask, mode) — a summary like any
        other (R14): derived from the payload, memoized, zero model calls."""
        params = self._summary(feature, mask)
        key = (feature, self._epoch.get(feature, 0), self._mask_key(mask), mode)
        return self._memo(
            key, lambda: self._compute_norm_const(feature, mode, params, mask)
        )

    # ------------------------------------------------------------------
    # shared derivations (pure numpy on top of the gates)
    # ------------------------------------------------------------------
    def _eval_mean(
        self, feature: int, x: np.ndarray, params: dict, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """The uncentered mean effect at `x` — default reads the payload.
        (d-)PDP overrides it: exact ICE columns from the growing position store
        (global), or the transient `data[mask]` retouch off the cached
        positions (masked) — the one documented model-touching exception."""
        return self._eval_payload(feature, params, x)

    def _compute_norm_const(
        self,
        feature: int,
        method: str,
        params: dict,
        mask: Optional[np.ndarray] = None,
    ):
        """Derive the centering constant from the payload: `zero_integral` =
        the mean over the (effective) feature interval, `zero_start` = the
        value at its left limit. Pure numpy; (d-)PDP overrides the two
        variants (not this dispatcher) with the per-instance form read off
        the cached ICE columns."""
        assert method in ["zero_integral", "zero_start"]
        if self._is_cat(feature):
            return self._compute_norm_const_cat(feature, method, params, mask)
        return self._compute_norm_const_cont(feature, method, params, mask)

    def _compute_norm_const_cat(
        self,
        feature: int,
        method: str,
        params: dict,
        mask: Optional[np.ndarray] = None,
    ):
        # discrete centering (method_semantics.md): zero_integral is the
        # frequency-weighted level mean (order-invariant); zero_start
        # zeroes the first level *in fit order* (a custom `order` makes
        # its first entry the reference level)
        levels, weights = self._level_weights(feature, mask)
        if method == "zero_integral":
            return float(
                np.average(self._eval_payload(feature, params, levels), weights=weights)
            )
        fit_levels = params.get("levels", levels)
        return self._eval_payload(feature, params, np.asarray(fit_levels[:1])).item()

    def _compute_norm_const_cont(
        self,
        feature: int,
        method: str,
        params: dict,
        mask: Optional[np.ndarray] = None,
    ):
        start, stop = self._effective_limits(feature, mask)
        if method == "zero_integral":
            return utils.mean_1d_linspace(
                lambda x: self._eval_payload(feature, params, x),
                start,
                stop,
                helpers.NOF_INTERNAL_POINTS,
            )
        return self._eval_payload(feature, params, np.array([start])).item()

    def _bin_local_effects(
        self,
        feature: int,
        col: np.ndarray,
        eff: np.ndarray,
        mask: Optional[np.ndarray],
        binning_method,
        binning_scope: str,
    ) -> dict:
        """The shared continuous-summary tail of the adaptive-binning methods
        (RHALE/ShapDP): bin the (already mask-sliced) local effects `eff` at
        positions `col` with the declared binner over the scope's x-range, and
        return the `compute_ale_params` payload (+ `alg_params`)."""
        binning = (
            ap.return_default(binning_method)
            if isinstance(binning_method, str)
            else binning_method
        )
        if mask is not None and binning_scope == "effective":
            limits_range = np.asarray(self._effective_limits(feature, mask))
        else:
            limits_range = self.axis_limits[:, feature]
        limits = binning.find_limits(col, eff, limits_range)
        utils.raise_if_no_binning(limits, feature, binning)
        params = utils.compute_ale_params(col, eff, limits)
        params["alg_params"] = binning
        return params

    def _mean_norm_const(self, norm_const):
        """The scalar amount the centered mean effect subtracts. It is a scalar
        for most methods (ALE/RHALE/ShapDP), so the value is returned as is;
        PDP overrides this because its constant is a per-instance array."""
        return norm_const

    def _avg_output(self, mask: Optional[np.ndarray], scale_y: Optional[dict]) -> float:
        """The (masked) average model output for plots — from the `_y_pred`
        cache, computed once per object (the plot layer never calls the model)."""
        if self._y_pred is None:
            self._y_pred = np.asarray(self.model(self.data))
        y = self._y_pred if mask is None else self._y_pred[mask]
        return helpers.prep_avg_output(None, None, float(np.mean(y)), scale_y)

    def _prep_mask(self, mask) -> Optional[np.ndarray]:
        """Normalize a user `mask` to a boolean `(N,)` array (`None` passes
        through). Strictly boolean — an integer index array is rejected rather
        than silently reinterpreted as truth values."""
        if mask is None:
            return None
        mask = np.asarray(mask)
        if mask.dtype != bool:
            raise ValueError(
                f"mask must be a boolean array of shape ({self.data.shape[0]},); "
                f"got dtype {mask.dtype}"
            )
        if mask.shape != (self.data.shape[0],):
            raise ValueError(
                f"mask must have shape ({self.data.shape[0]},); got {mask.shape}"
            )
        if not mask.any():
            raise ValueError("mask selects no instances")
        return mask

    def _resolve_mask(
        self, mask: Optional[np.ndarray] = None, rule: Union[None, str, Rule] = None
    ) -> Optional[np.ndarray]:
        """Normalize a query's `mask`/`rule` pair to a prepped boolean mask.
        `rule` is sugar over the mask path: a `Rule` (or a string parsed with
        this effect's metadata) applied to `self.data`. Exactly one of the two
        may be given."""
        if rule is None:
            return self._prep_mask(mask)
        if mask is not None:
            raise ValueError("pass either `mask` or `rule`, not both")
        if isinstance(rule, str):
            levels = {
                j: np.unique(self.data[:, j])
                for j in range(self.dim)
                if self._is_cat(j)
            }
            rule = Rule.parse(
                rule,
                feature_names=self.feature_names,
                feature_types=self.feature_types,
                levels=levels,
                category_names=self.feature_metadata.category_names,
            )
        return self._prep_mask(rule.contains(self.data))

    def _effective_limits(
        self, feature: int, mask: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """The transient x-interval a (feature, mask) pair lives on: the masked
        column's `[min, max]`; `None` = the immutable global frame. The global
        `axis_limits` are never mutated by a mask — anything region-shaped is
        derived per call. Raises on a degenerate masked interval."""
        if mask is None:
            return self.axis_limits[0, feature], self.axis_limits[1, feature]
        col = self.data[mask, feature]
        lo, hi = float(col.min()), float(col.max())
        if not lo < hi:
            raise ValueError(
                f"Feature {feature} has a degenerate interval [{lo}, {hi}] "
                f"within the masked subregion"
            )
        return lo, hi

    def _is_cat(self, feature: int) -> bool:
        """Does `feature` behave categorically (ordinal or nominal)?"""
        return ingestion.is_categorical(self.feature_types[feature])

    def _levels(self, feature: int) -> np.ndarray:
        """The observed levels of a discrete feature, ascending."""
        return np.unique(self.data[:, feature])

    def _level_weights(
        self, feature: int, mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """(levels, frequencies) of a discrete feature — the weights of every
        frequency-weighted quantity in method_semantics.md. With a `mask` the
        levels and frequencies are those *within* the masked subregion."""
        col = self.data[:, feature] if mask is None else self.data[mask, feature]
        levels, counts = np.unique(col, return_counts=True)
        return levels, counts / counts.sum()

    def _level_display(self, feature: int, levels=None):
        """(positions, tick labels) for categorical plots: positions are the
        level values; labels translate the `schema.category_names` map back to
        the original level names. `None` labels keep numeric ticks. `levels`
        overrides the ascending default (fit-order for nominal ALE)."""
        if levels is None:
            levels = self._levels(feature)
        cat_names = self.feature_metadata.category_names
        name_of = cat_names.get(feature) if cat_names else None
        if name_of is not None:
            # schema category_names, resolved to a {level_value: name} map at
            # ingest — maps by value, so level subsets (regional nodes) are fine
            labels = [name_of.get(float(v), f"{v:g}") for v in levels]
        elif self.feature_types[feature] == ingestion.NOMINAL:
            labels = [f"{v:g}" for v in levels]
        else:
            labels = None
        return levels, labels

    def _resolve_level_order(self, feature: int, levels: np.ndarray, order):
        """Resolve the `order` argument of (RH)ALE.fit for one discrete
        feature: `"similarity"` induces the order from the other features
        (effector.ordering); a list declares it explicitly."""
        if isinstance(order, str):
            if order != "similarity":
                raise ValueError(
                    f"invalid order {order!r}; use None, 'similarity', or an "
                    f"explicit list of the levels"
                )
            from effector import ordering

            return levels[
                ordering.similarity_order(
                    self.data, feature, levels, self.feature_types
                )
            ]
        arr = np.asarray(order, dtype=float)
        if sorted(arr.tolist()) != sorted(levels.tolist()):
            raise ValueError(
                f"order must be a permutation of the observed levels "
                f"{levels.tolist()}; got {np.asarray(order).tolist()}"
            )
        return arr

    def _check_feature_type_supported(self, feature: int) -> None:
        """The capability matrix as an error (method_semantics.md)."""
        check_feature_type_supported(
            self.method_name,
            self.SUPPORTED_FEATURE_TYPES,
            self.feature_types[feature],
            feature,
            self.feature_names[feature],
        )

    def _resolve_feature(self, feature: Union[int, str]) -> int:
        """Resolve a feature given by index or name to its index (R9 errors).

        Every public verb calls this first, so `pdp.plot("hour")` works
        wherever `pdp.plot(3)` does.
        """
        if isinstance(feature, str):
            return helpers.resolve_feature_name(feature, self.feature_names, self.dim)
        if isinstance(feature, bool) or not isinstance(feature, (int, np.integer)):
            raise TypeError(
                f"Invalid feature of type {type(feature).__name__}: {feature!r}; "
                "use an integer index or a feature name"
            )
        if not 0 <= feature < self.dim:
            raise ValueError(
                f"Feature index {feature} out of range for data with "
                f"{self.dim} features"
            )
        return int(feature)

    # ------------------------------------------------------------------
    # fit: declare the config + warm the caches (R1/R14)
    # ------------------------------------------------------------------
    @abstractmethod
    def fit(
        self,
        features: Union[int, str, list] = "all",
        centering: Union[bool, str] = False,
        **kwargs,
    ) -> None:
        """Declare the method configuration and do the expensive work now.

        ```python
        ale.fit("all")                                   # everything, defaults
        ale.fit(["hr", "temp"], binning_method="dp")     # custom config
        ```

        !!! tip "`fit` is optional"
            Any `eval`/`plot` on an unfitted feature silently computes what it
            needs with the defaults. `fit` is *the place to customize*: its
            kwargs (binning, order, scope) are deliberately not accepted by
            `eval`/`plot`.

        Args:
            features: which features to fit — an index/name, a list, or `"all"`.
            centering: the default centering mode for this feature:
                `False` (raw), `True`/`"zero_integral"` (zero mean), or
                `"zero_start"` (starts at 0).
            **kwargs: method-specific config — see each class's `fit`.
        """
        raise NotImplementedError

    def _fit_loop(
        self,
        features: Union[int, str, list],
        centering: Union[bool, str],
        **fit_feature_kwargs,
    ) -> None:
        """The one fit skeleton every method shares (R1): record the declared
        config, bump the epoch (the config may have changed — old summaries
        must become unreachable), then warm cache (a) and the all-ones payload
        (plus the centering constant, if a mode was declared)."""
        features = helpers.prep_features(features, self.dim, self.feature_names)
        centering = helpers.prep_centering(centering)
        for s in features:
            self._check_feature_type_supported(s)
            self.fit_args["feature_" + str(s)] = {
                "centering": centering,
                **fit_feature_kwargs,
            }
            self._bump_epoch(s)
            self._ensure_local(s)
            self._summary(s, None)
            if centering is not False:
                self._centering_const(s, None, centering)
            self.is_fitted[s] = True

    # ------------------------------------------------------------------
    # public queries (R1/R2/R11/R13) — thin wrappers over the two gates
    # ------------------------------------------------------------------
    def eval(
        self,
        feature: Union[int, str],
        xs: np.ndarray,
        centering: Union[None, bool, str] = None,
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> np.ndarray:
        """The mean effect of a feature at positions `xs`.

        ```python
        xs = np.linspace(0, 24, 100)
        y = pdp.eval("hr", xs)                            # (100,) mean effect
        y_wd = pdp.eval("hr", xs, rule="workingday == 0") # same, on a subregion
        ```

        !!! note "One array, one type (R1)"
            `eval` always returns the mean effect only. The spread around it
            has its own ladder: `eval_heter` (curve), `heter_score` (scalar),
            `payload` (the raw fitted object).

        !!! warning "Discrete features"
            Ordinal/nominal features are evaluated **only at observed
            levels** — any other `xs` value raises `ValueError`.

        Args:
            feature: index or name of the feature of interest.
            xs: where to evaluate, `(T,)`.
            centering: `None` (class default), `False`,
                `True`/`"zero_integral"`, or `"zero_start"`.
            mask: optional boolean `(N,)` selecting a subregion — the effect
                *within* it, re-summarized from cached local effects with zero
                model calls. Nothing is stored.
            rule: sugar over `mask` — an `effector.Rule` or a string like
                `"temp < 3 and season == 0"`. Mutually exclusive with `mask`.

        Returns:
            the mean effect at `xs`, shape `(T,)`.
        """
        feature = self._resolve_feature(feature)
        centering = self.DEFAULT_CENTERING if centering is None else centering
        centering = helpers.prep_centering(centering)
        mask = self._resolve_mask(mask, rule)

        if not self._is_cat(feature):
            if mask is not None:
                self._effective_limits(feature, mask)  # degeneracy guard
            elif not self.axis_limits[0, feature] < self.axis_limits[1, feature]:
                raise ValueError(
                    f"Feature {feature} has a degenerate axis interval "
                    f"[{self.axis_limits[0, feature]}, {self.axis_limits[1, feature]}]"
                )

        params = self._summary(feature, mask)
        y = self._eval_mean(feature, xs, params, mask)
        if centering is not False:
            y = y - self._mean_norm_const(
                self._centering_const(feature, mask, centering)
            )
        return y

    def eval_heter(
        self,
        feature: Union[int, str],
        xs: np.ndarray,
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> np.ndarray:
        """The heterogeneity curve h(xs): how much per-instance effects disagree at each x.

        ```python
        h = pdp.eval_heter("hr", xs)          # (T,) variance around the mean
        band = np.sqrt(h)                     # std-like band, plot-ready
        ```

        !!! note "It's a variance, and it's method-specific (R2)"
            PDP: variance of centered ICE; DerPDP: of d-ICE slopes; ALE/RHALE:
            per-bin slope variance as a step function; ShapDP: interpolated
            per-bin φ variance. Take the square root for a band.

        !!! note "No `centering` argument — by design"
            Heterogeneity is invariant to centering; the signature enforces it.

        Args:
            feature: index or name of the feature of interest.
            xs: where to evaluate, `(T,)`.
            mask: optional boolean `(N,)` subregion — re-summarized from cached
                local effects, zero model calls.
            rule: sugar over `mask` (an `effector.Rule` or a rule string);
                mutually exclusive with `mask`.

        Returns:
            the heterogeneity curve h(xs), `(T,)`, non-negative.
        """
        feature = self._resolve_feature(feature)
        mask = self._resolve_mask(mask, rule)
        params = self._summary(feature, mask)
        return self._eval_payload(feature, params, xs, heterogeneity=True)[1]

    def payload(self, feature: Union[int, str]) -> dict:
        """The raw fitted object behind `eval`/`eval_heter` — pure numpy, yours to inspect.

        ```python
        p = ale.payload("hr")     # e.g. {"limits": ..., "bin_effect": ..., "bin_variance": ...}
        ```

        Per method: per-bin effects and variances for (RH)ALE and ShapDP, the
        grid summaries for (d-)PDP. A copy — mutate freely.
        """
        return dict(self._summary(self._resolve_feature(feature), None))

    def heter_score(
        self,
        feature: Union[int, str],
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> float:
        """One number for a feature's heterogeneity — the scalar `find_regions` minimizes.

        ```python
        pdp.heter_score("hr")                            # global
        pdp.heter_score("hr", rule="workingday == 0")    # within a subregion
        ```

        In **output units** (units contract, method_semantics.md): the RMS of
        `eval_heter` over the feature's own (masked) data values
        (frequency-weighted over levels for categorical features), bridged by
        the feature's dispersion for the derivative-based methods
        (ALE/RHALE/DerPDP) so every feature type and every method lands on the
        same y-unit scale — "a typical instance's effect deviates from the
        mean effect by about this much". `eval_heter` itself stays a variance
        curve in the method's native units.

        !!! tip "Pair it with `importance`"
            `importance` measures the *mean* effect's strength; `heter_score`
            measures the spread around it — same units, mean/spread twins.
            High importance + high heterogeneity = the top-right corner of
            `effector.plot_triage` — where `find_regions` should look.

        Args:
            feature: index or name of the feature of interest.
            mask: optional boolean `(N,)` subregion — model-free.
            rule: sugar over `mask`; mutually exclusive with it.

        Returns:
            a non-negative scalar, in output units.
        """
        feature = self._resolve_feature(feature)
        self._check_feature_type_supported(feature)
        mask = self._resolve_mask(mask, rule)
        return float(self._heter(feature, mask))

    def _heter(self, feature: int, mask: Optional[np.ndarray]) -> float:
        """Default (PDP/ShapDP — native y-unit variances): the RMS of the
        heterogeneity curve over the data distribution — sqrt of the mean of
        `eval_heter` at the (masked) data values, frequency-weighted over
        levels for categorical features. Reads the summary payload directly
        (never `self.eval*`) so it stays model-free and store-safe (P1).
        Derivative-based methods override to multiply by the feature's frozen
        dispersion (the derivative→output unit bridge)."""
        params = self._summary(feature, mask)
        if self._is_cat(feature):
            # frequency-weighted over levels (method_semantics.md); with a mask
            # the levels/frequencies are those within the subregion
            levels, weights = self._level_weights(feature, mask)
            h = self._eval_payload(feature, params, levels, heterogeneity=True)[1]
            return float(np.sqrt(np.average(h, weights=weights)))
        xs = self.data[:, feature] if mask is None else self.data[mask, feature]
        h = self._eval_payload(feature, params, xs, heterogeneity=True)[1]
        return float(np.sqrt(np.mean(h)))

    def find_regions(
        self,
        feature: Union[int, str, None] = None,
        *,
        features: Union[list, str, None] = None,
        finder="best",
        candidate_conditioning_features="all",
    ):
        """Search for subregions that resolve a feature's heterogeneity.

        ```python
        part = pdp.find_regions("hr")                       # one feature -> Partition
        part.show()                                         # the tree + level stats
        pdp.plot("hr", rule=part.leaves[0].rule)            # drill into a leaf

        parts = pdp.find_regions(features="heterogeneous")  # several -> {name: Partition}
        effector.plot_triage(pdp, partitions=parts)         # the before/after picture
        ```

        !!! note "A query, not a mutation (R12)"
            The result is a value — nothing is stored on the effect. Don't
            like a partition? Search again with different finder kwargs;
            nothing needs resetting.

        !!! note "Model-free"
            Every candidate split is scored by `heter_score(feature, mask)`
            on the cached local effects — zero model calls, whatever the grid
            size. Binning/scope are those the feature was fitted with,
            replayed.

        Args:
            feature: index or name of the one feature to partition
                (→ `Partition`).
            features: several at once — a list, `"all"`, or `"heterogeneous"`
                (heter_score at/above the median, the same convention
                `effector.explain` uses) → `{feature_name: Partition}`.
                Exactly one of `feature`/`features` must be given.
            finder: `"best"` (default), `"best_level_wise"`, or a configured
                finder instance (e.g. `effector.space_partitioning.Best(...)`).
            candidate_conditioning_features: features allowed to define splits
                (`"all"` or a list of indices/names).

        Returns:
            a `Partition` bound to this effect — or `{feature_name: Partition}`
            with `features=`.
        """
        if (feature is None) == (features is None):
            raise ValueError(
                "find_regions takes exactly one of `feature` (singular -> "
                "Partition) or `features` (plural -> {name: Partition})"
            )
        if features is not None:
            return self._find_regions_plural(
                features,
                finder=finder,
                candidate_conditioning_features=candidate_conditioning_features,
            )

        from effector import space_partitioning  # lazy: one-way dep guard

        feature = self._resolve_feature(feature)
        if isinstance(candidate_conditioning_features, list):
            candidate_conditioning_features = [
                self._resolve_feature(f) for f in candidate_conditioning_features
            ]
        self._check_feature_type_supported(feature)
        self._ensure_local(feature)

        if isinstance(finder, str):
            finder = space_partitioning.return_default(finder)

        def score_fn(mask):
            return self.heter_score(feature, mask=mask)  # RAW; guard is the finder's

        partition = finder.find_regions(
            feature,
            self.data,
            score_fn,
            axis_limits=self.axis_limits,
            feature_types=self.feature_types,
            cat_limit=self.cat_limit,
            candidate_conditioning_features=candidate_conditioning_features,
            feature_names=self.feature_names,
            target_name=self.target_name,
        )
        return partition.bind(self)

    def _find_regions_plural(
        self,
        features: Union[list, str],
        *,
        finder,
        candidate_conditioning_features,
    ) -> dict:
        """The `features=` form of `find_regions`: one search per feature,
        keyed by feature name. `"all"`/`"heterogeneous"` iterate the supported
        features (one UserWarning for the skipped ones, mirroring
        `importances`); an explicit list is strict."""
        if isinstance(features, str):
            if features not in ("all", "heterogeneous"):
                raise ValueError(
                    f"Invalid features argument: {features!r}; use a list of "
                    "indices/names, 'all', or 'heterogeneous'"
                )
            supported, skipped = [], []
            for f in range(self.dim):
                try:
                    self._check_feature_type_supported(f)
                    supported.append(f)
                except ValueError:
                    skipped.append(self.feature_names[f])
            if skipped:
                warnings.warn(
                    f"find_regions skipped feature(s) {skipped} — this method "
                    f"does not support their feature type.",
                    UserWarning,
                    stacklevel=3,
                )
            if features == "heterogeneous":
                # the explain() threshold convention: at/above the median
                hs = {f: self.heter_score(f) for f in supported}
                thr = float(np.median(list(hs.values()))) if hs else 0.0
                supported = [f for f in supported if hs[f] >= thr]
            resolved = supported
        else:
            resolved = [self._resolve_feature(f) for f in features]

        return {
            self.feature_names[f]: self.find_regions(
                f,
                finder=finder,
                candidate_conditioning_features=candidate_conditioning_features,
            )
            for f in resolved
        }

    def importance(
        self,
        feature: Union[int, str],
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> float:
        """How much a feature's mean effect moves the prediction (R13).

        ```python
        pdp.importance("temp")                           # scalar
        pdp.importance("temp", rule="workingday == 1")   # within a subregion
        ```

        The dispersion of the **mean** effect in output units — the μ-twin of
        `heter_score` (which measures per-instance spread on the same scale).
        A flat curve scores ~0; a swinging curve scores high. Per method: std
        of the mean effect over the (masked) data values (PDP/ALE/RHALE; for
        a linear model this is `|coefficient| * std(x)`), `mean(|φ|)`
        (ShapDP), `mean(|derivative|) * std(x)` (DerPDP). Comparable across
        feature types and, in magnitude, across methods.

        !!! note "No `y`, ever"
            effector never sees ground-truth labels — this is a property of
            the fitted effect, not a loss/permutation importance.

        Args:
            feature: index or name of the feature of interest.
            mask: optional boolean `(N,)` subregion — model-free.
            rule: sugar over `mask`; mutually exclusive with it.

        Returns:
            a non-negative scalar.
        """
        feature = self._resolve_feature(feature)
        self._check_feature_type_supported(feature)
        mask = self._resolve_mask(mask, rule)
        if mask is None:
            # all-ones ≡ None (M1); the concrete array makes the per-method
            # `_importance` implementations mask-index without a null check
            mask = np.ones(self.data.shape[0], dtype=bool)
        self._ensure_local(feature)
        return float(self._importance(feature, mask))

    def _importance(self, feature: int, mask: np.ndarray) -> float:
        """Default (PDP/ALE/RHALE): the standard deviation of the mean effect
        over the data distribution — the μ-twin of `heter_score`, in output
        units. Continuous features evaluate at the (masked) data values of the
        feature (data-weighted std); discrete features weight by level
        frequency. Reads the summary payload directly (never `self.eval`) so
        it stays model-free and store-safe for every method (P1)."""
        params = self._summary(feature, mask)
        if self._is_cat(feature):
            levels, weights = self._level_weights(feature, mask)
            mu = self._eval_payload(feature, params, levels)
            mu_bar = float(np.average(mu, weights=weights))
            return float(np.sqrt(np.average((mu - mu_bar) ** 2, weights=weights)))
        xs = self.data[mask, feature]
        mu = self._eval_payload(feature, params, xs)
        return float(np.std(mu))

    def importances(
        self,
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> np.ndarray:
        """The whole importance vector — rank your features in one call.

        ```python
        imp = pdp.importances()                       # (D,)
        order = np.argsort(-np.nan_to_num(imp))       # most important first
        ```

        !!! warning "NaN means unsupported, not unimportant"
            Feature types this method cannot explain (e.g. DerPDP on a
            nominal feature) return `NaN`, with one `UserWarning` naming them.

        Args:
            mask: optional boolean `(N,)` subregion.
            rule: sugar over `mask`; mutually exclusive with it.

        Returns:
            the per-feature importance vector, `(D,)`.
        """
        mask = self._resolve_mask(mask, rule)
        out = np.full(self.dim, np.nan)
        skipped = []
        for f in range(self.dim):
            try:
                out[f] = self.importance(f, mask=mask)
            except ValueError:
                skipped.append(self.feature_names[f])
        if skipped:
            warnings.warn(
                f"importance is undefined for feature(s) {skipped} — this "
                f"method does not support their feature type; returned NaN.",
                UserWarning,
                stacklevel=2,
            )
        return out

    @abstractmethod
    def plot(
        self,
        feature: Union[int, str],
        heterogeneity: Union[bool, str] = False,
        centering: Union[bool, str] = False,
        **kwargs,
    ) -> None:
        """Draw the effect of one feature — a thin wrapper over the fitted state (R7).

        Args:
            feature: index or name of the feature to plot.
            heterogeneity: `False` (mean only), `True` (the method's default
                view, e.g. ICE for PDP), or a named view (`"std"`, `"ice"`, ...).
            centering: `False`, `True`/`"zero_integral"`, or `"zero_start"`.
            **kwargs: method-specific plot options — see each class's `plot`.
        """
        raise NotImplementedError

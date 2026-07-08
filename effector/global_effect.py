"""The global-effect engine: the two-block lifecycle (design contract R14).

Every effect object holds exactly two caches and one config, all owned here:

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

HINT_DERPDP = (
    " A derivative needs a continuous axis; use PDP instead — adjacent "
    "differences of the per-level PDP bars carry the same information."
)
HINT_RHALE_NOMINAL = (
    " No derivative exists for nominal features and grouping over an "
    "arbitrary order is not meaningful; use ALE or PDP instead."
)

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
    hints = {
        ("d-pdp", ingestion.ORDINAL): HINT_DERPDP,
        ("d-pdp", ingestion.NOMINAL): HINT_DERPDP,
        ("rhale", ingestion.NOMINAL): HINT_RHALE_NOMINAL,
    }
    hint = hints.get((method_name, ftype), "")
    raise ValueError(
        f"{method_name} does not support {ftype} features "
        f"(feature {feature} {feature_name!r} is {ftype})."
        f"{hint}"
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
        """Declare the method configuration for the given features and warm the
        caches. Nothing fit does is unavailable lazily: any `eval`/`plot` on an
        unfitted feature silently computes what it needs with the defaults.

        Args:
            features: the features to fit. If set to "all", all the features will be fitted.
            centering: the default centering mode this feature is queried with

                    - If `centering` is `False`, effects are not centered
                    - If `centering` is `True` or `zero_integral`, the effect is centered around the `y` axis.
                    - If `centering` is `zero_start`, the effect starts from zero.
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
        features = helpers.prep_features(features, self.dim)
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
        feature: int,
        xs: np.ndarray,
        centering: Union[None, bool, str] = None,
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> np.ndarray:
        """Evaluate the mean effect of the `feature`-th feature at positions `xs`.

        Notes:
            This is the one evaluation method of every effect class (R1): it
            always returns the mean effect as a single `(T,)` array.
            Heterogeneity lives on its own surface — `eval_heter(feature, xs)`
            for the curve, `heter_score(feature)` for the scalar, and
            `payload(feature)` for the method's raw object.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the effect at

              - `np.ndarray` of shape `(T, )`

            centering: whether to center the effect

                - `None` (default) uses the class default (`DEFAULT_CENTERING`)
                - `False`: no centering
                - `True` or `"zero_integral"`: center around the `y` axis
                - `"zero_start"`: the effect starts from `y=0`

            mask: optional boolean `(N,)` selecting a subregion. `None`
                (default) evaluates over all instances; a mask summarizes that
                subset of the cached local effects on the fly — the effect
                *within* the subregion, on the global frame, without model
                calls. Centering is then computed over the subregion's own
                interval. Nothing is stored.

            rule: sugar over `mask` — an `effector.Rule` (or a string like
                `"temp < 3 and season == 0"`, parsed with this effect's
                metadata) applied to the effect's data. Mutually exclusive
                with `mask`.

        Returns:
            the mean effect `y` at the given `xs`, `(T,)`
        """
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
        feature: int,
        xs: np.ndarray,
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> np.ndarray:
        """Evaluate the heterogeneity curve h(xs) of the `feature`-th feature.

        Notes:
            The values are *method-specific* (R2): the variance of the centered
            ICE curves (PDP), of the d-ICE curves (DerPDP), the per-bin variance
            of the local effects as a step function (ALE/RHALE), or the
            interpolated per-bin variance of the SHAP values (ShapDP). They are
            variances — take a square root for a std-like band.

            There is deliberately no `centering` argument: heterogeneity is
            invariant to centering.

        Args:
            feature: index of feature of interest
            xs: the points to evaluate the heterogeneity at, `(T,)`
            mask: optional boolean `(N,)` selecting a subregion. `None` (default)
                evaluates over all instances; a mask summarizes that subset of
                the cached local effects on the fly (the regional split search) —
                pure numpy, no model calls.
            rule: sugar over `mask` — an `effector.Rule` or a rule string,
                applied to the effect's data. Mutually exclusive with `mask`.

        Returns:
            the heterogeneity curve h(xs), `(T,)`, non-negative
        """
        mask = self._resolve_mask(mask, rule)
        params = self._summary(feature, mask)
        return self._eval_payload(feature, params, xs, heterogeneity=True)[1]

    def payload(self, feature: int) -> dict:
        """The method's raw fitted object for the `feature`-th feature — the
        honest method-specific state behind `eval`/`eval_heter` (per-bin
        effects and variances for (RH)ALE and ShapDP, the grid summaries for
        (d-)PDP): the all-ones summary (R14), pure numpy."""
        return dict(self._summary(feature, None))

    def heter_score(
        self,
        feature: int,
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> float:
        """The method-agnostic heterogeneity scalar of the `feature`-th
        feature: the mean of `eval_heter` over a uniform grid
        (`helpers.NOF_INTERNAL_POINTS` points) on the feature's interval — the
        single quantity regional splitting (and the future interaction module)
        consumes.

        With a `mask` (boolean `(N,)`), the score is computed over that
        subregion from the cached local effects — the entry point the regional
        split search calls for every candidate, model-free. `rule` is sugar
        over `mask` (an `effector.Rule` or a rule string; mutually
        exclusive)."""
        mask = self._resolve_mask(mask, rule)
        if self._is_cat(feature):
            # frequency-weighted over levels (method_semantics.md); with a mask
            # the levels/frequencies are those within the subregion
            levels, weights = self._level_weights(feature, mask)
            return float(
                np.average(self.eval_heter(feature, levels, mask), weights=weights)
            )
        xs = np.linspace(
            self.axis_limits[0, feature],
            self.axis_limits[1, feature],
            helpers.NOF_INTERNAL_POINTS,
        )
        return float(np.mean(self.eval_heter(feature, xs, mask)))

    def find_regions(
        self,
        feature: int,
        *,
        finder="best",
        candidate_conditioning_features="all",
    ):
        """Search for heterogeneity-reducing subregions of `feature` and return a
        `Partition` — a value (R12): nothing is stored on `self`.

        The search is model-free: every candidate's score is
        `heter_score(feature, mask)`, re-summarized from the cached local effects
        (and memoized). There are no method fit kwargs here — the binning/scope
        etc. are exactly those `feature` was fitted with, replayed.

        Args:
            feature: index of the feature to partition.
            finder: a region finder — either a name (`"best"` /
                `"best_level_wise"`) or any object implementing the finder
                protocol (`find_regions(feature, data, score_fn, ...) -> Partition`).
            candidate_conditioning_features: features allowed to define splits
                (`"all"` or a list of indices).

        Returns:
            a `Partition` bound to this effect (its `plot`/`eval` re-query `self`).
        """
        from effector import space_partitioning  # lazy: one-way dep guard

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

    def importance(
        self,
        feature: int,
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> float:
        """R13: how much the **mean effect** of `feature` varies over the
        (masked) data — the μ-twin of `heter_score` (which measures per-instance
        spread). Model-free (re-summarized from the cached local effects) and
        centering-invariant (the dispersion of the mean effect does not depend on
        the additive centering constant, so there is deliberately no `centering`
        argument). A `mask` restricts it to a subregion.

        Args:
            feature: index of the feature of interest.
            mask: optional boolean `(N,)` selecting a subregion.
            rule: sugar over `mask` — an `effector.Rule` or a rule string,
                applied to the effect's data. Mutually exclusive with `mask`.

        Returns:
            a non-negative scalar.
        """
        self._check_feature_type_supported(feature)
        mask = self._resolve_mask(mask, rule)
        if mask is None:
            # all-ones ≡ None (M1); the concrete array makes the per-method
            # `_importance` implementations mask-index without a null check
            mask = np.ones(self.data.shape[0], dtype=bool)
        self._ensure_local(feature)
        return float(self._importance(feature, mask))

    def _importance(self, feature: int, mask: np.ndarray) -> float:
        """Default (PDP/ALE/RHALE): the standard deviation of the mean effect —
        the μ-twin of `heter_score`, evaluated the same way it is. Continuous
        features use the uniform grid `heter_score` averages over; discrete
        features weight by level frequency. `centering=False` keeps it a pure
        query — the std is invariant to centering."""
        if self._is_cat(feature):
            levels, weights = self._level_weights(feature, mask)
            mu = self.eval(feature, levels, centering=False, mask=mask)
            mu_bar = float(np.average(mu, weights=weights))
            return float(np.sqrt(np.average((mu - mu_bar) ** 2, weights=weights)))
        xs = np.linspace(
            self.axis_limits[0, feature],
            self.axis_limits[1, feature],
            helpers.NOF_INTERNAL_POINTS,
        )
        mu = self.eval(feature, xs, centering=False, mask=mask)
        return float(np.std(mu))

    def importances(
        self,
        mask: Optional[np.ndarray] = None,
        rule: Union[None, str, "Rule"] = None,
    ) -> np.ndarray:
        """R13: the per-feature importance vector `(D,)`. Feature types this
        method cannot explain are `NaN`, with one `UserWarning` (R9) naming the
        skipped columns. `rule` is sugar over `mask` (mutually exclusive)."""
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
        feature: int,
        heterogeneity: Union[bool, str] = False,
        centering: Union[bool, str] = False,
        **kwargs,
    ) -> None:
        """

        Parameters
        ----------
        feature: index of the feature to plot
        heterogeneity: whether to plot the heterogeneity measures

            - If `heterogeneity=False`, the plot shows only the mean effect
            - If `heterogeneity=True`, the plot additionally shows the heterogeneity with the default visualization, e.g., ICE plots for PDPs
            - If `heterogeneity=<str>`, the plot shows the heterogeneity using the specified method

        centering: whether to center the PDP

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.
        **kwargs: all other plot-specific arguments
        """
        raise NotImplementedError

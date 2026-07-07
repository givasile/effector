import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Callable, Optional, Tuple, Union

import numpy as np

from effector import helpers, ingestion, utils

HINT_DERPDP = (
    " A derivative needs a continuous axis; use PDP instead — adjacent "
    "differences of the per-level PDP bars carry the same information."
)
HINT_RHALE_NOMINAL = (
    " No derivative exists for nominal features and grouping over an "
    "arbitrary order is not meaningful; use ALE or PDP instead."
)


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
    # the class-level centering default (R3): each subclass declares it once;
    # fit/eval/plot signatures converge on it during the homogenization
    DEFAULT_CENTERING: Union[bool, str] = False

    # capability contract per feature type (method_semantics.md): which of
    # continuous/ordinal/nominal the method supports, and the strategy code it
    # uses for discrete features (mirrored into the R5 registry)
    SUPPORTED_FEATURE_TYPES: frozenset = frozenset(
        {ingestion.CONTINUOUS, ingestion.ORDINAL, ingestion.NOMINAL}
    )
    CAT_STRATEGY: Optional[str] = None

    def __init__(
        self,
        method_name: str,
        data,
        model: Callable,
        model_jac: Optional[Callable] = None,
        *,
        data_effect: Optional[np.ndarray] = None,
        local_effects: Optional[dict] = None,
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
    ) -> None:
        """
        Constructor for the FeatureEffectBase class.
        """
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

        # state variable
        self.is_fitted: np.ndarray = np.ones([self.dim]) < 0

        # parameters used when fitting the feature effect
        self.fit_args: dict = {}

        # dict, like {"feature_i": {"quantity_1": value_1, "quantity_2": value_2, ...}} for the i-th
        self.feature_effect: dict = {}

        # step 2 output cache: {"feature_i": <per-instance local effect>} —
        # computed once (model-touching, `_compute_local_effects`) or injected
        # here at construction; steps 3-4 (summarize/eval/plot/heter) read it and
        # never re-touch the model (R: single-model-touch constitution)
        self.local_effects: dict = dict(local_effects) if local_effects else {}
        # feature keys whose local effects are cached, so a later model touch on
        # the cached path can be flagged by the (silent) diagnostic
        self._sealed: set = set()

        # Invisible performance memo (R12): masked summaries are pure functions
        # of (feature, fitted state, mask). Keyed by (feature, fit_epoch,
        # mask.tobytes()) so a refit — which bumps the epoch — invalidates it.
        # A cache is not API state; the partition it accelerates is the value.
        self._masked_cache: "OrderedDict" = OrderedDict()
        self._fit_epoch: dict = {}  # "feature_i" -> int, bumped on (re)fit/recompute
        self._MASKED_CACHE_MAX = 512

    @abstractmethod
    def fit(
        self,
        features: Union[int, str, list] = "all",
        centering: Union[bool, str] = False,
        **kwargs,
    ) -> None:
        """Fit, i.e., compute the quantities that are necessary for evaluating and plotting the feature effect, for the given features.

        Args:
            features: the features to fit. If set to "all", all the features will be fitted.
            centering: whether to center the feature effect plot

                    - If `centering` is `False`, the plot is not centered
                    - If `centering` is `True` or `zero_integral`, the plot is centered around the `y` axis.
                    - If `centering` is `zero_start`, the plot starts from zero.
        """
        raise NotImplementedError

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

    @abstractmethod
    def _fit_feature(self, feature: int, **kwargs) -> dict:
        """Compute and return the method-specific payload for one feature
        (everything `eval`/`plot` need, except the normalization constant)."""
        raise NotImplementedError

    @abstractmethod
    def _eval_unnorm(
        self,
        feature: int,
        x: np.ndarray,
        heterogeneity: bool = False,
        params: Optional[dict] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """The method-specific evaluation kernel: the *uncentered* mean effect
        at `x`, and — if `heterogeneity` — also the heterogeneity curve h(x) (a
        variance-like quantity in the method's own units, independent of any
        centering).

        `params` selects the payload to evaluate against: `None` reads the
        stored fitted state (`feature_effect["feature_i"]`); a passed dict (the
        output of `_summarize`) lets the masked heterogeneity path evaluate a
        *transient* subregion payload without disturbing the stored one."""
        raise NotImplementedError

    def _eval_masked_mean(
        self, feature: int, x: np.ndarray, params: dict, mask: np.ndarray
    ) -> np.ndarray:
        """The uncentered masked mean effect at `x` — default reads the
        transient payload (pure numpy). PDP overrides it: on the cached grid it
        reads the payload, off-grid it recomputes ICE on `data[mask]` (the
        exact-evaluation retouch, symmetric with the global PDP `eval`)."""
        return self._eval_unnorm(feature, x, heterogeneity=False, params=params)

    def _compute_local_effects(self, feature: int) -> None:
        """Step 2 (model-touching): compute the per-instance local effect for
        `feature` and store it in `self.local_effects["feature_i"]`. The single
        place the model is queried for the effect. Overridden per method."""
        raise NotImplementedError

    def _ensure_local_effects(self, feature: int) -> None:
        """Populate the local-effects cache for `feature` if absent — skipped
        when the effects were injected at construction. Seals the feature; a
        later recompute on a sealed feature (a parameter incompatible with the
        cache) is flagged by the silent diagnostic below."""
        key = "feature_" + str(feature)
        if key not in self.local_effects:
            if key in self._sealed:
                logging.getLogger("effector").debug(
                    "%s: recomputing local effects for feature %d after it was "
                    "sealed — a parameter is incompatible with the cache",
                    self.method_name,
                    feature,
                )
            self._compute_local_effects(feature)
            # local effects changed -> stale masked summaries must not be served
            self._fit_epoch[key] = self._fit_epoch.get(key, 0) + 1
        self._sealed.add(key)

    def _summarize(
        self, feature: int, mask: Optional[np.ndarray] = None, **fit_kwargs
    ) -> dict:
        """Step 3 (pure numpy): derive the effect payload for `feature` from the
        cached local effects restricted to `mask` (`None` = all instances),
        returning the same shape as the stored `feature_effect["feature_i"]`.
        Overridden per method."""
        raise NotImplementedError

    def _replay_fit_kwargs(self, feature: int) -> dict:
        """The method-specific fit kwargs recorded at the last `fit` (e.g.
        `binning_method`, `order`, `use_vectorized`), minus centering — what a
        transient `_summarize` must replay to match the fitted state."""
        prev = self.fit_args.get("feature_" + str(feature), {})
        return {
            k: v
            for k, v in prev.items()
            if k not in ("centering", "points_for_centering")
        }

    def _masked_params(self, feature: int, mask: np.ndarray) -> dict:
        """Bounded-LRU memo around the masked `_summarize` (with replayed fit
        kwargs). Semantically transparent: same (feature, fitted state, mask) ->
        same payload as calling `_summarize` directly, only faster on repeats
        (the split search re-proposes identical candidate masks; a plot after a
        search hits the exact node masks). The `_fit_epoch` term in the key means
        a refit invalidates stale entries. Callers must have ensured local
        effects first. Treat the return value as read-only."""
        key = (feature, self._fit_epoch.get(f"feature_{feature}", 0), mask.tobytes())
        cached = self._masked_cache.get(key)
        if cached is not None:
            self._masked_cache.move_to_end(key)
            return cached
        params = self._summarize(feature, mask, **self._replay_fit_kwargs(feature))
        self._masked_cache[key] = params
        if len(self._masked_cache) > self._MASKED_CACHE_MAX:
            self._masked_cache.popitem(last=False)
        return params

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

    def _fit_loop(
        self,
        features: Union[int, str, list],
        centering: Union[bool, str],
        points_for_centering: int = helpers.NOF_INTERNAL_POINTS,
        **fit_feature_kwargs,
    ) -> None:
        """The one fit skeleton every method shares (R1): normalize inputs,
        compute the per-feature payload, then the normalization constant."""
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            self._check_feature_type_supported(s)
            key = "feature_" + str(s)
            self.fit_args[key] = {
                "centering": centering,
                "points_for_centering": points_for_centering,
                **fit_feature_kwargs,
            }
            self.feature_effect[key] = self._fit_feature(s, **fit_feature_kwargs)
            self.feature_effect[key]["norm_const"] = (
                self._compute_norm_const(
                    s, method=centering, nof_points=points_for_centering
                )
                if centering is not False
                else None
            )
            self.is_fitted[s] = True
            # fitted state (fit_args/payload) changed -> invalidate masked memo
            self._fit_epoch[key] = self._fit_epoch.get(key, 0) + 1

    def _compute_norm_const(
        self,
        feature: int,
        method: str = "zero_integral",
        nof_points: int = helpers.NOF_INTERNAL_POINTS,
        params: Optional[dict] = None,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute the normalization constant from the evaluation kernel:
        `zero_integral` = the mean over the feature interval, `zero_start` =
        the value at its left limit.

        With `params`/`mask` (the masked path), the constant belongs to a
        *transient* subregion payload: the integral runs over the subregion's
        effective interval (its own `[min, max]`, not the global frame) and the
        level weights are those within the mask. Nothing is stored."""
        assert method in ["zero_integral", "zero_start"]

        def partial_eval(x):
            return self._eval_unnorm(feature, x, heterogeneity=False, params=params)

        if self._is_cat(feature):
            # discrete centering (method_semantics.md): zero_integral is the
            # frequency-weighted level mean (order-invariant); zero_start
            # zeroes the first level *in fit order* (a custom `order` makes
            # its first entry the reference level)
            levels, weights = self._level_weights(feature, mask)
            if method == "zero_integral":
                return float(np.average(partial_eval(levels), weights=weights))
            source = (
                params
                if params is not None
                else self.feature_effect.get("feature_" + str(feature), {})
            )
            fit_levels = source.get("levels", levels)
            return partial_eval(np.asarray(fit_levels[:1])).item()

        start, stop = self._effective_limits(feature, mask)

        if method == "zero_integral":
            return utils.mean_1d_linspace(partial_eval, start, stop, nof_points)
        return partial_eval(np.array([start])).item()

    def eval_heter(
        self, feature: int, xs: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Evaluate the heterogeneity curve h(xs) of the `feature`-th feature.

        Notes:
            The values are *method-specific* (R2): the variance of the centered
            ICE curves (PDP), of the d-ICE curves (DerPDP), the per-bin variance
            of the local effects as a step function (ALE/RHALE), or the residual
            spline around the SHAP curve (ShapDP). They are variances — take a
            square root for a std-like band.

            There is deliberately no `centering` argument: heterogeneity is
            invariant to centering.

        Args:
            feature: index of feature of interest
            xs: the points to evaluate the heterogeneity at, `(T,)`
            mask: optional boolean `(N,)` selecting a subregion. `None` (default)
                evaluates over the fitted state; a mask summarizes that subset of
                the cached local effects on the fly (the regional split search) —
                pure numpy, no model calls.

        Returns:
            the heterogeneity curve h(xs), `(T,)`, non-negative
        """
        mask = self._prep_mask(mask)
        if mask is None:
            if self.requires_refit(feature, centering=False):
                self._refit(feature)
            return self._eval_unnorm(feature, xs, heterogeneity=True)[1]
        self._ensure_local_effects(feature)
        params = self._masked_params(feature, mask)
        return self._eval_unnorm(feature, xs, heterogeneity=True, params=params)[1]

    def payload(self, feature: int) -> dict:
        """The method's raw fitted object for the `feature`-th feature — the
        honest method-specific state behind `eval`/`eval_heter` (bin effects
        and variances for (RH)ALE, splines and shap values for ShapDP, the
        normalization constants for PDP)."""
        if self.requires_refit(feature, centering=False):
            self._refit(feature)
        return dict(self.feature_effect["feature_" + str(feature)])

    def heter_score(self, feature: int, mask: Optional[np.ndarray] = None) -> float:
        """The method-agnostic heterogeneity scalar of the `feature`-th
        feature: the mean of `eval_heter` over a uniform grid
        (`helpers.NOF_INTERNAL_POINTS` points) on the feature's interval — the
        single quantity regional splitting (and the future interaction module)
        consumes.

        With a `mask` (boolean `(N,)`), the score is computed over that
        subregion from the cached local effects — the entry point the regional
        split search calls for every candidate, model-free."""
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
        if self.requires_refit(feature, centering=False):
            self._refit(feature)
        self._ensure_local_effects(feature)

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
        return partition._bind(self)

    def requires_refit(self, feature, centering):
        """Check if refitting is needed."""
        feature_key = f"feature_{feature}"

        # if the state variable is not set, refit
        if not self.is_fitted[feature]:
            return True

        # if the feature info does not exist, refit
        if self.feature_effect.get(feature_key) is None:
            return True

        # if the above are ok and centering is False, no need to refit
        if not centering:
            return False

        # if centering is not None and the norm_const is not set, refit
        norm_const = self.feature_effect.get(feature_key, {}).get("norm_const")
        if norm_const is None:
            return True

        # if centering is not None and is different from the centering when fitting, refit
        if self.fit_args.get(feature_key, {}).get("centering") != centering:
            return True

        return False

    def _mean_norm_const(self, norm_const):
        """The scalar amount the centered mean effect subtracts. It is a scalar
        for most methods (ALE/RHALE/ShapDP), so the stored value is returned as
        is; PDP overrides this because its norm_const is a per-instance array."""
        return norm_const

    def _refit(self, feature: int, centering=None) -> None:
        """Auto-refit for `feature`, replaying the kwargs of the user's last
        explicit `fit` and overriding **only** `centering`. This keeps a
        method-specific fit config (`order`, `binning_method`, …) intact when a
        later `eval`/`plot` forces a refit because centering changed; without
        it the refit would silently fall back to the method defaults.

        Falls back to a plain default fit when the feature was never fitted
        (nothing recorded to replay)."""
        prev = self.fit_args.get("feature_" + str(feature))
        if prev is None:
            if centering is None:
                self.fit(features=feature)
            else:
                self.fit(features=feature, centering=centering)
            return
        replay = {k: v for k, v in prev.items() if k != "centering"}
        eff_centering = prev["centering"] if centering is None else centering
        self._fit_loop(feature, eff_centering, **replay)

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        centering: Union[None, bool, str] = None,
        mask: Optional[np.ndarray] = None,
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
                (default) evaluates the fitted state; a mask summarizes that
                subset of the cached local effects on the fly — the effect
                *within* the subregion, on the global frame, without model
                calls. Centering is then computed over the subregion's own
                interval. Nothing is stored.

        Returns:
            the mean effect `y` at the given `xs`, `(T,)`
        """
        centering = self.DEFAULT_CENTERING if centering is None else centering
        centering = helpers.prep_centering(centering)
        mask = self._prep_mask(mask)

        if mask is not None:
            if not self._is_cat(feature):
                self._effective_limits(feature, mask)  # degeneracy guard
            self._ensure_local_effects(feature)
            params = self._masked_params(feature, mask)
            y = self._eval_masked_mean(feature, xs, params, mask)
            if centering is not False:
                norm_const = self._compute_norm_const(
                    feature, method=centering, params=params, mask=mask
                )
                y = y - self._mean_norm_const(norm_const)
            return y

        if self.requires_refit(feature, centering):
            self._refit(feature, centering)

        if not self.axis_limits[0, feature] < self.axis_limits[1, feature]:
            raise ValueError(
                f"Feature {feature} has a degenerate axis interval "
                f"[{self.axis_limits[0, feature]}, {self.axis_limits[1, feature]}]"
            )

        y = self._eval_unnorm(feature, xs)
        if centering is not False:
            norm_const = self.feature_effect["feature_" + str(feature)]["norm_const"]
            y = y - self._mean_norm_const(norm_const)
        return y

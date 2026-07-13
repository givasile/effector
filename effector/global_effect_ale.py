import typing
from typing import List, Optional, Union

import numpy as np

import effector.axis_partitioning as ap
import effector.helpers as helpers
import effector.utils as utils
import effector.visualization as vis
from effector import ingestion
from effector.global_effect import GlobalEffectBase, check_binning_scope


class ALEBase(GlobalEffectBase):
    DEFAULT_CENTERING: Union[bool, str] = "zero_integral"

    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        *,
        data_effect: typing.Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
        method_name: str = "ALE",
    ):
        super(ALEBase, self).__init__(
            method_name,
            data,
            model,
            model_jac,
            data_effect=data_effect,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
        )
        # cache (a′): all-pairs level differences for nominal features — the
        # order-free raw material of the nominal scalars. Model-touching like
        # cache (a), but its frame is the ascending level set (immutable per
        # object), so an `order=` refit never invalidates it and it never
        # bumps the epoch (chain summaries are unaffected).
        self._local_pairs: dict = {}

    def _cat_frame(self, feature: int) -> tuple:
        """The categorical frame (R14): the *declared* level order — the
        resolution ("similarity" seriation, ascending default) is deterministic
        given data + random_state, so comparing declarations equals comparing
        resolutions without paying the resolution on every access."""
        order = self._config(feature).get("order")
        if order is None:
            return ("order", "asc")
        if isinstance(order, str):
            return ("order", order)
        return ("order", tuple(float(v) for v in np.asarray(order, dtype=float)))

    def _eval_payload_cont(
        self, feature: int, params: dict, x: np.ndarray, heterogeneity: bool = False
    ):
        y = utils.compute_accumulated_effect(
            x, limits=params["limits"], bin_effect=params["bin_effect"], dx=params["dx"]
        )
        if heterogeneity:
            var = utils.apply_bin_value(
                x=x, bin_limits=params["limits"], bin_value=params["bin_variance"]
            )
            return y, var
        return y

    def _eval_payload_cat(
        self, feature: int, params: dict, x: np.ndarray, heterogeneity: bool = False
    ):
        # discrete kernel (method_semantics.md): accumulate in code space —
        # exact at levels, and h(v_j) is the variance of the step *into*
        # level j (h(v_1) = the first transition's variance)
        codes = utils.codes_from_levels(
            x, params["levels"], feature, self.feature_names[feature]
        )
        y = utils.compute_accumulated_effect(
            codes.astype(float),
            limits=params["limits"],
            bin_effect=params["bin_effect"],
            dx=params["dx"],
        )
        if heterogeneity:
            transition_pos = np.maximum(codes, 1) - 0.5
            var = utils.apply_bin_value(
                x=transition_pos,
                bin_limits=params["limits"],
                bin_value=params["bin_variance"],
            )
            return y, var
        return y

    def _heter(self, feature: int, mask=None) -> float:
        """(RH)ALE heterogeneity in output units: the local effects are
        slopes (secants/derivatives per unit x — continuous — or per unit
        code gap — ordinal), so the base RMS is bridged into output units by
        the feature's dispersion: slope disagreement × typical excursion =
        output-level disagreement. The bridge reads the FULL column (codes in
        fit order for ordinal) — frozen under masks, like the frame, so
        regional heterogeneity drops measure dispersion reduction, not range
        shrinkage. Nominal features are order-free instead: all-pairs raw
        differences, already in output units (no bridge)."""
        if self.feature_types[feature] == ingestion.NOMINAL:
            return self._heter_nominal_pairs(feature, mask)
        base = super()._heter(feature, mask)
        if not self._is_cat(feature):
            return base * float(np.std(self.data[:, feature]))
        levels = self._summary(feature, mask)["levels"]
        codes = utils.codes_from_levels(
            self.data[:, feature], levels, feature, self.feature_names[feature]
        )
        return base * float(np.std(codes))

    def _importance(self, feature: int, mask) -> float:
        """R13: nominal features go through the all-pairs means (the chain's
        accumulated values are path-dependent under an arbitrary order); every
        other type uses the shared data-weighted std of the mean effect."""
        if self.feature_types[feature] == ingestion.NOMINAL:
            return self._importance_nominal_pairs(feature, mask)
        return super()._importance(feature, mask)

    # ------------------------------------------------------------------
    # nominal scalars: all-pairs level differences (order-free)
    # ------------------------------------------------------------------
    def _ensure_local_pairs(self, feature: int) -> dict:
        """Gate to cache (a′): computed at most once per feature — the frame
        (the ascending level set) cannot change within an object's life."""
        entry = self._local_pairs.get(feature)
        if entry is None:
            levels = self._levels(feature)
            if len(levels) < 2:
                raise ValueError(
                    f"feature {feature} {self.feature_names[feature]!r} has a "
                    f"single level — no effect to compute"
                )
            pair_lo, pair_hi, effects, instance_idx = (
                utils.compute_local_effects_level_pairs(
                    self.data, self.model, levels, feature
                )
            )
            entry = {
                "levels": levels,
                "pair_lo": pair_lo,
                "pair_hi": pair_hi,
                "effects": effects,
                "instance_idx": instance_idx,
            }
            self._local_pairs[feature] = entry
        return entry

    def _pair_stats(self, feature: int, mask=None):
        """Masked per-pair mean/variance of the raw level differences — pure
        numpy over cache (a′), memoized like any summary (the `"pairs"` key
        suffix disambiguates; an epoch bump only re-runs the numpy)."""
        prim = self._ensure_local_pairs(feature)

        def build():
            K = len(prim["levels"])
            lo, hi, eff = prim["pair_lo"], prim["pair_hi"], prim["effects"]
            if mask is not None:
                keep = mask[prim["instance_idx"]]
                lo, hi, eff = lo[keep], hi[keep], eff[keep]
            mu = np.full((K, K), np.nan)
            var = np.full((K, K), np.nan)
            for a in range(K):
                for b in range(a + 1, K):
                    d = eff[(lo == a) & (hi == b)]
                    if len(d):
                        mu[a, b] = d.mean()
                        var[a, b] = d.var()
            return {"mu": mu, "var": var}

        key = (feature, self._epoch.get(feature, 0), self._mask_key(mask), "pairs")
        return self._memo(key, build)

    def _pair_weights(self, feature: int, mask=None) -> np.ndarray:
        """Level frequencies over the FULL ascending level set — zeros for
        levels absent under the mask (any pair with a zero-weight side drops
        out of the weighted sums, which also guards its NaN stats)."""
        levels = self._ensure_local_pairs(feature)["levels"]
        col = self.data[:, feature] if mask is None else self.data[mask, feature]
        counts = np.array([np.isclose(col, lv).sum() for lv in levels], dtype=float)
        total = counts.sum()
        if total == 0:
            raise ValueError(
                f"feature {feature}: no instances at any level within the mask"
            )
        return counts / total

    def _heter_nominal_pairs(self, feature: int, mask=None) -> float:
        """H² = ½ Σ_k Σ_a w_k w_a Var_i[d_{a→k}] — the ½ makes the all-pairs
        dispersion equal the level-value variance (E[(X−X′)²] = 2 Var), so a
        nominal H lands exactly where PDP's per-level H lands on the canaries.
        Output units, no order anywhere."""
        stats = self._pair_stats(feature, mask)
        w = self._pair_weights(feature, mask)
        acc = 0.0
        K = len(w)
        for a in range(K):
            for b in range(a + 1, K):
                wprod = w[a] * w[b]
                if wprod > 0:
                    acc += wprod * stats["var"][a, b]
        return float(np.sqrt(acc))

    def _importance_nominal_pairs(self, feature: int, mask=None) -> float:
        """I² = ½ Σ_k Σ_a w_k w_a μ_{ak}² — the μ-twin of the all-pairs H:
        the dispersion of the mean level effects, order-free."""
        stats = self._pair_stats(feature, mask)
        w = self._pair_weights(feature, mask)
        acc = 0.0
        K = len(w)
        for a in range(K):
            for b in range(a + 1, K):
                wprod = w[a] * w[b]
                if wprod > 0:
                    acc += wprod * stats["mu"][a, b] ** 2
        return float(np.sqrt(acc))

    def _fit_loop(self, features, centering, **fit_feature_kwargs) -> None:
        """(RH)ALE fit also warms cache (a′) for nominal features, keeping the
        model-free-after-fit contract for the scalars."""
        super()._fit_loop(features, centering, **fit_feature_kwargs)
        feats = helpers.prep_features(features, self.dim, self.feature_names)
        for s in feats:
            if self.feature_types[s] == ingestion.NOMINAL:
                self._ensure_local_pairs(s)

    def _validate_order_arg(self, features, order):
        if order is None or isinstance(order, str):
            return
        feats = helpers.prep_features(features, self.dim, self.feature_names)
        cats = [f for f in feats if self._is_cat(f)]
        if len(feats) != 1 or len(cats) != 1:
            raise ValueError(
                "an explicit `order` list applies to exactly one categorical "
                "feature — call fit per feature"
            )

    def _compute_local_cat(self, feature: int, frame: tuple) -> dict:
        """Cache-(a) kernel (categorical): two-sided adjacent-level differences
        in code space — the discrete derivative (method_semantics.md). The
        levels (and their order) are frozen from the full data, so a subregion
        re-bins the same per-instance differences without re-querying the
        model."""
        levels = self._levels(feature)
        if len(levels) < 2:
            raise ValueError(
                f"feature {feature} {self.feature_names[feature]!r} has a "
                f"single level — no effect to compute"
            )
        order = self._config(feature).get("order")
        if order is not None:
            levels = self._resolve_level_order(feature, levels, order)
        positions, effects, instance_idx = utils.compute_local_effects_categorical(
            self.data, self.model, levels, feature
        )
        return {
            "frame": frame,
            "positions": positions,
            "effects": effects,
            "instance_idx": instance_idx,
            "levels": levels,
        }

    def _summarize_levels(
        self, feature: int, mask=None, binning_method=None
    ) -> typing.Dict:
        """The shared categorical summary body: bin the cached adjacent-level
        differences over the subregion `mask` (None = all). ALE keeps one bin
        per transition (`binning_method=None`); RHALE merges adjacent
        transitions with Greedy/DP (adaptive grouping)."""
        prim = self._local[feature]
        positions = prim["positions"]
        effects = prim["effects"]
        levels = prim["levels"]
        if mask is not None:
            keep = mask[prim["instance_idx"]]
            positions = positions[keep]
            effects = effects[keep]

        if binning_method is None:
            limits = np.arange(len(levels), dtype=float)
        else:
            binning = ap.adapt_for_categorical(
                ap.return_default(binning_method), len(levels)
            )
            limits = binning.find_limits(
                positions, effects, np.array([0.0, len(levels) - 1.0])
            )
            utils.raise_if_no_binning(limits, feature, binning)

        params = utils.compute_ale_params(positions, effects, limits)
        params["alg_params"] = "categorical"
        params["levels"] = levels
        return params

    def plot(
        self,
        feature: Union[int, str],
        heterogeneity: Union[bool, str] = True,
        centering: Union[bool, str] = True,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        show_avg_output: bool = False,
        y_limits: Optional[List] = None,
        dy_limits: Optional[List] = None,
        show_only_aggregated: bool = False,
        show_plot: bool = True,
        mask: Optional[np.ndarray] = None,
        rule=None,
        feature_label: Optional[str] = None,
    ):
        """Plot the (RH)ALE effect of `feature`.

        ```python
        ale.plot("hr")                          # curve + heterogeneity
        ale.plot("hr", rule="workingday == 0")  # within a subregion
        ```

        For a continuous feature the figure has two panels: the accumulated
        curve on top, the per-bin average local effect (± std) below.
        Categorical features get one bar per level with std whiskers.

        Args:
            feature: index or name of the feature to plot.
            heterogeneity: `False` for the mean effect only; `True` or
                `"std"` (default) adds the per-bin std of the local effects.
            centering: `False` (none), `True`/`"zero_integral"` (center
                around the y axis), or `"zero_start"` (start at `y=0`).
            scale_x: `None` or `{"mean": m, "std": s}` to undo a
                standardization of the x axis for display.
            scale_y: same, for the y axis.
            show_avg_output: draw the model's average output as a
                horizontal line.
            y_limits: `(low, high)` for the y axis; `None` = automatic.
            dy_limits: `(low, high)` for the bottom (local-effect) panel;
                `None` = automatic.
            show_only_aggregated: draw only the accumulated curve, without
                the bottom panel.
            show_plot: if `False`, return the figure and axes instead of
                showing.
            mask: boolean `(N,)` selecting a subregion — plot the effect
                *within* it (re-binned from the cached local effects, no
                model calls), x axis windowed to the subregion's own
                interval.
            rule: sugar over `mask` — an `effector.Rule` or a rule string,
                applied to the effect's data. Mutually exclusive with `mask`.
            feature_label: display title for the figure (e.g. a regional
                node's label with its rule); defaults to the feature name.
                The x-axis always keeps the plain feature name.
        """
        feature = self._resolve_feature(feature)
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        centering = helpers.prep_centering(centering)
        scale_x = helpers.resolve_scale(
            scale_x, self.scale_x_list[feature] if self.scale_x_list else None
        )
        scale_y = helpers.resolve_scale(scale_y, self.scale_y)
        mask = self._resolve_mask(mask, rule)
        feature_names = self.feature_names
        # C2: title = feature (or leaf label with its rule); method · scope
        # context moves to the corner tag
        plot_title = (
            feature_label if feature_label is not None else feature_names[feature]
        )
        tag = (
            f"{'ALE' if self.method_name == 'ale' else 'RHALE'}"
            f" · {'regional' if mask is not None else 'global'}"
        )

        # one path for global and masked alike (R14): pick the payload, read it
        is_cat = self._is_cat(feature)
        params = self._summary(feature, mask)
        x_window = (
            self._effective_limits(feature, mask)
            if mask is not None and not is_cat
            else None
        )

        def centered_eval(xs):
            y = self._eval_payload(feature, params, xs)
            if centering is not False:
                y = y - self._centering_const(feature, mask, centering)
            return y

        # the accumulated curve is piecewise linear between bin limits, so
        # evaluating exactly at the limits draws it exactly (no resampling).
        # categoricals are drawn by the is_cat branch below (at their observed
        # level values); their limits are positional codes 0..K-1 that eval
        # would reject, so only build this grid for continuous features.
        if not is_cat:
            x = np.asarray(params["limits"], dtype=float)
            y = centered_eval(x)

        avg_output = self._avg_output(mask, scale_y) if show_avg_output else None

        if is_cat:
            # bars = accumulated per-level values (in fit order); whiskers =
            # the variance of the step into each level (method_semantics.md)
            levels, labels = self._level_display(feature, params["levels"])
            y_levels = centered_eval(levels)
            variances = (
                self._eval_payload(feature, params, levels, heterogeneity=True)[1]
                if heterogeneity is not False
                else None
            )
            level_kind = self.feature_types[feature]
            level_counts = self._level_counts_for(feature, mask, levels)
            positions = np.asarray(levels, dtype=float)
            plot_scale_x, sort = scale_x, None
            if level_kind == "ordinal" and np.any(np.diff(positions) < 0):
                # custom (declared/induced) order: draw by rank in fit order —
                # ranks are display geometry, so the feature scale must not
                # touch them — and label by level
                if labels is None:
                    labels = [f"{v:g}" for v in positions]
                positions = np.arange(len(positions), dtype=float)
                plot_scale_x, sort = None, False
            return vis.plot_categorical_effect(
                positions,
                y_levels,
                variances,
                feature,
                heterogeneity,
                title=plot_title,
                level_labels=labels,
                scale_x=plot_scale_x,
                scale_y=scale_y,
                avg_output=avg_output,
                feature_names=feature_names,
                target_name=self.target_name,
                y_limits=y_limits,
                # the accumulation path is meaningful only along an ordered
                # axis; sorted nominal bars would fake an interpolation
                connect_line=level_kind == "ordinal",
                show_plot=show_plot,
                tag=tag,
                level_kind=level_kind,
                sort=sort,
                level_counts=level_counts,
            )
        return vis.ale_plot(
            x,
            y,
            bin_effect=params["bin_effect"],
            bin_variance=params["bin_variance"],
            limits=params["limits"],
            dx=params["dx"],
            feature=feature,
            heterogeneity=heterogeneity,
            scale_x=scale_x,
            scale_y=scale_y,
            title=plot_title,
            avg_output=avg_output,
            feature_names=feature_names,
            target_name=self.target_name,
            y_limits=y_limits,
            dy_limits=dy_limits,
            show_only_aggregated=show_only_aggregated,
            show_plot=show_plot,
            x_limits=x_window,
            tag=tag,
        )


class ALE(ALEBase):
    r"""Accumulated Local Effects: the effect built bin-by-bin from local
    model differences — the safe choice for correlated features.

    ```python
    ale = effector.ALE(X, model)
    ale.plot("hr")
    ```

    The axis is split into $K$ fixed bins with limits $z_0 < \dots < z_K$.
    Each instance in bin $k$ contributes the secant of the model across the
    bin; the per-bin means $\mu_k$ are accumulated:

    $$
    \hat{f}^{ALE}(x) = \sum_{k=1}^{k_x - 1} (z_k - z_{k-1})\, \mu_k
                       + (x - z_{k_x - 1})\, \mu_{k_x},
    \qquad
    \mu_k = \frac{1}{|S_k|} \sum_{i \in S_k}
    \frac{f(x^i_{s=z_k}) - f(x^i_{s=z_{k-1}})}{z_k - z_{k-1}}
    $$

    Instances only move within their own bin, so ALE stays close to the data
    manifold where PDP would extrapolate. The heterogeneity at $x$ is the
    variance of the local effects within its bin.

    !!! tip "Differentiable model? Use RHALE"
        `effector.RHALE` reads the local effects off the model Jacobian:
        no dependence on bin width, and automatic bin sizing.
    """

    CAT_STRATEGY = "adjacent_level_diffs"

    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        *,
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
    ):
        r"""Build an ALE explainer. No model calls happen here.

        ??? note "Heterogeneity"
            `eval_heter` returns a step function: the variance of the local
            effects within the bin containing $x$,

            $$
            h(x) = \sigma^2_{k_x},
            \qquad
            \sigma^2_k = \frac{1}{|S_k|} \sum_{i \in S_k}
            (\mathtt{effect}_i - \mu_k)^2
            $$

            The bin plot draws $\sqrt{\sigma^2_k}$ as error bars.

        Args:
            data: the design matrix, shape `(N, D)` — numpy only.
            model: the black-box model — a `Callable` mapping `(N, D)`
                arrays to `(N,)` predictions.
            nof_instances: max instances kept (default `10_000`) — an `int`
                subsamples randomly, `"all"` keeps everything.
            axis_limits: per-feature plot limits, shape `(2, D)`; `None`
                (default) infers them from `data`.
            schema: input metadata — an `effector.Schema` or a plain `dict`
                with any of `feature_names`, `feature_types`, `cat_limit`,
                `target_name`, `scale_x_list`, `scale_y`; omitted fields are
                inferred from `data`, explicit ones win. Coming from a
                DataFrame? Use `effector.from_dataframe`.
            random_state: seed for every internal random step (default `21`,
                reproducible); `None` for non-deterministic behavior.
        """
        super(ALE, self).__init__(
            data,
            model,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
            method_name="ALE",
        )

    @staticmethod
    def _check_binning_is_fixed(binning_method) -> None:
        if not (binning_method == "fixed" or isinstance(binning_method, ap.Fixed)):
            raise ValueError(
                f"Invalid binning_method: {binning_method!r}; ALE works only with "
                "the fixed binning method ('fixed' or an ap.Fixed instance)"
            )

    def _frame_from_config(self, feature: int) -> tuple:
        """ALE's local effect is *edge-bound* (the secant depends on the bin
        limits), so the fixed grid IS the frame: changing it invalidates the
        cached secants."""
        if self._is_cat(feature):
            return self._cat_frame(feature)
        binning_method = self._config(feature).get("binning_method", "fixed")
        binning = ap.Fixed() if isinstance(binning_method, str) else binning_method
        return (
            "fixed",
            binning.params["nof_bins"],
            binning.constraints.min_points_per_bin,
        )

    def _compute_local_cont(self, feature: int, frame: tuple) -> dict:
        """Cache-(a) kernel: the secant of the model across each fixed bin —
        computed on the frame's grid, one model query pair for all instances."""
        binning_method = self._config(feature).get("binning_method", "fixed")
        binning = ap.Fixed() if isinstance(binning_method, str) else binning_method
        limits = binning.find_limits(
            self.data[:, feature], None, self.axis_limits[:, feature]
        )
        utils.raise_if_no_binning(limits, feature, binning)
        secants = utils.compute_local_effects(self.data, self.model, limits, feature)
        return {"frame": frame, "effects": secants, "limits": limits}

    def _summarize_cont(
        self, feature: int, mask=None, binning_method="fixed", order=None
    ) -> typing.Dict:
        """Summary kernel (pure numpy): re-bin the cached secants over the
        subregion `mask` (None = all) on the frozen frame bins → bin
        effects/variances."""
        prim = self._local[feature]
        secants, limits = prim["effects"], prim["limits"]
        col = self.data[:, feature]
        if mask is not None:
            secants = secants[mask]
            col = col[mask]
        dale_params = utils.compute_ale_params(col, secants, limits)
        dale_params["alg_params"] = "fixed"
        return dale_params

    def _summarize_cat(
        self, feature: int, mask=None, binning_method="fixed", order=None
    ) -> typing.Dict:
        # one bin per transition, hard-coded `None` — the config's "fixed"
        # binning is a continuous-axis knob and must not group levels
        return self._summarize_levels(feature, mask, None)

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        *,
        centering: typing.Union[bool, str] = True,
        binning_method: typing.Union[str, ap.Fixed] = "fixed",
        order: typing.Union[None, str, list] = None,
    ) -> None:
        """Declare per-feature defaults and warm the caches.

        ```python
        ale.fit("hr", binning_method=Fixed(nof_bins=30))
        ```

        !!! note "fit is optional"
            `eval`, `plot`, `heter_score` compute what they need lazily with
            these defaults; `fit` declares the config once and pays the model
            cost upfront.

        Args:
            features: feature(s) to fit — index, name, list, or `"all"`.
            centering: default centering for this feature's queries —
                `False` (none), `True`/`"zero_integral"` (center around the
                y axis), or `"zero_start"` (start at `y=0`).
            binning_method: `"fixed"` (default: 20 equal-width bins) or an
                `effector.axis_partitioning.Fixed` instance for custom
                parameters, e.g. `Fixed(nof_bins=30, min_points_per_bin=0)`.
                ALE accepts only fixed binning — for adaptive bins use
                `effector.RHALE`.
            order: level order for a *categorical* feature of interest:

                - `None` (default): ascending encoded order — exact for
                  ordinal features; for nominal ones it is arbitrary-but-
                  deterministic, and only the adjacent-level differences are
                  meaningful (see docs/method_semantics.md)
                - `"similarity"`: induce the order from the other features
                  (KS-distance seriation, Molnar/iml)
                - a list of the levels: declare it explicitly (applies to
                  exactly one categorical feature)

                Changing `order` invalidates the cached local effects: the
                next query recomputes them; re-fitting the same `order` is a
                cache hit.

        Raises:
            ValueError: if `binning_method` is not fixed, or an explicit
                `order` list targets more than one categorical feature.
        """
        self._check_binning_is_fixed(binning_method)
        self._validate_order_arg(features, order)

        self._fit_loop(
            features,
            centering,
            binning_method=binning_method,
            order=order,
        )


class RHALE(ALEBase):
    r"""Robust and Heterogeneity-aware ALE: ALE computed from the model
    Jacobian, with automatic variable-size binning.

    ```python
    rhale = effector.RHALE(X, model, model_jac)
    rhale.plot("hr")
    ```

    The local effect of an instance is the pointwise derivative instead of
    ALE's bin secant, so it does not depend on the bin width — bins can be
    sized automatically (dynamic programming by default) to balance bias and
    variance. Accumulation and heterogeneity are then identical to ALE:

    $$
    \hat{f}^{RHALE}(x) = \sum_{k=1}^{k_x - 1} (z_k - z_{k-1})\, \mu_k
                         + (x - z_{k_x - 1})\, \mu_{k_x},
    \qquad
    \mu_k = \frac{1}{|S_k|} \sum_{i \in S_k}
    \frac{\partial f}{\partial x_s}(x^i)
    $$

    !!! warning "Needs derivatives"
        Pass `model_jac` (or a precomputed `data_effect`); otherwise the
        Jacobian is estimated with slower, less exact numerical
        differentiation. The Jacobian only serves the *continuous* features:
        ordinal ones use discrete differences with adaptive level grouping,
        and nominal ones fall back to ALE exactly (one bin per transition,
        no grouping — "adjacent" is not real under an arbitrary order).
    """

    CAT_STRATEGY = "level_diffs_grouped"

    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        *,
        data_effect: typing.Optional[np.ndarray] = None,
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Optional[np.ndarray] = None,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: typing.Optional[int] = 21,
    ):
        r"""Build a RHALE explainer. No model calls happen here.

        ??? note "Heterogeneity"
            `eval_heter` returns a step function: the variance of the
            per-instance derivatives within the bin containing $x$,

            $$
            h(x) = \sigma^2_{k_x},
            \qquad
            \sigma^2_k = \frac{1}{|S_k|} \sum_{i \in S_k}
            (\mathtt{effect}_i - \mu_k)^2
            $$

            The bin plot draws $\sqrt{\sigma^2_k}$ as error bars.

        Args:
            data: the design matrix, shape `(N, D)` — numpy only.
            model: the black-box model — a `Callable` mapping `(N, D)`
                arrays to `(N,)` predictions.
            model_jac: the model Jacobian — a `Callable` mapping `(N, D)`
                arrays to `(N, D)` derivatives. If `None` (and no
                `data_effect`), the Jacobian is computed numerically.
            data_effect: precomputed Jacobian on `data`, shape `(N, D)`;
                skips calling `model_jac`.
            nof_instances: max instances kept (default `10_000`) — an `int`
                subsamples randomly, `"all"` keeps everything.
            axis_limits: per-feature plot limits, shape `(2, D)`; `None`
                (default) infers them from `data`.
            schema: input metadata — an `effector.Schema` or a plain `dict`
                with any of `feature_names`, `feature_types`, `cat_limit`,
                `target_name`, `scale_x_list`, `scale_y`; omitted fields are
                inferred from `data`, explicit ones win. Coming from a
                DataFrame? Use `effector.from_dataframe`.
            random_state: seed for every internal random step (default `21`,
                reproducible); `None` for non-deterministic behavior.
        """
        super(RHALE, self).__init__(
            data,
            model,
            model_jac,
            data_effect=data_effect,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
            method_name="RHALE",
        )

    def _fill_jacobian(self):
        """Compute the model Jacobian on the data (shared cache-(a) raw
        material): exact via `model_jac`, else numerically. Runs at most once
        per object — every feature's local effect is a column view of it."""
        if self.data_effect is None and self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        elif self.data_effect is None and self.model_jac is None:
            self.data_effect = utils.compute_jacobian_numerically(self.model, self.data)

    def _frame_from_config(self, feature: int) -> tuple:
        """RHALE's continuous local effect is the pointwise derivative —
        instance-anchored, no frame (binning is a summary-stage parameter).
        Categorical features share the (RH)ALE order frame."""
        if self._is_cat(feature):
            return self._cat_frame(feature)
        return ()

    def _compute_local_cont(self, feature: int, frame: tuple) -> dict:
        """Cache-(a) kernel: the pointwise derivative — a column view of the
        shared jacobian table, filled once per object. (The categorical kernel
        is the inherited adjacent-level differences; the jacobian is ignored
        for those features.)"""
        self._fill_jacobian()
        return {"frame": frame, "effects": self.data_effect[:, feature]}

    def _summarize_cont(
        self,
        feature: int,
        mask=None,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        order=None,
        binning_scope: str = "global",
    ) -> typing.Dict:
        """Summary kernel (pure numpy): bin the cached per-instance jacobian
        over the subregion `mask` (None = all) and derive the bin
        effects/variances.

        `binning_scope` (masked only): the x-range handed to the binner —
        `"global"` keeps the frozen global frame (one frame for the split
        search and every node), `"effective"` packs the bins into the masked
        column's own `[min, max]` (finer subregion resolution)."""
        eff = self._local[feature]["effects"]
        col = self.data[:, feature]
        if mask is not None:
            eff = eff[mask]
            col = col[mask]
        return self._bin_local_effects(
            feature, col, eff, mask, binning_method, binning_scope
        )

    def _summarize_cat(
        self,
        feature: int,
        mask=None,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        order=None,
        binning_scope: str = "global",
    ) -> typing.Dict:
        # ordinal: merge adjacent transitions with the chosen binning
        # (code-space bins — binning_scope does not apply); nominal: ALE
        # exactly — one bin per transition, since grouping presumes the
        # adjacency is real
        if self.feature_types[feature] == ingestion.NOMINAL:
            return self._summarize_levels(feature, mask, None)
        return self._summarize_levels(feature, mask, binning_method)

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        *,
        centering: typing.Union[bool, str] = True,
        binning_method: typing.Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        order: typing.Union[None, str, list] = None,
        binning_scope: str = "global",
    ) -> None:
        """Declare per-feature defaults and warm the caches.

        ```python
        rhale.fit("hr", binning_method="dp")
        ```

        !!! note "fit is optional"
            `eval`, `plot`, `heter_score` compute what they need lazily with
            these defaults; `fit` declares the config once and pays the model
            cost upfront.

        Args:
            features: feature(s) to fit — index, name, list, or `"all"`.
            centering: default centering for this feature's queries —
                `False` (none), `True`/`"zero_integral"` (center around the
                y axis), or `"zero_start"` (start at `y=0`).
            binning_method: how the axis is split into bins:

                - `"dp"` (default): dynamic programming — optimal
                  variable-size bins
                - `"agglomerative"`: bottom-up merging of small bins
                  (`"greedy"` is a deprecated alias)
                - `"quantile"`: equal-frequency bins
                - `"fixed"`: equal-width bins

                For custom parameters pass an instance from
                `effector.axis_partitioning`, e.g.
                `DynamicProgramming(max_nof_bins=30)`.

            order: level order for a *categorical* feature of interest:

                - `None` (default): ascending encoded order
                - `"similarity"`: induce the order from the other features
                  (KS-distance seriation, Molnar/iml)
                - a list of the levels: declare it explicitly (applies to
                  exactly one categorical feature)

                Changing `order` invalidates the cached local effects: the
                next query recomputes them; re-fitting the same `order` is a
                cache hit.

            binning_scope: the x-range the binner covers when a *masked*
                summary re-bins a subregion (`eval`/`eval_heter`/`plot`/
                `heter_score` with `mask=`; the regional split search):

                - `"global"` (default): the frozen global `axis_limits` —
                  one frame for every subregion, directly comparable
                - `"effective"`: the masked column's own `[min, max]` —
                  bins packed into the subregion, finer resolution

                Recorded at fit and replayed by every masked call. Ignored
                when no mask is involved.
        """
        # validation is the resolver's job (R6): one table, one error message
        binning_method = ap.return_default(binning_method)
        self._validate_order_arg(features, order)
        check_binning_scope(binning_scope)

        self._fit_loop(
            features,
            centering,
            binning_method=binning_method,
            order=order,
            binning_scope=binning_scope,
        )

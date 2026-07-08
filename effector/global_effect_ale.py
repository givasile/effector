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
        """
        Plot the (RH)ALE feature effect of feature `feature`.

        Notes:
            This is a common method inherited by both ALE and RHALE.

        Parameters:
            feature: index or name of the feature to plot
            heterogeneity: whether to plot the heterogeneity

                  - `False`, plots only the mean effect
                  - `True` or `"std"`, the std of the bin-effects will be plotted using a red vertical bar

            centering: whether to center the plot:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.

            scale_x: None or Dict with keys ['std', 'mean']

                - If set to None, no scaling will be applied.
                - If set to a dict, the x-axis will be scaled by the standard deviation and the mean.
            scale_y: None or Dict with keys ['std', 'mean']

                - If set to None, no scaling will be applied.
                - If set to a dict, the y-axis will be scaled by the standard deviation and the mean.
            show_avg_output: if True, the average output will be shown as a horizontal line.
            y_limits: None or tuple, the limits of the y-axis

                - If set to None, the limits of the y-axis are set automatically
                - If set to a tuple, the limits are manually set

            dy_limits: None or tuple, the limits of the dy-axis

                - If set to None, the limits of the dy-axis are set automatically
                - If set to a tuple, the limits are manually set

            show_only_aggregated: if True, only the main ale plot will be shown
            show_plot: if True, the plot will be shown
            mask: optional boolean `(N,)` selecting a subregion — plot the
                effect *within* it (re-binned from the cached local effects,
                no model calls) with the x-axis windowed to the subregion's
                own interval
            rule: sugar over `mask` — an `effector.Rule` or a rule string,
                applied to the effect's data. Mutually exclusive with `mask`.
            feature_label: optional display name for the feature axis (e.g. a
                regional node's name), overriding `feature_names[feature]`
        """
        feature = self._resolve_feature(feature)
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        centering = helpers.prep_centering(centering)
        scale_x = helpers.resolve_scale(
            scale_x, self.scale_x_list[feature] if self.scale_x_list else None
        )
        scale_y = helpers.resolve_scale(scale_y, self.scale_y)
        mask = self._resolve_mask(mask, rule)
        feature_names = list(self.feature_names)
        if feature_label is not None:
            feature_names[feature] = feature_label

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

        title = (
            "Accumulated Local Effects (ALE)"
            if self.method_name == "ale"
            else "Robust and Heterogeneity-Aware ALE (RHALE)"
        )
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
            positions = np.asarray(levels, dtype=float)
            if np.any(np.diff(positions) < 0):
                # custom (declared/induced) order: draw by rank, label by level
                if labels is None:
                    labels = [f"{v:g}" for v in positions]
                positions = np.arange(len(positions), dtype=float)
            return vis.plot_categorical_effect(
                positions,
                y_levels,
                variances,
                feature,
                heterogeneity,
                title=title,
                level_labels=labels,
                scale_x=scale_x,
                scale_y=scale_y,
                avg_output=avg_output,
                feature_names=feature_names,
                target_name=self.target_name,
                y_limits=y_limits,
                connect_line=True,  # (RH)ALE bars accumulate: show the step path
                show_plot=show_plot,
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
            title=title,
            avg_output=avg_output,
            feature_names=feature_names,
            target_name=self.target_name,
            y_limits=y_limits,
            dy_limits=dy_limits,
            show_only_aggregated=show_only_aggregated,
            show_plot=show_plot,
            x_limits=x_window,
        )


class ALE(ALEBase):
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
        r"""
        Constructor for the ALE plot.

        Definition:
            ALE reveals the effect of $x_s$ by accumulating, bin by bin, the
            average *local effect* of the feature. The axis of $x_s$ is split
            into $K$ fixed bins by the limits $z_0 < z_1 < \dots < z_K$. For an
            instance $x^i$ whose $x_s^i$ falls in bin $k$, the local effect is
            the secant of the model across that bin:
            $$
            \mathtt{effect}_i = \frac{f(x^i_{s=z_k}) - f(x^i_{s=z_{k-1}})}{z_k - z_{k-1}}
            $$
            where $x^i_{s=z}$ is $x^i$ with its $s$-th coordinate set to $z$.
            The bin effect is the mean local effect over the instances $S_k$
            that fall in bin $k$, and ALE at a point $x$ lying in bin $k_x$
            accumulates the completed bins plus the partial contribution of the
            current one:
            $$
            \mu_k = \frac{1}{|S_k|} \sum_{i \in S_k} \mathtt{effect}_i
            \qquad
            \hat{f}^{ALE}(x) = \sum_{k=1}^{k_x - 1} (z_k - z_{k-1})\, \mu_k
                               + (x - z_{k_x - 1})\, \mu_{k_x}
            $$
            The curve is centered afterwards (by default `zero_integral`,
            subtracting its mean over the axis).

            The heterogeneity is the variance of the local effects within the
            bin containing $x$; `eval_heter` returns it as a step function:
            $$
            H(x) = \sigma^2_{k_x},
            \qquad
            \sigma^2_k = \frac{1}{|S_k|} \sum_{i \in S_k} (\mathtt{effect}_i - \mu_k)^2
            $$

            The std of the bin-effects is $\sqrt{\sigma^2_k}$, drawn as the
            error bars on the bin plot.

        Notes:
            - The required parameters are `data` and `model`. The rest are optional.

        Args:
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N, )`

            nof_instances: the number of instances to use for the explanation

                - use an `int`, to specify the number of instances
                - use `"all"`, to use all the instances

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            schema: input metadata (R10) — an `effector.Schema` or a plain `dict`
                with any of the keys `feature_names`, `feature_types`,
                `cat_limit`, `target_name`, `scale_x_list`, `scale_y`

                - omitted fields are auto-inferred from `data` (numpy
                  heuristics) or synthesized (`["x_0", ...]`, `"y"`); to start
                  from a DataFrame use `effector.from_dataframe`
                - explicit fields always win over inference

            random_state: seed for every internal random step (e.g. `nof_instances` subsampling)

                - use an `int` (default: `21`), for reproducible output; two identical constructions give identical results
                - use `None`, for non-deterministic behavior
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
        """Fit the ALE plot.

        Args:
            features: the features to fit. If set to "all", all the features will be fitted.

            binning_method:

                - If set to `"fixed"`, the ALE plot will be computed with the  default values, which are
                `20` bins with at least `10` points per bin and the feature is considered as categorical if it has
                less than `15` unique values.
                - If you want to change the parameters of the method, you pass an instance of the
                class `effector.axis_partitioning.Fixed` with the desired parameters.
                For example: `Fixed(nof_bins=20, min_points_per_bin=0)`

            centering: the default centering mode for this feature's queries:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.

            order: level order for a *categorical* feature of interest

                - `None` (default): ascending encoded order — exact for
                  ordinal features; for nominal features it is arbitrary-but-
                  deterministic, and the accumulated curve's *shape* depends
                  on it (the meaningful quantities are the adjacent-level
                  differences — see docs/method_semantics.md)
                - `"similarity"`: induce the order from the other features
                  (KS-distance seriation, Molnar/iml)
                - a list of the levels: declare it explicitly (applies to
                  exactly one categorical feature)

                Changing `order` changes the frame (R14): the next query
                recomputes the local effects; the same `order` re-fitted is
                a cache hit.
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
    SUPPORTED_FEATURE_TYPES = frozenset({ingestion.CONTINUOUS, ingestion.ORDINAL})
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
        r"""
        Constructor for RHALE.

        Definition:
            RHALE is ALE with the *pointwise derivative* as the local effect.
            Because the effect is read at the instance instead of as a secant
            across the bin, it no longer depends on the bin width, which makes
            the accumulated curve and the per-bin heterogeneity robust to the
            binning. The axis of $x_s$ is split into $K$ bins by the limits
            $z_0 < z_1 < \dots < z_K$, and for an instance $x^i$ whose $x_s^i$
            falls in bin $k$ the local effect is
            $$
            \mathtt{effect}_i = \frac{\partial f}{\partial x_s}(x^i)
            $$
            taken from the model Jacobian (exact if `model_jac` is provided,
            otherwise numerical). The bin effect, the accumulation and the
            heterogeneity are then identical to ALE:
            $$
            \mu_k = \frac{1}{|S_k|} \sum_{i \in S_k} \mathtt{effect}_i
            \qquad
            \hat{f}^{RHALE}(x) = \sum_{k=1}^{k_x - 1} (z_k - z_{k-1})\, \mu_k
                                 + (x - z_{k_x - 1})\, \mu_{k_x}
            $$
            The curve is centered afterwards (by default `zero_integral`).

            The heterogeneity is the variance of the local effects within the
            bin containing $x$; `eval_heter` returns it as a step function:
            $$
            H(x) = \sigma^2_{k_x},
            \qquad
            \sigma^2_k = \frac{1}{|S_k|} \sum_{i \in S_k} (\mathtt{effect}_i - \mu_k)^2
            $$

            The std of the bin-effects is $\sqrt{\sigma^2_k}$, drawn as the
            error bars on the bin plot.

        Notes:
            The required parameters are `data` and `model`. The rest are optional.

        Args:
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N, )`

            model_jac: the Jacobian of the model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N, D)`

            nof_instances: the number of instances to use for the explanation

                - use an `int`, to specify the number of instances
                - use `"all"`, to use all the instances

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            data_effect:
                - if np.ndarray, the model Jacobian computed on the `data`
                - if None, the Jacobian will be computed using model_jac

            schema: input metadata (R10) — an `effector.Schema` or a plain `dict`
                with any of the keys `feature_names`, `feature_types`,
                `cat_limit`, `target_name`, `scale_x_list`, `scale_y`

                - omitted fields are auto-inferred from `data` (numpy
                  heuristics) or synthesized (`["x_0", ...]`, `"y"`); to start
                  from a DataFrame use `effector.from_dataframe`
                - explicit fields always win over inference

            random_state: seed for every internal random step (e.g. `nof_instances` subsampling)

                - use an `int` (default: `21`), for reproducible output; two identical constructions give identical results
                - use `None`, for non-deterministic behavior
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
        # (code-space bins — binning_scope does not apply)
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
        """Fit the model.

        Args:
            features (int, str, list): the features to fit.

                - If set to "all", all the features will be fitted.

            binning_method (str): the binning method to use.

                - Use `"greedy"` for using the Greedy binning solution with the default parameters.
                  For custom parameters initialize a `axis_partitioning.Greedy` object
                - Use `"dp"` for using a Dynamic Programming binning solution with the default parameters.
                  For custom parameters initialize a `axis_partitioning.DynamicProgramming` object
                - Use `"fixed"` for using a Fixed binning solution with the default parameters.
                  For custom parameters initialize a `axis_partitioning.Fixed` object

            centering: the default centering mode for this feature's queries:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis
                - `zero_start` starts the plot from `y=0`

            order: level order for a *categorical* feature of interest

                - `None` (default): ascending encoded order — exact for
                  ordinal features; for nominal features it is arbitrary-but-
                  deterministic, and the accumulated curve's *shape* depends
                  on it (the meaningful quantities are the adjacent-level
                  differences — see docs/method_semantics.md)
                - `"similarity"`: induce the order from the other features
                  (KS-distance seriation, Molnar/iml)
                - a list of the levels: declare it explicitly (applies to
                  exactly one categorical feature)

                Changing `order` changes the frame (R14): the next query
                recomputes the local effects; the same `order` re-fitted is
                a cache hit.

            binning_scope: the x-range the binner covers when a *masked*
                summary re-bins a subregion (`eval`/`eval_heter`/`plot`/
                `heter_score` with `mask=`; the regional split search)

                - `"global"` (default): the frozen global `axis_limits` — one
                  frame for every subregion, directly comparable
                - `"effective"`: the masked column's own `[min, max]` — bins
                  packed into the subregion, finer resolution

                Recorded at fit and replayed by every masked call, so the
                split search and the display always share the same scope.
                Ignored when no mask is involved.
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

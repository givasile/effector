import typing
from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np

import effector.axis_partitioning as ap
import effector.helpers as helpers
import effector.utils as utils
import effector.visualization as vis
from effector import ingestion
from effector.global_effect import GlobalEffectBase


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
        # per-feature raw local effects + bin limits (regional reuse and the
        # discrete kernel store here; ALE also uses them on the continuous path)
        self.data_effect_ale: dict = {}
        self.bin_limits: dict = {}
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

    @abstractmethod
    def fit(self, features: typing.Union[int, str, list] = "all", **kwargs) -> None:
        raise NotImplementedError

    def _eval_unnorm(
        self, feature: int, x: np.ndarray, heterogeneity: bool = False, params=None
    ):
        # `params` (from `_summarize`) lets the masked heterogeneity path evaluate
        # a transient subregion payload without disturbing the stored fitted one
        if params is None:
            params = self.feature_effect["feature_" + str(feature)]
        if params.get("is_cat"):
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
        y = utils.compute_accumulated_effect(
            x, limits=params["limits"], bin_effect=params["bin_effect"], dx=params["dx"]
        )
        if heterogeneity:
            var = utils.apply_bin_value(
                x=x, bin_limits=params["limits"], bin_value=params["bin_variance"]
            )
            return y, var
        else:
            return y

    def _validate_order_arg(self, features, order):
        if order is None or isinstance(order, str):
            return
        feats = helpers.prep_features(features, self.dim)
        cats = [f for f in feats if self._is_cat(f)]
        if len(feats) != 1 or len(cats) != 1:
            raise ValueError(
                "an explicit `order` list applies to exactly one categorical "
                "feature — call fit per feature"
            )

    def _fit_feature_cat(
        self, feature: int, binning_method=None, order=None
    ) -> typing.Dict:
        """The discrete (RH)ALE kernel: two-sided adjacent-level differences in
        code space (method_semantics.md). ALE keeps one bin per transition;
        RHALE additionally merges adjacent transitions with Greedy/DP —
        adaptive level grouping."""
        levels = self._levels(feature)
        if len(levels) < 2:
            raise ValueError(
                f"feature {feature} {self.feature_names[feature]!r} has a "
                f"single level — no effect to compute"
            )
        if order is not None:
            levels = self._resolve_level_order(feature, levels, order)
        positions, effects, instance_idx = utils.compute_local_effects_categorical(
            self.data, self.model, levels, feature
        )
        self.data_effect_ale["feature_" + str(feature)] = {
            "positions": positions,
            "effects": effects,
            "instance_idx": instance_idx,
            "levels": levels,
        }

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
        self.bin_limits["feature_" + str(feature)] = limits

        params = utils.compute_ale_params(positions, effects, limits)
        params["alg_params"] = "categorical"
        params["levels"] = levels
        params["is_cat"] = True
        return params

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = True,
        centering: Union[bool, str] = True,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        show_avg_output: bool = False,
        y_limits: Optional[List] = None,
        dy_limits: Optional[List] = None,
        show_only_aggregated: bool = False,
        show_plot: bool = True,
    ):
        """
        Plot the (RH)ALE feature effect of feature `feature`.

        Notes:
            This is a common method inherited by both ALE and RHALE.

        Parameters:
            feature: the feature to plot
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
        """
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        centering = helpers.prep_centering(centering)
        scale_x = helpers.resolve_scale(
            scale_x, self.scale_x_list[feature] if self.scale_x_list else None
        )
        scale_y = helpers.resolve_scale(scale_y, self.scale_y)

        # fit the feature if needed (the eval below reuses the stored state)
        self.eval(
            feature, np.array([self.axis_limits[0, feature]]), centering=centering
        )
        params = self.feature_effect["feature_" + str(feature)]

        # the accumulated curve is piecewise linear between bin limits, so
        # evaluating exactly at the limits draws it exactly (no resampling).
        # categoricals are drawn by the is_cat branch below (at their observed
        # level values); their limits are positional codes 0..K-1 that eval
        # would reject, so only build this grid for continuous features.
        if not params.get("is_cat"):
            x = np.asarray(params["limits"], dtype=float)
            y = self.eval(feature, x, centering=centering)

        if show_avg_output:
            avg_output = helpers.prep_avg_output(self.data, self.model, None, scale_y)
        else:
            avg_output = None

        title = (
            "Accumulated Local Effects (ALE)"
            if self.method_name == "ale"
            else "Robust and Heterogeneity-Aware ALE (RHALE)"
        )
        if params.get("is_cat"):
            # bars = accumulated per-level values (in fit order); whiskers =
            # the variance of the step into each level (method_semantics.md)
            levels, labels = self._level_display(feature, params["levels"])
            y_levels = self.eval(feature, levels, centering=centering)
            variances = (
                self._eval_unnorm(feature, levels, heterogeneity=True)[1]
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
                feature_names=self.feature_names,
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
            feature_names=self.feature_names,
            target_name=self.target_name,
            y_limits=y_limits,
            dy_limits=dy_limits,
            show_only_aggregated=show_only_aggregated,
            show_plot=show_plot,
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

    def _fit_feature(
        self, feature: int, binning_method="fixed", order=None
    ) -> typing.Dict:

        data = self.data
        if self._is_cat(feature):
            return self._fit_feature_cat(feature, order=order)
        if not (binning_method == "fixed" or isinstance(binning_method, ap.Fixed)):
            raise ValueError(
                f"Invalid binning_method: {binning_method!r}; ALE works only with "
                "the fixed binning method ('fixed' or an ap.Fixed instance)"
            )

        if isinstance(binning_method, str):
            binning_method = ap.Fixed()
        limits = binning_method.find_limits(
            data[:, feature], None, self.axis_limits[:, feature]
        )
        utils.raise_if_no_binning(limits, feature, binning_method)

        # compute data effect on bin limits
        data_effect = utils.compute_local_effects(data, self.model, limits, feature)
        self.data_effect_ale["feature_" + str(feature)] = data_effect
        self.bin_limits["feature_" + str(feature)] = limits

        # compute the bin effect
        dale_params = utils.compute_ale_params(data[:, feature], data_effect, limits)
        dale_params["alg_params"] = "fixed"
        return dale_params

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        *,
        centering: typing.Union[bool, str] = True,
        points_for_centering: int = helpers.NOF_INTERNAL_POINTS,
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

            centering: whether to compute the normalization constant for centering the plot:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.

            points_for_centering: the number of points to use for centering the plot. Default is 30.

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

                Changing `order` requires calling `fit` again; `eval`/`plot`
                reuse the fitted order.
        """
        if not (binning_method == "fixed" or isinstance(binning_method, ap.Fixed)):
            raise ValueError(
                f"Invalid binning_method: {binning_method!r}; ALE works only with "
                "the fixed binning method ('fixed' or an ap.Fixed instance)"
            )
        self._validate_order_arg(features, order)

        self._fit_loop(
            features,
            centering,
            points_for_centering,
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
        """Compute the model Jacobian on the data (step 2 raw material): exact
        via `model_jac`, else numerically. Runs at most once."""
        if self.data_effect is None and self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        elif self.data_effect is None and self.model_jac is None:
            self.data_effect = utils.compute_jacobian_numerically(self.model, self.data)

    def _compute_local_effects(self, feature: int) -> None:
        """Step 2: RHALE's local effect is the pointwise derivative — a column
        of the model Jacobian, independent of the binning (so re-binning a
        subregion needs no model calls)."""
        if self._is_cat(feature):
            # ordinal kernel handled by the (combined) `_fit_feature_cat`; not
            # reached on the continuous split-search path
            raise NotImplementedError
        self._fill_jacobian()
        self.local_effects["feature_" + str(feature)] = self.data_effect[:, feature]

    def _summarize(
        self,
        feature: int,
        mask=None,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        order=None,
    ) -> typing.Dict:
        """Step 3 (pure numpy): bin the cached per-instance jacobian over the
        subregion `mask` (None = all) and derive the bin effects/variances."""
        self._ensure_local_effects(feature)
        eff = self.local_effects["feature_" + str(feature)]
        col = self.data[:, feature]
        if mask is not None:
            eff = eff[mask]
            col = col[mask]

        binning = (
            ap.return_default(binning_method)
            if isinstance(binning_method, str)
            else binning_method
        )
        limits = binning.find_limits(col, eff, self.axis_limits[:, feature])
        utils.raise_if_no_binning(limits, feature, binning)

        dale_params = utils.compute_ale_params(col, eff, limits)
        dale_params["alg_params"] = binning
        return dale_params

    def _fit_feature(
        self,
        feature: int,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        order=None,
    ) -> typing.Dict:
        if self._is_cat(feature):
            # ordinal kernel: adjacent-level differences are the discrete
            # derivative; the jacobian (if any) is ignored for this feature
            return self._fit_feature_cat(feature, binning_method, order=order)
        return self._summarize(feature, None, binning_method=binning_method, order=order)

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        *,
        centering: typing.Union[bool, str] = True,
        points_for_centering: int = helpers.NOF_INTERNAL_POINTS,
        binning_method: typing.Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        order: typing.Union[None, str, list] = None,
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

            centering: whether to compute the normalization constant for centering the plot:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis
                - `zero_start` starts the plot from `y=0`

            points_for_centering: the number of points to use for centering the plot. Default is 30.

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

                Changing `order` requires calling `fit` again; `eval`/`plot`
                reuse the fitted order.
        """
        # validation is the resolver's job (R6): one table, one error message
        binning_method = ap.return_default(binning_method)
        self._validate_order_arg(features, order)

        self._fit_loop(
            features,
            centering,
            points_for_centering,
            binning_method=binning_method,
            order=order,
        )

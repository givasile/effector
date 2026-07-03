import typing
from abc import abstractmethod
from typing import List, Optional, Union

import numpy as np

import effector.axis_partitioning as ap
import effector.helpers as helpers
import effector.utils as utils
import effector.visualization as vis
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
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
        method_name: str = "ALE",
    ):
        super(ALEBase, self).__init__(
            method_name,
            data,
            model,
            model_jac,
            data_effect,
            nof_instances,
            axis_limits,
            feature_names,
            target_name,
        )

    @abstractmethod
    def fit(self, features: typing.Union[int, str, list] = "all", **kwargs) -> None:
        raise NotImplementedError

    def _eval_unnorm(self, feature: int, x: np.ndarray, heterogeneity: bool = False):
        params = self.feature_effect["feature_" + str(feature)]
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

    def plot(
        self,
        feature: int,
        heterogeneity: bool = True,
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
                  - `True`, the std of the bin-effects will be plotted using a red vertical bar

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

        # hack to fit the feature if not fitted
        self.eval(
            feature, np.array([self.axis_limits[0, feature]]), centering=centering
        )

        # compat shim for the vis layer's old (feature, x, het, centering)
        # callable, until the visualization pass (step 4) redraws it on
        # eval/eval_heter directly
        def curve(feature_, x_, heterogeneity_, centering_):
            y = self.eval(feature_, x_, centering=centering_)
            if heterogeneity_:
                return y, self.eval_heter(feature_, x_)
            return y

        if show_avg_output:
            avg_output = helpers.prep_avg_output(self.data, self.model, None, scale_y)
        else:
            avg_output = None

        title = (
            "Accumulated Local Effects (ALE)"
            if self.method_name == "ale"
            else "Robust and Heterogeneity-Aware ALE (RHALE)"
        )
        ret = vis.ale_plot(
            self.feature_effect["feature_" + str(feature)],
            curve,
            feature,
            centering=centering,
            error=heterogeneity,
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

        if not show_plot:
            fig, ax = ret
            return fig, ax


class ALE(ALEBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        *,
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ):
        r"""
        Constructor for the ALE plot.

        Definition:
            ALE is defined as:
            $$
            \hat{f}^{ALE}(x_s) = TODO
            $$

            The heterogeneity is:
            $$
            TODO
            $$

            The std of the bin-effects is:
            $$
            TODO
            $$

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

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `                  ["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """
        self.bin_limits = {}
        self.data_effect_ale = {}
        super(ALE, self).__init__(
            data,
            model,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            feature_names=feature_names,
            target_name=target_name,
            method_name="ALE",
        )

    def _fit_feature(self, feature: int, binning_method="fixed") -> typing.Dict:

        data = self.data
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
        binning_method: typing.Union[str, ap.Fixed] = "fixed",
        centering: typing.Union[bool, str] = True,
        points_for_centering: int = 30,
    ) -> None:
        """Fit the ALE plot.

        Args:
            features: the features to fit. If set to "all", all the features will be fitted.

            binning_method:

                - If set to `"fixed"`, the ALE plot will be computed with the  default values, which are
                `20` bins with at least `10` points per bin and the feature is considered as categorical if it has
                less than `15` unique values.
                - If you want to change the parameters of the method, you pass an instance of the
                class `effector.binning_methods.Fixed` with the desired parameters.
                For example: `Fixed(nof_bins=20, min_points_per_bin=0, cat_limit=10)`

            centering: whether to compute the normalization constant for centering the plot:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.

            points_for_centering: the number of points to use for centering the plot. Default is 100.
        """
        if not (binning_method == "fixed" or isinstance(binning_method, ap.Fixed)):
            raise ValueError(
                f"Invalid binning_method: {binning_method!r}; ALE works only with "
                "the fixed binning method ('fixed' or an ap.Fixed instance)"
            )

        self._fit_loop(
            features, centering, points_for_centering, binning_method=binning_method
        )


class RHALE(ALEBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        *,
        data_effect: typing.Optional[np.ndarray] = None,
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Optional[np.ndarray] = None,
        feature_names: typing.Optional[list] = None,
        target_name: typing.Optional[str] = None,
    ):
        r"""
        Constructor for RHALE.

        Definition:
            RHALE is defined as:
            $$
            \hat{f}^{RHALE}(x_s) = TODO
            $$

            The heterogeneity is:
            $$
            TODO
            $$

            The std of the bin-effects is:
            $$
            TODO
            $$

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

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """
        super(RHALE, self).__init__(
            data,
            model,
            model_jac,
            data_effect=data_effect,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            feature_names=feature_names,
            target_name=target_name,
            method_name="RHALE",
        )

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points."""
        if self.data_effect is None and self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        elif self.data_effect is None and self.model_jac is None:
            self.data_effect = utils.compute_jacobian_numerically(self.model, self.data)

    def _fit_feature(
        self,
        feature: int,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Greedy, ap.Fixed
        ] = "greedy",
    ) -> typing.Dict:
        if self.data_effect is None:
            self.compile()

        data = self.data
        data_effect = self.data_effect

        if isinstance(binning_method, str):
            binning_method = ap.return_default(binning_method)
        limits = binning_method.find_limits(
            data[:, feature], self.data_effect[:, feature], self.axis_limits[:, feature]
        )
        utils.raise_if_no_binning(limits, feature, binning_method)

        # compute the bin effect
        dale_params = utils.compute_ale_params(
            data[:, feature], data_effect[:, feature], limits
        )

        dale_params["alg_params"] = binning_method
        return dale_params

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        binning_method: typing.Union[
            str, ap.DynamicProgramming, ap.Greedy, ap.Fixed
        ] = "greedy",
        centering: typing.Union[bool, str] = True,
        points_for_centering: int = 30,
    ) -> None:
        """Fit the model.

        Args:
            features (int, str, list): the features to fit.

                - If set to "all", all the features will be fitted.

            binning_method (str): the binning method to use.

                - Use `"greedy"` for using the Greedy binning solution with the default parameters.
                  For custom parameters initialize a `binning_methods.Greedy` object
                - Use `"dp"` for using a Dynamic Programming binning solution with the default parameters.
                  For custom parameters initialize a `binning_methods.DynamicProgramming` object
                - Use `"fixed"` for using a Fixed binning solution with the default parameters.
                  For custom parameters initialize a `binning_methods.Fixed` object

            centering: whether to compute the normalization constant for centering the plot:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis
                - `zero_start` starts the plot from `y=0`

            points_for_centering: the number of points to use for centering the plot. Default is 100.
        """
        # validation is the resolver's job (R6): one table, one error message
        binning_method = ap.return_default(binning_method)

        self._fit_loop(
            features, centering, points_for_centering, binning_method=binning_method
        )

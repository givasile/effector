import typing
from typing import Callable, List, Optional, Union, Tuple
import effector.utils as utils
import effector.visualization as vis
import effector.binning_methods as bm
import effector.helpers as helpers
import effector.utils_integrate as utils_integrate
from effector.global_effect import GlobalEffectBase
import numpy as np
from abc import abstractmethod


class ALEBase(GlobalEffectBase):

    def __init__(
            self,
            data: np.ndarray,
            model: callable,
            nof_instances: Union[int, str] = "all",
            axis_limits: Optional[np.ndarray] = None,
            avg_output: Optional[float] = None,
            feature_names: Optional[List] = None,
            target_name: Optional[str] = None,
            method_name: str = "ALE",
    ):
        self.method_name = method_name
        super(ALEBase, self).__init__(
            method_name,
            data,
            model,
            nof_instances,
            axis_limits,
            avg_output,
            feature_names,
            target_name
        )

    @abstractmethod
    def _fit_feature(self,
                     feature: int,
                     binning_method: typing.Union[str, bm.DynamicProgramming, bm.Greedy, bm.Fixed] = "greedy"
                     ) -> typing.Dict:
        raise NotImplementedError

    @abstractmethod
    def fit(self,
            features: typing.Union[int, str, list] = "all",
            **kwargs) -> None:
        raise NotImplementedError

    def _compute_norm_const(
        self, feature: int, method: str = "zero_integral", nof_points: int = 100
    ) -> float:
        """Compute the normalization constant."""
        assert method in ["zero_integral", "zero_start"]

        def create_partial_eval(feat):
            return lambda x: self._eval_unnorm(feat, x, heterogeneity=False)

        partial_eval = create_partial_eval(feature)
        start = self.axis_limits[0, feature]
        stop = self.axis_limits[1, feature]

        if method == "zero_integral":
            z = utils_integrate.mean_1d_linspace(partial_eval, start, stop, nof_points)
        else:
            z = partial_eval(np.array([start])).item()
        return z

    def _fit_loop(self, features, binning_method, centering):
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            # compute all information required for plotting and evaluating the feature effect
            self.feature_effect["feature_" + str(s)] = self._fit_feature(
                s, binning_method
            )

            # append the "norm_const" to the feature effect if centering is not False
            if centering is not False:
                self.feature_effect["feature_" + str(s)]["norm_const"] = self._compute_norm_const(s, method=centering)
            else:
                self.feature_effect["feature_" + str(s)]["norm_const"] = self.empty_symbol

            self.is_fitted[s] = True
            self.method_args["feature_" + str(s)] = {
                "centering": centering,
            }

    def _eval_unnorm(self, feature: int, x: np.ndarray, heterogeneity: bool = False):
        params = self.feature_effect["feature_" + str(feature)]
        y = utils.compute_accumulated_effect(
            x, limits=params["limits"], bin_effect=params["bin_effect"], dx=params["dx"]
        )
        if heterogeneity:
            std = utils.compute_accumulated_effect(
                x,
                limits=params["limits"],
                bin_effect=np.sqrt(params["bin_variance"]),
                dx=params["dx"],
            )

            return y, std
        else:
            return y

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        heterogeneity: bool = False,
        centering: typing.Union[bool, str] = False,
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray]]:
        """Evalueate the (RH)ALE feature effect of feature `feature` at points `xs`.

        Notes:
            This is a common method inherited by both ALE and RHALE.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot
              - `np.ndarray` of shape `(T, )`
            heterogeneity: whether to return heterogeneity:

                  - `False`, returns the mean effect `y` at the given `xs`
                  - `True`, returns a tuple `(y, H)` of two `ndarrays`; `y` is the mean effect and `H` is the
                  heterogeneity evaluated at `xs`

            centering: whether to center the plot:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.
        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, std)` otherwise

        """
        centering = helpers.prep_centering(centering)

        if self.refit(feature, centering):
            self.fit(features=feature, centering=centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        # Evaluate the feature
        yy = self._eval_unnorm(feature, xs, heterogeneity=heterogeneity)
        y, std = yy if heterogeneity else (yy, None)

        # Center if asked
        y = (
            y - self.feature_effect["feature_" + str(feature)]["norm_const"]
            if centering
            else y
        )

        return (y, std) if heterogeneity is not False else y

    def plot(
            self,
            feature: int,
            heterogeneity: bool = False,
            centering: Union[bool, str] = False,
            scale_x: Optional[dict] = None,
            scale_y: Optional[dict] = None,
            show_avg_output: bool = False,
            y_limits: Optional[List] = None,
            dy_limits: Optional[List] = None
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
        """
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        centering = helpers.prep_centering(centering)

        # hack to fit the feature if not fitted
        self.eval(
            feature, np.array([self.axis_limits[0, feature]]), centering=centering
        )

        if show_avg_output:
            avg_output = helpers.prep_avg_output(self.data, self.model, self.avg_output, scale_y)
        else:
            avg_output = None

        vis.ale_plot(
            self.feature_effect["feature_" + str(feature)],
            self.eval,
            feature,
            centering=centering,
            error=heterogeneity,
            scale_x=scale_x,
            scale_y=scale_y,
            title=self.method_name + " plot",
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            y_limits=y_limits,
            dy_limits=dy_limits
        )


class ALE(ALEBase):
    def __init__(
            self,
            data: np.ndarray,
            model: callable,
            nof_instances: Union[int, str] = "all",
            axis_limits: Optional[np.ndarray] = None,
            avg_output: Optional[float] = None,
            feature_names: Optional[List] = None,
            target_name: Optional[str] = None,
    ):
        """
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

            avg_output: the average output of the model on the data

                - use a `float`, to specify it manually
                - use `None`, to be inferred as `np.mean(model(data))`

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `                  ["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """
        super(ALE, self).__init__(
            data, model, nof_instances, axis_limits, avg_output, feature_names, target_name, "ALE"
        )

    def _fit_feature(self, feature: int, binning_method="fixed") -> typing.Dict:

        # drop points outside of limits
        ind = np.logical_and(
            self.data[:, feature] >= self.axis_limits[0, feature],
            self.data[:, feature] <= self.axis_limits[1, feature],
        )
        data = self.data[ind, :]

        # assertion
        assert binning_method == "fixed" or isinstance(
            binning_method, bm.Fixed
        ), "ALE can work only with the fixed binning method!"

        if isinstance(binning_method, str):
            binning_method = bm.Fixed(nof_bins=20, min_points_per_bin=0)
        bin_est = bm.find_limits(data, None, feature, self.axis_limits, binning_method)
        bin_name = bin_est.__class__.__name__

        # assert bins can be computed else raise error
        assert bin_est.limits is not False, (
            "Impossible to compute bins with enough points for feature "
            + str(feature + 1)
            + " and binning strategy: "
            + bin_name
            + ". Change bin strategy or "
            "the parameters of the method"
        )

        # compute data effect on bin limits
        data_effect = utils.compute_local_effects(
            data, self.model, bin_est.limits, feature
        )

        # compute the bin effect
        dale_params = utils.compute_ale_params(
            data[:, feature], data_effect, bin_est.limits
        )
        dale_params["alg_params"] = "fixed"
        return dale_params

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        binning_method: typing.Union[str, bm.Fixed] = "fixed",
        centering: typing.Union[bool, str] = "zero_integral",
    ) -> None:
        """Fit the ALE plot.

        Args:
            features: the features to fit. If set to "all", all the features will be fitted.

            binning_method:

                - If set to `"fixed"`, the ALE plot will be computed with the  default values, which are
                `20` bins with at least `10` points per bin and the featue is considered as categorical if it has
                less than `15` unique values.
                - If you want to change the parameters of the method, you pass an instance of the
                class `effector.binning_methods.Fixed` with the desired parameters.
                For example: `Fixed(nof_bins=20, min_points_per_bin=0, cat_limit=10)`

            centering: whether to compute the normalization constant for centering the plot:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.
        """
        assert binning_method == "fixed" or isinstance(
            binning_method, bm.Fixed
        ), "ALE can work only with the fixed binning method!"

        self._fit_loop(features, binning_method, centering)


class RHALE(ALEBase):
    def __init__(
            self,
            data: np.ndarray,
            model: callable,
            model_jac: typing.Union[None, callable] = None,
            nof_instances: typing.Union[int, str] = "all",
            axis_limits: typing.Optional[np.ndarray] = None,
            data_effect: typing.Optional[np.ndarray] = None,
            avg_output: typing.Optional[float] = None,
            feature_names: typing.Optional[list] = None,
            target_name: typing.Optional[str] = None,
    ):
        """
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

            avg_output: the average output of the model on the data

                - use a `float`, to specify it manually
                - use `None`, to be inferred as `np.mean(model(data))`

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """
        self.model_jac = model_jac

        # select nof_instances from the data
        nof_instances, indices = helpers.prep_nof_instances(nof_instances, data.shape[0])
        data = data[indices, :]
        data_effect = data_effect[indices, :] if data_effect is not None else None
        self.data_effect = data_effect

        super(RHALE, self).__init__(
            data, model, "all", axis_limits, avg_output, feature_names, target_name, "RHALE"
        )

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points.
        """
        if self.data_effect is None and self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        elif self.data_effect is None and self.model_jac is None:
            self.data_effect = utils.compute_jacobian_numerically(self.model, self.data)

    def _fit_feature(
            self,
            feature: int,
            binning_method: Union[str, bm.DynamicProgramming, bm.Greedy, bm.Fixed] = "greedy"
    ) -> typing.Dict:
        if self.data_effect is None:
            self.compile()

        # drop points outside of limits
        ind = np.logical_and(
            self.data[:, feature] >= self.axis_limits[0, feature],
            self.data[:, feature] <= self.axis_limits[1, feature],
        )
        data = self.data[ind, :]
        data_effect = self.data_effect[ind, :]

        # bin estimation
        bin_est = bm.find_limits(
            data, data_effect, feature, self.axis_limits, binning_method
        )
        bin_name = bin_est.__class__.__name__

        # assert bins can be computed else raise error
        assert bin_est.limits is not False, (
            "Impossible to compute bins with enough points for feature "
            + str(feature + 1)
            + " and binning strategy: "
            + bin_name
            + ". Change bin strategy or "
            "the parameters of the method"
        )

        # compute the bin effect
        dale_params = utils.compute_ale_params(
            data[:, feature], data_effect[:, feature], bin_est.limits
        )
        dale_params["alg_params"] = binning_method
        return dale_params

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        binning_method: typing.Union[str, bm.DynamicProgramming, bm.Greedy, bm.Fixed] = "greedy",
        centering: typing.Union[bool, str] = False,
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
        """
        assert binning_method in [
            "greedy",
            "dynamic",
            "fixed"
        ] or isinstance(
            binning_method, bm.Greedy
        ) or isinstance(
            binning_method, bm.DynamicProgramming
        ) or isinstance(
            binning_method, bm.Fixed
        ), "Unknown binning method!"

        self._fit_loop(features, binning_method, centering)
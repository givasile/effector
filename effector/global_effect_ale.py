import typing
import effector.utils as utils
import effector.visualization as vis
import effector.binning_methods as bm
import effector.helpers as helpers
from effector.global_effect import GlobalEffect
import numpy as np


class ALE(GlobalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: None | np.ndarray = None,
        avg_output: None | float = None,
        feature_names: None | list = None,
        target_name: None | str = None,
    ):
        """
        Constructor for the ALE plot.

        Definition:
            RHALE is defined as:
            $$
            \hat{f}^{RHALE}(x_s) = TODO
            $$

            The heterogeneity is:
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
            data, model, axis_limits, avg_output, feature_names, target_name
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
                - If set to "fixed", the greedy binning method with default values will be used.
                - If you want to change the parameters of the method, you can pass an instance of the Fixed class.
            centering: whether to center the RHALE plot

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.
        """
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(
                s, binning_method
            )
            if centering is not False:
                self.norm_const[s] = self._compute_norm_const(s, method=centering)
            self.is_fitted[s] = True
            self.method_args["feature_" + str(s)] = {
                "centering": centering,
            }

    def _eval_unnorm(self, feature: int, x: np.ndarray, uncertainty: bool = False):
        params = self.feature_effect["feature_" + str(feature)]
        y = utils.compute_accumulated_effect(
            x, limits=params["limits"], bin_effect=params["bin_effect"], dx=params["dx"]
        )
        if uncertainty:
            std = utils.compute_accumulated_effect(
                x,
                limits=params["limits"],
                bin_effect=np.sqrt(params["bin_variance"]),
                dx=params["dx"],
            )
            std_err = utils.compute_accumulated_effect(
                x,
                limits=params["limits"],
                bin_effect=np.sqrt(params["bin_estimator_variance"]),
                dx=params["dx"],
            )

            return y, std, std_err
        else:
            return y

    # TODO: add latex formula
    def plot(
        self,
        feature: int = 0,
        heterogeneity: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = False,
        scale_x: typing.Union[None, dict] = None,
        scale_y: typing.Union[None, dict] = None,
        show_avg_output: bool = False,
        y_limits: None | list = None,
    ):
        """
        Plot the ALE plot for a given feature.

        Parameters:
            feature: the feature to plot
            heterogeneity:
                - If set to False, no confidence interval will be shown.
                - If set to "std" or True, the accumulated standard deviation will be shown.
                - If set to "stderr", the accumulated standard error of the mean will be shown.
            centering:
                - If set to False, no centering will be applied.
                - If set to "zero_integral" or True, the integral of the feature effect will be set to zero.
                - If set to "zero_mean", the mean of the feature effect will be set to zero.
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
            title="ALE plot",
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            y_limits = y_limits
        )

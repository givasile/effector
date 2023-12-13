import typing
import copy
import effector.utils as utils
import effector.visualization as vis
import effector.binning_methods as bm
import effector.helpers as helpers
from effector.global_effect import GlobalEffect
import numpy as np


class RHALE(GlobalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        axis_limits: typing.Union[None, np.ndarray] = None,
        data_effect: typing.Union[None, np.ndarray] = None,
        avg_output: typing.Union[None, float] = None,
        feature_names: typing.Union[None, list] = None,
        target_name: typing.Union[None, str] = None,
    ):
        """
        RHALE constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            model_jac: the black-box model Jacobian (N,D) -> (N,D)
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            data_effect:
                - if np.ndarray, the model Jacobian computed on the `data`
                - if None, the Jacobian will be computed using model_jac

        """
        self.model_jac = model_jac

        # if data_effect is None, it will be computed after compile
        self.data_effect = data_effect

        super(RHALE, self).__init__(
            data, model, axis_limits, avg_output, feature_names, target_name
        )

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points.
        """
        if self.data_effect is None and self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        elif self.data_effect is None and self.model_jac is None:
            self.data_effect = np.zeros_like(self.data)

            # use finite difference
            for i in range(self.data.shape[1]):
                data_1 = copy.deepcopy(self.data)
                data_2 = copy.deepcopy(self.data)

                data_1[:, i] = data_1[:, i] + 1e-6
                data_2[:, i] = data_2[:, i] - 1e-6
                self.data_effect[:, i] = (
                    self.model(data_1) - self.model(data_2)
                ) / 2e-6

    def _fit_feature(self, feature: int, binning_method) -> typing.Dict:
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
        binning_method: typing.Union[
            str, bm.DynamicProgramming, bm.Greedy, bm.Fixed
        ] = "greedy",
        centering: typing.Union[bool, str] = "zero_integral",
    ) -> None:
        """Fit the model.

        Args:
            features (int, str, list): the features to fit.
                - If set to "all", all the features will be fitted.
            binning_method (str):
                - If set to "greedy" or bm.Greedy, the greedy binning method will be used.
                - If set to "dynamic" or bm.DynamicProgramming, the dynamic programming binning method will be used.
                - If set to "fixed" or bm.Fixed, the fixed binning method will be used.
            centering (bool, str):
                - If set to False, no centering will be applied.
                - If set to "zero_integral" or True, the integral of the feature effect will be set to zero.
                - If set to "zero_mean", the mean of the feature effect will be set to zero.
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

    # TODO: add latex formula and add to documentation
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

    # TODO: add latex formula and add to documentation
    def plot(
        self,
        feature: int = 0,
        confidence_interval: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = False,
        scale_x: typing.Union[None, dict] = None,
        scale_y: typing.Union[None, dict] = None,
        show_avg_output: bool = False,
        not_show: bool = False,
    ):
        """
        Plot the ALE plot for a given feature.

        Parameters:
            feature: the feature to plot
            confidence_interval:
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
            show_avg_output: bool, if True, the average output is shown
            not_show: bool, if True, the plot is not shown, but returned as a matplotlib object
        """
        confidence_interval = helpers.prep_confidence_interval(confidence_interval)
        centering = helpers.prep_centering(centering)

        # hack to fit the feature if not fitted
        self.eval(
            feature, np.array([self.axis_limits[0, feature]]), centering=centering
        )

        avg_output = helpers.prep_avg_output(
            self.data, self.model, show_avg_output, self.avg_output, scale_y
        )

        fig, ax1, ax2 = vis.ale_plot(
            self.feature_effect["feature_" + str(feature)],
            self.eval,
            feature,
            centering=centering,
            error=confidence_interval,
            scale_x=scale_x,
            scale_y=scale_y,
            title="RHALE Plot",
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            not_show=not_show,
        )
        if not_show:
            return fig, ax1, ax2

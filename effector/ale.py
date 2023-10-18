import typing
import effector.utils as utils
import effector.visualization as vis
import effector.binning_methods as bm
import effector.helpers as helpers
from effector.fe_base import FeatureEffectBase
import numpy as np


class ALE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        """
        ALE constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N,).
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.

        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )

        super(ALE, self).__init__(axis_limits)

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
        ), "Only fixed binning method is supported for now"

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
            centering:
                - If set to False, no centering will be applied.
                - If set to "zero_integral" or True, the integral of the feature effect will be set to zero.
                - If set to "zero_mean", the mean of the feature effect will be set to zero.
        """
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(s, binning_method)
            if centering is not False:
                self.norm_const[s] = self._compute_norm_const(s, method=centering)
            self.is_fitted[s] = True

    # TODO: add latex formula
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
        confidence_interval: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = False,
        scale_x: typing.Union[None, dict] = None,
        scale_y: typing.Union[None, dict] = None,
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
        """
        confidence_interval = helpers.prep_confidence_interval(confidence_interval)
        centering = helpers.prep_centering(centering)

        # hack to fit the feature if not fitted
        self.eval(
            feature, np.array([self.axis_limits[0, feature]]), centering=centering
        )

        fig, ax1, ax2 = vis.ale_plot(
            self.feature_effect["feature_" + str(feature)],
            self.eval,
            feature,
            centering=centering,
            error=confidence_interval,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        return fig, ax1, ax2

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
        RHALE constructor

        Args
        ---
            data (numpy.ndarray (N,D)): X matrix.
            model (Callable (N,D) -> (N,)): the black-box model.
            axis_limits (numpy.ndarray or None): axis limits of the data. If set to None, the axis limits will be auto-computed from the input data.

        Returns
        -------
        The function returns an instance of the RHALE class, which can be used to estimate the loss of a given input.

        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        axis_limits = (helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits)

        super(ALE, self).__init__(axis_limits)


    def _fit_feature(self, feature: int, binning_method) -> typing.Dict:

        # drop points outside of limits
        ind = np.logical_and(
            self.data[:, feature] >= self.axis_limits[0, feature],
            self.data[:, feature] <= self.axis_limits[1, feature],
        )
        data = self.data[ind, :]

        # assert binning_method is either "fixed" or an instance of the class binning_method.Fixed
        assert isinstance(binning_method, bm.Fixed) or binning_method == "fixed", (
            "binning_method must be either 'fixed' or an instance of the class binning_method.Fixed"
        )

        # bin estimation
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
        data_effect = utils.compute_local_effects(data, self.model, bin_est.limits, feature)

        # compute the bin effect
        dale_params = utils.compute_ale_params(data[:, feature], data_effect, bin_est.limits)
        dale_params["alg_params"] = "fixed"
        return dale_params

    def fit(self,
            features: typing.Union[int, str, list] = "all",
            binning_method="fixed",
            centering: typing.Union[bool, str] = "zero_integral",
            ) -> None:
        """Fit the model.

        Args
        ---
            features (int, str, list): the features to fit.
                - If set to "all", all the features will be fitted.
            centering (bool, str):
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

    def plot(
        self,
        feature: int = 0,
        confidence_interval: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = False,
        scale_x=None,
        scale_y=None,
        savefig=False,
    ):
        """

        Parameters
        ----------
        feature:
        confidence_interval:
        centering:
        scale_x:
        scale_y:
        savefig:
        """
        confidence_interval = helpers.prep_confidence_interval(confidence_interval)
        centering = helpers.prep_centering(centering)

        # hack to fit the feature if not fitted
        self.eval(feature, np.array([self.axis_limits[0, feature]]), centering=centering)

        vis.ale_plot(
            self.feature_effect["feature_" + str(feature)],
            self.eval,
            feature,
            centering=centering,
            error=confidence_interval,
            scale_x=scale_x,
            scale_y=scale_y,
            savefig=savefig,
        )
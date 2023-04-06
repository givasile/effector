import typing
import pythia.binning_methods
import pythia.utils as utils
import pythia.visualization as vis
import pythia.binning_methods as bm
import pythia.helpers as helpers
from pythia.fe_base import FeatureEffectBase
import numpy as np


class RHALE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable],
        axis_limits: typing.Union[None, np.ndarray] = None,
        data_effect: typing.Union[None, np.ndarray] = None,
    ):
        """
        Initializes DALE.

        Parameters
        ----------
        data: [N, D] np.array, X matrix
        model: Callable [N, D] -> [N,], prediction function
        model_jac: Callable [N, D] -> [N,D], jacobian function
        axis_limits: [2, D] np.ndarray or None, if None they will be auto computed from the data
        data_effect: [N, D] np.ndarray or None, if None they will be computed from the model_jac
        """
        # assertions
        assert data.ndim == 2
        assert (model_jac is not None) or (data_effect is not None)

        # setters
        self.model = model
        self.model_jac = model_jac
        self.data = data
        axis_limits = (helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits)

        super(RHALE, self).__init__(axis_limits)

        # init as None, it will get gradients after compile
        self.data_effect = None if data_effect is None else data_effect

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points.
        TODO add numerical approximation
        """
        if self.data_effect is None:
            self.data_effect = self.model_jac(self.data)
        else:
            # TODO add numerical approximation
            pass

    def _fit_feature(self, feature: int, binning_method) -> typing.Dict:
        """Fit a specific feature, using DALE.

        Parameters
        ----------
        feature: index of the feature
        binning_method: str or instance of appropriate binning class
        """
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
        bin_est = bm.find_limits(data, data_effect, feature, self.axis_limits, binning_method)
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
        dale_params = utils.compute_ale_params_from_data(data[:, feature], data_effect[:, feature], bin_est.limits)
        dale_params["alg_params"] = binning_method
        return dale_params

    def fit(self,
            features: typing.Union[int, str, list] = "all",
            binning_method="greedy",
            centering: typing.Union[bool, str] = "zero_integral",
            ) -> None:
        """Fit the model."""
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
        uncertainty: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = False,
        scale_x=None,
        scale_y=None,
        savefig=False,
    ):
        """

        Parameters
        ----------
        feature:
        uncertainty:
        centering:
        scale_x:
        scale_y:
        savefig:
        """
        uncertainty = helpers.prep_uncertainty(uncertainty)
        centering = helpers.prep_centering(centering)
        vis.ale_plot(
            self.feature_effect["feature_" + str(feature)],
            self.eval,
            feature,
            centering=centering,
            error=uncertainty,
            scale_x=scale_x,
            scale_y=scale_y,
            savefig=savefig,
        )
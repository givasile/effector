import typing
import pythia.utils as utils
import pythia.visualization as vis
import numpy as np
from pythia.fe_base import FeatureEffectBase
from pythia import helpers
from pythia import bin_estimation as be
import pythia

empty_symbol = 1e10


class ALE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        """
        Initializes ALE.

        Parameters
        ----------
        data: [N, D] np.array, X matrix
        model: Callable [N, D] -> [N,], prediction function
        axis_limits: [2, D] np.ndarray or None, if None they will be auto computed from the data
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

        # init as None, it will get gradients after compile
        self.data_effect = None

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points.
        TODO add numerical approximation
        """
        self.data_effect = np.ones_like(self.data) * empty_symbol

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        """Fit a specific feature, using ALE.

        Parameters
        ----------
        feat: index of the feature
        params: Dict, with fitting-specific parameters
            - "nof_bins": int (default 100), how many bins to create

        Returns
        -------

        """
        params = helpers.prep_ale_fit_params(params)

        if self.data_effect is None:
            self.compile()

        # drop points outside of limits
        ind = np.logical_and(
            self.data[:, feat] >= self.axis_limits[0, feat],
            self.data[:, feat] <= self.axis_limits[1, feat],
        )
        data = self.data[ind, :]


        # Compute data effect
        limits, dx = np.linspace(
            self.axis_limits[0, feat],
            self.axis_limits[1, feat],
            num=params["nof_bins"] + 1, endpoint=True, retstep=True
        )

        data_effect = utils.compute_local_effects_at_bin_limits(
            self.data, self.model, limits, feat
        )
        self.data_effect[:, feat] = data_effect
        data_effect = self.data_effect

        # bin estimation
        bin_est = be.Fixed(
                data, data_effect, feature=feat, axis_limits=self.axis_limits
            )
        bin_est.find(nof_bins=params["nof_bins"], min_points=None)

        dale_params = utils.compute_ale_params_from_data(
            data[:, feat], data_effect[:, feat], bin_est.limits
        )

        dale_params["alg_params"] = params
        return dale_params


    def fit(self,
            features: typing.Union[int, str, list] = "all",
            binning_method="fixed",
            normalize: bool = True,
            ) -> None:
        """Fit feature effect plot for the asked features

        Parameters
        ----------
        features: features to compute the normalization constant
            - "all", all features
            - int, the index of the feature
            - list, list of indexes of the features
        binning_method: dictionary with method-specific parameters for fitting the FE plots
        normalize: bool, whether to compute the normalization constants
        """

        # if binning_method is a string -> make it a class
        if isinstance(binning_method, str):
            assert binning_method in ["fixed", "greedy", "dynamic_programming"]
            if binning_method == "fixed":
                tmp = pythia.binning_methods.Fixed()
            elif binning_method == "greedy":
                tmp = pythia.binning_methods.Greedy()
            else:
                tmp = pythia.binning_methods.DynamicProgramming()
            binning_method = tmp

        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(s, binning_method)
            if normalize:
                self.norm_const[s] = self._compute_norm_const(s)
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
        error: typing.Union[None, str] = "std",
        scale_x=None,
        scale_y=None,
        savefig=False,
    ):
        """

        Parameters
        ----------
        s
        error:
        scale_x:
        scale_y:
        savefig:
        """
        vis.ale_plot(
            self.feature_effect["feature_" + str(feature)],
            self.eval,
            feature,
            error=error,
            scale_x=scale_x,
            scale_y=scale_y,
            savefig=savefig,
        )
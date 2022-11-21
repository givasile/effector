import typing
import pythia.utils as utils
import pythia.visualization as vis
import pythia.bin_estimation as be
import pythia.helpers as helpers
from pythia.fe_base import FeatureEffectBase
import numpy as np


class DALE(FeatureEffectBase):
    def __init__(self,
                 data: np.ndarray,
                 model: callable,
                 model_jac: callable,
                 axis_limits: typing.Union[None, np.ndarray] = None):
        """
        Initializes DALE.

        Parameters
        ----------
        data: [N, D] np.array, X matrix
        model: Callable [N, D] -> [N,], prediction function
        model_jac: Callable Callable [N, D] -> [N,], jacobian function
        axis_limits: [2, D] np.ndarray or None, if None they will be auto computed from the data
        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.model_jac = model_jac
        self.data = data
        self.D = data.shape[1]
        axis_limits = helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        super(DALE, self).__init__(axis_limits)

        # init as None, it will get gradients after compile
        self.data_effect = None

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points.
        TODO add numerical approximation
        """
        if self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        else:
            # TODO add numerical approximation
            pass

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        """Fit a specific feature, using DALE.

        Parameters
        ----------
        feat: index of the feature
        params: Dict, with fitting-specific parameters
            - "bin_method": in ["fixed", "greedy", "dp"], which method to use
            - "nof_bins": int (default 100), how many bins to create -> important only if "fixed" is used
            - "max_nof_bins: int (default 20), max number of bins -> important only if "greedy" or "dp" is used
            - "min_points_per_bin" int (default 10), min numbers per bin -> important in all cases

        Returns
        -------

        """
        params = helpers.prep_dale_fit_params(params)

        if self.data_effect is None:
            self.compile()

        # drop points outside of limits
        if self.axis_limits is not None:
            ind = np.logical_and(self.data[:, feat] > self.axis_limits[0, feat],
                                 self.data[:, feat] < self.axis_limits[1, feat])
            data = self.data[ind, :]
            data_effect = self.data_effect[ind, :]
        else:
            data = self.data
            data_effect = self.data_effect

        # bin estimation
        if params["bin_method"] == "fixed":
            bin_est = be.Fixed(data,
                               data_effect,
                               feature=feat,
                               axis_limits=self.axis_limits)
            bin_est.find(nof_bins=params["nof_bins"],
                         min_points=params["min_points_per_bin"])

        elif params["bin_method"] == "greedy":
            bin_est = be.Greedy(data, data_effect, feature=feat, axis_limits=self.axis_limits)
            bin_est.find(min_points=params["min_points_per_bin"],
                         n_max=params["nof_bins"])
        elif params["bin_method"] == "dp":
            bin_est = be.DP(data,
                            data_effect,
                            feature=feat,
                            axis_limits=self.axis_limits)
            bin_est.find(min_points=params["min_points_per_bin"],
                         k_max=params["max_nof_bins"])

        # stats per bin
        assert bin_est.limits is not False, "Impossible to compute bins with enough points for feature " + str(feat + 1) + " and binning strategy: " + params["bin_method"] + ". Change bin strategy or the parameters of the method"
        dale_params = utils.compute_ale_parameters(data[:, feat],
                                                   data_effect[:, feat],
                                                   bin_est.limits)

        dale_params["limits"] = bin_est.limits
        dale_params["alg_params"] = params
        return dale_params

    def _eval_unnorm(self, x: np.ndarray, s: int, uncertainty: bool = False):
        params = self.feature_effect["feature_" + str(s)]
        y = utils.compute_accumulated_effect(x,
                                             limits=params["limits"],
                                             bin_effect=params["bin_effect"],
                                             dx=params["dx"])
        if uncertainty:
            std = utils.compute_accumulated_effect(x,
                                                   limits=params["limits"],
                                                   bin_effect=np.sqrt(params["bin_variance"]),
                                                   dx=params["dx"],
                                                   square=False)
            std_err = utils.compute_accumulated_effect(x,
                                                             limits=params["limits"],
                                                             bin_effect=np.sqrt(params["bin_estimator_variance"]),
                                                             dx=params["dx"],
                                                             square=False)

            return y, std, std_err
        else:
            return y

    def plot(self,
             s: int = 0,
             error: str = "std",
             scale_x=None,
             scale_y=None,
             savefig=False):
        """

        Parameters
        ----------
        s
        error:
        scale_x
        scale_y
        gt
        gt_bins
        block
        savefig
        """
        vis.ale_plot(self.feature_effect["feature_" + str(s)],
                     self.eval,
                     s,
                     error=error,
                     scale_x=scale_x,
                     scale_y=scale_y,
                     savefig=savefig)


class DALEGroundTruth(FeatureEffectBase):
    def __init__(self, mean, mean_int, var, var_int, axis_limits):
        super(DALEGroundTruth, self).__init__(axis_limits)
        self.mean = mean
        self.mean_int = mean_int
        self.var = var
        self.var_int = var_int

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        return {}

    def _eval_unnorm(self, x: np.ndarray, s: int, uncertainty: bool = False):
        if not uncertainty:
            return self.mean_int(x)
        else:
            return self.mean_int(x), self.var_int(x), None

    def plot(self, s: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature
        """
        # getters
        x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], nof_points)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self._eval_unnorm(x, s)
        vis.plot_1D(x, y, title="Ground-truth ALE for feature %d" % (s+1))


class DALEBinsGT(FeatureEffectBase):
    def __init__(self, mean, var, axis_limits):
        super(DALEBinsGT, self).__init__(axis_limits)
        self.mean = mean
        self.var = var

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        params = helpers.prep_dale_fit_params(params)

        # bin estimation
        if params["bin_method"] == "fixed":
            bin_est = be.FixedGT(self.mean, self.var, self.axis_limits, feature=feat)
            bin_est.find(nof_bins=params["nof_bins"],
                         min_points=params["min_points_per_bin"])
        elif params["bin_method"] == "greedy":
            bin_est = be.GreedyGT(self.mean, self.var, self.axis_limits, feature=feat)
        elif params["bin_method"] == "dp":
            bin_est = be.DPGT(self.mean, self.var, self.axis_limits, feature=feat)

        # stats per bin
        dale_params = utils.compute_bin_statistics_gt(self.mean, self.var, bin_est.limits)
        dale_params["limits"] = bin_est.limits
        return dale_params

    def _eval_unnorm(self, x: np.ndarray, s: int, uncertainty: bool = False):
        params = self.feature_effect["feature_" + str(s)]
        y = utils.compute_accumulated_effect(x,
                                             limits=params["limits"],
                                             bin_effect=params["bin_effect"],
                                             dx=params["dx"])
        if uncertainty:
            var = utils.compute_accumulated_effect(x,
                                                   limits=params["limits"],
                                                   bin_effect=params["bin_variance"],
                                                   dx=params["dx"],
                                                   square=True)
            return y, var
        else:
            return y

    def plot(self, s: int, normalized: bool = True, nof_points: int = 30) -> None:
        x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], nof_points)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self._eval_unnorm(x, s)
        vis.plot_1D(x, y, title="ALE GT Bins for feature %d" % (s+1))

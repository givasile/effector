import typing
import feature_effect.utils as utils
import feature_effect.visualization as vis
import feature_effect.bin_estimation as be
import feature_effect.helpers as helpers
from feature_effect.pdp import FeatureEffectBase
import numpy as np


class DALEGroundTruth(FeatureEffectBase):
    def __init__(self, mean, mean_int, var, var_int, axis_limits):
        super(DALEGroundTruth, self).__init__(axis_limits)
        self.mean = mean
        self.mean_int = mean_int
        self.var = var
        self.var_int = var_int

    def fit_feature(self, s: int, alg_params: typing.Dict = None) -> typing.Dict:
        return {}

    def eval_unnorm(self, x: np.ndarray, s: int, uncertainty: bool = False):
        if not uncertainty:
            return self.mean_int(x)
        else:
            return self.mean_int(x), self.var_int(x)

    def plot(self, s: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature
        """
        # getters
        x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], nof_points)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self.eval_unnorm(x, s)
        vis.plot_1D(x, y, title="Ground-truth ALE for feature %d" % (s+1))


class DALEBinsGT(FeatureEffectBase):
    def __init__(self, mean, var, axis_limits):
        super(DALEBinsGT, self).__init__(axis_limits)
        self.mean = mean
        self.var = var

    def fit_feature(self, s: int, alg_params: typing.Dict = None) -> typing.Dict:
        alg_params = helpers.prep_dale_fit_params(alg_params)

        # bin estimation
        if alg_params["bin_method"] == "fixed":
            bin_est = be.FixedSizeGT(self.axis_limits, feature=s)
            bin_est.solve(min_points = alg_params["min_points_per_bin"],
                          K = alg_params["nof_bins"])
        elif alg_params["bin_method"] == "greedy":
            bin_est = be.GreedyGroundTruth(self.mean, self.var, self.axis_limits, feature=s)
        elif alg_params["bin_method"] == "dp":
            bin_est = be.DPGroundTruth(self.mean, self.var, self.axis_limits, feature=s)
        self.bin_est = bin_est

        # stats per bin
        dale_params = utils.compute_bin_statistics_gt(self.mean, self.var, bin_est.limits)
        dale_params["limits"] = bin_est.limits
        return dale_params


    def eval_unnorm(self, x: np.ndarray, s: int, uncertainty: bool = False):
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
            y = self.eval_unnorm(x, s)
        vis.plot_1D(x, y, title="ALE GT Bins for feature %d" % (s+1))

    # def plot_local_effects(self, s: int = 0, limits=True, block=False):
    #     xs = self.data[:, s]
    #     data_effect = self.data_effect[:, s]
    #     if limits:
    #         limits = self.bin_est.limits
    #     vis.plot_local_effects(s, xs, data_effect, limits, block)

        # """Plot the s-th feature
        # """
        # # getters
        # x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], nof_points)
        # if normalized:
        #     y = self.eval(x, s)
        # else:
        #     y = self.eval_unnorm(x, s)
        # vis.plot_1D(x, y, title="Ground-truth ALE for feature %d" % (s+1))



def compute_dale_parameters(data: np.ndarray, data_effect: np.ndarray, feature: int, method: str, alg_params: typing.Dict) -> typing.Dict:
    """Compute the DALE dale_params for a single feature.

    Performs all actions to compute the dale_params that are required for
    the s-th feature DALE plot

    Parameters
    ----------
    data: ndarray
      The training-set points, shape: (N,D)
    data_effect: ndarray
      The feature effect contribution of the training-set points, shape: (N,D)
    feature: int
      Index of the feature of interest
    method: int
      Number of bins
    alg_params: Dict

    Returns
    -------
    dale_params: Dict
      - limits: ndarray (K+1,) with the bin limits
      - bin_effect_interp: ndarray (K,) with the effect of each bin
      - dx: float, bin length
      - z: float, the normalizer
    """
    assert method in ["fixed-size", "variable-size"]
    if method == "fixed-size":
        # get hyperparameters
        K = alg_params["nof_bins"] if "nof_bins" in alg_params.keys() else 30
        min_points_per_bin = alg_params["min_points_per_bin"] if "min_points_per_bin" in alg_params.keys() else 10

        # estimate bins
        limits = utils.create_fix_size_bins(data[:, feature], K)

        # compute dale params
        dale_params = utils.compute_fe_parameters(data[:, feature], data_effect[:, feature], limits, min_points_per_bin)

        # store dale parameters
        dale_params["method"] = method
        dale_params["min_points_per_bin"] = min_points_per_bin
        dale_params["nof_bins"] = K
    elif method == "variable-size":
        # get hyperparameters
        K = alg_params["max_nof_bins"] if "max_nof_bins" in alg_params.keys() else 30
        min_points_per_bin = alg_params["min_points_per_bin"] if "min_points_per_bin" in alg_params.keys() else 10

        # estimate bins
        if "limits" in alg_params.keys():
            limits = alg_params["limits"]
        else:
            bin_estimator = be.DP(data, data_effect, feature)
            limits = bin_estimator.solve(min_points_per_bin, K)

        # compute dale parameters
        dale_params = utils.compute_fe_parameters(data[:, feature], data_effect[:, feature], limits, min_points_per_bin)
        if "limits" not in alg_params.keys():
            dale_params["bin_estimator"] = bin_estimator

        # store dale parameters
        dale_params["method"] = method
        dale_params["max_nof_bins"] = K
        dale_params["min_points_per_bin"] = min_points_per_bin
    return dale_params


class DALE:
    def __init__(self, data: np.ndarray, model: typing.Callable, model_jac: typing.Union[typing.Callable, None] = None):
        self.data = data
        self.model = model
        self.model_jac = model_jac

        self.data_effect = None
        self.feature_effect = None
        self.dale_params = None

    @staticmethod
    def _dale_func(params):
        """Returns the DALE function on for the s-th feature.

        Parameters
        ----------
        params: Dict
          Dictionary with all parameters required to create a 1D DALE function

        Returns
        -------
        dale_function: Callable
          The dale_function on the s-th feature
        """
        def dale_function(x):
            y = utils.compute_accumulated_effect(x, limits=params["limits"], bin_effect=params["bin_effect"],
                                                 dx=params["dx"])
            y -= params["z"]
            estimator_var = utils.compute_accumulated_effect(x,
                                                   limits=params["limits"],
                                                   bin_effect=params["bin_estimator_variance"],
                                                   dx=params["dx"],
                                                   square=True)

            var = utils.compute_accumulated_effect(x,
                                                   limits=params["limits"],
                                                   bin_effect=params["bin_variance"],
                                                   dx=params["dx"],
                                                   square=True)
            return y, estimator_var, var
        return dale_function

    def compile(self):
        if self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        else:
            # TODO add numerical approximation
            pass

    def fit(self, features: typing.Union[str, list] = "all", method="fixed-size", alg_params={}):
        assert method in ["fixed-size", "variable-size"]
        assert features == "all" or type(features) == list

        if features == "all":
            features = [i for i in range(self.data.shape[1])]

        if self.data_effect is None:
            self.compile()

        # fit and store dale parameters
        funcs = {}
        dale_params = {}
        for s in features:
            dale_params["feature_" + str(s)] = compute_dale_parameters(self.data, self.data_effect, s, method, alg_params)
            funcs["feature_" + str(s)] = self._dale_func(dale_params["feature_" + str(s)])

        # TODO change it to append, instead of overwriting
        self.feature_effect = funcs
        self.dale_params = dale_params

    def eval(self, x: np.ndarray, s: int):
        return self.feature_effect["feature_" + str(s)](x)

    def plot(self, s: int = 0, error="standard error", block=False, gt=None, gt_bins=None, savefig=False):
        title = "DALE: Effect of feature %d" % (s + 1)
        vis.feature_effect_plot(self.dale_params["feature_"+str(s)],
                                self.eval,
                                s,
                                error=error,
                                min_points_per_bin=self.dale_params["feature_"+str(s)]["min_points_per_bin"],
                                title=title,
                                block=block,
                                gt=gt,
                                gt_bins=gt_bins,
                                savefig=savefig)

    def plot_local_effects(self, s: int = 0, limits=True, block=False):
        xs = self.data[:, s]
        data_effect = self.data_effect[:, s]
        if limits:
            limits = self.dale_params["feature_" + str(s)]["limits"]
        vis.plot_local_effects(s, xs, data_effect, limits, block)

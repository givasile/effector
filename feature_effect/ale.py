import typing
import feature_effect.utils as utils
import feature_effect.visualization as vis
import numpy as np


def compute_ale_parameters(data: np.ndarray, model: np.ndarray, feature: int, alg_params: typing.Dict) -> typing.Dict:
    """Compute the ALE parameters for the s-th feature

    Performs all actions to compute the parameters that are required for
    the s-th feature ALE effect

    Parameters
    ----------
    data
    model
    feature
    k

    Returns
    -------

    """
    K = alg_params["nof_bins"] if "nof_bins" in alg_params.keys() else 30
    min_points_per_bin = alg_params["min_points_per_bin"] if "min_points_per_bin" in alg_params.keys() else 10
    limits = utils.create_fix_size_bins(data[:, feature], K)

    # compute local data effects, based on the bins
    data_effect = utils.compute_data_effect(data, model, limits, feature)

    # compute parameters
    ale_params = utils.compute_fe_parameters(data[:, feature], data_effect, limits, min_points_per_bin)
    ale_params["method"] = "fixed-size"
    ale_params["min_points_per_bin"] = min_points_per_bin
    return ale_params

class ALE:
    def __init__(self, data: np.ndarray, model: typing.Callable):
        self.data = data
        self.model = model

        self.data_effect = None
        self.feature_effect = None
        self.ale_params = None

    @staticmethod
    def _ale_func(params):
        """Returns the DALE function on for the s-th feature.

        Parameters
        ----------
        points: ndarray
          The training-set points, shape: (N,D)
        f: ndarray
          The feature effect contribution of the training-set points, shape: (N,)
        s: int
          Index of the feature of interest
        k: int
          Number of bins

        Returns
        -------
        dale_function: Callable
          The dale_function on the s-th feature
        parameters: Dict
          - limits: ndarray (K+1,) with the bin limits
          - bin_effects: ndarray (K,) with the effect of each bin
          - dx: float, bin length
          - z: float, the normalizer

        """
        def ale_function(x):
            y = utils.compute_accumulated_effect(x,
                                                 limits=params["limits"],
                                                 bin_effect=params["bin_effect"],
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
        return ale_function

    def compile(self):
        pass

    def fit(self, features: typing.Union[str, list] = "all", alg_params={}):
        assert features == "all" or type(features) == list
        if features == "all":
            features = [i for i in range(self.data.shape[1])]

        # fit and store dale parameters
        funcs = {}
        ale_params = {}
        for s in features:
            ale_params["feature_" + str(s)] = compute_ale_parameters(self.data,
                                                                     self.model,
                                                                     s,
                                                                     alg_params)
            funcs["feature_" + str(s)] = self._ale_func(ale_params["feature_" + str(s)])

        # TODO change it to append, instead of overwriting
        self.feature_effect = funcs
        self.ale_params = ale_params

    def eval(self, x: np.ndarray, s: int):
        return self.feature_effect["feature_" + str(s)](x)

    def plot(self, s: int = 0, error="standard error", block=False, gt=None, gt_bins=None, savefig=None):
        title = "ALE: Effect of feature %d" % (s + 1)
        vis.feature_effect_plot(self.ale_params["feature_" + str(s)],
                                self.eval,
                                s,
                                error=error,
                                min_points_per_bin=self.ale_params["feature_" + str(s)]["min_points_per_bin"],
                                title=title,
                                block=block,
                                gt=gt,
                                gt_bins=gt_bins,
                                savefig=savefig)

    def plot_local_effects(self, s: int = 0, K: int = 10, limits=True, block=False):
        data = self.data
        model = self.model

        # compute local effects
        bin_limits = utils.create_fix_size_bins(data[:, s], K)

        # compute local data effects, based on the bins
        data_effect = utils.compute_data_effect(data, model, bin_limits, s)

        # plot
        xs = self.data[:, s]
        if limits:
            limits = bin_limits
        vis.plot_local_effects(s, xs, data_effect, limits, block)

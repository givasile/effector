import typing
import feature_effect.utils as utils
import feature_effect.visualization as vis
import feature_effect.bin_estimation as be
import numpy as np


def compute_dale_parameters(data: np.ndarray, data_effect: np.ndarray, feature: int, k: int,
                            min_points_per_bin: int, method: str) -> typing.Dict:
    """Compute the DALE parameters for a single feature.

    Performs all actions to compute the parameters that are required for
    the s-th feature DALE plot

    Parameters
    ----------
    data: ndarray
      The training-set points, shape: (N,D)
    data_effect: ndarray
      The feature effect contribution of the training-set points, shape: (N,D)
    feature: int
      Index of the feature of interest
    k: int
      Number of bins

    Returns
    -------
    parameters: Dict
      - limits: ndarray (K+1,) with the bin limits
      - bin_effect_interp: ndarray (K,) with the effect of each bin
      - dx: float, bin length
      - z: float, the normalizer
    """
    assert method in ["fix-size", "variable-size"]
    # create bins
    if method == "fix-size":
        limits, dx = utils.create_fix_size_bins(data[:, feature], k)
        parameters = utils.compute_fe_parameters(data[:, feature], data_effect[:, feature], limits, min_points_per_bin)
    elif method == "variable-size":
        bin_estimator = be.BinEstimatorDP(data, data_effect, None, feature, k)
        limits, dx = bin_estimator.solve_dp(min_points_per_bin)
        parameters = utils.compute_fe_parameters(data[:, feature], data_effect[:, feature], limits, min_points_per_bin)
        parameters["bin_estimator"] = bin_estimator

    return parameters


def dale(x: np.ndarray, data: np.ndarray, data_effect: np.ndarray, feature: int, k: int = 100,
         min_points_per_bin: int = 10, method: str = "fix-size"):
    """Compute DALE at points x.

    Functional implementation of DALE at a single feature. Computation is
    made on-the-fly.

    Parameters
    ----------
    x: ndarray, shape (N,)
      The points we want to evaluate the feature effect plot
    data: ndarray
      The training-set points, shape: (N,D)
    data_effect: ndarray
      The feature effect contribution of the training-set points, shape: (N,)
    feature: int
      Index of the feature of interest
    k: int
      Number of bins

    Returns
    -------
    y: ndarray, shape (N,)
      Feature effect evaluation at points x.
    """
    parameters = compute_dale_parameters(data, data_effect, feature, k, min_points_per_bin, method)
    y = utils.compute_accumulated_effect(x,
                                         limits=parameters["limits"],
                                         bin_effect=parameters["bin_effect"],
                                         dx=parameters["dx"][0])
    y -= parameters["z"]
    var = utils.compute_accumulated_effect(x,
                                           limits=parameters["limits"],
                                           bin_effect=parameters["bin_estimator_variance"],
                                           dx=parameters["dx"][0],
                                           square=True)
    return y, var


class DALE:
    def __init__(self, data: np.ndarray, model: typing.Callable, model_jac: typing.Union[typing.Callable, None] = None):
        self.data = data
        self.model = model
        self.model_jac = model_jac

        self.data_effect = None
        self.feature_effect = None
        self.parameters = None

    @staticmethod
    def _dale_func(points, point_effects, s, k, min_points_per_bin, method):
        """Returns the DALE function on for the s-th feature.

        Parameters
        ----------
        points: ndarray
          The training-set points, shape: (N,D)
        point_effects: ndarray
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
        parameters = compute_dale_parameters(points, point_effects, s, k, min_points_per_bin, method)

        def dale_function(x):
            y = utils.compute_accumulated_effect(x,
                                                 limits=parameters["limits"],
                                                 bin_effect=parameters["bin_effect"],
                                                 dx=parameters["dx"][0])
            y -= parameters["z"]
            var = utils.compute_accumulated_effect(x,
                                                   limits=parameters["limits"],
                                                   bin_effect=parameters["bin_estimator_variance"],
                                                   dx=parameters["dx"][0],
                                                   square=True)
            return y, var
        return dale_function, parameters

    def compile(self):
        if self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        else:
            # TODO add numerical approximation
            pass

    def fit(self, features: list, k: int, min_points_per_bin: int = 10, method="fix-size"):
        if self.data_effect is None:
            self.compile()

        # (b) compute DALE function for the features
        funcs = {}
        parameters = {}
        for s in features:
            func, param = self._dale_func(self.data, self.data_effect, s, k, min_points_per_bin, method)
            funcs["feature_" + str(s)] = func
            parameters["feature_" + str(s)] = param

        self.feature_effect = funcs
        self.parameters = parameters

    def eval(self, x: np.ndarray, s: int):
        func = self.feature_effect["feature_" + str(s)]
        y, var = func(x)
        return y, var

    def plot(self, s: int, block=True, gt=None, gt_bins=None):
        title = "DALE: Effect of feature %d" % (s + 1)
        vis.feature_effect_plot(self.parameters["feature_"+str(s)], self.eval, s, title=title, block=block, gt=gt, gt_bins=gt_bins)

import typing
import feature_effect.utils as utils
import feature_effect.visualization as vis
import numpy as np


def compute_ale_parameters(data: np.ndarray, model: np.ndarray, feature: int, k: int) -> typing.Dict:
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
    # create bins
    limits, dx = utils.create_fix_size_bins(data[:, feature], k)

    # compute local data effects, based on the bins
    data_effect = utils.compute_data_effect(data, model, limits, dx, feature)

    # compute parameters
    parameters = utils.compute_fe_parameters(data[:, feature], data_effect, limits, dx)
    return parameters


def ale(x: np.ndarray, data: np.ndarray, model: typing.Callable, feature: int, k: int = 100):
    """Compute ALE at points x.

    Functional implementation of DALE at the s-th feature. Computation is
    made on-the-fly.

    Parameters
    ----------
    x: ndarray, shape (N,)
      The points to evaluate DALE on
    data: ndarray, shape (N,D)
      The training set
    model: Callable
    feature: int
      Index of the feature
    k: int
      number of bins

    Returns
    -------

    """
    # compute
    parameters = compute_ale_parameters(data, model, feature, k)
    y = utils.compute_accumulated_effect(x,
                                         limits=parameters["limits"],
                                         bin_effect=parameters["bin_effect"],
                                         dx=parameters["dx"])
    y -= parameters["z"]
    var = utils.compute_accumulated_effect(x,
                                           limits=parameters["limits"],
                                           bin_effect=parameters["bin_estimator_variance"],
                                           dx=parameters["dx"],
                                           square=True)
    return y, var


class ALE:
    def __init__(self, data: np.ndarray, model: typing.Callable):
        self.data = data
        self.model = model

        self.data_effect = None
        self.feature_effect = None
        self.parameters = None

    @staticmethod
    def _ale_func(points, f, s, k):
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
        parameters = compute_ale_parameters(points, f, s, k)

        def ale_function(x):
            y = utils.compute_accumulated_effect(x,
                                                 limits=parameters["limits"],
                                                 bin_effect=parameters["bin_effect"],
                                                 dx=parameters["dx"])
            y -= parameters["z"]
            var = utils.compute_accumulated_effect(x,
                                                   limits=parameters["limits"],
                                                   bin_effect=parameters["bin_estimator_variance"],
                                                   dx=parameters["dx"],
                                                   square=True)
            return y, var
        return ale_function, parameters

    def compile(self):
        pass

    def fit(self, features: list, k: int):
        funcs = {}
        parameters = {}
        for s in features:
            func, param = self._ale_func(self.data, self.model, s, k)
            funcs["feature_" + str(s)] = func
            parameters["feature_" + str(s)] = param

        self.feature_effect = funcs
        self.parameters = parameters

    def eval(self, x: np.ndarray, s: int):
        func = self.feature_effect["feature_" + str(s)]
        return func(x)

    def plot(self, s: int, block=False, gt=None):
        title = "ALE: Effect of feature %d" % (s + 1)
        vis.feature_effect_plot(self.parameters["feature_" + str(s)], self.eval, s, title=title, block=block, gt=gt)

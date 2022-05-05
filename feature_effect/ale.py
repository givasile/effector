import numpy as np
import typing
import feature_effect.utils as utils
import numpy as np
import matplotlib.pyplot as plt


def compute_ale_parameters(points: np.ndarray, f: np.ndarray, s: int, k: int):
    """Compute the ALE parameters for the s-th feature

    Performs all actions to compute the parameters that are required for
    the s-th feature DALE effect

    Parameters
    ----------
    points
    f
    s
    k

    Returns
    -------

    """
    # create bins
    limits, dx = utils.create_bins(points[:, s], k)

    # compute bin effects
    point_effects = utils.compute_data_effect(points, f, limits, dx, s)

    # compute effect on each bin
    bin_effects, points_per_bin = utils.compute_bin_effect_mean(points[:, s], point_effects, limits)

    # fill bins with NaN values
    bin_effects = utils.fill_nans(bin_effects)

    # compute Z
    z = utils.compute_normalizer(points[:, s], limits, bin_effects, dx)

    parameters = {"limits": limits,
                  "dx": dx,
                  "bin_effect": bin_effects,
                  "z": z}
    return parameters


def ale(x, data, model, s, k=100):
    """Compute ALE at points x.

    Functional implementation of DALE at the s-th feature. Computation is
    made on-the-fly.

    Parameters
    ----------
    x: ndarray, shape (N,)
      The points to evaluate DALE on
    s: int
      Index of the feature
    k: int
      number of bins
    data: ndarray, shape (N,D)
      The training set
    effects: ndarray, shape (N,D)
      The training set

    Returns
    -------

    """
    # compute
    parameters = compute_ale_parameters(data, model, s, k)
    y = utils.compute_accumulated_effect(x,
                                         limits=parameters["limits"],
                                         bin_effect=parameters["bin_effect"],
                                         dx=parameters["dx"]) - parameters["z"]
    return y


class ALE:
    def __init__(self, data: np.ndarray, model: typing.Callable):
        self.data = data
        self.model = model

        self.effects = None
        self.funcs = None
        self.parameters = None

    @staticmethod
    def create_ale_function(points, f, s, k):
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

        def dale_function(x):
            y = utils.compute_accumulated_effect(x,
                                                 limits=parameters["limits"],
                                                 bin_effect=parameters["bin_effect"],
                                                 dx=parameters["dx"])
            y -= parameters["z"]
            return y

        return dale_function, parameters

    def compile(self):
        pass

    def fit(self, features: list, k: int):
        funcs = {}
        parameters = {}
        for s in features:
            func, param = self.create_ale_function(self.data, self.model, s, k)
            funcs["feature_" + str(s)] = func
            parameters["feature_" + str(s)] = param

        self.funcs = funcs
        self.parameters = parameters

    def eval(self, x: np.ndarray, s: int):
        func = self.funcs["feature_" + str(s)]
        return func(x)

    def plot(self, s: int, block=False):
        params = self.parameters["feature_" + str(s)]
        x = np.linspace(params["limits"][0] - .01, params["limits"][-1] + .01, 10000)
        y = self.eval(x, s)
        plt.figure()
        plt.title("ALE plot for feature %d" % (s+1))
        plt.plot(x, y, "b-")
        if block is False:
            plt.show(block=False)
        else:
            plt.show()


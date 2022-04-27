import typing
import mdale.utils as utils
import numpy as np
import matplotlib.pyplot as plt


def compute_dale_parameters(points: np.ndarray, point_effects: np.ndarray, s: int, k: int):
    """Compute the DALE parameters for a single feature.

    Performs all actions to compute the parameters that are required for
    the s-th feature DALE plot

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
    parameters: Dict
      - limits: ndarray (K+1,) with the bin limits
      - bin_effects: ndarray (K,) with the effect of each bin
      - dx: float, bin length
      - z: float, the normalizer
    """
    points = points[:, s]
    point_effects = point_effects[:, s]

    # create bins
    limits, dx = utils.create_bins(points, k)

    # compute mean effect on each bin
    bin_effects = utils.compute_bin_effects(points, point_effects, limits)

    # compute effect variance in each bin
    bin_estimator_variance = utils.compute_bin_estimator_variance(points, point_effects, limits, bin_effects)

    # fill bins with NaN values
    bin_effects = utils.fill_nans(bin_effects)

    # fill bins with NaN values
    bin_estimator_variance = utils.fill_nans(bin_estimator_variance)

    # compute Z
    z = utils.compute_normalizer(points, limits, bin_effects, dx)

    parameters = {"limits": limits,
                  "dx": dx,
                  "bin_effects": bin_effects,
                  "bin_estimator_variance": bin_estimator_variance,
                  "z": z}
    return parameters


def dale(x: np.ndarray, points: np.ndarray, point_effects: np.ndarray, s: int, k: int = 100):
    """Compute DALE at points x.

    Functional implementation of DALE at a single feature. Computation is
    made on-the-fly.

    Parameters
    ----------
    x: ndarray, shape (N,)
      The points we want to evaluate the feature effect plot
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
    y: ndarray, shape (N,)
      Feature effect evaluation at points x.
    """
    parameters = compute_dale_parameters(points, point_effects, s, k)
    y = utils.compute_accumulated_effect(x,
                                         limits=parameters["limits"],
                                         bin_effects=parameters["bin_effects"],
                                         dx=parameters["dx"])
    y -= parameters["z"]
    var = utils.compute_accumulated_effect(x,
                                           limits=parameters["limits"],
                                           bin_effects=parameters["bin_estimator_variance"],
                                           dx=parameters["dx"]**2)
    return y, var


class DALE:
    def __init__(self, points: np.ndarray, f: typing.Callable, f_der: typing.Union[typing.Callable, None] = None):
        self.points = points
        self.f = f
        self.f_der = f_der
        self.effects = None
        self.funcs = None
        self.parameters = None

    @staticmethod
    def create_dale_function(points, point_effects, s, k):
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
        parameters = compute_dale_parameters(points, point_effects, s, k)

        def dale_function(x):
            y = utils.compute_accumulated_effect(x,
                                                 limits=parameters["limits"],
                                                 bin_effects=parameters["bin_effects"],
                                                 dx=parameters["dx"])
            y -= parameters["z"]
            var = utils.compute_accumulated_effect(x,
                                                   limits=parameters["limits"],
                                                   bin_effects=parameters["bin_estimator_variance"],
                                                   dx=parameters["dx"] ** 2)
            return y, var

        return dale_function, parameters

    def compile(self):
        if self.f_der is not None:
            self.effects = self.f_der(self.points)
        else:
            # TODO add numerical approximation
            pass

    def fit(self, features: list, k: int):
        if self.effects is None:
            self.compile()

        # (b) compute DALE function for the features
        funcs = {}
        parameters = {}
        for s in features:
            func, param = self.create_dale_function(self.points, self.effects, s, k)
            funcs["feature_" + str(s)] = func
            parameters["feature_" + str(s)] = param

        self.funcs = funcs
        self.parameters = parameters

    def evaluate(self, x: np.ndarray, s: int):
        func = self.funcs["feature_" + str(s)]
        y, var = func(x)
        return y, var

    def plot(self, s: int, block=True):
        params = self.parameters["feature_" + str(s)]
        x = np.linspace(params["limits"][0] - .01, params["limits"][-1] + .01, 10000)
        y, var = self.evaluate(x, s)
        plt.figure()
        plt.title("DALE plot for feature %d" % (s+1))
        plt.plot(x, y, "b-")
        plt.fill_between(x, y - np.sqrt(var), y + np.sqrt(var), color='gray', alpha=0.4)
        if block is False:
            plt.show(block=False)
        else:
            plt.show()


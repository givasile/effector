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

    # list with points
    list_with_bin_points, list_with_bin_points_effects, is_bin_empty = utils.allocate_points_in_bins(points, point_effects, limits)

    # compute mean effect on each bin
    bin_effects_with_nans = utils.compute_bin_effects(points, point_effects, limits)

    # compute effect variance in each bin
    bin_estimator_variance_with_nans = utils.compute_bin_estimator_variance(points, point_effects, limits, bin_effects_with_nans)

    # fill bins with NaN values
    bin_effects = utils.fill_nans(bin_effects_with_nans)

    # fill bins with NaN values
    bin_estimator_variance = utils.fill_nans(bin_estimator_variance_with_nans)

    # first empty bin
    first_empty_bin = utils.find_first_nan_bin(bin_effects_with_nans)

    # compute Z
    z = utils.compute_normalizer(points, limits, bin_effects, dx)

    parameters = {"limits": limits,
                  "dx": dx,
                  "list_with_bin_points": list_with_bin_points,
                  "list_with_bin_points_effects": list_with_bin_points_effects,
                  "is_bin_empty": is_bin_empty,
                  "bin_effects": bin_effects,
                  "bin_effects_with_nans": bin_effects_with_nans,
                  "bin_estimator_variance": bin_estimator_variance,
                  "bin_estimator_variance_with_nans": bin_estimator_variance_with_nans,
                  "z": z,
                  "first_empty_bin": first_empty_bin}
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
                                           dx=parameters["dx"],
                                           with_squares=True)
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
                                                   dx=parameters["dx"],
                                                   with_squares=True)
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

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

        fig.suptitle("Effect of feature %d" % (s+1))

        # first subplot
        ax1.set_title("Plot")
        ax1.plot(x, y, "b-", label="feature effect")

        first_empty = params["first_empty_bin"]
        if first_empty is not None:
            ax1.axvspan(xmin=params["limits"][first_empty], xmax=x[-1], ymin=np.min(y), ymax=np.max(y), alpha=.2, color="red", label="not-trusted-area")

        ax1.fill_between(x, y - np.sqrt(var), y + np.sqrt(var), color='green', alpha=0.8, label="standard error")
        # ax1.fill_between(x, y - 2*np.sqrt(var), y + 2*np.sqrt(var), color='green', alpha=0.4)
        ax1.legend()

        # second subplot
        ax2.set_title("Effects per bin")
        data = params["list_with_bin_points_effects"]
        bin_centers = params["limits"][:-1] + params["dx"]/2
        is_bin_full = ~np.array(params["is_bin_empty"])
        positions = bin_centers[is_bin_full]
        std_err = params["bin_estimator_variance"][is_bin_full]
        data1 = params["bin_effects"][is_bin_full]
        ax2.bar(x=positions, height=data1, width=params["dx"], color=(0.1, 0.1, 0.1, 0.1),  edgecolor='blue',
                yerr=std_err, label="bin effect")
        ax2.legend()

        if block is False:
            plt.show(block=False)
        else:
            plt.show()


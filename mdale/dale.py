import numpy as np
import matplotlib.pyplot as plt


def create_bins(x: np.array, k: int):
    """Find the bin limits.

    Parameters
    ----------
    x: ndarray (N,)
      array of points
    k: int
      number of bins

    Returns
    -------
    limits: np.array
      bin limits, shape: [k+1,]
    dx: float
      step between bins

    """

    z_start = np.min(x)
    z_stop = np.max(x)
    limits, dx = np.linspace(z_start, z_stop, num=k + 1, endpoint=True, retstep=True)
    return limits, dx


def compute_bin_effects(points: np.ndarray, point_effects: np.ndarray, limits: np.ndarray):
    """Compute the effect of each bin.

    The function (a) allocates the points in the bins and (b) computes the mean effect of the points that lie
    in each bin as the bin effect. If no points lie in a bin, then NaN is passed as the bin effect.

    Parameters
    ----------
    points: ndarray
      The points we evaluate, shape (N,)
    point_effects: ndarray
      The effect of each point (N,)
    limits: ndarray
      The bin limits, shape: [k+1,]

    Returns
    -------
    bin_effects: ndarray
      The effect of each bin.
    """
    empty_symbol = np.NaN

    # find bin-index of points
    eps = 1e-8
    limits[-1] += eps
    ind = np.digitize(points, limits)
    assert np.alltrue(ind > 0)

    # bin effect is the mean of all points that lie in the bin
    nof_bins = limits.shape[0] - 1
    tmp1 = np.bincount(ind - 1, point_effects, minlength=nof_bins)
    tmp2 = np.bincount(ind - 1, minlength=nof_bins)

    # if no point lies in a bin, store Nan
    bin_effects = np.divide(tmp1, tmp2, out=np.ones(tmp1.shape, dtype=float)*empty_symbol, where=tmp2 != 0)
    return bin_effects


def fill_nans(bin_effects):
    """Interpolate the bin_effects with Nan values.

    Parameters
    ----------
    bin_effects: ndarray
      The bin effects with NaNs

    Returns
    -------
    bin_effects: ndarray
      The bin effects without NaNs

    """
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = nan_helper(bin_effects)
    bin_effects[nans] = np.interp(x(nans), x(~nans), bin_effects[~nans])
    return bin_effects


def compute_accumulated_effect(xs: np.ndarray, limits: np.ndarray, bin_effects: np.ndarray, dx: float):
    """Compute the accumulated effect.

    Parameters
    ----------
    xs: ndarray
      points we want to evaluate, shape: (N,)
    limits: ndarray
      bin limits, shape: (k+1,)
    bin_effects: ndarray
      effect of the bins, shape: (k,)
    dx: float
      step between bins

    Returns
    -------
    y: ndarray
      the accumulated effect, shape: (N,)

    Notes
    -----
    The function implements the following formula

    .. math:: f(x) = dx*\sum_{i=0}^{k_x - 1} bin[i] + (x - limits[k_x-1])*bin[k_x]
    """
    big_m = 100000000000000.

    # find where each point belongs to
    ind = np.digitize(xs, limits)

    # find for each point, the accumulated full-bin effect
    x_cumsum = bin_effects.cumsum()
    tmp = np.concatenate([[0, 0], x_cumsum])
    full_bin_effect = tmp[ind] * dx

    # find for each point, the remaining effect
    tmp = np.concatenate([[limits[0]], limits[:-1], [big_m]])
    deltas = xs - tmp[ind]

    # if xs < left_limit or xs > right_limit, delta = 0
    deltas[deltas < 0] = 0

    tmp = np.concatenate([[0.], bin_effects, [bin_effects[-1]]])
    remaining_effect = deltas * tmp[ind]

    # final effect
    y = full_bin_effect + remaining_effect
    return y


def compute_normalizer(xs, limits, bin_effects, dx):
    """Compute the feature effect normalizer.

    Parameters
    ----------
    xs: ndarray
      points we want to evaluate, shape: (N,)
    limits: ndarray
      bin limits, shape: (k+1,)
    bin_effects: ndarray
      effect of the bins, shape: (k,)
    dx: float

    Returns
    -------
    y: float
      The normalizer

    """
    y = compute_accumulated_effect(xs, limits, bin_effects, dx)
    z = np.mean(y)
    return z


def compute_dale_parameters(points, effects, s, k):
    """Compute the DALE parameters for the s-th feature

    Performs all actions to compute the parameters that are required for
    the s-th feature DALE effect

    Parameters
    ----------
    points
    effects
    s
    k

    Returns
    -------

    """
    points = points[:, s]
    effects = effects[:, s]

    # create bins
    limits, dx = create_bins(points, k)

    # compute effect on each bin
    bin_effects = compute_bin_effects(points, effects, limits)

    # fill bins with NaN values
    bin_effects = fill_nans(bin_effects)

    # compute Z
    z = compute_normalizer(points, limits, bin_effects, dx)

    parameters = {"limits": limits,
                  "dx": dx,
                  "bin_effects": bin_effects,
                  "z": z}
    return parameters


def create_dale_function(points, effects, s, k):
    """Returns the DALE function on for the s-th feature.

    Parameters
    ----------
    points: dataset
    effects: effect of each point
    s: feature
    k: nof bins

    Returns
    -------

    """
    parameters = compute_dale_parameters(points, effects, s, k)

    def dale_function(x):
        y = compute_accumulated_effect(x,
                                       limits=parameters["limits"],
                                       bin_effects=parameters["bin_effects"],
                                       dx=parameters["dx"]) - parameters["z"]
        return y
    return dale_function


def dale(x, s, k, points, effects):
    """Compute DALE at points x.

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
    points: ndarray, shape (N,D)
      The training set
    effects: ndarray, shape (N,D)
      The training set

    Returns
    -------

    """
    parameters = compute_dale_parameters(points, effects, s, k)
    y = compute_accumulated_effect(x,
                                   limits=parameters["limits"],
                                   bin_effects=parameters["bin_effects"],
                                   dx=parameters["dx"]) - parameters["z"]
    return y


class DALE:
    def __init__(self, f, f_der):
        self.f = f
        self.f_der = f_der
        self.dale_effects = None
        self.dale_parameters = None

    def compile(self):
        pass

    def fit(self, X, features, k, effects):
        dic = {}
        dic1 = {}
        for s in features:
            # TODO: fix for not recomputing
            dic1["dale_params_feature_" + str(s)] = compute_dale_parameters(X, effects, s, k)
            dic["dale_feature_" + str(s)] = create_dale_function(X, effects, s, k)
        self.dale_effects = dic
        self.dale_parameters = dic1

    def evaluate(self, x, s):
        func = self.dale_effects["dale_feature_" + str(s)]
        return func(x)

    def plot(self, s):
        params = self.dale_parameters["dale_params_feature_" + str(s)]
        x = np.linspace(params["limits"] - .01, params["limits"] + .01, 10000)
        y = self.evaluate(x, s)
        plt.figure()
        plt.plot(x, y, "b-")
        plt.show()


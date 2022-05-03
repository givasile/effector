import numpy as np
import pandas as pd
import PyALE
import copy


def pdp(x, points, f, s):
    y = []
    for i in range(x.shape[0]):
        points1 = copy.deepcopy(points)
        points1[:, s] = x[i]
        y.append(np.mean(f(points1)))
    return np.array(y)


def mplot(x, points, f, s, tau):
    y = []
    for i in range(x.shape[0]):
        points1 = copy.deepcopy(points)
        points1 = points1[np.abs(points[:, s] - x[i]) < tau, :]
        points1[:, s] = x[i]
        y.append(np.mean(f(points1)))
    return np.array(y)


# ale1
def ale(X, f_bb, s, K, feature_type="auto"):
    X_df = pd.DataFrame(X, columns=["feat_" + str(i) for i in range(X.shape[-1])])

    class model():
        def __init__(self, f, X_df):
            self.predict = self.func
            self.f_bb = f

        def func(self, X_df):
            return self.f_bb(X_df.to_numpy())

    model_bb = model(f_bb, X_df)

    ale_computation = PyALE.ale(X_df, model_bb, feature= ["feat_" + str(s)],
                                feature_type=feature_type,
                                grid_size=K,
                                plot=False)

    x = ale_computation["eff"].index.to_numpy()
    y = ale_computation["eff"].to_numpy()
    return x, y


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


def allocate_points_in_bins(points, point_effects, limits):
    # find bin-index of points
    eps = 1e-8
    limits[-1] += eps
    ind = np.digitize(points, limits)
    assert np.alltrue(ind > 0)

    list_with_bin_point_effects = []
    list_with_bin_points = []
    is_bin_empty = []
    for i in range(limits.shape[0]-1):
        if sum(ind == i+1) == 0:
            is_bin_empty.append(True)
        else:
            is_bin_empty.append(False)

        list_with_bin_points.append(points[ind == i+1].tolist())
        list_with_bin_point_effects.append(point_effects[ind == i+1].tolist())
    return list_with_bin_points, list_with_bin_point_effects, is_bin_empty


def compute_point_effects(points, limits, f, dx, s):
    # find bin-index of points
    eps = 1e-8
    limits[-1] += eps
    ind = np.digitize(points[:, s], limits)
    assert np.alltrue(ind > 0)

    right_lim = copy.deepcopy(points)
    left_lim = copy.deepcopy(points)
    right_lim[:, s] = limits[ind]
    left_lim[:, s] = limits[ind-1]
    point_effects = (f(right_lim) - f(left_lim))/dx

    return point_effects


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


def compute_bin_estimator_variance(points, point_effects, limits, mean_effects):
    empty_symbol = np.NaN

    # find bin-index of points
    eps = 1e-8
    limits[-1] += eps
    ind = np.digitize(points, limits)
    assert np.alltrue(ind > 0)

    variance_per_point = (point_effects - mean_effects[ind-1])**2
    nof_bins = limits.shape[0] - 1
    aggregated_variance_per_bin = np.bincount(ind - 1, variance_per_point, minlength=nof_bins)
    points_per_bin = np.bincount(ind - 1, minlength=nof_bins)

    # if no point lies in a bin, store Nan
    bin_variance = np.divide(aggregated_variance_per_bin,
                             points_per_bin,
                             out=np.ones(aggregated_variance_per_bin.shape,dtype=float)*empty_symbol,
                             where=points_per_bin != 0)

    bin_estimator_variance = np.divide(bin_variance,
                                       points_per_bin,
                                       out=np.ones(aggregated_variance_per_bin.shape, dtype=float)*empty_symbol,
                                       where=points_per_bin != 0)
    return bin_estimator_variance


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
    bin_effects1 = copy.deepcopy(bin_effects)

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = nan_helper(bin_effects1)
    bin_effects1[nans] = np.interp(x(nans), x(~nans), bin_effects1[~nans])
    return bin_effects1


def compute_accumulated_effect(xs: np.ndarray, limits: np.ndarray, bin_effects: np.ndarray, dx: float, with_squares=False):
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

    .. math:: f(x) = dx * \sum_{i=0}^{k_x - 1} bin[i] + (x - limits[k_x-1])*bin[k_x]
    """
    big_m = 100000000000000.

    # find where each point belongs to
    ind = np.digitize(xs, limits)

    # find for each point, the accumulated full-bin effect
    x_cumsum = bin_effects.cumsum()
    tmp = np.concatenate([[0, 0], x_cumsum])
    if with_squares:
        full_bin_effect = tmp[ind] * dx**2
    else:
        full_bin_effect = tmp[ind] * dx

    # find for each point, the remaining effect
    tmp = np.concatenate([[limits[0]], limits[:-1], [big_m]])
    deltas = xs - tmp[ind]

    # if xs < left_limit or xs > right_limit, delta = 0
    deltas[deltas < 0] = 0
    if with_squares:
        deltas = deltas**2

    tmp = np.concatenate([[0.], bin_effects, [bin_effects[-1]]])
    remaining_effect = deltas * tmp[ind]

    # final effect
    y = full_bin_effect + remaining_effect
    return y


def find_first_nan_bin(x: np.ndarray):
    nans = np.where(np.isnan(x))[0]

    if nans.size == 0:
        first_empty_bin = None
    else:
        first_empty_bin = nans[0]
    return first_empty_bin


def compute_normalizer(xs: np.ndarray, limits: np.ndarray, bin_effects: np.ndarray, dx: float):
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

import pythia.utils_integrate as integrate
import typing
import numpy as np
import copy


# def create_fix_size_bins(data: np.array, k: int) -> typing.Tuple[np.ndarray, float]:
#     """Find the bin limits.
#
#     Parameters
#     ----------
#     data: ndarray (N,)
#       array of points
#     k: int
#       number of bins
#
#     Returns
#     -------
#     limits: np.array, shape: (k+1,)
#       The bin limits
#     dx: float
#       The bin size
#     """
#     assert data.ndim == 1
#     assert data.size > 1, "The dataset must contain more than one point!"
#
#     z_start = np.min(data)
#     z_stop = np.max(data)
#     assert z_stop > z_start, "The dataset must contain more than one discrete points."
#
#     limits, dx = np.linspace(z_start, z_stop, num=k + 1, endpoint=True, retstep=True)
#     return limits


def compute_local_effects_at_bin_limits(
    data: np.ndarray, model: typing.Callable, limits: np.ndarray, feature: int
) -> np.ndarray:
    """Compute the local effects by evaluating the model at bin limits.

    The function (a) allocates the points in the bins based on the feature of interest
    and (b) computes the effect of each point measuring the difference in the output at the bin limits.

    # TODO add equation

    Parameters
    ----------
    data: ndarray, shape (N,D)
      The training set
    model: Callable: ndarray (N,D) -> (N)
      The black-box model
    limits: np.array, shape: (k+1,)
      The bin limits
    dx: float
      The bin size
    feature: int
      Index of the feature-of-interest

    Returns
    -------
    data_effect: ndarray, shape (N,)
      The local effect of each data point.
    """
    assert data.ndim == 2

    assert limits[0] <= np.min(data[:, feature])
    assert limits[-1] >= np.max(data[:, feature])

    # find bin-index of points
    eps = 1e-8
    limits[-1] += eps
    ind = np.digitize(data[:, feature], limits)
    assert np.alltrue(ind > 0)

    # compute effect
    right_lim = copy.deepcopy(data)
    left_lim = copy.deepcopy(data)
    right_lim[:, feature] = limits[ind]
    left_lim[:, feature] = limits[ind - 1]
    dx = limits[1] - limits[0]
    data_effect = model(right_lim) - model(left_lim)
    return np.squeeze(data_effect) / dx


def filter_points_in_bin(data: np.ndarray, data_effect: np.ndarray, limits: np.ndarray):
    filt = np.logical_and(limits[0] <= data, data <= limits[1])
    data_effect = data_effect[filt]
    data = data[filt]
    return data, data_effect


# def compute_data_effect_single_bin(data: np.ndarray, model: typing.Callable, limits: np.ndarray,
#                                    dx: float, feature: int) -> np.ndarray:
#
#     # compute effect
#     right_lim = copy.deepcopy(data)
#     left_lim = copy.deepcopy(data)
#     right_lim[:, feature] = limits[-1]
#     left_lim[:, feature] = limits[0]
#     data_effect = (model(right_lim) - model(left_lim))/dx
#     return data_effect


def compute_bin_effect(
    data: np.ndarray, data_effect: np.ndarray, limits: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute the mean effect in each bin.

    The function (a) allocates the points in the bins and (b) computes the mean effect of the points that lie
    in each bin as the bin effect. If no points lie in a bin, then NaN is passed as the bin effect.

    # TODO add equation

    Parameters
    ----------
    data: ndarray
      The points we evaluate, shape (N,)
    data_effect: ndarray
      The effect of each point (N,)
    limits: ndarray
      The bin limits, shape: [k+1,]

    Returns
    -------
    bin_effects: ndarray, shape (K,)
      The effect of each bin.
    points_per_bin: ndarray, shape (K,)
      How many points lie in each bin.
    """
    empty_symbol = np.NaN

    # find bin-index of points
    eps = 1e-8
    limits[-1] += eps
    ind = np.digitize(data, limits)
    # assert np.alltrue(ind > 0)

    # bin effect is the mean of all points that lie in the bin
    nof_bins = limits.shape[0] - 1
    aggregated_effect = np.bincount(ind - 1, data_effect, minlength=nof_bins)
    points_per_bin = np.bincount(ind - 1, minlength=nof_bins)

    # if no point lies in a bin, store Nan
    bin_effect_mean = np.divide(
        aggregated_effect,
        points_per_bin,
        out=np.ones(aggregated_effect.shape, dtype=float) * empty_symbol,
        where=points_per_bin != 0,
    )
    return bin_effect_mean, points_per_bin


def compute_bin_variance(data, data_effect, limits, bin_effect_mean):
    """Compute the variance of the effect in each bin.

    The function (a) allocates the points in the bins and (b) computes the variance in each bin and the variance
     of the estimated variance in each bin. If no points lie in a bin, then NaN is passed as the bin effect.

    # TODO add equation

    Parameters
    ----------
    data: ndarray, shape (N,)
      The points we evaluate
    data_effect: ndarray
      The effect of each point (N,)
    limits: ndarray, shape: (k+1,)
      The bin limits
    bin_effect_mean: ndarray, shape: (k,)
      Mean effect in each bin

    Returns
    -------
    bin_variance: ndarray, shape (K,)
      The variance in each bin
    bin_estimator_variance: ndarray, shape (K,)
      The variance of the estimated variance in each bin
    """
    empty_symbol = np.NaN

    # find bin-index of points
    eps = 1e-8
    limits[-1] += eps
    ind = np.digitize(data, limits)
    # assert np.alltrue(ind > 0)

    # variance of the effect in each bin
    variance_per_point = (data_effect - bin_effect_mean[ind - 1]) ** 2
    nof_bins = limits.shape[0] - 1
    aggregated_variance_per_bin = np.bincount(
        ind - 1, variance_per_point, minlength=nof_bins
    )
    points_per_bin = np.bincount(ind - 1, minlength=nof_bins)

    # if less that two points in a bin, store Nan
    bin_variance = np.divide(
        aggregated_variance_per_bin,
        points_per_bin,
        out=np.ones(aggregated_variance_per_bin.shape, dtype=float) * empty_symbol,
        where=points_per_bin > 1,
    )

    # the variance of the estimator
    bin_estimator_variance = np.divide(
        bin_variance,
        points_per_bin,
        out=np.ones(aggregated_variance_per_bin.shape, dtype=float) * empty_symbol,
        where=points_per_bin > 1,
    )
    return bin_variance, bin_estimator_variance


def fill_nans(bin_effect_mean):
    """Interpolate the bin_effects with Nan values.

    Parameters
    ----------
    bin_effect_mean: ndarray
      The bin effects with NaNs

    Returns
    -------
    bin_effects: ndarray
      The bin effects without NaNs

    """
    bin_effect_1 = copy.deepcopy(bin_effect_mean)

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = nan_helper(bin_effect_1)
    bin_effect_1[nans] = np.interp(x(nans), x(~nans), bin_effect_1[~nans])
    return bin_effect_1


def compute_accumulated_effect(
    x: np.ndarray, limits: np.ndarray, bin_effect: np.ndarray, dx: float, square=False
):
    """Compute the accumulated effect.

    Parameters
    ----------
    x: ndarray
      points we want to evaluate, shape: (N,)
    limits: ndarray
      bin limits, shape: (k+1,)
    bin_effect: ndarray
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
    big_m = 100000000000000.0

    # find where each point belongs to
    ind = np.digitize(x, limits)
    # find for each point, the accumulated full-bin effect
    if square:
        x_cumsum = (bin_effect * dx**2).cumsum()
        tmp = np.concatenate([[0, 0], x_cumsum])
        full_bin_effect = tmp[ind]  # * tmp2[ind] ** 2
    else:
        x_cumsum = (bin_effect * dx).cumsum()
        tmp = np.concatenate([[0, 0], x_cumsum])
        full_bin_effect = tmp[ind]  # * tmp2[ind]

    # find for each point, the remaining effect
    tmp = np.concatenate([[limits[0]], limits[:-1], [big_m]])
    deltas = x - tmp[ind]

    # if xs < left_limit or xs > right_limit, delta = 0
    deltas[deltas < 0] = 0
    if square:
        deltas = deltas**2

    tmp = np.concatenate([[0.0], bin_effect, [bin_effect[-1]]])
    remaining_effect = deltas * tmp[ind]

    # final effect
    y = full_bin_effect + remaining_effect
    return y


def compute_ale_params_from_data(data, data_effect, limits):

    dx = np.array([limits[i + 1] - limits[i] for i in range(len(limits) - 1)])

    # compute mean effect on each bin
    bin_effect_nans, points_per_bin = compute_bin_effect(data, data_effect, limits)

    # compute effect variance in each bin
    bin_variance_nans, bin_estimator_variance_nans = compute_bin_variance(
        data, data_effect, limits, bin_effect_nans
    )

    # interpolate NaNs
    bin_effect = fill_nans(bin_effect_nans)
    bin_variance = fill_nans(bin_variance_nans)
    bin_estimator_variance = fill_nans(bin_estimator_variance_nans)

    parameters = {
        "limits": limits,
        "dx": dx,
        "points_per_bin": points_per_bin,
        "bin_effect": bin_effect,
        "bin_variance": bin_variance,
        "bin_estimator_variance": bin_estimator_variance,
    }
    return parameters


def compute_ale_params_from_gt(mean, limits):
    dx = np.array([limits[i + 1] - limits[i] for i in range(len(limits) - 1)])

    bin_mean = []
    bin_var = []
    for i in range(limits.shape[0] - 1):
        start = limits[i]
        stop = limits[i + 1]
        mu_bin = integrate.integrate_1d_quad(mean, start, stop) / (stop - start)
        mean_var = lambda x: (mean(x) - mu_bin) ** 2
        bin_var1 = integrate.integrate_1d_quad(mean_var, start, stop)
        bin_var1 = bin_var1 / (stop - start)

        bin_mean.append(mu_bin)
        bin_var.append(bin_var1)

    bin_mean = np.array(bin_mean)
    bin_var = np.array(bin_var)
    parameters = {
        "dx": dx,
        "limits": limits,
        "bin_effect": bin_mean,
        "bin_variance": bin_var,
    }
    return parameters

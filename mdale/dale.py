import numpy as np


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


def compute_dale(x, s, k, points, effects):
    """Compute DALE at points x.

    Parameters
    ----------
    x
    s
    K
    points
    effects

    Returns
    -------

    """
    points = points[:, s]
    effects = effects[:, s]

    limits, dx = create_bins(points, k)

    # step 2-3
    bin_effects = compute_bin_effects(points, effects, limits)

    # step 4 - fill NaNs
    bin_effects = fill_nans(bin_effects)

    # step 5 - compute the normalizer
    z = compute_normalizer(points, limits, bin_effects, dx)

    # step 6 - create the normalized feature effect function
    return compute_accumulated_effect(x, limits, bin_effects, dx) - z


def create_dale_function(X, X_der, s, K):
    """Return DALE function on for the s-th feature.

    Parameters
    ----------
    X
    X_der
    s
    K

    Returns
    -------

    """
    def dale_function(x):
        return compute_dale(x, s, K, X, X_der)
    return dale_function


# def plot(X, X_der, s, K, title=None, savefig=None):
#     S, dx = create_bins(X, s, K)
#     z0 = S[0,0]
#     zK = S[-1,1]

#     ale_grad2 = create_ale_gradients(X, X_der, s, K)
#     plt.figure()
#     if title is not None:
#         plt.title(title)
#     x = np.arange(z0, zK, .01)
#     y = [ale_grad2(i) for i in x]
#     plt.plot(x,y,"b-")
#     if savefig is not None:
#         plt.savefig(savefig)
#     else:
#         plt.show(block=False)

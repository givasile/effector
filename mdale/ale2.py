import numpy as np
import matplotlib.pyplot as plt
import typing
import timeit


# def create_bins(X: np.array, s: int, K: int):
#     """Compute the limits of the bins.
#
#     :param X: (N,D)
#     :param s: index of the feature
#     :param K: number of bins
#     :returns:
#         S: (K,2) A numpy array with the limits of the bins ()
#         dx: float, the step
#     """
#     z0 = np.min(X[:,s])
#     zK = np.max(X[:,s])
#
#     points, dx = np.linspace(z0, zK, num=K+1, endpoint=True, retstep=True)
#     S = np.array([[points[i], points[i+1]] for i in range(K)])
#     return S, dx


def create_bins(X: np.array, s: int, K: int):
    """Compute the limits of the bins.

    :param X: (N,D)
    :param s: index of the feature
    :param K: number of bins
    :returns:
        S: (K,2) A numpy array with the limits of the bins ()
        dx: float, the step
    """
    z_start = np.min(X[:, s])
    z_stop = np.max(X[:, s])
    bins, dx = np.linspace(z_start, z_stop, num=K+1, endpoint=True, retstep=True)
    return bins, dx


def alloc_points_to_bins(X: np.array, X_der, S: np.array, s: int):
    """Allocates the points in the bins

    :param X:
    :param X_der:
    :param S:
    :param s:
    :returns:
      X_bins: a list with the points that belong to each bins
      X_bins_der: a list with the derivatives of the points that belong to each bins
    """
    N, D = X.shape
    allocated = np.zeros((N,), dtype=bool)
    X_bins = []
    X_bins_der = []
    for k, window in enumerate(S):
        # find which indices belong to current array
        l_ind = window[0] <= X[:,s]
        r_ind = X[:,s] <= window[1]
        ind_now = np.logical_and(l_ind,r_ind)

        # ensure that they have not been allocated to other bin
        ind_now_unalloc = np.logical_and(ind_now, np.logical_not(allocated))

        # updated allocated array
        allocated = np.logical_or(ind_now_unalloc, allocated)

        # insert current unnalocated indices to current bin
        X_bins.append(X[ind_now_unalloc,:])

        X_bins_der.append(X_der[ind_now_unalloc,:])

    # assert all points have been allocated
    assert np.sum([arr.shape[0] for arr in X_bins]) == N
    assert np.sum([arr.shape[0] for arr in X_bins_der]) == N
    return X_bins, X_bins_der


def mean_var_of_bins(X_bins, X_bins_der, s):
    """List of

    :param X_bins:
    :param X_bins_der:
    :param s:
    :returns:

    """
    X_bins_mean = []
    X_bins_std = []
    for i, x in enumerate(X_bins):
        tmp = X_bins_der[i]
        if tmp.size == 0:
            X_bins_mean.append(np.NaN)
            X_bins_std.append(np.NaN)
        else:
            X_bins_mean.append(np.mean(tmp[:,s]))
            X_bins_std.append(np.std(tmp[:,s]))
    X_bins_mean = np.array(X_bins_mean)
    X_bins_std = np.array(X_bins_std)
    return X_bins_mean, X_bins_std


def mean_of_bins(X: np.array, X_der, bins: np.array, s: int):
    eps = 1e-8
    bins[-1] += eps
    inds = np.digitize(X[:,s], bins)
    res = np.bincount(inds-1, X_der[:,s])/np.bincount(inds-1)
    return res, res


def interp(arr):
    """Interpolates missing values (NaN) of a numpy array

    :param arr: the np array with NaNs
    :returns: the np array without NaNs

    """
    def nan_helper(y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
              to convert logical indices of NaNs to 'equivalent' indices
        Example:
            >>> # linear interpolation of NaNs
            >>> nans, x= nan_helper(y)
            >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        """

        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = nan_helper(arr)
    arr[nans]= np.interp(x(nans), x(~nans), arr[~nans])
    return arr


def f_eff_un(xs: float,
             S: np.ndarray,
             X_bins_mean: np.ndarray,
             dx: float):

    """The unnormalized feature effect - it is not vectorized

    :param xs: float
    :param S: np.ndarray (K,2) bins limits
    :param X_bins_mean: np.ndarray (K,) mean value of each bin
    :param dx: float
    :returns:

    """
    # TODO implement function with a look-up table
    K = S.shape[0] - 1
    ind = np.digitize(xs, S)
    y = []
    for i, x in enumerate(xs):
        if ind[i] <= 0:
            y.append(0.) # X_bins_mean[0]*dx
        elif ind[i] > K:
            y.append(np.sum(X_bins_mean)*dx) ## + X_bins_mean[-1]*(xs-S[-1,-1]))
        else:
            y.append(np.sum(X_bins_mean[:ind[i]-1])*dx + X_bins_mean[ind[i]-1]*(x-S[ind[i]-1]))
    return np.array(y)


def f_eff_un2(xs: np.ndarray,
              S: np.ndarray,
              X_bins_mean: np.ndarray,
              dx: float):

    """The unnormalized feature effect - it is not vectorized

    :param xs: float
    :param S: np.ndarray (K,2) bins limits
    :param X_bins_mean: np.ndarray (K,) mean value of each bin
    :param dx: float
    :returns:

    """
    big_M = 100000000000000.
    X_cumsum = X_bins_mean.cumsum()
    tmp = np.concatenate([[0, 0], X_cumsum])
    ind = np.digitize(xs, S)
    full_dx = tmp[ind]*dx

    # find dx
    tmp2 = np.concatenate([[S[0]], S[:-1], [big_M]])
    tmp3 = np.concatenate([[0.], X_bins_mean, [X_bins_mean[-1]]])
    deltas = xs - tmp2[ind]
    deltas[deltas < 0] = 0
    deltas = deltas*tmp3[ind]
    # deltas[deltas == big_M] = 0
    # deltas[deltas == -big_M] = 0
    y = full_dx + deltas
    return y


def compute_normalizer(X, S, s, X_bins_mean, dx):
    """Computes the normalizer Z. Mean value of the unnormalized feature
    effect at evaluated at all training points.

    :param X:
    :param S:
    :param s:
    :param X_bins_mean:
    :param dx:
    :returns:

    """
    # TODO implement function with a look-up table
    K = S.shape[0] - 1
    ind = np.digitize(X[:,s], S)
    y = []
    for i, x in enumerate(X):
        if ind[i] <= 0:
            y.append(0.) # X_bins_mean[0]*dx
        elif ind[i] > K:
            y.append(np.sum(X_bins_mean)*dx) ## + X_bins_mean[-1]*(xs-S[-1,-1]))
        else:
            y.append(np.sum(X_bins_mean[:ind[i]-1])*dx + X_bins_mean[ind[i]-1]*(x[s]-S[ind[i]-1]))

    Z = np.mean(y)
    return Z


def compute_normalizer2(X, S, s, X_bins_mean, dx):
    """Computes the normalizer Z. Mean value of the unnormalized feature
    effect at evaluated at all training points.

    :param X:
    :param S:
    :param s:
    :param X_bins_mean:
    :param dx:
    :returns:

    """
    # TODO implement function with a look-up table
    K = S.shape[0] - 1
    X_cumsum = X_bins_mean.cumsum()
    tmp = np.concatenate([[0, 0], X_cumsum])
    ind = np.digitize(X[:,s], S)
    full_dx = tmp[ind]*dx

    # find dx
    tmp2 = np.concatenate([[S[0]], S[:-1], [np.inf]])
    tmp3 = np.concatenate([[np.inf], X_bins_mean, [X_bins_mean[-1]]])
    deltas = (X[:,s] - tmp2[ind])*tmp3[ind]
    deltas[deltas == np.inf] = 0
    deltas[deltas == -np.inf] = 0
    y = full_dx + deltas
    Z = np.mean(y)
    return Z


def f_eff_norm(x: float,
               S: np.ndarray,
               X: np.ndarray,
               Z: float,
               s: int,
               dx: float,
               X_bins_mean: np.ndarray):
    """The normalized (centerized) feature effect.

    :param x: float
    :param S:
    :param X:
    :param Z:
    :param s:
    :param dx:
    :param X_bins_mean:
    :returns:

    """
    return f_eff_un2(x, S, X_bins_mean, dx) - Z


def create_ale_gradients(X, X_der, s, K):
    # # step 1 - compute the limits of the bins
    # S, dx = create_bins(X, s, K)

    # # step 2 - allocate X points and X_der into bins
    # X_bins, X_bins_der = alloc_points_to_bins(X, X_der, S, s)

    # # step 3 - compute mean and variance inside each bin
    # X_bins_mean, X_bins_std = mean_var_of_bins(X_bins, X_bins_der, s)

    # step 1
    S, dx = create_bins(X, s, K)

    # step 2-3
    X_bins_mean, X_bins_std = mean_of_bins(X, X_der, S, s)

    # step 4 - fill NaNs
    X_bins_mean = interp(X_bins_mean)
    X_bins_std = interp(X_bins_std)

    # step 5 - compute the normalizer
    Z2 = compute_normalizer2(X, S, s, X_bins_mean, dx)

    # step 6 - create the normalized feature effect function
    def tmp(xs):
        return f_eff_norm(xs, S, X, Z2, s, dx, X_bins_mean)

    return tmp

def plot(X, X_der, s, K, title=None, savefig=None):
    S, dx = create_bins(X, s, K)
    z0 = S[0,0]
    zK = S[-1,1]

    ale_grad2 = create_ale_gradients(X, X_der, s, K)
    plt.figure()
    if title is not None:
        plt.title(title)
    x = np.arange(z0, zK, .01)
    y = [ale_grad2(i) for i in x]
    plt.plot(x,y,"b-")
    if savefig is not None:
        plt.savefig(savefig)
    else:
        plt.show(block=False)

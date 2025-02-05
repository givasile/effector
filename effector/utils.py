import typing
import numpy as np
import copy
from effector import helpers

BIG_M = helpers.BIG_M
EPS = helpers.EPS


class AllBinsHaveAtMostOnePointError(ValueError):
    """Custom exception raised when all values in the input are NaN."""
    pass


def compute_local_effects(
    data: np.ndarray, model: typing.Callable, limits: np.ndarray, feature: int
) -> np.ndarray:
    """Compute the local effects, permuting the feature of interest using the bin limits.

    Notes:
        The function (a) allocates the points in the bins based on the feature of interest (foi)
        and (b) computes the effect as the difference when evaluating the output setting the foi at the right and the
        left limit of the bin.

        Given that the bins are defined as a list [l_0, l_1, ..., l_k], and x_s of the i-th point belongs to the k-th bin:

        $$
        {df \over dx_s}(x^i) = {f(x_0^i, ... ,x_s=l_k, ..., x_D^i) - f(x_0^i, ... ,x_s=l_{k-1}, ..., x_D^i)
         \over l_k - l_{k-1}}
        $$


    Examples:
        >>> data = np.array([[1, 2], [2, 3.0]])
        >>> model = lambda x: np.sum(x, axis=1)
        >>> limits = np.array([1.0, 2.0])
        >>> data_effect = compute_local_effects(data, model, limits, feature=0)
        >>> data_effect
        array([1., 1.])

    Args:
        data: The training set, (N, D)
        model: The black-box model ((N, D) -> (N))
        limits: The bin limits, (K+1)
        feature: Index of the feature-of-interest

    Returns:
        data_effect: The local effect of each data point, (N)

    """
    assert data.ndim == 2

    # check that limits cover all data points
    assert limits[0] <= np.min(data[:, feature])
    assert limits[-1] >= np.max(data[:, feature])

    # for each point, find the bin-index it belongs to
    limits[-1] += EPS
    ind = np.digitize(data[:, feature], limits)
    assert np.all(ind > 0)

    # compute effect
    right_lim = copy.deepcopy(data)
    left_lim = copy.deepcopy(data)
    right_lim[:, feature] = limits[ind]
    left_lim[:, feature] = limits[ind - 1]
    dx = limits[1] - limits[0]
    data_effect = model(right_lim) - model(left_lim)
    return np.squeeze(data_effect) / dx


def filter_points_in_bin(
    xs: np.ndarray, df_dxs: typing.Union[np.ndarray, None], limits: np.ndarray
) -> typing.Tuple[np.ndarray, typing.Union[np.ndarray, None]]:
    """
    Filter the points inside the bin defined by the `limits`.

    Notes:
        Filtering depends on whether `xs` lies in the interval [limits[0], limits[1]], not `df_dxs`.

    Examples:
        >>> xs = np.array([1, 2, 3])
        >>> df_dxs = np.array([32, 34, 36])
        >>> limits = np.array([1, 2])
        >>> xs, df_dxs = filter_points_in_bin(xs, df_dxs, limits)
        >>> xs
        array([1, 2])
        >>> df_dxs
        array([32, 34])

    Args:
        xs: The instances, (N)
        df_dxs: The instance-effects (N) or None
        limits: [Start, Stop] of the bin

    Returns:
        data: The instances in the bin, (nof_points_in_bin, D)
        data_effect: The instance-effects in the bin, (nof_points_in_bin, D) or None

    """
    filt = np.logical_and(limits[0] <= xs, xs <= limits[1])

    # return data
    xs = xs[filt]

    # return data effect if not None
    if df_dxs is not None:
        df_dxs = df_dxs[filt]
    return xs, df_dxs


def compute_bin_effect(
    xs: np.ndarray, df_dxs: np.ndarray, limits: np.ndarray
) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Compute the mean effect in each bin.

    Notes:
        The function (a) allocates the instances in the bins and (b) aggregates the instance-level effects to compute
        the average bin-effect. If no instances lie in a bin, then the bin effect is NaN.

        $$
        \mathtt{bin\_effect}_k = {1 \over |i \in bin_k|} \sum_{i \in bin_k} \mathtt{effect}_i
        $$

    Examples:
        >>> n = 100
        >>> xs = np.ones([n]) - 0.5
        >>> df_dxs = np.ones_like(xs) * 10
        >>> limits = np.array([0., 1., 2.0])
        >>> bin_effects, ppb = compute_bin_effect(xs, df_dxs, limits)
        >>> bin_effects
        array([10., nan])
        >>> ppb
        array([100,   0])

    Parameters:
        xs: The s-th feature of the instances, (N)
        df_dxs: The effect wrt the s-th feature for each instance, (N)
        limits: The bin limits, (K+1)

    Returns:
        bin_effects: The average effect per bin, (K)
        points_per_bin: The number of points per bin, (K)
    """
    empty_symbol = np.nan

    # find bin-index of points
    limits_enh = copy.deepcopy(limits).astype(float)
    limits_enh[-1] += EPS
    ind = np.digitize(xs, limits_enh)
    # assert np.all(ind > 0)

    # bin effect is the mean of all points that lie in the bin
    nof_bins = limits.shape[0] - 1
    aggregated_effect = np.bincount(ind - 1, df_dxs, minlength=nof_bins)
    points_per_bin = np.bincount(ind - 1, minlength=nof_bins)

    # if no point lies in a bin, store Nan
    bin_effect_mean = np.divide(
        aggregated_effect,
        points_per_bin,
        out=np.ones(aggregated_effect.shape, dtype=float) * empty_symbol,
        where=points_per_bin != 0,
    )
    return bin_effect_mean, points_per_bin


def compute_bin_variance(
    xs: np.ndarray, df_dxs: np.ndarray, limits: np.ndarray, bin_effect_mean: np.ndarray
) -> np.ndarray:
    """
    Compute the variance of the effect in each bin.

    Notes:
        The function (a) allocates the points in the bins and (b) computes the variance and the variance/nof points.
        If less than two points in a bin, NaN is passed.

        $$
        \mathtt{bin\_variance}_k = {1 \over |i \in bin_k|} \sum_{i \in bin_k}
        (\mathtt{effect}_i - \mathtt{bin\_effect}_k)^2
        $$

        $$
        \mathtt{bin\_estimator\_variance_k} = {\mathtt{bin\_variance}_k \over |i \in bin_k|}
        $$

    Examples:
        >>> n = 100
        >>> xs = np.ones([n]) - 0.5
        >>> df_dxs = np.ones_like(xs) * 10
        >>> limits = np.array([0., 1., 2.0])
        >>> bin_effect_mean, ppb = compute_bin_effect(xs, df_dxs, limits)
        >>> bin_variance= compute_bin_variance(xs, df_dxs, limits, bin_effect_mean)
        >>> bin_variance
        array([ 0., nan])

        >>> xs = np.ones(4) * 0.5
        >>> df_dxs = np.array([1.0, 3.0, 3.0, 5.0])
        >>> limits = np.array([0, 1, 2.0])
        >>> bin_effect_mean = np.array([np.mean(df_dxs), np.nan])
        >>> compute_bin_variance(xs, df_dxs, limits, bin_effect_mean)
        array([ 2., nan])

    Parameters:
        xs: The points we evaluate, (N)
        df_dxs: The effect of each point, (N, )
        limits: The bin limits (K+1)
        bin_effect_mean: Mean effect in each bin, (K)

    Returns:
        bin_variance: The variance in each bin, (K, )
        bin_estimator_variance: The variance of the estimator in each bin, (K, )

    """
    empty_symbol = np.nan

    # find bin-index of points
    eps = 1e-8
    limits_enh = copy.deepcopy(limits).astype(float)
    limits_enh[-1] += eps
    ind = np.digitize(xs, limits_enh)
    # assert np.all(ind > 0)

    # variance of the effect in each bin
    variance_per_point = (df_dxs - bin_effect_mean[ind - 1]) ** 2
    nof_bins = limits.shape[0] - 1
    aggregated_variance_per_bin = np.bincount(
        ind - 1, variance_per_point, minlength=nof_bins
    )
    points_per_bin = np.bincount(ind - 1, minlength=nof_bins)

    # if less than two points in a bin, store Nan
    bin_variance = np.divide(
        aggregated_variance_per_bin,
        points_per_bin,
        out=np.ones(aggregated_variance_per_bin.shape, dtype=float) * empty_symbol,
        where=points_per_bin > 1,
    )

    return bin_variance


def fill_nans(x: np.ndarray) -> np.ndarray:
    """Replace NaNs with interpolated values.

    Examples:
        >>> x = np.array([1.0, np.nan, 2.0])
        >>> fill_nans(x)
        array([1. , 1.5, 2. ])

        >>> x = np.array([1.0, np.nan, np.nan, np.nan, 2.0])
        >>> fill_nans(x)
        array([1.  , 1.25, 1.5 , 1.75, 2.  ])

        >>> x = np.array([0.5, 1.0, np.nan, np.nan, np.nan])
        >>> fill_nans(x)
        array([0.5, 1. , 1. , 1. , 1. ])

    Parameters:
        x: Time-series with NaNs, (T)

    Returns:
        x: Time-series values without NaNs, (T)
    """
    if np.all(np.isnan(x)):
        raise AllBinsHaveAtMostOnePointError(
            "Input array contains only NaN values. "
            "This is probably because in all bins there is at most one point, "
            "which is not enough to compute the bin variance. "
            "Please consider decreasing the number of bins or changing the bin splitting strategy."
        )

    bin_effect_1 = copy.deepcopy(x)

    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]

    nans, x = nan_helper(bin_effect_1)
    bin_effect_1[nans] = np.interp(x(nans), x(~nans), bin_effect_1[~nans])
    return bin_effect_1


def apply_bin_value(
        x: np.ndarray,
        bin_limits: np.ndarray,
        bin_value: np.ndarray,
):
    """Apply the bin effect to the points.

    Args:
        x: A np.array of points, shape (T,)
        bin_limits: The bin limits, shape (K+1,)
        bin_value: The bin value, shape (K,)

    Returns:

    """
    # assertions
    assert x.ndim == 1
    assert bin_limits.ndim == 1
    assert bin_value.ndim == 1
    assert bin_value.shape[0] == bin_limits.shape[0] - 1

    # find where each point belongs to
    ind = np.digitize(x, bin_limits)

    # add the first and last bin value to the bin value
    bin_value = np.concatenate(
        [
            bin_value[0, np.newaxis],
            bin_value,
            bin_value[-1, np.newaxis]
        ]
    )

    # for each point return the bin effect
    return bin_value[ind]

def compute_accumulated_effect(
    x: np.ndarray,
    limits: np.ndarray,
    bin_effect: np.ndarray,
    dx: np.ndarray,
    square: bool = False,
) -> np.ndarray:
    """Compute the accumulated effect at each point `x`.

    Notes:
        The function implements the following formula:

        $$
        \mathtt{dx}[i] = \mathtt{limits}[i+1] - \mathtt{limits}[i]
        $$

        $$
        \mathtt{full\_bin\_acc} = \sum_{i=0}^{k_x - 1} \mathtt{dx}[i] * \mathtt{bin\_effect}[i]
        $$

        $$
        \mathtt{remainder} = (x - \mathtt{limits}[k_x-1])* \mathtt{bin\_effect}[k_x]
        $$

        $$
        f(x) =  \mathtt{full\_bin\_acc} + \mathtt{remainder}
        $$

    Notes:
        if `square=True`, then the formula is:
        $$
        \mathtt{full\_bin\_acc} = \sum_{i=0}^{k_x - 1} \mathtt{dx}^2[i] * \mathtt{bin\_effect}[i]
        $$

        $$
        \mathtt{remainder} = (x - \mathtt{limits}[k_x-1])^2* \mathtt{bin\_effect}[k_x]
        $$

    Examples:
        >>> x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        >>> limits = np.array([0, 1.5, 2.0])
        >>> bin_effect = np.array([1.0, -1.0])
        >>> dx = np.array([1.5, 0.5])
        >>> compute_accumulated_effect(x, limits, bin_effect, dx)
        array([0. , 0. , 0. , 0.5, 1. , 1.5, 1. , 1. , 1. ])

        >>> x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
        >>> limits = np.array([0, 1.5, 2.0])
        >>> bin_effect = np.array([1.0, 1.0])
        >>> dx = np.array([1.5, 0.5])
        >>> compute_accumulated_effect(x, limits, bin_effect, dx)
        array([0. , 0. , 0. , 0.5, 1. , 1.5, 2. , 2. , 2. ])



    Parameters:
        x: The points we want to evaluate at, (T)
        limits: The bin limits, (K+1)
        bin_effect: The effect in each bin, (K)
        dx: The bin-widths, (K)
        square: Whether to square the width. If true, the effect is bin_effect * dx^2, otherwise bin_effect * dx

    Returns:
        y: The accumulated effect at each point, (T)


    """
    # find where each point belongs to
    ind = np.digitize(x, limits)

    # for each point, find the accumulated full-bin effect
    x_cumsum = (bin_effect * dx**2).cumsum() if square else (bin_effect * dx).cumsum()
    tmp = np.concatenate([[0, 0], x_cumsum])
    full_bin_effect = tmp[ind]

    # for each point, find the remaining effect
    tmp = np.concatenate([[limits[0]], limits[:-1], [BIG_M]])
    deltas = x - tmp[ind]
    deltas[deltas < 0] = 0  # if xs < left_limit => delta = 0
    deltas = deltas**2 if square else deltas
    tmp = np.concatenate([[0.0], bin_effect, [bin_effect[-1]]])
    remaining_effect = deltas * tmp[ind]

    # final effect
    y = full_bin_effect + remaining_effect
    return y


def compute_ale_params(xs: np.ndarray, bin_values: np.ndarray, bin_limits: np.ndarray) -> dict:
    """
    Compute all important parameters for the ALE plot.

    Examples:
        >>> # Example without interpolation
        >>> xs = np.array([0.5, 1.2, 2, 2.3])
        >>> df_dxs = np.array([30, 34, 15, 17])
        >>> limits = np.array([0, 1.5, 3.])
        >>> compute_ale_params(xs, bin_values, bin_limits)
        {'limits': array([0. , 1.5, 3. ]), 'dx': array([1.5, 1.5]), 'points_per_bin': array([2, 2]), 'bin_effect': array([32., 16.]), 'bin_variance': array([4., 1.]), 'bin_estimator_variance': array([2. , 0.5])}

        >>> # Example with interpolation
        >>> xs = np.array([1, 2, 2.8, 4])
        >>> df_dxs = np.array([31, 34, 37, 40])
        >>> limits = np.array([1, 3, 4])
        >>> compute_ale_params(xs, bin_values, bin_limits)
        {'limits': array([1, 3, 4]), 'dx': array([2, 1]), 'points_per_bin': array([3, 1]), 'bin_effect': array([34., 40.]), 'bin_variance': array([6., 6.]), 'bin_estimator_variance': array([2., 2.])}

    Args:
        xs: The values of s-th feature, (N)
        bin_values: The effect wrt the s-th feature, (N)
        bin_limits: The bin limits, (K+1)

    Returns:
        parameters: dict

    """
    # compute bin-widths
    dx = np.array([bin_limits[i + 1] - bin_limits[i] for i in range(len(bin_limits) - 1)])

    # compute mean effect on each bin
    bin_effect_nans, points_per_bin = compute_bin_effect(xs, bin_values, bin_limits)

    # compute effect variance in each bin
    bin_variance_nans = compute_bin_variance(
        xs, bin_values, bin_limits, bin_effect_nans
    )

    # interpolate NaNs
    bin_effect = fill_nans(bin_effect_nans)
    bin_variance = fill_nans(bin_variance_nans)

    parameters = {
        "limits": bin_limits,
        "dx": dx,
        "points_per_bin": points_per_bin,
        "bin_effect": bin_effect,
        "bin_variance": bin_variance,
    }
    return parameters


def get_feature_types(
    data: np.ndarray, categorical_limit: int = 10
) -> typing.List[str]:
    """Determine the type of each feature.

    Notes:
        A feature is considered as categorical if it has less than `cat_limit` unique values.

    Args:
        data: The dataset, (N, D)
        categorical_limit: Maximum unique values for a feature to be considered as categorical


    Returns:
        types: A list of strings, where each string is either `"cat"` or `"cont"`

    """

    types = [
        "cat" if len(np.unique(data[:, f])) < categorical_limit else "cont"
        for f in range(data.shape[1])
    ]
    return types


def compute_jacobian_numerically(
    model: typing.Callable, data: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Compute the Jacobian of the model using finite differences.

    Notes:
        The function computes the Jacobian of the model using finite differences. The formula is:

        $$
        \mathtt{J} = {\mathtt{model}(x + \mathtt{eps}) - \mathtt{model}(x) \over \mathtt{eps}}
        $$

    Examples:
        >>> data = np.array([[1, 2], [2, 3.0]])
        >>> model = lambda x: np.sum(x, axis=1)
        >>> compute_jacobian_numerically(model, data)
        array([[1., 1.],
               [1., 1.]])

    Args:
        data: The dataset, (N, D)
        model: The black-box model ((N, D) -> (N))
        eps: The finite difference step

    Returns:
        jacobian: The Jacobian of the model, (N, D)

    """
    assert data.ndim == 2
    jacobian = np.zeros_like(data)
    for f in range(data.shape[1]):
        data_plus = copy.deepcopy(data)
        data_plus[:, f] += eps
        jacobian[:, f] = (model(data_plus) - model(data)) / eps
    return jacobian

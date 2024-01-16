import numpy as np
import typing
import effector.bin_splitting as be


class DynamicProgramming:
    def __init__(self,
                 max_nof_bins: int = 20,
                 min_points_per_bin: int = 10,
                 discount: float = 0.3,
                 cat_limit: int = 15):
        self.max_nof_bins = max_nof_bins
        self.min_points_per_bin = min_points_per_bin
        self.discount = discount
        self.cat_limit = cat_limit


class Greedy:
    def __init__(self,
                 init_nof_bins: int = 100,
                 min_points_per_bin: int = 10,
                 discount: float = 0.3,
                 cat_limit: int = 15
                 ):
        self.max_nof_bins = init_nof_bins
        self.min_points_per_bin = min_points_per_bin
        self.discount = discount
        self.cat_limit = cat_limit


class Fixed:
    def __init__(self,
                 nof_bins: int = 100,
                 min_points_per_bin=10,
                 cat_limit: int = 15
                 ):
        self.nof_bins = nof_bins
        self.min_points_per_bin = min_points_per_bin
        self.cat_limit = cat_limit


def find_limits(
        data: np.ndarray,
        data_effect: typing.Union[None, np.ndarray],
        feature: int,
        axis_limits: np.ndarray,
        binning_method: typing.Union[str, DynamicProgramming, Greedy, Fixed]):
    """Find the limits of the bins for a specific feature.

    Parameters
    ----------
    data: numpy array, shape (n_samples, n_features)
        Data matrix
    data_effect: numpy array, shape (n_samples, n_features)
        Jacobian matrix
    feature: int
        Index of the feature of interest
    axis_limits: numpy array
        Axis limits, shape (n_features, 2)
    binning_method: str or instance of appropriate binning class
        Binning method to use

    Returns
    -------
    bin_est: instance of appropriate binning class
        Binning estimator
    """
    if isinstance(binning_method, str):
        assert binning_method in ["dp", "greedy", "fixed"]
    assert isinstance(axis_limits, np.ndarray)

    if isinstance(binning_method, Fixed):
        bin_est = be.Fixed(data, data_effect, feature=feature, axis_limits=axis_limits)
        bin_est.find(nof_bins=binning_method.nof_bins, min_points=binning_method.min_points_per_bin, cat_limit=binning_method.cat_limit)
    elif binning_method == "fixed":
        bin_est = be.Fixed(data, data_effect, feature=feature, axis_limits=axis_limits)
        bin_est.find(nof_bins=20, min_points=0, cat_limit=15)
    elif isinstance(binning_method, Greedy):
        bin_est = be.Greedy(data, data_effect, feature=feature, axis_limits=axis_limits)
        bin_est.find(
            min_points=binning_method.min_points_per_bin, init_nof_bins=binning_method.max_nof_bins, discount=binning_method.discount, cat_limit=binning_method.cat_limit
        )
    elif binning_method == "greedy":
        bin_est = be.Greedy(data, data_effect, feature=feature, axis_limits=axis_limits)
        bin_est.find(min_points=0, init_nof_bins=100, discount=0.3, cat_limit=15)
    elif isinstance(binning_method, DynamicProgramming):
        bin_est = be.DP(data, data_effect, feature=feature, axis_limits=axis_limits)
        bin_est.find(
            min_points=binning_method.min_points_per_bin, max_nof_bins=binning_method.max_nof_bins, discount=binning_method.discount, cat_limit=binning_method.cat_limit
        )
    elif binning_method == "dp":
        bin_est = be.DP(data, data_effect, feature=feature, axis_limits=axis_limits)
        bin_est.find(min_points=0, max_nof_bins=20, discount=0.3, cat_limit=15)
    return bin_est

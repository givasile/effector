import typing
import numpy as np


def prep_features(feat: typing.Union[str, list], D) -> list:
    assert type(feat) in [list, str, int]
    if feat == "all":
        feat = [i for i in range(D)]
    elif type(feat) == int:
        feat = [feat]
    return feat


def prep_normalize(normalize):
    assert type(normalize) in [bool, str]
    if type(normalize) is str:
        assert normalize in ["zero_start", "zero_integral"]

    if normalize is True:
        normalize = "zero_integral"
    return normalize


def axis_limits_from_data(data: np.ndarray) -> np.ndarray:
    """

    :param data: np.ndarray (N, D)
    :returns: np.ndarray (2, D)

    """
    D = data.shape[1]
    axis_limits = np.zeros([2, D])
    for d in range(D):
        axis_limits[0, d] = data[:, d].min()
        axis_limits[1, d] = data[:, d].max()
    return axis_limits


def prep_dale_fit_params(par: dict):
    if par is None:
        par = {}

    if "bin_method" in par.keys():
        assert par["bin_method"] in ["fixed", "greedy", "dp"]
    else:
        par["bin_method"] = "fixed"

    if "nof_bins" in par.keys():
        assert type(par["nof_bins"]) == int
    else:
        par["nof_bins"] = 100

    if "max_nof_bins" in par.keys():
        assert type(par["max_nof_bins"]) == int
    else:
        par["max_nof_bins"] = 20

    if "min_points_per_bin" in par.keys():
        assert type(par["max_nof_bins"]) == int
    else:
        par["min_points_per_bin"] = None

    return par


def prep_ale_fit_params(par: dict):
    if "nof_bins" in par.keys():
        assert type(par["nof_bins"]) == int
    else:
        par["nof_bins"] = 100
    return par
import typing
import numpy as np


def prep_features(feat: typing.Union[str, list]) -> list:
    assert type(feat) in [list, str, int]
    if feat == "all":
        feat = [i for i in range(self.D)]
    elif type(feat) == int:
        feat = [feat]
    return feat



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

# functions for automatically generating regions minimizing some criterion
import typing
import numpy as np

foc = [0, 2]
foi = 1
nof_splits = 10



def find_optimal_split(criterion: typing.Callable, D1: list, foi: int, foc: list, nof_splits: int, axis_limits: np.ndarray):
    """
    Find the optimal split defined by the criterion
    on the list of features of conditioning (foc) for a given feature of interest (foi).

    Parameters:
        criterion: function that maps a dataset to a loss value
        D1: list of datasets
        foi: feature of interest
        foc: list of candidate features for conditioning
        nof_splits: number of split positions to check along each feature of conditioning
    """
    big_M = 1000000
    I = np.ones([len(foc), nof_splits]) * big_M

    I_start = np.mean([criterion(D) for D in D1])
    for i, foc_i in enumerate(foc):

        for j in range(nof_splits):
            position = axis_limits[0, foc_i] + j * (axis_limits[1, foc_i] - axis_limits[0, foc_i]) / nof_splits

            # evaluate criterion (integral) after split
            D2 = []
            for d in D1:
                D2.append(d[d[:, foc_i] < position])
                D2.append(d[d[:, foc_i] >= position])
            I[i, j] = np.mean([criterion(d) for d in D2])

    # argmin over all splits
    i, j = np.unravel_index(np.argmin(I, axis=None), I.shape)
    feature = foc[i]
    position = axis_limits[0, foc[i]] + j * (axis_limits[1, foc[i]] - axis_limits[0, foc[i]]) / nof_splits
    return I_start, I, i, j, feature, position



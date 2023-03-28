# functions for automatically generating regions minimizing some criterion
import typing
import numpy as np
from pythia import pdp

foc = [0, 2]
foi = 1
nof_splits = 10


def pdp_heterogeneity(data, model, model_jac, axis_limits, nof_instances):
    # if data is empty, return zero
    if data.shape[0] == 0:
        return 1000000
    feat = 1
    dpdp = pdp.dPDP(data, model, model_jac, axis_limits, nof_instances)
    dpdp.fit(features="all", normalize=False)
    start = axis_limits[:, feat][0]
    stop = axis_limits[:, feat][1]
    x = np.linspace(start, stop, 1000)
    x = 0.5 * (x[:-1] + x[1:])
    pdp_m, pdp_std, pdp_stderr = dpdp.eval(feature=feat, x=x, uncertainty=True)
    z = np.mean(pdp_std)
    return z


def find_optimal_split(model, model_jac, data, D1: list, foi: int, feature_types: list, foc: list, nof_splits: int, axis_limits: np.ndarray, nof_instances: int):
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
    # initialize I[i,j] where i is the feature of conditioning and j is the split position
    big_M = 1000000
    I = np.ones([len(foc), nof_splits]) * big_M

    # evaluate heterogeneity before split
    I_start = np.mean([pdp_heterogeneity(D, model, model_jac, axis_limits, nof_instances) for D in D1])

    # for each feature of conditioning
    for i, foc_i in enumerate(foc):
        # if feature is categorical, split at each possible value
        if feature_types[i] == "categorical":
            for j, position in enumerate(np.unique(data[:, foc_i])):
                # evaluate criterion (integral) after split
                D2 = []
                for d in D1:
                    D2.append(d[d[:, foc_i] == position])
                    D2.append(d[d[:, foc_i] != position])
                I[i, j] = np.mean([pdp_heterogeneity(d, model, model_jac, axis_limits, nof_instances) for d in D2])
        # else split at nof_splits positions
        else:
            step = (axis_limits[1, foc_i] - axis_limits[0, foc_i]) / nof_splits
            for j in range(nof_splits):
                # find split position as the middle of the j-th interval
                position = axis_limits[0, foc_i] + (j + 0.5) * step

                # evaluate criterion (integral) after split
                D2 = []
                for d in D1:
                    D2.append(d[d[:, foc_i] < position])
                    D2.append(d[d[:, foc_i] >= position])
                I[i, j] = np.mean([pdp_heterogeneity(d, model, model_jac, axis_limits, nof_instances) for d in D2])

    # find minimum heterogeneity split
    i, j = np.unravel_index(np.argmin(I, axis=None), I.shape)
    feature = foc[i]
    if feature_types[i] == "categorical":
        position = np.unique(data[:, feature])[j]
    else:
        position = axis_limits[0, feature] + (j + 0.5) * step
    return I_start, I, i, j, feature, position, feature_types[i]


def find_dICE_splits(
        nof_levels: int,
        nof_split_positions: int,
        foi: int,
        foc: typing.Union[list, str],
        data: np.ndarray,
        model: typing.Callable,
        model_jac: typing.Callable,
        axis_limits: np.ndarray,
        nof_instances: int):

    # preprocess foc
    if foc == "all":
        foc = list(range(data.shape[1]))
        foc.remove(foi)

    # find which features inside foc are categorical and which are continuous
    feature_types = []
    for f in foc:
        if len(np.unique(data[:, f])) < 10:
            feature_types.append("categorical")
        else:
            feature_types.append("continuous")

    # find optimal split for each level
    splitting_positions = []
    splitting_features = []
    splitting_feature_types = []

    list_of_heterogeneity = []
    list_of_X = [data]
    for lev in range(nof_levels):

        I_start, I, i, j, feature, position, f_type = find_optimal_split(model, model_jac, data, list_of_X, foi, feature_types, foc, nof_split_positions, axis_limits, nof_instances)
        list_of_heterogeneity.append(I_start)
        if lev == nof_levels - 1:
            list_of_heterogeneity.append(I[i, j])
        splitting_positions.append(position)
        splitting_features.append(feature)
        splitting_feature_types.append(f_type)

        # split all datasets based on optimal split
        new_list_of_X = []
        for x in list_of_X:
            # split X on the optimal feature and position
            if f_type == "categorical":
                X1 = x[x[:, feature] == position]
                X2 = x[x[:, feature] != position]
            else:
                X1 = x[x[:, feature] < position]
                X2 = x[x[:, feature] >= position]
            new_list_of_X.append(X1)
            new_list_of_X.append(X2)
        list_of_X = new_list_of_X

    return splitting_features, splitting_feature_types, splitting_positions, list_of_heterogeneity



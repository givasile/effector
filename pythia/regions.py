# functions for automatically generating regions minimizing some criterion
import typing
import numpy as np

import pythia
from pythia import pdp
from tqdm import tqdm

foc = [0, 2]
foi = 1
nof_splits = 10


def pdp_heter(data, model, model_jac, foi):
    # if data is empty, return zero
    if data.shape[0] < 50:
        return 1000000
    feat = foi
   
    # Initialize dpdp
    axis_limits = pythia.helpers.axis_limits_from_data(data)
    nof_instances = 100
    dpdp = pdp.dPDP(data, model, model_jac, axis_limits, nof_instances)
   
    # Fit dpdp
    dpdp.fit(features=foi, normalize=False)

    start = axis_limits[:, feat][0]
    stop = axis_limits[:, feat][1]

    x = np.linspace(start, stop, 21)
    x = 0.5 * (x[:-1] + x[1:])

    pdp_m, pdp_std, pdp_stderr = dpdp._eval_unnorm(feature=feat, x=x, uncertainty=True)
    z = np.mean(pdp_std)
    return z


def ale_heter(x, x_jac, model, foi):
    # if data is empty, return big number
    if x.shape[0] < 50:
        return 1000000

    # Initialize rhale
    rhale = pythia.RHALE(x, model, None, None, x_jac)
    binning_method = pythia.binning_methods.Fixed(nof_bins=40)
    rhale.fit(features=foi, binning_method=binning_method)

    # heterogeneity is the accumulated std at the end of the curve
    axis_limits = pythia.helpers.axis_limits_from_data(x)
    stop = np.array([axis_limits[:, foi][1]])
    _, z, _ = rhale.eval(feature=foi, x=stop, uncertainty=True)
    return z.item()


def split_pdp(model, model_jac, data, D1: list, foi: int, feature_types: list, foc: list, nof_splits: int, cat_limit):
    heterogen = pdp_heter

    # initialize I[i,j] where i is the feature of conditioning and j is the split position
    big_M = 1000000
    I = np.ones([len(foc), np.max(nof_splits, cat_limit)]) * big_M

    nof_instances = 100
    axis_limits = pythia.helpers.axis_limits_from_data(data)

    # evaluate heterogeneity before split
    print('evaluate heterogeneity before split')
    I_start = np.mean([heterogen(D, model, model_jac, foi) for D in D1])

    # for each feature of conditioning
    for i, foc_i in enumerate(tqdm(foc)):
        # if feature is categorical, split at each possible value
        if feature_types[i] == "cat":
            for j, position in enumerate(np.unique(data[:, foc_i])):
                # evaluate criterion (integral) after split
                D2 = []
                for d in D1:
                    D2.append(d[d[:, foc_i] == position])
                    D2.append(d[d[:, foc_i] != position])
                I[i, j] = np.mean([heterogen(d, model, model_jac, foi) for d in D2])
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
                I[i, j] = np.mean([heterogen(d, model, model_jac, foi) for d in D2])

    # find minimum heterogeneity split
    i, j = np.unravel_index(np.argmin(I, axis=None), I.shape)
    feature = foc[i]
    if feature_types[i] == "cat":
        position = np.unique(data[:, feature])[j]
    else:
        position = axis_limits[0, feature] + (j + 0.5) * step
    return I_start, I, i, j, feature, position, feature_types[i]


def split_rhale(model, model_jac, data, x_list: list, x_jac_list: list, foi: int, foc_types: list, foc: list, nof_splits: int, cat_limit, heter_before):
    # init heter_mat[i,j] (i index of foc and j index of split position)
    big_M = -10000000000
    heter_mat = np.ones([len(foc), max(nof_splits, cat_limit)]) * big_M

    # find all split positions
    positions = [find_positions_cat(data, foc_i) if foc_types[i] == "cat" else find_positions_cont(data, foc_i, nof_splits) for i, foc_i in enumerate(foc)]


    # exhaustive search on all split positions
    for i, foc_i in enumerate(tqdm(foc)):
        for j, position in enumerate(positions[i]):
            # split datasets
            x_list_2 = flatten_list([split_dataset(x, None, foc_i, position, foc_types[i]) for x in x_list])
            x_jac_list_2 = flatten_list([split_dataset(x, x_jac, foc_i, position, foc_types[i]) for x, x_jac in zip(x_list, x_jac_list)])

            # evaluate heterogeneity after split
            sub_heter = [ale_heter(x, x_jac, model, foi) for x, x_jac in zip(x_list_2, x_jac_list_2)]
            heter_drop = np.array(flatten_list([[heter_bef - sub_heter[int(2*i)], heter_bef - sub_heter[int(2*i + 1)]] for i, heter_bef in enumerate(heter_before)]))
            populations = np.array([len(xx) for xx in x_list_2])
            weights = (populations+1) / (np.sum(populations + 1))
            heter_mat[i, j] = np.sum(heter_drop * weights)

    # find min heterogeneity split
    i, j = np.unravel_index(np.argmax(heter_mat, axis=None), heter_mat.shape)
    feature = foc[i]
    position = positions[i][j]
    split_positions = positions[i]
    # how many instances in each dataset after the min split
    x_list_2 = flatten_list([split_dataset(x, None, foc[i], position, foc_types[i]) for x in x_list])
    x_jac_list_2 = flatten_list([split_dataset(x, x_jac, foc[i], position, foc_types[i]) for x, x_jac in zip(x_list, x_jac_list)])
    nof_instances = [len(x) for x in x_list_2]
    sub_heter = [ale_heter(x, x_jac, model, foi) for x, x_jac in zip(x_list_2, x_jac_list_2)]
    split = {"feature": feature, "position": position, "candidate_split_positions": split_positions, "nof_instances": nof_instances, "type": foc_types[i], "heterogeneity": sub_heter, "split_i": i, "split_j": j, "foc": foc, "weighted_heter": heter_mat[i, j]}
    return split


def find_splits(nof_levels: int, nof_splits: int, foi: int, foc: typing.Union[list, str], cat_limit: int, data: np.ndarray, model: typing.Callable, model_jac: typing.Callable, criterion="ale"):
    # prepare foc
    foc = [f for f in range(data.shape[1]) if f != foi] if foc == "all" else foc

    assert nof_levels <= len(foc), "nof_levels must be smaller than the number of features of conditioning"

    # find foc types
    foc_types = ["cat" if len(np.unique(data[:, f])) < cat_limit else "cont" for f in foc]

    # initial heterogeneity
    heter_init = ale_heter(data, model_jac(data), model, foi) if criterion == "ale" else pdp_heter(data, model, model_jac, foi)

    # find optimal split for each level
    x_list = [data]
    x_jac_list = [model_jac(data)]
    splits = [{"heterogeneity": [heter_init], "weighted_heter": heter_init, "nof_instances": [len(data)], "split_i": -1, "split_j": -1, "foc": foc}]
    for lev in range(nof_levels):
        # find split
        split_fn = split_pdp if criterion == "pdp" else split_rhale
        split = split_fn(model, model_jac, data, x_list, x_jac_list, foi, foc_types, foc, nof_splits, cat_limit, splits[-1]["heterogeneity"])
        splits.append(split)

        # split X
        feat, pos, typ = split["feature"], split["position"], split["type"]
        x_jac_list = flatten_list([split_dataset(x, x_jac, feat, pos, typ) for x, x_jac in zip(x_list, x_jac_list)])
        x_list = flatten_list([split_dataset(x, None, feat, pos, typ) for x in x_list])

        # if splits[-1]["weighted_heter"] > splits[-2]["weighted_heter"]:
        #     splits.remove(splits[-1])
        #     break

    return splits


def split_dataset(x, x_jac, feature, position, feat_type):
    if feat_type == "cat":
        ind_1 = x[:, feature] == position
        ind_2 = x[:, feature] != position
    else:
        ind_1 = x[:, feature] < position
        ind_2 = x[:, feature] >= position

    if x_jac is None:
        X1 = x[ind_1, :]
        X2 = x[ind_2, :]
    else:
        X1 = x_jac[ind_1, :]
        X2 = x_jac[ind_2, :]

    return X1, X2


def find_positions_cat(x, feature):
    return np.unique(x[:, feature])


def find_positions_cont(x, feature, nof_splits):
    step = (np.max(x[:, feature]) - np.min(x[:, feature])) / nof_splits
    return np.min(x[:, feature]) + (np.arange(nof_splits) + 0.5) * step


def flatten_list(l):
    return [item for sublist in l for item in sublist]


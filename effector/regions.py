# functions for automatically generating regions minimizing some criterion
import typing
import numpy as np
import itertools
from effector import pdp, helpers, utils
from effector.rhale import RHALE
from effector import binning_methods
from tqdm import tqdm


BIG_M = helpers.BIG_M

foc = [0, 2]
foi = 1
nof_splits = 10


class Regions:
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[callable, None],
        feature_types: typing.Union[list, None] = None,
        cat_limit: int = 10,
    ):
        # setters
        self.data = data
        self.dim = self.data.shape[1]
        self.model = model
        self.model_jac = model_jac
        self.cat_limit = cat_limit

        # on-init
        if feature_types is None:
            self.feature_types = utils.get_feature_types(data, cat_limit)
        else:
            self.feature_types = feature_types

        # init method args
        self.method_args = {}

        # init splits
        self.splits = {}
        self.important_splits = {}

        # state variable
        self.splits_found: np.ndarray = np.ones([self.dim]) < 0
        self.important_splits_selected: np.ndarray = np.ones([self.dim]) < 0

        # state variables
        self.regions_found = False

    def find_splits_single_feature(
        self, feat, nof_levels, nof_candidate_splits, method
    ):
        # if feature is categorical, skip
        if self.feature_types[feat] == "cat":
            self.splits["feat_{}".format(feat)] = []
        else:
            foi, foc = feat, [i for i in range(self.dim) if i != feat]

            # get feature types
            foc_types = [self.feature_types[i] for i in foc]

            # get splits
            splits = find_splits(
                nof_levels=nof_levels,
                nof_splits=nof_candidate_splits,
                foi=foi,
                foc=foc,
                foc_types=foc_types,
                cat_limit=self.cat_limit,
                data=self.data,
                model=self.model,
                model_jac=self.model_jac,
                criterion=method,
            )

            self.splits["feat_{}".format(feat)] = splits
            self.splits_found[feat] = True

    def find_splits(
        self, nof_levels: int, nof_candidate_splits: int = 10, method: str = "rhale"
    ):
        assert method in ["pdp", "rhale"]

        # set method args
        self.method_args["nof_levels"] = nof_levels
        self.method_args["nof_candidate_splits"] = nof_candidate_splits
        self.method_args["method"] = method

        # method
        for feat in tqdm(range(self.dim)):
            self.find_splits_single_feature(
                feat, nof_levels, nof_candidate_splits, method
            )

        # update state
        self.regions_found = True

    def choose_important_splits(self, heter_thres=0.1, pcg=0.2):
        assert self.regions_found is True
        optimal_splits = {}
        for feat in range(self.dim):
            self.important_splits_selected[feat] = self.splits_found[feat]

            # if split is empy, skip
            if len(self.splits["feat_{}".format(feat)]) == 0:
                optimal_splits["feat_{}".format(feat)] = {}
                continue

            # if initial heterogeneity is BIG_M, skip
            if self.splits["feat_{}".format(feat)][0]["weighted_heter"] == BIG_M:
                optimal_splits["feat_{}".format(feat)] = {}
                continue

            # if initial heterogeneity is small right from the beginning, skip
            if self.splits["feat_{}".format(feat)][0]["weighted_heter"] < heter_thres:
                optimal_splits["feat_{}".format(feat)] = {}
                continue

            feat_splits = self.splits["feat_{}".format(feat)]
            # accept split if heterogeneity drops over 20%
            heter = np.array(
                [feat_splits[i]["weighted_heter"] for i in range(len(feat_splits))]
            )
            heter_drop = (heter[:-1] - heter[1:]) / heter[:-1]
            split_valid = heter_drop > pcg

            # if all are negative, return nothing
            if np.sum(split_valid) == 0:
                optimal_splits["feat_{}".format(feat)] = {}
                continue

            # if all are positive, return all
            if np.sum(split_valid) == len(split_valid):
                optimal_splits["feat_{}".format(feat)] = feat_splits[1:]
                continue

            # find first negative split
            first_negative = np.where(split_valid == False)[0][0]

            # if first negative is the first split, return nothing
            if first_negative == 0:
                optimal_splits["feat_{}".format(feat)] = {}
                continue
            else:
                optimal_splits["feat_{}".format(feat)] = feat_splits[
                    1 : first_negative + 1
                ]

        self.important_splits = optimal_splits
        return optimal_splits

    def print_split(self, feature, only_importants=True):
        if only_importants:
            assert self.important_splits_selected[
                feature
            ], "No important splits found for feature {}".format(feature)
        else:
            assert self.splits_found[feature], "No splits found for feature {}".format(
                feature
            )

        if only_importants:
            splits = self.important_splits["feat_{}".format(feature)]
            print("Important splits for feature {}".format(feature))
        else:
            splits = self.splits["feat_{}".format(feature)][1:]
            print("All splits for feature {}".format(feature))

        for i, split in enumerate(splits):
            type_of_split_feature = self.feature_types[split["feature"]]
            print(
                "- On feature {} ({})".format(split["feature"], type_of_split_feature)
            )

            candidate_splits_formatted = ", ".join(
                ["{:.2f}".format(x) for x in split["candidate_split_positions"]]
            )
            print(
                "  - Candidate split positions: {}".format(candidate_splits_formatted)
            )
            print("  - Position of split: {:.2f}".format(split["position"]))

            print(
                "  - Heterogeneity before split: {:.2f}".format(
                    split["weighted_heter"] + split["weighted_heter_drop"]
                )
            )
            print(
                "  - Heterogeneity after split: {:.2f}".format(split["weighted_heter"])
            )
            print(
                "  - Heterogeneity drop: {:.2f} ({:.2f} %)".format(
                    split["weighted_heter_drop"],
                    (split["weighted_heter_drop"] / split["weighted_heter"] * 100),
                )
            )

            nof_instances_before = (
                sum(split["nof_instances"])
                if i == 0
                else splits[i - 1]["nof_instances"]
            )
            print(
                "  - Number of instances before split: {}".format(nof_instances_before)
            )
            print(
                "  - Number of instances after split: {}".format(split["nof_instances"])
            )


class DataTransformer:
    def __init__(self, splits: typing.Dict):
        self.splits = splits

    def transform(self, X):
        # feat_mapping <- to how many features each feature is mapped
        feat_mapping = []
        for split in self.splits.values():
            if len(split) == 0:
                feat_mapping.append(1)
            else:
                feat_mapping.append(2 ** len(split))

        # the enhaned data, without masking
        new_data = []
        for i in range(X.shape[1]):
            new_data.append(np.repeat(X[:, i, np.newaxis], feat_mapping[i], axis=-1))
        new_data = np.concatenate(new_data, axis=-1)

        # create mask, based on splits
        mask = np.ones(new_data.shape)
        new_columns = []
        for feat in range(X.shape[1]):
            # jj = j in the enhanced dataset
            pos = int(np.sum(feat_mapping[:feat]))

            if feat_mapping[feat] == 1:
                new_columns.append("x{}".format(feat))
                continue
            else:
                feat_splits = self.splits["feat_{}".format(feat)]
                lst = [
                    list(i) for i in itertools.product([0, 1], repeat=len(feat_splits))
                ]
                for ii, bin in enumerate(lst):
                    new_name = "x{} | ".format(feat)
                    init_col_mask = np.ones(new_data.shape[0]) * True
                    for jj, b in enumerate(bin):
                        if b == 0:
                            if feat_splits[jj]["type"] == "cat":
                                init_col_mask = np.logical_and(
                                    init_col_mask,
                                    X[:, feat_splits[jj]["feature"]]
                                    == feat_splits[jj]["position"],
                                )
                                # add with two decimals
                                new_name += "x{}={:.2f} & ".format(
                                    feat_splits[jj]["feature"],
                                    feat_splits[jj]["position"],
                                )
                            else:
                                init_col_mask = np.logical_and(
                                    init_col_mask,
                                    X[:, feat_splits[jj]["feature"]]
                                    <= feat_splits[jj]["position"],
                                )
                                new_name += "x{}<={:.2f} & ".format(
                                    feat_splits[jj]["feature"],
                                    feat_splits[jj]["position"],
                                )
                        else:
                            if feat_splits[jj]["type"] == "cat":
                                init_col_mask = np.logical_and(
                                    init_col_mask,
                                    X[:, feat_splits[jj]["feature"]]
                                    != feat_splits[jj]["position"],
                                )
                                new_name += "x{}!={:.2f} & ".format(
                                    feat_splits[jj]["feature"],
                                    feat_splits[jj]["position"],
                                )
                            else:
                                init_col_mask = np.logical_and(
                                    init_col_mask,
                                    X[:, feat_splits[jj]["feature"]]
                                    > feat_splits[jj]["position"],
                                )
                                new_name += "x{}>{:.2f} & ".format(
                                    feat_splits[jj]["feature"],
                                    feat_splits[jj]["position"],
                                )
                    # current position in mask
                    mask[:, pos + ii] = init_col_mask
                    new_columns.append(new_name[:-3])
        self.mask = mask
        self.new_data = new_data * mask
        self.new_names = new_columns
        return self.new_data


def rename_features():
    pass


def pdp_heter(data, model, model_jac, foi, min_points=15):
    # if data is empty, return zero
    if data.shape[0] < min_points:
        return BIG_M
    feat = foi

    # Initialize dpdp
    axis_limits = helpers.axis_limits_from_data(data)
    nof_instances = 100
    dpdp = pdp.DerivativePDP(data, model, model_jac, axis_limits, nof_instances)

    # Fit dpdp
    dpdp.fit(features=foi, normalize=False)

    start = axis_limits[:, feat][0]
    stop = axis_limits[:, feat][1]

    x = np.linspace(start, stop, 21)
    x = 0.5 * (x[:-1] + x[1:])

    pdp_m, pdp_std, pdp_stderr = dpdp._eval_unnorm(feature=feat, x=x, uncertainty=True)
    z = np.mean(pdp_std)
    return z


def rhale_heter(
    data: np.ndarray,
    data_effect: np.ndarray,
    model: callable,
    foi: int,
    min_points: int = 15,
):
    """Computes the accumulated heterogeneity of the RHALE curve at the end of the curve."""
    if data.shape[0] < min_points:
        return BIG_M

    rhale = RHALE(data, model, None, None, data_effect)
    binning_method = binning_methods.Greedy(
        init_nof_bins=50, min_points_per_bin=10, discount=0.1
    )
    # rhale fit throws an error if there are not enough points. In this case, return a big z value
    try:
        rhale.fit(features=foi, binning_method=binning_method)
    except:
        return BIG_M

    # heterogeneity is the accumulated std at the end of the curve
    axis_limits = helpers.axis_limits_from_data(data)
    stop = np.array([axis_limits[:, foi][1]])
    _, z, _ = rhale.eval(feature=foi, xs=stop, uncertainty=True)
    return z.item()


def split_pdp(
    model,
    model_jac,
    data,
    D1: list,
    foi: int,
    feature_types: list,
    foc: list,
    nof_splits: int,
    cat_limit,
    min_points=15,
):
    heterogen = pdp_heter

    # initialize I[i,j] where i is the feature of conditioning and j is the split position
    big_M = 1000000
    I = np.ones([len(foc), np.max(nof_splits, cat_limit)]) * big_M

    nof_instances = 100
    axis_limits = helpers.axis_limits_from_data(data)

    # evaluate heterogeneity before split
    print("evaluate heterogeneity before split")
    I_start = np.mean([heterogen(D, model, model_jac, foi, min_points) for D in D1])

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
                I[i, j] = np.mean(
                    [heterogen(d, model, model_jac, foi, min_points) for d in D2]
                )
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
                I[i, j] = np.mean(
                    [heterogen(d, model, model_jac, foi, min_points) for d in D2]
                )

    # find minimum heterogeneity split
    i, j = np.unravel_index(np.argmin(I, axis=None), I.shape)
    feature = foc[i]
    if feature_types[i] == "cat":
        position = np.unique(data[:, feature])[j]
    else:
        position = axis_limits[0, feature] + (j + 0.5) * step
    return I_start, I, i, j, feature, position, feature_types[i]


def split_rhale(
    model: callable,
    model_jac: callable,
    data: np.ndarray,
    x_list: list,
    x_jac_list: list,
    foi: int,
    foc_types: list,
    foc: list,
    nof_splits: int,
    cat_limit,
    heter_before: float,
    min_points: int = 15,
):
    """Find the best split from all features in the list of foc according to the RHALE criterion.

    Return
    ------
    split: dict, All information about the best split.
        Dictionary with keys: feature, position, candidate_split_positions, nof_instances, type, heterogeneity, split_i, split_j, foc, weighted_heter_drop, weighted_heter
    """
    big_M = -BIG_M

    # weighted_heter_drop[i,j] (i index of foc and j index of split position) is
    # the accumulated heterogeneity drop if I split foc[i] at index j
    weighted_heter_drop = np.ones([len(foc), max(nof_splits, cat_limit)]) * big_M

    # weighted_heter[i,j] (i index of foc and j index of split position) is
    # the accumulated heterogeneity if I split foc[i] at index j
    weighted_heter = np.ones([len(foc), max(nof_splits, cat_limit)]) * big_M

    # limits of the feature of interest for each dataset in x_list
    # lims = np.array([effector.helpers.axis_limits_from_data(xx)[:,foi] for xx in x_list])

    # list with len(foc) elements
    # each element is a list with the split positions for the corresponding feature of conditioning
    positions = [
        find_positions_cat(data, foc_i)
        if foc_types[i] == "cat"
        else find_positions_cont(data, foc_i, nof_splits)
        for i, foc_i in enumerate(foc)
    ]

    # exhaustive search on all split positions
    for i, foc_i in enumerate(foc):
        for j, position in enumerate(positions[i]):
            # split datasets
            x_list_2 = flatten_list(
                [split_dataset(x, None, foc_i, position, foc_types[i]) for x in x_list]
            )
            x_jac_list_2 = flatten_list(
                [
                    split_dataset(x, x_jac, foc_i, position, foc_types[i])
                    for x, x_jac in zip(x_list, x_jac_list)
                ]
            )

            # sub_heter: list with the heterogeneity after split of foc_i at position j
            sub_heter = [
                rhale_heter(x, x_jac, model, foi, min_points)
                for x, x_jac in zip(x_list_2, x_jac_list_2)
            ]
            # heter_drop: list with the heterogeneity drop after split of foc_i at position j
            heter_drop = np.array(
                flatten_list(
                    [
                        [
                            heter_bef - sub_heter[int(2 * i)],
                            heter_bef - sub_heter[int(2 * i + 1)],
                        ]
                        for i, heter_bef in enumerate(heter_before)
                    ]
                )
            )
            # populations: list with the number of instances in each dataset after split of foc_i at position j
            populations = np.array([len(xx) for xx in x_list_2])
            # weights analogous to the populations in each split
            weights = (populations + 1) / (np.sum(populations + 1))
            # weighted_heter_drop[i,j] is the weighted accumulated heterogeneity drop if I split foc[i] at index j
            weighted_heter_drop[i, j] = np.sum(heter_drop * weights)
            # weighted_heter[i,j] is the weighted accumulated heterogeneity if I split foc[i] at index j
            weighted_heter[i, j] = np.sum(weights * np.array(sub_heter))

    # find the split with the largest weighted heterogeneity drop
    i, j = np.unravel_index(
        np.argmax(weighted_heter_drop, axis=None), weighted_heter_drop.shape
    )
    feature = foc[i]
    position = positions[i][j]
    split_positions = positions[i]
    # how many instances in each dataset after the min split
    x_list_2 = flatten_list(
        [split_dataset(x, None, foc[i], position, foc_types[i]) for x in x_list]
    )
    x_jac_list_2 = flatten_list(
        [
            split_dataset(x, x_jac, foc[i], position, foc_types[i])
            for x, x_jac in zip(x_list, x_jac_list)
        ]
    )
    nof_instances = [len(x) for x in x_list_2]
    sub_heter = [
        rhale_heter(x, x_jac, model, foi) for x, x_jac in zip(x_list_2, x_jac_list_2)
    ]
    split = {
        "feature": feature,
        "position": position,
        "candidate_split_positions": split_positions,
        "nof_instances": nof_instances,
        "type": foc_types[i],
        "heterogeneity": sub_heter,
        "split_i": i,
        "split_j": j,
        "foc": foc,
        "weighted_heter_drop": weighted_heter_drop[i, j],
        "weighted_heter": weighted_heter[i, j],
    }
    return split


def find_splits(
    nof_levels: int,
    nof_splits: int,
    foi: int,
    foc: list,
    foc_types: list,
    cat_limit: int,
    data: np.ndarray,
    model: typing.Callable,
    model_jac: typing.Callable,
    min_points: int = 10,
    criterion="rhale",
):
    """Find nof_level best splits for a given feature of interest and a list of features of conditioning"""

    assert nof_levels <= len(
        foc
    ), "nof_levels must be smaller than the number of features of conditioning"

    # initial heterogeneity
    heter_init = (
        rhale_heter(data, model_jac(data), model, foi, min_points)
        if criterion == "rhale"
        else pdp_heter(data, model, model_jac, foi, min_points)
    )

    # find optimal split for each level
    x_list = [data]
    x_jac_list = [model_jac(data)]
    splits = [
        {
            "heterogeneity": [heter_init],
            "weighted_heter": heter_init,
            "nof_instances": [len(data)],
            "split_i": -1,
            "split_j": -1,
            "foc": foc,
        }
    ]
    for lev in range(nof_levels):
        # if any x in x_list is empty then stop
        if any([len(x) == 0 for x in x_list]):
            break

        # find split
        split_fn = split_pdp if criterion == "pdp" else split_rhale
        split = split_fn(
            model,
            model_jac,
            data,
            x_list,
            x_jac_list,
            foi,
            foc_types,
            foc,
            nof_splits,
            cat_limit,
            splits[-1]["heterogeneity"],
            min_points,
        )
        splits.append(split)

        # split data and data_effect based on the optimal split found above
        feat, pos, typ = split["feature"], split["position"], split["type"]
        x_jac_list = flatten_list(
            [
                split_dataset(x, x_jac, feat, pos, typ)
                for x, x_jac in zip(x_list, x_jac_list)
            ]
        )
        x_list = flatten_list([split_dataset(x, None, feat, pos, typ) for x in x_list])
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

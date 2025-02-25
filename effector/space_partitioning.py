import typing
import numpy as np
from effector import helpers, utils
from effector.tree import Tree
import copy
import matplotlib.pyplot as plt

BIG_M = helpers.BIG_M


class Best:
    def __init__(
            self,
            min_heterogeneity_decrease_pcg: float = 0.1,
            heter_small_enough: float = 0.0,
            max_depth: int = 2,
            min_samples_leaf: int = 10,
            numerical_features_grid_size: int = 20,
            search_partitions_when_categorical: bool = False,
    ):
        """Choose the algorithm `Best`.
        The algorithm is a greedy algorithm that finds the best split for each level in a greedy fashion.


        Args:
            min_heterogeneity_decrease_pcg: Minimum percentage of heterogeneity decrease to accept a split.

                ??? Example "Example"
                    - `0.1`: if the heterogeneity before any split is 1, the heterogeneity after the first split must be at most 0.9 to be accepted. Otherwise, no split will be accepted.

            heter_small_enough: When heterogeneity is smaller than this value, no more splits are performed.

                ??? Note "Default is `0.0`"
                    By default, the algorithm will never stop due to heterogeneity being small enough; it will stop only if `max_depth` is reached.

                ??? Note "Custom value"
                    If you know a priori that a specific heterogeneity value is small enough, you can set this parameter to that value to speed up the algorithm.

            max_depth: Maximum number of splits to perform

                ??? Note "Default is `2`"
                    2 splits already create 4 subregions, i.e. 4 regional plots per feature, which are already enough.
                    Setting this value to a higher number will increase the number of subregions and plots, which may be too much for the user to analyze.

            min_samples_leaf: Minimum number of instances per subregion

                ??? Note "Default is `10`"
                    If a subregion has less than 10 instances, it may not be representative enough to be analyzed.

            numerical_features_grid_size: Number of candidate split positions for numerical features

                ??? Note "Default is `20`"
                    For numerical features, the algorithm will create a grid of 20 equally spaced values between the minimum and maximum values of the feature.

            search_partitions_when_categorical: Whether to search for partitions when the feature is categorical

                ??? warning "refers to a categorical feature of interest"
                    This argument asks whether to search for partitions when the feature of interest is categorical.
                    If the feature of interest is numerical, the algorithm will always search for partitions and will consider
                    categorical features for conditioning.

                ??? Note "Default is `False`"
                    It is difficult to compute the heterogeneity for categorical features, so by default, the algorithm will not search for partitions when the feature of interest is categorical.

        """
        # setters
        self.min_points_per_subregion = min_samples_leaf
        self.nof_candidate_splits_for_numerical = numerical_features_grid_size
        self.max_split_levels = max_depth
        self.heter_pcg_drop_thres = min_heterogeneity_decrease_pcg
        self.heter_small_enough = heter_small_enough
        self.split_categorical_features = search_partitions_when_categorical

        # to be set latter
        self.feature = None
        self.foi = None
        self.data = None
        self.dim = None
        self.heter_func = None
        self.axis_limits = None
        self.feature_types = None
        self.cat_limit = None
        self.feature_names = None
        self.target_name = None
        self.foc_types = None
        self.candidate_conditioning_features = None

        # init method args
        self.method_args = {}

        # init splits
        self.splits: dict = {}
        self.important_splits: dict = {}

        self.splits_tree: typing.Union[Tree, None] = None
        self.important_splits_tree: typing.Union[Tree, None] = None

        # state variable
        self.split_found: bool = False
        self.important_splits_selected: bool = False

    def _prapare_data(self, data):
        pass

    def find_subregions(self,
                        feature: int,
                        data: np.ndarray,
                        heter_func: callable,
                        axis_limits: np.ndarray,
                        feature_types: typing.Union[list, None] = None,
                        categorical_limit: int = 10,
                        candidate_conditioning_features: typing.Union[None, list] = None,
                        feature_names: typing.Union[None, list] = None,
                        target_name: typing.Union[None, str] = None
                        ):
        self.feature = feature
        self.foi = feature
        self.data = data
        self.dim = self.data.shape[1]
        self.heter_func = heter_func
        self.axis_limits = axis_limits
        self.cat_limit = categorical_limit
        self.feature_names = feature_names
        self.target_name = target_name

        self.candidate_conditioning_features = (
            [i for i in range(self.dim) if i != self.feature]
            if candidate_conditioning_features == "all"
            else candidate_conditioning_features
        )

        self.feature_types = utils.get_feature_types(data, categorical_limit) if feature_types is None else feature_types

        # on-init
        self.foc_types = [self.feature_types[i] for i in self.candidate_conditioning_features]

        self.search_all_splits()
        self.choose_important_splits()


    def search_all_splits(self):
        """
        Iterate over all features of conditioning and choose the best split for each level in a greedy fashion.
        """
        if (
            self.feature_types[self.feature] == "cat"
            and not self.split_categorical_features
        ):
            self.splits = []
        else:
            if self.max_split_levels > len(self.candidate_conditioning_features):
                self.max_split_levels = len(self.candidate_conditioning_features)

            active_indices = np.ones((self.data.shape[0]))
            heter_init = self.heter_func(active_indices)
            splits = [
                {
                    "after_split_active_indices_list": [active_indices],
                    "after_split_heter_list": [heter_init],
                    "after_split_weighted_heter": heter_init,
                    "after_split_nof_instances": [len(self.data)],
                    "split_i": -1,
                    "split_j": -1,
                    "candidate_conditioning_features": self.candidate_conditioning_features,
                }
            ]

            for lev in range(self.max_split_levels):
                # TODO: check this, as it seems redundant;
                # if any subregion had less than min_points, the
                # specific split should not have been selected
                if any([np.sum(x) < self.min_points_per_subregion for x in splits[-1]["after_split_active_indices_list"]]):
                    break

                # find optimal split
                new_split = self.single_level_splits(
                    splits[-1]["after_split_active_indices_list"]
                )
                splits.append(new_split)
            self.splits = splits

        # update state
        self.split_found = True
        return self.splits

    def single_level_splits(
        self,
        before_split_active_indices_list: typing.Union[list, None] = None,
    ):
        """Find all splits for a single level."""
        foc_types = self.foc_types
        ccf = self.candidate_conditioning_features
        nof_splits = self.nof_candidate_splits_for_numerical
        heter_func = self.heter_func

        data = self.data

        # matrix_weighted_heter[i,j] (i index of ccf and j index of split position) is
        # the accumulated heterogeneity if I split ccf[i] at index j
        matrix_weighted_heter = np.ones([
            len(self.candidate_conditioning_features),
            max(self.nof_candidate_splits_for_numerical-1, self.cat_limit)
        ]) * BIG_M

        # list with len(ccf) elements
        # each element is a list with the split positions for the corresponding feature of conditioning
        candidate_split_positions = [(self.find_positions_cat(data, foc_i) if foc_types[i] == "cat" else self.find_positions_cont(foc_i, nof_splits)) for i, foc_i in enumerate(self.candidate_conditioning_features)]

        # exhaustive search on all split positions
        for i, foc_i in enumerate(self.candidate_conditioning_features):
            for j, position in enumerate(candidate_split_positions[i]):
                after_split_active_indices_list = self.flatten_list([ self.split_dataset(active_indices, foc_i, position, foc_types[i])
                    for active_indices in before_split_active_indices_list
                ])

                heter_list_after_split = [heter_func(x) for x in after_split_active_indices_list]

                # populations: list with the number of instances in each dataset after split of foc_i at position j
                populations = np.array([np.sum(x) for x in after_split_active_indices_list])

                # after_split_weight_list analogous to the populations in each split
                after_split_weight_list = (populations + 1) / (np.sum(populations + 1))

                # first: computed the weighted heterogeneity after the split
                after_split_weighted_heter = np.sum(after_split_weight_list * np.array(heter_list_after_split))

                # matrix_weighted_heter[i,j] is the weighted accumulated heterogeneity if I split ccf[i] at index j
                matrix_weighted_heter[i, j] = after_split_weighted_heter

        # find the split with the largest weighted heterogeneity drop
        i, j = np.unravel_index(
            np.argmin(matrix_weighted_heter, axis=None), matrix_weighted_heter.shape
        )
        feature = ccf[i]
        position = candidate_split_positions[i][j]
        split_positions = candidate_split_positions[i]

        after_split_active_indices_list = self.flatten_list([self.split_dataset(active_indices, ccf[i], position, foc_types[i]) for active_indices in before_split_active_indices_list])

        nof_instances_l = [np.sum(x) for x in after_split_active_indices_list]

        # TODO change that
        after_split_heter_l = [heter_func(ai) for ai in after_split_active_indices_list]
        split = {
            "foc_index": ccf[i],
            "foc_split_position": position,
            "foc_range": [np.min(data[:, feature]), np.max(data[:, feature])],
            "foc_type": foc_types[i],
            "split_i": i,
            "split_j": j,
            "candidate_split_positions": split_positions,
            "candidate_conditioning_features": ccf,

            "after_split_nof_instances": nof_instances_l,
            "after_split_heter_list": after_split_heter_l,
            "after_split_active_indices_list": after_split_active_indices_list,
            "after_split_weighted_heter": matrix_weighted_heter[i, j],

            # "matrix_weighted_heter_drop": matrix_weighted_heter_drop,
            "matrix_weighted_heter": matrix_weighted_heter,
        }
        return split

    def choose_important_splits(self):
        assert self.split_found, "No splits found for feature {}".format(self.feature)

        # if split is empty, skip
        if len(self.splits) == 0:
            optimal_splits = {}
        # if initial heterogeneity is BIG_M, skip
        elif self.splits[0]["after_split_weighted_heter"] == BIG_M:
            optimal_splits = {}
        # if initial heterogeneity is small right from the beginning, skip
        elif self.splits[0]["after_split_weighted_heter"] <= self.heter_small_enough:
            optimal_splits = {}
        else:
            splits = self.splits

            # accept split if heterogeneity drops over 20%
            heter = np.array([splits[i]["after_split_weighted_heter"] for i in range(len(splits))])
            heter_drop = (heter[:-1] - heter[1:]) / heter[:-1]
            split_valid = heter_drop > self.heter_pcg_drop_thres

            # if all are negative, return nothing
            if np.sum(split_valid) == 0:
                optimal_splits = {}
            # if all are positive, return all
            elif np.sum(split_valid) == len(split_valid):
                optimal_splits = splits[1:]
            else:
                # find first negative split
                first_negative = np.where(split_valid == False)[0][0]

                # if first negative is the first split, return nothing
                if first_negative == 0:
                    optimal_splits = {}
                else:
                    optimal_splits = splits[1 : first_negative + 1]

        # update state variable
        self.important_splits_selected = True
        self.important_splits = optimal_splits
        return optimal_splits

    def split_dataset(self, active_indices, feature, position, feat_type):
        if feat_type == "cat":
            ind_1 = self.data[:, feature] == position
            ind_2 = self.data[:, feature] != position
        else:
            ind_1 = self.data[:, feature] < position
            ind_2 = self.data[:, feature] >= position

        # active indices is a (N,) array with 1s and 0s, where N is the number of the total instances
        # all instances in x and x_jac have a 1 in active_indices, else 0
        active_indices_1 = np.copy(active_indices)
        active_indices_2 = np.copy(active_indices)
        active_indices_1 = np.logical_and(active_indices_1, ind_1)
        active_indices_2 = np.logical_and(active_indices_2, ind_2)
        return active_indices_1, active_indices_2

    def find_positions_cat(self, x, feature):
        return np.unique(x[:, feature])

    def find_positions_cont(self, feature, nof_splits):
        start = self.axis_limits[0, feature]
        stop = self.axis_limits[1, feature]
        pos = np.linspace(start, stop, nof_splits+1)
        return pos[1:-1]

    def flatten_list(self, l):
        return [item for sublist in l for item in sublist]

    def splits_to_tree(self, only_important=True, scale_x_list=None):
        if len(self.splits) == 0:
            return None

        nof_instances = self.splits[0]["after_split_nof_instances"][0]
        tree = Tree()
        # format with two decimals
        data = {
            "heterogeneity": self.splits[0]["after_split_heter_list"][0],
            "feature_name": self.feature,
            "nof_instances": self.splits[0]["after_split_nof_instances"][0],
            "weight": 1.0,
            "active_indices": np.ones((self.data.shape[0])),
        }

        feature_name = self.feature_names[self.feature]
        data["level"] = 0
        tree.add_node(feature_name, None, data=data)
        parent_level_nodes = [feature_name]
        parent_level_active_indices = [np.ones((self.data.shape[0]))]
        splits = self.important_splits if only_important else self.splits[1:]

        for i, split in enumerate(splits):

            # nof nodes to add
            nodes_to_add = len(split["after_split_nof_instances"])

            new_parent_level_nodes = []

            new_parent_level_active_indices = []

            # find parent
            for j in range(nodes_to_add):
                parent_name = parent_level_nodes[int(j / 2)]

                parent_active_indices = parent_level_active_indices[int(j / 2)]

                # prepare data

                foc_name = self.feature_names[split["foc_index"]]
                foc = split["foc_index"]
                pos = split["foc_split_position"]
                if scale_x_list is not None:
                    mean = scale_x_list[foc]["mean"]
                    std = scale_x_list[foc]["std"]
                    pos_scaled = std * split["foc_split_position"] + mean
                else:
                    pos_scaled = pos

                pos_small = pos_scaled.round(2)

                active_indices_1, active_indices_2 = self.split_dataset(
                    parent_active_indices,
                    foc, pos, split["foc_type"],
                    )

                active_indices_new = active_indices_1 if j % 2 == 0 else active_indices_2
                if j % 2 == 0:
                    if split["foc_type"] == "cat":
                        name = foc_name + " == {}".format(pos_small)
                        comparison = "=="
                    else:
                        name = foc_name + " <= {}".format(pos_small)
                        comparison = "<="
                else:
                    if split["foc_type"] == "cat":
                        name = foc_name + " != {}".format(pos_small)
                        comparison = "!="
                    else:
                        name = foc_name + "  > {}".format(pos_small)
                        comparison = ">"

                name = (
                    parent_name + " | " + name
                    if nodes_to_add == 2
                    else parent_name + " and " + name
                )

                data = {
                    "heterogeneity": split["after_split_heter_list"][j],
                    "weight": float(split["after_split_nof_instances"][j]) / nof_instances,
                    "position": split["foc_split_position"],
                    "foc_name": foc_name,
                    "feature": split["foc_index"],
                    "feature_type": split["foc_type"],
                    "range": split["foc_range"],
                    "candidate_split_positions": split["candidate_split_positions"],
                    "nof_instances": split["after_split_nof_instances"][j],
                    "active_indices": active_indices_new,
                    "comparison": comparison,
                }

                data["level"] = i + 1
                tree.add_node(name, parent_name=parent_name, data=data)

                new_parent_level_nodes.append(name)
                new_parent_level_active_indices.append(active_indices_new)

            # update parent_level_nodes
            parent_level_nodes = new_parent_level_nodes
            parent_level_active_indices = new_parent_level_active_indices

        return tree

    def visualize_all_splits(self, split_ind):
        split_ind = split_ind + 1
        heter_matr = copy.deepcopy(self.splits[split_ind]["matrix_weighted_heter"])
        heter_matr[heter_matr > 1e6] = np.nan

        plt.figure()
        plt.title("split {}, parent heter: {:.2f}".format(split_ind, self.splits[split_ind - 1]["after_split_weighted_heter"]))
        plt.imshow(heter_matr)
        plt.colorbar()
        plt.yticks([i for i in range(len(self.candidate_conditioning_features))], [self.feature_names[foc] for foc in self.candidate_conditioning_features])
        plt.show(block=False)


def return_default(partitioner_name):
    if partitioner_name == "best":
        return Best()
    else:
        raise ValueError("Partitioner not found")

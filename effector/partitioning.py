import typing
import numpy as np
import itertools
from effector import helpers, utils
import copy
import matplotlib.pyplot as plt

BIG_M = helpers.BIG_M


class Regions:
    def __init__(
            self,
            heter_pcg_drop_thres=0.1,
            heter_small_enough=0.,
            max_split_levels=2,
            min_points_per_subregion: int = 10,
            nof_candidate_splits_for_numerical=20,
            split_categorical_features=False,
    ):
        # setters
        self.min_points_per_subregion = min_points_per_subregion
        self.nof_candidate_splits_for_numerical = nof_candidate_splits_for_numerical
        self.max_split_levels = max_split_levels
        self.heter_pcg_drop_thres = heter_pcg_drop_thres
        self.heter_small_enough = heter_small_enough
        self.split_categorical_features = split_categorical_features

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
        tree.add_node(feature_name, None, data=data, level=0)
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

                tree.add_node(name, parent_name=parent_name, data=data, level=i + 1)

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



class Node:
    def __init__(
            self,
            idx: int,
            name: str,
            parent_node: typing.Union[None, "Node"],
            info: dict,
            level: int,
    ):
        """A node in the tree.

        Args:
            idx: Index, unique for each node in the tree
            name: Name of the node, e.g. "x1 | x2 <= 0.5"
            parent_node: Parent node
            info: A dictionary with the following keys

                - heterogeneity: Heterogeneity of the node
                - weight: Weight of the node
                - position: float, position of conditioning, e.g. 0.5
                - feature: index (zero-based) of the feature of conditioning, e.g. 1
                - feature_type: type of the feature of conditioning, e.g. "cont" or "cat"
                - range: list with two elements, min and max of the feature of conditioning
                - candidate_split_positions: list with the candidate split positions
                - nof_instances: Number of instances in the node
                - data: the data in the node
                - data_effect: the effect data in the node
                - comparison: the comparison operator, e.g. "<="
            level: Level of the node in the tree
        """
        self.idx = idx
        self.name = name
        self.parent_node = parent_node
        self.info = info
        self.level = level

        self.heterogeneity = info.get("heterogeneity")
        self.weight = info.get("weight")
        self.nof_instances = info.get("nof_instances")

        self.foc_index = info.get("feature")
        self.foc_name = info.get("foc_name")
        self.foc_type = info.get("feature_type")
        self.foc_position = info.get("position")
        self.foc_comparison_operator = info.get("comparison")
        self.foc_candidate_split_values = info.get("candidate_split_positions")
        self.foc_range = info.get("range")

    def show(self, show_extensive_info: bool = False):
        """

        Args:
            show_extensive_info: Whether to show extensive information, e.g. candidate split positions,

        Returns:

        """
        print("Node id: ", self.idx)
        print("name: ", self.name)
        print("parent name: ", getattr(self.parent_node, "name", None))
        print("level: ", self.level)

        print("heterogeneity: ", self.heterogeneity)
        print("weight: ", self.weight)
        print("nof_instances: ", self.nof_instances)

        print("foc_index: ", self.foc_index)
        print("foc_type: ", self.foc_type)
        print("foc_position: ", self.foc_position)
        print("fox_comparison_operator: ", self.foc_comparison_operator)

        print("info: ", self.info) if show_extensive_info else None


class Tree:
    def __init__(self):
        self.nodes = []
        self.idx = 0

    def scale_node_name(self, name, scale_x_list):

        node = self.get_node_by_name(name)

        # get all parents
        parents = []
        cur_node = node
        while cur_node.parent_node is not None:
            parents.append(cur_node.parent_node)
            cur_node = cur_node.parent_node

        if len(parents) == 0:
            new_name = name
        else:
            parents.reverse()
            new_name = ""
            for ii, parent in enumerate(parents):
                if ii == 0:
                    new_name = new_name + parent.name + " | "
                else:
                    foc = parent.info["feature"]
                    foc_name = parent.info["foc_name"]
                    pos = parent.info["position"]
                    scaled_pos = scale_x_list[foc]["std"] * pos + scale_x_list[foc]["mean"] if scale_x_list is not None else pos
                    conditioning_string = "{} {} {:.2f} and ".format(foc_name, parent.info["comparison"], scaled_pos)
                    new_name = new_name + conditioning_string
            foc = node.info["feature"]
            foc_name = node.info["foc_name"]
            pos = node.info["position"]
            scaled_pos = scale_x_list[foc]["std"] * pos + scale_x_list[foc]["mean"] if scale_x_list is not None else pos
            conditioning_string = "{} {} {:.2f}".format(foc_name, node.info["comparison"], scaled_pos)
            new_name = new_name + conditioning_string
        return new_name

    def add_node(
            self,
            name: str,
            parent_name: typing.Union[None, str],
            data: dict,
            level: int
            ):
        if parent_name is None:
            parent_node = None
        else:
            assert parent_name in [node.name for node in self.nodes]
            parent_node = self.get_node_by_name(parent_name)

        idx = self.idx
        self.idx += 1
        node = Node(idx, name, parent_node, data, level)
        self.nodes.append(node)

    def get_node_by_name(self, name):
        node = None
        for node_i in self.nodes:
            if node_i.name == name:
                node = node_i
                break
        return node

    def get_node_by_idx(self, idx):
        node = None
        for node_i in self.nodes:
            if node_i.idx == idx:
                node = node_i
                break
        return node

    def get_level_nodes(self, level):
        nodes = []
        for node_i in self.nodes:
            if node_i.level == level:
                nodes.append(node_i)
        return nodes

    def get_root(self):
        node = None
        for node_i in self.nodes:
            if node_i.parent_node is None:
                node = node_i
                break
        assert node is not None
        return node

    def get_children(self, name):
        children = []
        for node_i in self.nodes:
            if node_i.parent_node is not None:
                if node_i.parent_node.name == name:
                    children.append(node_i)
        return children

    def get_level_stats(self, level):
        level_nodes = self.get_level_nodes(level)

        w_heter = 0
        for nod in level_nodes:
            w_heter += nod.info["heterogeneity"] * nod.info["weight"]

        return {"heterogeneity": w_heter}

    def show_full_tree(self, node=None, scale_x_list=None):
        if node is None:
            node = self.get_root()

        indent = node.level * 2
        print(
            "    " * indent
            + "Node id: %d, name: %s, heter: %.2f || nof_instances: %5d || weight: %.2f"
            % (
                node.idx,
                node.name if scale_x_list is None else self.scale_node_name(node.name, scale_x_list),
                node.info["heterogeneity"],
                node.info["nof_instances"],
                node.info["weight"],
            )
        )
        children = self.get_children(node.name)
        for child in children:
            self.show_full_tree(child, scale_x_list)

    def show_level_stats(self):
        max_level = max([node.level for node in self.nodes])
        prev_heter = 0
        for lev in range(max_level + 1):
            level_stats = self.get_level_stats(lev)
            if lev == 0:
                print(
                    "    " * lev * 2
                    + "Level %.d, heter: %.2f" % (lev, level_stats["heterogeneity"])
                )
            else:
                print(
                    "    " * lev * 2
                    + "Level %.d, heter: %.2f || heter drop : %.2f (units), %.2f%% (pcg)"
                    % (
                        lev,
                        level_stats["heterogeneity"],
                        prev_heter - level_stats["heterogeneity"],
                        100 * (prev_heter - level_stats["heterogeneity"]) / prev_heter,
                    )
                )
            prev_heter = level_stats["heterogeneity"]


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

        # the enhanced data, without masking
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

def return_default(partitioner_name):
    if partitioner_name == "greedy":
        return Regions()
    else:
        raise ValueError("Partitioner not found")

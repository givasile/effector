import typing
import itertools
import numpy as np


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


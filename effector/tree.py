import typing
import itertools
import numpy as np


class Tree:
    def __init__(self):
        self.nodes = []
        self.idx = 0

    def scale_node_name(self, name, scale_x_list):
        node = self.get_node_by_name(name)
        chain = []
        cur = node
        while cur.parent_node:
            chain.append(cur.parent_node)
            cur = cur.parent_node
        if not chain:
            return name
        chain.reverse()
        # First element: use parent's name
        first, *rest = chain
        conds = [
            f"{n.info['foc_name']} {n.info['comparison']} "
            f"{(scale_x_list[n.info['feature']]['std'] * n.info['position'] + scale_x_list[n.info['feature']]['mean']) if scale_x_list else n.info['position']:.2f}"
            for n in (rest + [node])
        ]
        return f"{first.name} | " + " and ".join(conds)

    def add_node(
        self,
        name: str,
        parent_name: typing.Union[None, str],
        data: dict,
        level: int,
    ):
        if parent_name is not None:
            parent_node = self.get_node_by_name(parent_name)
            if parent_node is None:
                raise ValueError(f"Parent node '{parent_name}' not found.")
        else:
            parent_node = None
        node = Node(self.idx, name, parent_node, data, level)
        self.idx += 1
        self.nodes.append(node)

    def get_node_by_name(self, name):
        return next((node for node in self.nodes if node.name == name), None)

    def get_node_by_idx(self, idx):
        return next((node for node in self.nodes if node.idx == idx), None)

    def get_level_nodes(self, level):
        return [node for node in self.nodes if node.level == level]

    def get_root(self):
        root = next((node for node in self.nodes if node.parent_node is None), None)
        assert root is not None, "Root not found."
        return root

    def get_children(self, name):
        return [node for node in self.nodes if node.parent_node and node.parent_node.name == name]

    def update_display_names(self, scale_x_list):
        for node in self.nodes:
            node.display_name = self.scale_node_name(node.name, scale_x_list)

    def get_level_stats(self, level):
        level_nodes = self.get_level_nodes(level)
        w_heter = sum(n.info["heterogeneity"] * n.info["weight"] for n in level_nodes)
        return {"heterogeneity": w_heter}

    def show_full_tree(self, node=None, scale_x_list=None):
        self.update_display_names(scale_x_list)
        if node is None:
            node = self.get_root()
        self._recursive_print_full_tree(node)

    def _recursive_print_full_tree(self, node=None):
        indent = "    " * node.level
        disp_name = node.display_name #  self.scale_node_name(node.name, scale_x_list)
        print(
            f"{indent}Node id: {node.idx}, name: {disp_name}, heter: {node.info['heterogeneity']:.2f} "
            f"|| nof_instances: {node.info['nof_instances']:5d} || weight: {node.info['weight']:.2f}"
        )
        for child in self.get_children(node.name):
            self._recursive_print_full_tree(child)

    def show_level_stats(self):
        max_level = max(node.level for node in self.nodes)
        prev_heter = 0
        for lev in range(max_level + 1):
            stats = self.get_level_stats(lev)
            if lev == 0:
                print(f"Level {lev}, heter: {stats['heterogeneity']:.2f}")
            else:
                drop = prev_heter - stats["heterogeneity"]
                perc = 100 * drop / prev_heter if prev_heter else 0
                print(
                    f"Level {lev}, heter: {stats['heterogeneity']:.2f} || heter drop: {drop:.2f} (units), {perc:.2f}% (pcg)"
                )
            prev_heter = stats["heterogeneity"]


# class DataTransformer:
#     def __init__(self, splits: typing.Dict):
#         self.splits = splits
#
#     def transform(self, X):
#         # Determine how many times each feature should be repeated.
#         feat_mapping = [1 if len(split) == 0 else 2 ** len(split) for split in self.splits.values()]
#
#         # Repeat each column according to feat_mapping.
#         new_data = np.concatenate(
#             [np.repeat(X[:, i, np.newaxis], rep, axis=-1) for i, rep in enumerate(feat_mapping)],
#             axis=-1,
#         )
#
#         mask = np.ones(new_data.shape, dtype=bool)
#         new_columns = []
#         for feat in range(X.shape[1]):
#             pos = int(sum(feat_mapping[:feat]))
#             if feat_mapping[feat] == 1:
#                 new_columns.append(f"x{feat}")
#                 continue
#
#             feat_splits = self.splits[f"feat_{feat}"]
#             # Create all binary combinations for the splits.
#             for ii, bin_choice in enumerate(list(itertools.product([0, 1], repeat=len(feat_splits)))):
#                 col_name = f"x{feat} | "
#                 col_mask = np.ones(new_data.shape[0], dtype=bool)
#                 for jj, b in enumerate(bin_choice):
#                     split = feat_splits[jj]
#                     if b == 0:
#                         if split["type"] == "cat":
#                             col_mask &= (X[:, split["feature"]] == split["position"])
#                             col_name += f"x{split['feature']}={split['position']:.2f} & "
#                         else:
#                             col_mask &= (X[:, split["feature"]] <= split["position"])
#                             col_name += f"x{split['feature']}<={split['position']:.2f} & "
#                     else:
#                         if split["type"] == "cat":
#                             col_mask &= (X[:, split["feature"]] != split["position"])
#                             col_name += f"x{split['feature']}!={split['position']:.2f} & "
#                         else:
#                             col_mask &= (X[:, split["feature"]] > split["position"])
#                             col_name += f"x{split['feature']}>{split['position']:.2f} & "
#                 mask[:, pos + ii] = col_mask
#                 new_columns.append(col_name[:-3])
#         self.mask = mask
#         self.new_data = new_data * mask
#         self.new_names = new_columns
#         return self.new_data



class Node:
    def __init__(
        self,
        idx: int,
        name: str,
        parent_node: typing.Union[None, "Node"],
        info: dict,
        level: int,
    ):
        self.idx = idx
        self.name = name
        self.display_name = None
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
        self.active_indices = info.get("active_indices")

    def show(self, show_extensive_info: bool = False):
        print(f"Node id: {self.idx}")
        print(f"name: {self.name}")
        print(f"display_name: {self.display_name}")
        print(f"parent name: {getattr(self.parent_node, 'name', None)}")
        print(f"level: {self.level}")
        print(f"heterogeneity: {self.heterogeneity}")
        print(f"weight: {self.weight}")
        print(f"nof_instances: {self.nof_instances}")
        print(f"foc_index: {self.foc_index}")
        print(f"foc_type: {self.foc_type}")
        print(f"foc_position: {self.foc_position}")
        print(f"fox_comparison_operator: {self.foc_comparison_operator}")
        if show_extensive_info:
            print(f"info: {self.info}")

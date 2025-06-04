import typing
from dataclasses import dataclass, field


class Tree:
    """A class to represent a tree structure."""

    def __init__(self):
        self.nodes = []
        self.node_dict = {}  # Fast lookup by name
        self.idx = 0

    def set_display_name(self, name, scale_x_list, full=True):
        node = self.get_node_by_name(name)
        if node is None:
            return name

        chain = self._get_parent_chain(node)
        if not chain:
            return name

        first, *rest = chain
        conds = [self._get_condition_string(n, scale_x_list) for n in (rest + [node])]

        return f"{first.name} | " + " and ".join(conds) if full else conds[-1]

    def _get_condition_string(self, node, scale_x_list):
        """Generate the condition string for a node."""
        pos = node.info["foc_split_position"]
        if scale_x_list:
            feature_stats = scale_x_list[node.info["foc_index"]]
            pos = feature_stats["std"] * pos + feature_stats["mean"]
        return f"{node.info['foc_name']} {self._comparison_str(node.info['comparison'])} {pos:.2f}"

    @staticmethod
    def _comparison_str(comparison):
        return {">=": "≥", "<=": "≤", "!=": "≠", "==": "="}.get(comparison, comparison)

    def create_node_name(self, name, parent=None, comp=None, pos=None):
        if parent is None:
            return name

        assert (
            comp is not None and pos is not None
        ), "Comparison and position must be provided if parent is specified."

        name = f"{name} {self._comparison_str(comp)} {pos}"

        if parent.info["level"] == 0:
            return f"{parent.name} | {name}"

        return f"{parent.name} and {name}"

    def add_node(self, name: str, parent_name: typing.Optional[str], data: dict):
        parent_node = self.node_dict.get(parent_name) if parent_name else None
        if parent_name and parent_node is None:
            raise ValueError(f"Parent node '{parent_name}' not found.")

        node = Node(self.idx, name, parent_node, data)
        self.idx += 1
        self.nodes.append(node)
        self.node_dict[name] = node  # Store in dictionary for fast lookup

    def get_node_by_name(self, name):
        return self.node_dict.get(name)

    def get_node_by_idx(self, idx):
        return next((node for node in self.nodes if node.idx == idx), None)

    def get_root(self):
        return next((node for node in self.nodes if node.parent_node is None), None)

    def get_children(self, name):
        parent = self.get_node_by_name(name)
        return [node for node in self.nodes if node.parent_node == parent]

    def update_display_names(self, scale_x_list):
        for node in self.nodes:
            node.display_name = self.set_display_name(node.name, scale_x_list)
            node.display_name_short = self.set_display_name(
                node.name, scale_x_list, full=False
            )

    def show_full_tree(self, scale_x_list=None):
        """Print the full tree structure.

        Examples:
            >>> tree.show_full_tree()
            🌳 Full Tree Structure:
            ───────────────────────
            x1 🔹 [id: 0 | heter: 0.50 | inst: 100 | w: 1.00]
                x2 ≥ 3.00 🔹 [id: 1 | heter: 0.30 | inst: 50 | w: 0.50]
                x2 < 3.00 🔹 [id: 2 | heter: 0.20 | inst: 50 | w: 0.50]

            >>> tree.show_full_tree({"mean": 3, "std":2}, {"mean": 3, "std":3}, {"mean": 3, "std":2})
            🌳 Full Tree Structure:
            ───────────────────────
            x1 🔹 [id: 0 | heter: 0.50 | inst: 100 | w: 1.00]
                x2 ≥ 12.00 🔹 [id: 1 | heter: 0.30 | inst: 50 | w: 0.50]
                x2 < 12.00 🔹 [id: 2 | heter: 0.20 | inst: 50 | w: 0.50]

        Args:
            scale_x_list: A list of dictionaries with the mean and standard deviation for each feature.
                               - Example: [{"mean": 3, "std":2}, {"mean": 3, "std":2}, ...]


        """
        self.update_display_names(scale_x_list)
        root = self.get_root()
        if root is None:
            print("Tree is empty.")
            return

        print("🌳 Full Tree Structure:")
        print("───────────────────────")
        self._recursive_print_full_tree(root)

    def _recursive_print_full_tree(self, node):
        indent = "    " * node.info["level"]
        print(
            f"{indent}{node.display_name_short} 🔹 [id: {node.idx} | heter: {node.info['heterogeneity']:.2f} "
            f"| inst: {node.info['nof_instances']:d} | w: {node.info['weight']:.2f}]"
        )
        for child in self.get_children(node.name):
            self._recursive_print_full_tree(child)

    def show_level_stats(self):
        """Print the heterogeneity drop at each level of the tree.

        Examples:
            >>> tree.show_level_stats()
            🌳 Tree Summary:
            ─────────────────
            Level 0🔹heter: 0.50
                Level 1🔹heter: 0.25 | 🔻0.25 (50.00%)
        """

        max_level = max((node.info["level"] for node in self.nodes), default=0)
        prev_heter = 0
        print("🌳 Tree Summary:")
        print("─────────────────")
        for lev in range(max_level + 1):
            indent = "    " * lev
            stats = self.get_level_stats(lev)
            if lev == 0:
                print(f"Level {lev}🔹heter: {stats['heterogeneity']:.2f}")
            else:
                drop = prev_heter - stats["heterogeneity"]
                perc = 100 * drop / prev_heter if prev_heter else 0
                print(
                    f"{indent}Level {lev}🔹heter: {stats['heterogeneity']:.2f} | 🔻{drop:.2f} ({perc:.2f}%)"
                )
            prev_heter = stats["heterogeneity"]

    def get_level_stats(self, level):
        level_nodes = [node for node in self.nodes if node.info["level"] == level]
        return {
            "heterogeneity": sum(
                n.info["heterogeneity"] * n.info["weight"] for n in level_nodes
            )
        }

    @staticmethod
    def _get_parent_chain(node):
        """Returns the chain from the root to the given node."""
        chain = []
        cur = node
        while cur.parent_node:
            chain.append(cur.parent_node)
            cur = cur.parent_node
        chain.reverse()
        return chain


@dataclass
class Node:
    idx: int  # name must be unique
    name: str
    parent_node: typing.Optional["Node"]
    info: dict
    display_name: str = field(default=None, init=False)
    display_name_short: str = field(default=None, init=False)

    def __post_init__(self):
        required_keys = {"level", "heterogeneity", "active_indices"}
        if not required_keys.issubset(self.info):
            missing = required_keys - self.info.keys()
            raise KeyError(f"Missing required keys in node info: {missing}")

        self.info["nof_instances"] = int(self.info["active_indices"].sum())
        self.info["weight"] = float(
            self.info["nof_instances"] / self.info["active_indices"].shape[0]
        )
        self.info["weighted_heterogeneity"] = float(
            self.info["heterogeneity"] * self.info["weight"]
        )

        if self.info["level"] > 0:
            extra_keys = {
                "foc_index",
                "foc_name",
                "foc_type",
                "foc_split_position",
                "comparison",
            }
            if not extra_keys.issubset(self.info):
                missing = extra_keys - self.info.keys()
                raise KeyError(f"Missing required keys for non-root node: {missing}")

    def __repr__(self):
        return f"Node---idx: {self.idx}, name: {self.name}"


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

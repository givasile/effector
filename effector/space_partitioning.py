import copy
import typing

import numpy as np

from effector import helpers, ingestion
from effector.partition import Partition, Region, partition_from_tree
from effector.tree import Tree

BIG_M = helpers.BIG_M


class Base:
    def __init__(
        self,
        name: str,
        min_heterogeneity_decrease_pcg: float = 0.1,
        heter_small_enough: float = 0.001,
        max_depth: int = 2,
        min_samples_leaf: int = 10,
        numerical_features_grid_size: int = 20,
        search_partitions_when_categorical: bool = True,
    ):
        """Shared configuration of the space partitioners.

        Args:
            min_heterogeneity_decrease_pcg: Minimum percentage of heterogeneity decrease to accept a split.

                ??? Example "Example"
                    - `0.1`: if the heterogeneity before any split is 1, the heterogeneity after the first split must be at most 0.9 to be accepted. Otherwise, no split will be accepted.

            heter_small_enough: When heterogeneity is smaller than this value, no more splits are performed.

                ??? Note "Default is `0.001`"
                    Value 0.001 is small enough for most cases.
                    It is advisable to set this value to a small number to avoid unnecessary splits.

                ??? Note "Custom value"
                    If you know a priori that a specific heterogeneity value is small enough,
                    you can set this parameter to a higher value than the default.

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
        self.name = helpers.camel_to_snake(name)

        self.min_points_per_subregion = min_samples_leaf
        self.nof_candidate_splits_for_numerical = numerical_features_grid_size
        self.max_split_levels = max_depth
        self.heter_pcg_drop_thres = min_heterogeneity_decrease_pcg
        self.heter_small_enough = heter_small_enough
        self.split_categorical_features = search_partitions_when_categorical

        # all methods will set these attributes
        self.feature = None  # feature of interest
        self.foi = None  # feature of interest
        self.data = None  # dataset (N, D)
        self.dim = None  # dimensionality of the dataset (= D)
        self.heter_func = None  # heterogeneity function (callable (mask) -> float)
        self.axis_limits = (
            None  # axis limits (min and max for each feature), shape (2, D)
        )
        self.feature_types = None  # feature types (continuous/ordinal/nominal)
        self.cat_limit = None  # categorical limit
        self.feature_names = None  # feature names
        self.target_name = None  # target name
        self.foc_types = None  # feature-of-conditioning types (three-way taxonomy)
        self.candidate_conditioning_features = None  # candidate conditioning features

        self.splits_tree: typing.Union[Tree, None] = None  # the output of the algorithm

    def compile(
        self,
        feature: int,
        data: np.ndarray,
        heter_func: callable,
        axis_limits: np.ndarray,
        feature_types: typing.Union[list, None] = None,
        cat_limit: int = 10,
        candidate_conditioning_features: typing.Union[str, list] = "all",
        feature_names: typing.Union[None, list] = None,
        target_name: typing.Union[None, str] = None,
    ):
        "Tidy up the input data."
        self.feature = feature
        self.foi = feature
        self.data = data
        self.dim = self.data.shape[1]
        self.heter_func = heter_func
        self.axis_limits = axis_limits
        self.cat_limit = cat_limit
        self.feature_names = feature_names
        self.target_name = target_name

        self.candidate_conditioning_features = helpers.prep_conditioning_features(
            candidate_conditioning_features, feature, self.dim
        )

        self.feature_types = (
            ingestion.infer_feature_types(data, cat_limit)
            if feature_types is None
            else feature_types
        )
        self.foc_types = [
            self.feature_types[i] for i in self.candidate_conditioning_features
        ]

    def find_regions(
        self,
        feature,
        data,
        score_fn,
        *,
        axis_limits,
        feature_types,
        cat_limit,
        candidate_conditioning_features,
        feature_names,
        target_name,
    ) -> Partition:
        """Finder protocol: given a ``mask -> float`` ``score_fn`` plus the data
        and metadata needed to PROPOSE candidate splits, return a `Partition`.

        Any object exposing this method (returning a `Partition`) is a valid
        region finder. ``score_fn`` is RAW — it may raise ``ValueError`` or
        return nan; this adapter owns the min-points and degeneracy guard, so the
        ``BIG_M`` vocabulary lives here and never leaks into the effect.
        """
        if self.min_points_per_subregion < 2:
            raise ValueError("min_points_per_subregion must be >= 2")

        from effector import utils  # local import for the except tuple

        def guarded(active_indices):
            mask = active_indices.astype(bool)
            if mask.sum() < self.min_points_per_subregion:
                return BIG_M
            try:
                score = score_fn(mask)
            except (utils.AllBinsHaveAtMostOnePointError, ValueError):
                return BIG_M
            return score if np.isfinite(score) else BIG_M

        # compile() mutates the partitioner in place; work on a copy so the
        # caller's instance stays reusable (RC5 non-mutation contract).
        worker = copy.deepcopy(self)
        worker.compile(
            feature,
            data,
            guarded,
            axis_limits,
            feature_types,
            cat_limit,
            candidate_conditioning_features,
            feature_names,
            target_name,
        )
        tree = worker.fit()

        if tree is None or len(tree.nodes) == 0:
            # BestLevelWise no-search path: a root-only Partition.
            n = data.shape[0]
            root = Region(
                idx=0,
                name=feature_names[feature],
                mask=np.ones(n, dtype=bool),
                heterogeneity=float(guarded(np.ones(n))),
                nof_instances=n,
                weight=1.0,
                level=0,
                parent_idx=None,
            )
            return Partition(
                [root],
                feature=feature,
                feature_name=feature_names[feature],
                finder_name=self.name,
            )

        return partition_from_tree(
            tree,
            feature=feature,
            feature_name=feature_names[feature],
            finder_name=self.name,
        )

    def fit(self) -> Tree:
        """Find the subregions."""
        raise NotImplementedError

    def _split_dataset(self, active_indices, feature, position, feat_type):
        if ingestion.is_categorical(feat_type):
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

    def _find_positions_cat(self, x, feature):
        return np.unique(x[:, feature])

    def _find_positions_cont(self, feature, nof_splits):
        start = self.axis_limits[0, feature]
        stop = self.axis_limits[1, feature]
        pos = np.linspace(start, stop, nof_splits + 1)
        return pos[1:-1]

    def _flatten_list(self, nested_list):
        return [item for sublist in nested_list for item in sublist]

    @staticmethod
    def _get_comparison_symbol(foc_type, i):
        if ingestion.is_categorical(foc_type):
            return "==" if i == 0 else "!="
        else:
            return "<=" if i == 0 else ">"

    def _evaluate_splits(self, before_split_active_indices_list: list) -> dict:
        """The shared split search (§2.7): exhaustive scan over every
        (conditioning feature, split position) pair, applied to *each* set of
        active indices in the list (one set = node-wise, a whole level =
        level-wise), and return the split minimizing the weighted
        heterogeneity."""
        foc_types = self.foc_types
        ccf = self.candidate_conditioning_features
        nof_splits = self.nof_candidate_splits_for_numerical
        heter_func = self.heter_func
        data = self.data

        # candidate_split_positions[i] is the list of split positions for the
        # i-th feature of conditioning
        candidate_split_positions = [
            (
                self._find_positions_cat(data, foc_i)
                if ingestion.is_categorical(foc_types[i])
                else self._find_positions_cont(foc_i, nof_splits)
            )
            for i, foc_i in enumerate(ccf)
        ]

        # matrix_weighted_heter[i,j] (i index of ccf and j index of split position) is
        # the weighted heterogeneity if the node(s) split in ccf[i] at position j.
        # The column count must fit the LARGEST candidate-position list: a categorical
        # conditioning feature can have more levels than cat_limit or the numerical grid
        # (e.g. hour-of-day = 24 levels), which would otherwise index j out of bounds.
        max_positions = max((len(p) for p in candidate_split_positions), default=1)
        matrix_weighted_heter = (
            np.ones([len(ccf), max(nof_splits - 1, self.cat_limit, max_positions)])
            * BIG_M
        )

        def apply_split(foc_i, position, foc_type):
            return self._flatten_list(
                [
                    self._split_dataset(active_indices, foc_i, position, foc_type)
                    for active_indices in before_split_active_indices_list
                ]
            )

        # exhaustive search on all split positions
        for i, foc_i in enumerate(ccf):
            for j, position in enumerate(candidate_split_positions[i]):
                after_split_active_indices_list = apply_split(
                    foc_i, position, foc_types[i]
                )
                heter_list_after_split = [
                    heter_func(x) for x in after_split_active_indices_list
                ]

                # weights analogous to the populations in each split
                populations = np.array(
                    [np.sum(x) for x in after_split_active_indices_list]
                )
                after_split_weight_list = (populations + 1) / (np.sum(populations + 1))

                matrix_weighted_heter[i, j] = np.sum(
                    after_split_weight_list * np.array(heter_list_after_split)
                )

        # the split with the minimum weighted heterogeneity
        i, j = np.unravel_index(
            np.argmin(matrix_weighted_heter, axis=None), matrix_weighted_heter.shape
        )
        position = candidate_split_positions[i][j]
        after_split_active_indices_list = apply_split(ccf[i], position, foc_types[i])

        return {
            "foc_index": ccf[i],
            "foc_split_position": position,
            "foc_range": [np.min(data[:, ccf[i]]), np.max(data[:, ccf[i]])],
            "foc_type": foc_types[i],
            "split_i": i,
            "split_j": j,
            "candidate_split_positions": candidate_split_positions[i],
            "candidate_conditioning_features": ccf,
            "after_split_nof_instances": [
                np.sum(x) for x in after_split_active_indices_list
            ],
            "after_split_heter_list": [
                heter_func(x) for x in after_split_active_indices_list
            ],
            "after_split_active_indices_list": after_split_active_indices_list,
            "after_split_weighted_heter": matrix_weighted_heter[i, j],
            "matrix_weighted_heter": matrix_weighted_heter,
        }


class Best(Base):
    """Node-wise recursive partitioning: find the best split for each node,
    recurse into the children (see `Base.__init__` for the parameters)."""

    def __init__(
        self,
        min_heterogeneity_decrease_pcg: float = 0.1,
        heter_small_enough: float = 0.001,
        max_depth: int = 2,
        min_samples_leaf: int = 10,
        numerical_features_grid_size: int = 20,
        search_partitions_when_categorical: bool = True,
    ):
        super().__init__(
            "Best",
            min_heterogeneity_decrease_pcg,
            heter_small_enough,
            max_depth,
            min_samples_leaf,
            numerical_features_grid_size,
            search_partitions_when_categorical,
        )

    def fit(self) -> Tree:
        self.splits_tree = Tree()

        root_info = {
            "active_indices": np.ones((self.data.shape[0])),
            "heterogeneity": self.heter_func(np.ones((self.data.shape[0]))),
            "level": 0,
        }
        self.splits_tree.add_node(
            name=self.feature_names[self.feature], parent_name=None, data=root_info
        )

        self._recursive_split(self.splits_tree.get_root())
        return self.splits_tree

    def _recursive_split(self, parent_node) -> None:
        """Recursively split the tree."""

        # if any of the following, stop before splitting
        conditions = [
            parent_node.info["level"]
            >= self.max_split_levels,  # Max split levels reached
            np.sum(parent_node.info["active_indices"])
            < self.min_points_per_subregion,  # Not enough points,
            parent_node.info["heterogeneity"]
            < self.heter_small_enough,  # Heterogeneity is already small enough
        ]

        if any(conditions):
            return None

        # find the best split
        split = self._evaluate_splits([parent_node.info["active_indices"]])

        # weighted heterogeneity of the best split
        weights = split["after_split_nof_instances"] / np.sum(
            split["after_split_nof_instances"]
        )
        heter_after = np.sum(weights * np.array(split["after_split_heter_list"]))
        heter_before = parent_node.info["heterogeneity"]

        heter_drop_pcg = (heter_before - heter_after) / heter_before
        if heter_drop_pcg < self.heter_pcg_drop_thres:
            return None
        else:
            for i in [0, 1]:
                node_data = {
                    "foc_split_position": split["foc_split_position"],
                    "comparison": self._get_comparison_symbol(split["foc_type"], i),
                    "foc_index": split["foc_index"],
                    "foc_name": self.feature_names[split["foc_index"]],
                    "foc_type": split["foc_type"],
                    "active_indices": split["after_split_active_indices_list"][i],
                    "heterogeneity": split["after_split_heter_list"][i],
                    "level": parent_node.info["level"] + 1,
                }

                node_name = self.splits_tree.create_node_name(
                    node_data["foc_name"],
                    parent_node,
                    node_data["comparison"],
                    f"{node_data['foc_split_position']:.2g}",
                )

                self.splits_tree.add_node(node_name, parent_node.name, node_data)
                child_node = self.splits_tree.get_node_by_name(node_name)

                self._recursive_split(child_node)


class BestLevelWise(Base):
    """Level-wise partitioning: find the single best split for each level
    (applied to every node of that level at once), then keep the prefix of
    levels whose heterogeneity drop is large enough (see `Base.__init__` for
    the parameters)."""

    def __init__(
        self,
        min_heterogeneity_decrease_pcg: float = 0.1,
        heter_small_enough: float = 0.001,
        max_depth: int = 2,
        min_samples_leaf: int = 10,
        numerical_features_grid_size: int = 20,
        search_partitions_when_categorical: bool = True,
    ):
        super().__init__(
            "best_level_wise",
            min_heterogeneity_decrease_pcg,
            heter_small_enough,
            max_depth,
            min_samples_leaf,
            numerical_features_grid_size,
            search_partitions_when_categorical,
        )

        # init splits
        self.splits: dict = {}
        self.important_splits: dict = {}

        self.important_splits_tree: typing.Union[Tree, None] = None

        # state variable
        self.split_found: bool = False
        self.important_splits_selected: bool = False

    def fit(self):
        self._search_all_splits()
        self._choose_important_splits()
        self.splits_tree = self._splits_to_tree(True)
        return self.splits_tree

    def _search_all_splits(self):
        """
        Iterate over all features of conditioning and choose the best split for each level in a greedy fashion.
        """
        if (
            ingestion.is_categorical(self.feature_types[self.feature])
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
                if any(
                    [
                        np.sum(x) < self.min_points_per_subregion
                        for x in splits[-1]["after_split_active_indices_list"]
                    ]
                ):
                    break

                # find optimal split
                new_split = self._evaluate_splits(
                    splits[-1]["after_split_active_indices_list"]
                )
                splits.append(new_split)
            self.splits = splits

        # update state
        self.split_found = True
        return self.splits

    def _choose_important_splits(self):
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

            # accept split if heterogeneity drops over `heter_pcg_drop_thres`
            heter = np.array(
                [splits[i]["after_split_weighted_heter"] for i in range(len(splits))]
            )
            heter_drop = (heter[:-1] - heter[1:]) / heter[:-1]
            split_valid = heter_drop > self.heter_pcg_drop_thres

            # accept split if heterogeneity is not already small enough
            heter_not_too_small = heter[:-1] > self.heter_small_enough
            split_valid = np.logical_and(split_valid, heter_not_too_small)

            # if all are negative, return nothing
            if np.sum(split_valid) == 0:
                optimal_splits = {}
            # if all are positive, return all
            elif np.sum(split_valid) == len(split_valid):
                optimal_splits = splits[1:]
            else:
                # find first negative split
                first_negative = np.where(~split_valid)[0][0]

                # if first negative is the first split, return nothing
                if first_negative == 0:
                    optimal_splits = {}
                else:
                    optimal_splits = splits[1 : first_negative + 1]

        # update state variable
        self.important_splits_selected = True
        self.important_splits = optimal_splits
        return optimal_splits

    def _splits_to_tree(self, only_important=True, scale_x_list=None):
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

                active_indices_1, active_indices_2 = self._split_dataset(
                    parent_active_indices,
                    foc,
                    pos,
                    split["foc_type"],
                )

                active_indices_new = (
                    active_indices_1 if j % 2 == 0 else active_indices_2
                )
                comparison = self._get_comparison_symbol(split["foc_type"], j % 2)

                name = tree.create_node_name(
                    foc_name,
                    tree.get_node_by_name(parent_name),
                    comparison,
                    f"{pos_small:.2g}",
                )

                data = {
                    "heterogeneity": split["after_split_heter_list"][j],
                    "weight": float(split["after_split_nof_instances"][j])
                    / nof_instances,
                    "foc_split_position": split["foc_split_position"],
                    "foc_name": foc_name,
                    "foc_index": split["foc_index"],
                    "foc_type": split["foc_type"],
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


def return_default(partitioner_name):
    if partitioner_name == "best":
        return Best()
    elif partitioner_name == "best_level_wise":
        return BestLevelWise()
    else:
        raise ValueError("Partitioner not found")

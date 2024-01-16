import numpy as np
import effector.binning_methods as binning_methods
import effector.helpers as helpers
import effector.utils as utils
from effector.partitioning import Regions, Tree
from effector.global_effect_ale import RHALE, ALE
from effector.global_effect_pdp import PDP, DerivativePDP
from effector.global_effect_shap import SHAPDependence
import typing
from typing import Callable, Optional, Union, List
from tqdm import tqdm
import copy

BIG_M = helpers.BIG_M


class RegionalEffectBase:
    empty_symbol = helpers.EMPTY_SYMBOL

    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        data_effect: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 100,
        axis_limits: Optional[np.ndarray] = None,
        feature_types: Optional[List] = None,
        cat_limit: Optional[int] = 10,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ) -> None:
        """
        Constructor for the RegionalEffect class.
        """
        self.method_name = method_name.lower()
        self.model = model
        self.model_jac = model_jac

        # select nof_instances from the data
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        self.data = data[self.indices, :]
        self.instance_effects = data_effect[self.indices, :] if data_effect is not None else None
        self.dim = self.data.shape[1]

        # set axis_limits
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        self.axis_limits: np.ndarray = axis_limits

        # set feature types
        self.cat_limit = cat_limit
        feature_types = (
            utils.get_feature_types(data, cat_limit)
            if feature_types is None
            else feature_types
        )
        self.feature_types: list = feature_types

        # set feature names
        feature_names: list[str] = (
            helpers.get_feature_names(axis_limits.shape[1])
            if feature_names is None
            else feature_names
        )
        self.feature_names: list = feature_names

        # set target name
        self.target_name = "y" if target_name is None else target_name

        # state variables
        self.is_fitted: np.ndarray = np.ones([self.dim]) < 0

        # parameters used when fitting the regional effect
        self.method_args: typing.Dict = {}

        # dictionary with all the information required for plotting or evaluating the regional effects
        self.partitioners: typing.Dict[str, Regions] = {}
        self.tree_full: typing.Dict[str, Tree] = {}
        self.tree_pruned: typing.Dict[str, Tree] = {}
        self.tree_full_scaled: typing.Dict[str, Tree] = {}
        self.tree_pruned_scaled: typing.Dict[str, Tree] = {}

    def _fit_feature(
        self,
        feature: int,
        heter_func: Callable,
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        max_split_levels: int = 2,
        candidate_positions_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_foc: Union[str, List] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the subregions for a single feature.
        """
        # init Region Extractor
        regions = Regions(
            feature,
            heter_func,
            self.data,
            self.instance_effects,
            self.feature_types,
            self.feature_names,
            self.target_name,
            self.cat_limit,
            candidate_foc,
            min_points_per_subregion,
            candidate_positions_for_numerical,
            max_split_levels,
            heter_pcg_drop_thres,
            heter_small_enough,
            split_categorical_features,
        )

        # apply partitioning
        regions.search_all_splits()
        regions.choose_important_splits()
        self.tree_full["feature_{}".format(feature)] = regions.splits_to_tree()
        self.tree_pruned["feature_{}".format(feature)] = regions.splits_to_tree(True)

        # store the partitioning object
        self.partitioners["feature_{}".format(feature)] = regions

        # update state
        self.is_fitted[feature] = True

    def refit(self, feature):
        if not self.is_fitted[feature]:
            self.fit(feature)

    def get_node_info(self, feature, node_idx):
        assert self.is_fitted[feature], "Feature {} has not been fitted yet".format(feature)
        assert self.tree_pruned["feature_{}".format(feature)] is not None, "Feature {} has no splits".format(feature)

        if self.tree_pruned_scaled is not None and "feature_{}".format(feature) in self.tree_pruned_scaled.keys():
            tree = self.tree_pruned_scaled["feature_{}".format(feature)]
        else:
            tree = self.tree_pruned["feature_{}".format(feature)]

        # assert node id exists
        assert node_idx in [node.idx for node in tree.nodes], "Node {} does not exist".format(node_idx)

        # find the node
        node = [node for node in tree.nodes if node.idx == node_idx][0]

        # get data
        data = node.data["data"]
        data_effect = node.data["data_effect"]
        name = node.name
        return data, data_effect, name

    def _create_fe_object(self, data, data_effect, feature_names):
        if self.method_name == "rhale":
            return RHALE(data, self.model, self.model_jac, data_effect=data_effect, feature_names=feature_names, target_name=self.target_name)
        elif self.method_name == "ale":
            return ALE(data, self.model, feature_names=feature_names, target_name=self.target_name)
        elif self.method_name == "shap":
            return SHAPDependence(data, self.model, feature_names=feature_names, target_name=self.target_name)
        elif self.method_name == "pdp":
            return PDP(data, self.model, feature_names=feature_names, target_name=self.target_name)
        elif self.method_name == "d-pdp":
            return DerivativePDP(data, self.model, self.model_jac, feature_names=feature_names, target_name=self.target_name)
        else:
            raise NotImplementedError

    def eval(self, feature, node_idx, xs, heterogeneity=False, centering=False):
        """
        Evaluate the regional effect for a given feature and node.

        Args:
            feature: the feature to evaluate
            node_idx: the node corresponding to the subregion to evaluate
            xs: the points at which to evaluate the regional effect
            heterogeneity: whether to return the heterogeneity.

                  - if `heterogeneity=False`, the function returns the mean effect at the given `xs`
                  - If `heterogeneity=True`, the function returns `(y, std)` where `y` is the mean effect and `std` is the standard deviation of the mean effect

            centering: whether to center the regional effect. The following options are available:

                - If `centering` is `False`, the regional effect is not centered
                - If `centering` is `True` or `zero_integral`, the regional effect is centered around the `y` axis.
                - If `centering` is `zero_start`, the regional effect starts from `y=0`.

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, std)` otherwise

        """
        self.refit(feature)
        centering = helpers.prep_centering(centering)
        data, data_effect, _ = self.get_node_info(feature, node_idx)
        fe_method = self._create_fe_object(data, data_effect, None)
        return fe_method.eval(feature, xs, heterogeneity, centering)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self,
             feature,
             node_idx,
             heterogeneity=False,
             centering=False,
             scale_x_list=None,
             scale_y=None,
             y_limits=None):

        self.refit(feature)

        if scale_x_list is not None:
            self.tree_full_scaled["feature_{}".format(feature)] = self.partitioners["feature_{}".format(feature)].splits_to_tree(False, scale_x_list)
            self.tree_pruned_scaled["feature_{}".format(feature)] = self.partitioners["feature_{}".format(feature)].splits_to_tree(True, scale_x_list)

        data, data_effect, name = self.get_node_info(feature, node_idx)
        feature_names = copy.deepcopy(self.feature_names)
        feature_names[feature] = name
        fe_method = self._create_fe_object(data, data_effect, feature_names)

        return fe_method.plot(
            feature=feature,
            heterogeneity=heterogeneity,
            centering=centering,
            scale_x=scale_x_list[feature] if scale_x_list is not None else None,
            scale_y=scale_y,
            y_limits=y_limits
            )

    def show_partitioning(self, features, only_important=True, scale_x_list=None):
        features = helpers.prep_features(features, self.dim)

        for feat in features:
            self.refit(feat)

            if scale_x_list is not None:
                tree_full_scaled = self.partitioners["feature_{}".format(feat)].splits_to_tree(True, scale_x_list)
                tree_pruned_scaled = self.partitioners["feature_{}".format(feat)].splits_to_tree(False, scale_x_list)
                tree_dict = tree_full_scaled if only_important else tree_pruned_scaled
            else:
                tree_dict = self.tree_pruned["feature_{}".format(feat)] if only_important else self.tree_full["feature_{}".format(feat)]

            print("Feature {} - Full partition tree:".format(feat))

            if tree_dict is None:
                print("No splits found for feature {}".format(feat))
            else:
                tree_dict.show_full_tree()

            print("-" * 50)
            print("Feature {} - Statistics per tree level:".format(feat))

            if tree_dict is None:
                print("No splits found for feature {}".format(feat))
            else:
                tree_dict.show_level_stats()

    def describe_subregions(
        self,
        features,
        only_important=True,
        scale_x_list: typing.Union[None, typing.List[dict]] = None,
    ):
        features = helpers.prep_features(features, self.dim)
        for feature in features:
            self.refit(feature)

            # it means it a categorical feature
            if self.tree_full["feature_{}".format(feature)] is None:
                continue

            feature_name = self.feature_names[feature]
            if only_important:
                tree = self.tree_pruned["feature_{}".format(feature)]
                if len(tree.nodes) == 1:
                    print("No important splits found for feature {}".format(feature))
                    continue
                else:
                    print("Important splits for feature {}".format(feature_name))
            else:
                print("All splits for feature {}".format(feature_name))
                tree = self.tree_full["feature_{}".format(feature)]

            max_level = max([node.level for node in tree.nodes])
            for level in range(1, max_level+1):
                previous_level_nodes = tree.get_level_nodes(level-1)
                level_nodes = tree.get_level_nodes(level)
                type_of_split_feature = level_nodes[0].data["feature_type"]
                foc_name = self.feature_names[level_nodes[0].data["feature"]]
                print("- On feature {} ({})".format(foc_name, type_of_split_feature))

                position_split_formatted = (
                    "{:.2f}".format(level_nodes[0].data["position"])
                    if scale_x_list is None
                    else "{:.2f}".format(
                        level_nodes[0].data["position"] * scale_x_list[level_nodes[0].data["feature"]]["std"]
                        + scale_x_list[level_nodes[0].data["feature"]]["mean"]
                    )
                )
                print("  - Position of split: {}".format(position_split_formatted))

                weight_heter_before = np.sum([node.data["weight"] * node.data["heterogeneity"] for node in previous_level_nodes])
                print("  - Heterogeneity before split: {:.2f}".format(weight_heter_before))

                weight_heter = np.sum([node.data["weight"] * node.data["heterogeneity"] for node in level_nodes])
                print("  - Heterogeneity after split: {:.2f}".format(weight_heter))
                weight_heter_drop = weight_heter_before - weight_heter
                print("  - Heterogeneity drop: {:.2f} ({:.2f} %)".format(
                    weight_heter_drop, weight_heter_drop / weight_heter_before * 100)
                )

                nof_instances_before = [nod.data["nof_instances"] for nod in previous_level_nodes]
                print("  - Number of instances before split: {}".format(nof_instances_before))
                nof_instances = [nod.data["nof_instances"] for nod in level_nodes]
                print("  - Number of instances after split: {}".format(nof_instances))

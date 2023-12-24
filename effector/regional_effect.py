import numpy as np
import effector.binning_methods as binning_methods
import effector.helpers as helpers
import effector.utils as utils
from effector.partitioning import Regions
from effector.global_effect_ale import RHALE, ALE
from effector.global_effect_pdp import PDP, DerivativePDP
from effector.global_effect_shap import SHAPDependence
import typing
from tqdm import tqdm
import copy

BIG_M = helpers.BIG_M


class RegionalEffectBase:
    empty_symbol = helpers.EMPTY_SYMBOL

    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: callable,
        nof_instances: int | str = 100,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ) -> None:
        """
        Constructor for the RegionalEffect class.
        """
        self.method_name = method_name.lower()

        # select nof_instances from the data
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        data = data[self.indices, :]
        self.data = data
        self.dim = self.data.shape[1]
        self.model = model

        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        feature_types = (
            utils.get_feature_types(data, cat_limit)
            if feature_types is None
            else feature_types
        )
        feature_names: list[str] = (
            helpers.get_feature_names(axis_limits.shape[1])
            if feature_names is None
            else feature_names
        )

        self.cat_limit = cat_limit

        self.axis_limits: np.ndarray = axis_limits
        self.feature_types: list = feature_types
        self.feature_names: list = feature_names
        self.target_name = target_name
        self.dim = self.axis_limits.shape[1]

        # state variables
        self.splits_full_depth_found: np.ndarray = np.ones([self.dim]) < 0
        self.splits_only_important_found: np.ndarray = np.ones([self.dim]) < 0

        # parameters used when fitting the regional effect
        self.method_args: typing.Dict = {}

        # dictionary with all the information required for plotting or evaluating the regional effects
        self.regions: typing.Dict[str, Regions] = {}
        # self.splits_full_depth: typing.Dict[str, list[dict]] = {}
        self.splits_full_depth_tree: typing.Dict = {}
        # self.splits_only_important: typing.Dict[str, list[dict]] = {}
        self.splits_only_important_tree: dict = {}

    def _fit_feature(
        self,
        feature: int,
        heter: callable,
        heter_pcg_drop_thres=0.1,
        heter_small_enough=0.1,
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional RHALE for a single feature.
        """

        # init Region Extractor
        regions = Regions(
            feature,
            heter,
            self.data,
            None,
            self.feature_types,
            self.feature_names,
            self.target_name,
            self.cat_limit,
            candidate_conditioning_features,
            min_points_per_subregion,
            nof_candidate_splits_for_numerical,
            max_split_levels,
            heter_pcg_drop_thres,
            heter_small_enough,
            split_categorical_features,
        )

        self.regions["feature_{}".format(feature)] = regions
        splits = regions.search_all_splits()

        # self.splits_full_depth["feature_{}".format(feature)] = splits
        self.splits_full_depth_found[feature] = True

        important_splits = regions.choose_important_splits()
        # self.splits_only_important[
        #     "feature_{}".format(feature)
        # ] = important_splits
        self.splits_only_important_found[feature] = True

        self.splits_full_depth_tree["feature_{}".format(feature)] = regions.splits_to_tree()
        self.splits_only_important_tree["feature_{}".format(feature)] = regions.splits_to_tree(True)

    def refit(self, feature, centering):
        "Checks if refitting is needed"
        if not self.splits_full_depth_found[feature]:
            return True
        else:
            # TODO: change that, we want refiting in many cases
            if centering is not False:
                if self.method_args["feature_" + str(feature)]["centering"] != centering:
                    return True
        return False

    def _get_data_from_node(self, feature, node_idx):
        # find the region and get data (and data_effect)
        if not self.splits_full_depth_found[feature]:
            self._fit_feature(feature)

        # assert node id exists
        assert node_idx in [node.idx for node in self.splits_only_important_tree["feature_{}".format(feature)].nodes], "Node {} does not exist".format(node_idx)

        # find the node
        node = [node for node in self.splits_only_important_tree["feature_{}".format(feature)].nodes if node.idx == node_idx][0]

        # get data
        data = node.data["data"]
        data_effect = node.data["data_effect"]
        name = node.name
        return data, data_effect, name

    def _create_object(self, method_name, data, data_effect, feature_names):
        if method_name == "rhale":
            return RHALE(data, self.model, self.model_jac, data_effect=data_effect, feature_names=feature_names)
        elif method_name == "ale":
            return ALE(data, self.model, feature_names=feature_names)
        elif method_name == "shap":
            return SHAPDependence(data, self.model, feature_names=feature_names)
        elif method_name == "pdp":
            return PDP(data, self.model, feature_names=feature_names)
        elif method_name == "d-pdp":
            return DerivativePDP(data, self.model, self.model_jac, feature_names=feature_names)
        else:
            raise NotImplementedError

    def eval(self, feature, node_idx, xs, heterogeneity=False, centering=False):
        data, data_effect, feature_names = self._get_data_from_node(feature, node_idx)
        fe_method = self._create_object(self.method_name, data, data_effect, feature_names)
        return fe_method.eval(feature, xs, heterogeneity, centering)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self,
             feature,
             node_idx,
             heterogeneity=False,
             centering=False,
             scale_x=None,
             scale_y=None,
             y_limits=None):
        data, data_effect, name = self._get_data_from_node(feature, node_idx)
        feature_names = copy.deepcopy(self.feature_names)
        feature_names[feature] = name
        fe_method = self._create_object(self.method_name, data, data_effect, feature_names)
        # TODO add ylimits
        return fe_method.plot(
            feature=feature,
            heterogeneity=heterogeneity,
            centering=centering,
            scale_x=scale_x,
            scale_y=scale_y
            )


    def print_tree(self, features, only_important=True):
        features = helpers.prep_features(features, self.dim)
        for feat in features:
            if not self.splits_full_depth_found[feat]:
                self._fit_feature(feat)

            print("Feature {}".format(feat))
            if only_important:
                if self.splits_only_important_tree["feature_{}".format(feat)] is None:
                    print("No important splits found for feature {}".format(feat))
                else:
                    self.splits_only_important_tree["feature_{}".format(feat)].show()
            else:
                if self.splits_full_depth_tree["feature_{}".format(feat)] is None:
                    print("No important splits found for feature {}".format(feat))
                else:
                    self.splits_full_depth_tree["feature_{}".format(feat)].show()

    def print_level_stats(self, features):
        features = helpers.prep_features(features, self.dim)
        for feat in features:
            if not self.splits_full_depth_found[feat]:
                self._fit_feature(feat)

            print("Feature {}".format(feat))
            if self.splits_full_depth_tree["feature_{}".format(feat)] is None:
                print("No important splits found for feature {}".format(feat))
            else:
                self.splits_full_depth_tree["feature_{}".format(feat)].show1()

    def describe_subregions(
        self,
        features,
        only_important=True,
        scale_x: typing.Union[None, list[dict]] = None,
    ):
        features = helpers.prep_features(features, self.dim)
        for feature in features:
            if not self.splits_full_depth_found[feature]:
                self._fit_feature(feature)

            # it means it a categorical feature
            if self.splits_full_depth_tree["feature_{}".format(feature)] is None:
                continue

            feature_name = self.feature_names[feature]
            if only_important:
                tree = self.splits_only_important_tree["feature_{}".format(feature)]
                if len(tree.nodes) == 1:
                    print("No important splits found for feature {}".format(feature))
                    continue
                else:
                    print("Important splits for feature {}".format(feature_name))
            else:
                print("All splits for feature {}".format(feature_name))
                tree = self.splits_full_depth_tree["feature_{}".format(feature)]

            max_level = max([node.level for node in tree.nodes])
            for level in range(1, max_level+1):
                previous_level_nodes = tree.get_level_nodes(level-1)
                level_nodes = tree.get_level_nodes(level)
                type_of_split_feature = level_nodes[0].data["feature_type"]
                foc_name = self.feature_names[level_nodes[0].data["feature"]]
                print("- On feature {} ({})".format(foc_name, type_of_split_feature))

                position_split_formatted = (
                    "{:.2f}".format(level_nodes[0].data["position"])
                    if scale_x is None
                    else "{:.2f}".format(
                        level_nodes[0].data["position"] * scale_x[level_nodes[0].data["feature"]]["std"]
                        + scale_x[level_nodes[0].data["feature"]]["mean"]
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


class RegionalRHALEBase(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        instance_effects: None | np.ndarray = None,
        nof_instances: int | str = 100,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        """
        Regional RHALE constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            model_jac: the black-box model Jacobian (N,D) -> (N,D)
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
        """
        self.model_jac = model_jac
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        instance_effects = self.model_jac(data)
        data = data[self.indices, :]

        instance_effects = instance_effects[self.indices, :]
        self.instance_effects = instance_effects
        super(RegionalRHALEBase, self).__init__(
            "rhale",
            data,
            model,
            "all",
            axis_limits,
            feature_types,
            cat_limit, feature_names, target_name
        )

    def _create_heterogeneity_function(self, foi, binning_method, min_points=10):
        binning_method = helpers.prep_binning_method(binning_method)

        def heter(data, instance_effects=None) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            rhale = RHALE(data, self.model, self.model_jac,  "all", None, instance_effects)
            try:
                rhale.fit(features=foi, binning_method=binning_method)
            except:
                return BIG_M

            # heterogeneity is the accumulated std at the end of the curve
            axis_limits = helpers.axis_limits_from_data(data)
            stop = np.array([axis_limits[:, foi][1]])
            _, z = rhale.eval(feature=foi, xs=stop, heterogeneity=True)
            return z.item()

        return heter

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        binning_method: typing.Union[
            str,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional RHALE for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            binning_method: binning method to use
            max_split_levels: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            heter = self._create_heterogeneity_function(
                feat, binning_method, min_points_per_subregion
            )

            self._fit_feature(
                feat,
                heter,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_split_levels,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )


class RegionalPDP(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        nof_instances: int | str = 100,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
    ):
        """
        Regional PDP constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
        """
        super(RegionalPDP, self).__init__("pdp", data, model, nof_instances, axis_limits, feature_types, cat_limit, feature_names)

    def _create_heterogeneity_function(self, foi, min_points=10):
        def heter(data) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            pdp = PDP(data, self.model, None, nof_instances=100)
            try:
                pdp.fit(features=foi)
            except:
                return BIG_M

            # heterogeneity is the mean heterogeneity over the curve
            axis_limits = helpers.axis_limits_from_data(data)

            xx = np.linspace(axis_limits[:, foi][0], axis_limits[:, foi][1], 10)
            try:
                _, z = pdp.eval(feature=foi, xs=xx, heterogeneity=True)
            except:
                return BIG_M
            return np.mean(z)

        return heter

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional PDP for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            max_split_levels: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
            split_categorical_features
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            heter = self._create_heterogeneity_function(feat, min_points_per_subregion)

            self._fit_feature(
                feat,
                heter,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_split_levels,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )

import numpy as np
import effector.helpers as helpers
import effector.utils as utils
from effector.partitioning import Regions, Tree
from effector.global_effect_ale import RHALE, ALE
from effector.global_effect_pdp import PDP, DerPDP
from effector.global_effect_shap import ShapDP
import typing
from typing import Callable, Optional, Union, List
import copy

class RegionalEffectBase:
    empty_symbol = helpers.EMPTY_SYMBOL

    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        data_effect: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        feature_types: Optional[List] = None,
        cat_limit: Optional[int] = 10,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ) -> None:
        """
        Constructor for the RegionalEffect class.
        """
        assert data.ndim == 2

        self.method_name = method_name.lower()
        self.model = model
        self.model_jac = model_jac

        self.dim = data.shape[1]

        # data preprocessing (i): if axis_limits passed manually,
        # keep only the points within,
        # otherwise, compute the axis limits from the data
        if axis_limits is not None:
            assert axis_limits.shape == (2, self.dim)
            assert np.all(axis_limits[0, :] <= axis_limits[1, :])

            # drop points outside of limits
            accept_indices = helpers.indices_within_limits(data, axis_limits)
            data = data[accept_indices, :]
            data_effect = data_effect[accept_indices, :] if data_effect is not None else None
        else:
            axis_limits = helpers.axis_limits_from_data(data)
        self.axis_limits: np.ndarray = axis_limits


        # data preprocessing (ii): select nof_instances from the remaining data
        self.nof_instances, self.indices = helpers.prep_nof_instances(nof_instances, data.shape[0])
        data = data[self.indices, :]
        data_effect = data_effect[self.indices, :] if data_effect is not None else None

        # store the data
        self.data: np.ndarray = data
        self.data_effect: Optional[np.ndarray] = data_effect

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
            self.data_effect,
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

    def _create_fe_object(self, feature, node_idx, scale_x_list):
        feature_tree = self.tree_pruned["feature_{}".format(feature)]
        assert feature_tree is not None, "Feature {} has no splits".format(feature)
        node = feature_tree.get_node_by_idx(node_idx)
        name = feature_tree.scale_node_name(node.name, scale_x_list)
        data = node.info["data"]
        data_effect = node.info["data_effect"]
        feature_names = copy.deepcopy(self.feature_names)
        feature_names[feature] = name

        if self.method_name == "rhale":
            return RHALE(
                data,
                self.model,
                self.model_jac,
                data_effect=data_effect,
                feature_names=feature_names,
                target_name=self.target_name,
            )
        elif self.method_name == "ale":
            return ALE(
                data,
                self.model,
                feature_names=feature_names,
                target_name=self.target_name,
            )
        elif self.method_name == "shap":
            return ShapDP(
                data,
                self.model,
                feature_names=feature_names,
                target_name=self.target_name,
            )
        elif self.method_name == "pdp":
            return PDP(
                data,
                self.model,
                feature_names=feature_names,
                target_name=self.target_name,
            )
        elif self.method_name == "d-pdp":
            return DerPDP(
                data,
                self.model,
                self.model_jac,
                feature_names=feature_names,
                target_name=self.target_name,
            )
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
        fe_method = self._create_fe_object(feature, node_idx, None)
        return fe_method.eval(feature, xs, heterogeneity, centering)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def plot(
        self,
        feature,
        node_idx,
        heterogeneity=False,
        centering=False,
        scale_x_list=None,
        scale_y=None,
        y_limits=None,
    ):

        self.refit(feature)
        fe_method = self._create_fe_object(feature, node_idx, scale_x_list)

        return fe_method.plot(
            feature=feature,
            heterogeneity=heterogeneity,
            centering=centering,
            scale_x=scale_x_list[feature] if scale_x_list is not None else None,
            scale_y=scale_y,
            y_limits=y_limits,
        )

    def summary(self, features, only_important=True, scale_x_list=None):
        features = helpers.prep_features(features, self.dim)

        for feat in features:
            self.refit(feat)

            feat_str = "feature_{}".format(feat)
            tree_dict = self.tree_pruned[feat_str] if only_important else self.tree_full[feat_str]

            print("\n")
            print("Feature {} - Full partition tree:".format(feat))

            if tree_dict is None:
                print("No splits found for feature {}".format(feat))
            else:
                tree_dict.show_full_tree(node=None, scale_x_list=scale_x_list)

            print("-" * 50)
            print("Feature {} - Statistics per tree level:".format(feat))

            if tree_dict is None:
                print("No splits found for feature {}".format(feat))
            else:
                tree_dict.show_level_stats()
            print("\n")

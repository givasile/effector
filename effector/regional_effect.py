import numpy as np
import effector.helpers as helpers
import effector.space_partitioning
import effector.utils as utils
from effector.space_partitioning import Best, Tree
from effector.global_effect_ale import RHALE, ALE
from effector.global_effect_pdp import PDP, DerPDP
from effector.global_effect_shap import ShapDP
from typing import Callable, Optional, Union, List, Tuple
import typing
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
        # self.method_args: typing.Dict = {}
        self.kwargs_subregion_detection: typing.Dict = {} # subregion specific arguments
        self.kwargs_fitting: typing.Dict = {} # fitting specific arguments

        # dictionary with all the information required for plotting or evaluating the regional effects
        self.partitioners: typing.Dict[str, Best] = {}
        # self.tree_full: typing.Dict[str, Tree] = {}
        self.tree: typing.Dict[str, Tree] = {}

    def _fit_feature(
        self,
        feature: int,
        heter_func: Callable,
        space_partitioner: Union["str", effector.space_partitioning.Best] = "best",
        candidate_foc: Union[str, List] = "all",
    ):
        """
        Find the subregions for a single feature.
        """
        assert feature < self.dim, "Feature index out of bounds"
        if isinstance(space_partitioner, str):
            assert space_partitioner in ["best", "cart"], "space_partitioner must be 'best' or 'cart'"
            space_partitioner = space_partitioning.return_default(space_partitioner)
        else:
            space_partitioner = copy.deepcopy(space_partitioner)

        # apply partitioning
        space_partitioner.compile(
            feature,
            self.data,
            heter_func,
            self.axis_limits,
            self.feature_types,
            self.cat_limit,
            candidate_foc,
            self.feature_names,
            self.target_name
        )
        self.tree["feature_{}".format(feature)] = space_partitioner.fit()

        # # self.tree_full["feature_{}".format(feature)] = regions.splits_to_tree()
        # self.tree["feature_{}".format(feature)] = space_partitioner.splits_to_tree(True)

        # store the partitioning object
        self.partitioners["feature_{}".format(feature)] = space_partitioner

        # update state
        self.is_fitted[feature] = True

    def refit(self, feature):
        if not self.is_fitted[feature]:
            self.fit(feature)

    def _create_fe_object(self, feature, node_idx, scale_x_list):
        feature_tree = self.tree["feature_{}".format(feature)]
        assert feature_tree is not None, "Feature {} has no splits".format(feature)
        assert node_idx < len(feature_tree.nodes)

        node = feature_tree.get_node_by_idx(node_idx)
        name = feature_tree.set_display_name(node.name, scale_x_list)
        active_indices = node.info["active_indices"]
        data = self.data[active_indices.astype(bool), :]
        data_effect = self.data_effect[active_indices.astype(bool), :] if self.data_effect is not None else None
        feature_names = copy.deepcopy(self.feature_names)
        feature_names[feature] = name

        if self.method_name == "rhale":
            return RHALE(
                data,
                self.model,
                self.model_jac,
                nof_instances="all",
                data_effect=data_effect,
                feature_names=feature_names,
                target_name=self.target_name,
            )
        elif self.method_name == "ale":
            return ALE(
                data,
                self.model,
                nof_instances="all",
                feature_names=feature_names,
                target_name=self.target_name,
            )
        elif self.method_name == "shap":
            return ShapDP(
                data,
                self.model,
                nof_instances="all",
                feature_names=feature_names,
                target_name=self.target_name,
                shap_values=self.global_shap_values[active_indices.astype(bool), :],
            )
        elif self.method_name == "pdp":
            return PDP(
                data,
                self.model,
                nof_instances="all",
                feature_names=feature_names,
                target_name=self.target_name,
            )
        elif self.method_name == "d-pdp":
            return DerPDP(
                data,
                self.model,
                self.model_jac,
                nof_instances="all",
                feature_names=feature_names,
                target_name=self.target_name,
            )
        else:
            raise NotImplementedError

    def eval(
            self,
            feature: int,
            node_idx: int,
            xs: np.ndarray,
            heterogeneity: bool = False,
            centering: Union[bool, str] = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        :point_right: Evaluate the regional effect for a given feature and node.

        !!! note "This is a common method for all regional effect methods, so use the arguments carefully."

            - `centering=True` is a good option for most methods, but not for all.
                - `DerPDP`, use `centering=False`
                - `[RegionalPDP, RegionalShapDP]`, it depends on you :sunglasses:
                - `[RegionalALE, RegionalRHALE]`, use `centering=True`

        !!! note "The `heterogeneity` argument changes the return value of the function."

            - If `heterogeneity=False`, the function returns `y`
            - If `heterogeneity=True`, the function returns a tuple `(y, std)`

        Args:
            feature: index of the feature
            node_idx: index of the node
            xs: horizontal grid of points to evaluate on
            heterogeneity: whether to return the heterogeneity.

                  - if `heterogeneity=False`, the function returns `y`, a numpy array of the mean effect at grid points `xs`
                  - If `heterogeneity=True`, the function returns `(y, std)` where `y` is the mean effect and `std` is the standard deviation of the mean effect at grid points `xs`

            centering: whether to center the regional effect. The following options are available:

                - If `centering` is `False`, the regional effect is not centered
                - If `centering` is `True` or `zero_integral`, the regional effect is centered around the `y` axis.
                - If `centering` is `zero_start`, the regional effect starts from `y=0`.

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, std)` otherwise

        """
        self.refit(feature)
        centering = helpers.prep_centering(centering)

        kwargs = copy.deepcopy(self.kwargs_fitting)
        kwargs['centering'] = centering

        # select only the three out of all
        fe_method = self._create_fe_object(feature, node_idx, None)
        fe_method.fit(features=feature, **kwargs)
        return fe_method.eval(feature, xs, heterogeneity, centering)

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def _plot(self, kwargs):
        # assert "feature", "node_idx" are in the dict
        assert "feature" in kwargs, "feature not found in kwargs"
        assert "node_idx" in kwargs, "node_idx not found in kwargs"

        self.refit(kwargs["feature"])

        # select only the three out of all kwargs
        fe_method = self._create_fe_object(kwargs["feature"], kwargs["node_idx"], kwargs["scale_x_list"])

        kwargs_fitting = copy.deepcopy(self.kwargs_fitting)
        kwargs_fitting['centering'] = kwargs["centering"]
        fe_method.fit(features=kwargs["feature"], **kwargs_fitting)

        plot_kwargs = copy.deepcopy(kwargs)
        plot_kwargs["scale_x"] = kwargs["scale_x_list"][kwargs["feature"]] if kwargs["scale_x_list"] is not None else None
        plot_kwargs.pop("scale_x_list")
        plot_kwargs.pop("node_idx")
        return fe_method.plot(**plot_kwargs)

    def summary(self, features: List[int], scale_x_list: Optional[List] = None):
        """:point_right: Summarize the partition tree for the selected features.

        ???+ Example "Example output"

            ```python
            Feature 3 - Full partition tree:
            🌳 Full Tree Structure:
            ───────────────────────
            hr 🔹 [id: 0 | heter: 0.43 | inst: 3476 | w: 1.00]
                workingday = 0.00 🔹 [id: 1 | heter: 0.36 | inst: 1129 | w: 0.32]
                    temp ≤ 6.50 🔹 [id: 3 | heter: 0.17 | inst: 568 | w: 0.16]
                    temp > 6.50 🔹 [id: 4 | heter: 0.21 | inst: 561 | w: 0.16]
                workingday ≠ 0.00 🔹 [id: 2 | heter: 0.28 | inst: 2347 | w: 0.68]
                    temp ≤ 6.50 🔹 [id: 5 | heter: 0.19 | inst: 953 | w: 0.27]
                    temp > 6.50 🔹 [id: 6 | heter: 0.20 | inst: 1394 | w: 0.40]
            --------------------------------------------------
            Feature 3 - Statistics per tree level:
            🌳 Tree Summary:
            ─────────────────
            Level 0🔹heter: 0.43
                Level 1🔹heter: 0.31 | 🔻0.12 (28.15%)
                    Level 2🔹heter: 0.19 | 🔻0.11 (37.10%)
            ```

        Args:
            features: indices of the features to summarize
            scale_x_list: list of scaling factors for each feature

                - `None`, for no scaling
                - `[{"mean": 0, "std": 1}, {"mean": 3, "std": 0.1}]`, to manually scale the features

        """
        features = helpers.prep_features(features, self.dim)

        for feat in features:
            self.refit(feat)

            feat_str = "feature_{}".format(feat)
            tree_dict = self.tree[feat_str]

            print("\n")
            print("Feature {} - Full partition tree:".format(feat))

            if tree_dict is None:
                print("No splits found for feature {}".format(feat))
            else:
                tree_dict.show_full_tree(scale_x_list=scale_x_list)

            print("-" * 50)
            print("Feature {} - Statistics per tree level:".format(feat))

            if tree_dict is None:
                print("No splits found for feature {}".format(feat))
            else:
                tree_dict.show_level_stats()
            print("\n")

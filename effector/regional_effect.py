import copy
import typing
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

import effector.helpers as helpers
import effector.space_partitioning
import effector.utils as utils
from effector.method_registry import resolve as resolve_method
from effector.space_partitioning import Best, Tree


class RegionalEffectBase:
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
        self.method_name = method_name.lower()
        self.model = model
        self.model_jac = model_jac

        self.dim = data.shape[1]

        # shared preprocessing: filter to axis_limits (or infer them), then
        # subsample nof_instances (helpers.prep_data)
        data, data_effect, axis_limits, self.nof_instances, self.indices = (
            helpers.prep_data(data, axis_limits, nof_instances, data_effect)
        )
        self.axis_limits: np.ndarray = axis_limits

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

        # parameters used when fitting the regional effect: what detected the
        # subregions, and what eval/plot must refit the node objects with —
        # written out explicitly by each subclass's fit (no locals(); B1)
        self.kwargs_subregion_detection: typing.Dict = {}
        self.kwargs_fitting: typing.Dict = {}

        # dictionary with all the information required for plotting or evaluating the regional effects
        self.partitioners: typing.Dict[str, Best] = {}
        self.tree: typing.Dict[str, Tree] = {}

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def _precompute_global(self, feature: int) -> None:
        """Hook: method-specific global precompute for `feature` (ICE table,
        global ALE effects, shap values), run once per feature before the
        heterogeneity function is built."""

    def _create_heterogeneity_function(self, feature: int, min_points: int) -> Callable:
        """Hook: the heterogeneity function the partitioner minimizes —
        `active_indices -> float` (BIG_M when the region is invalid)."""
        raise NotImplementedError

    def _fit_loop(
        self,
        features: Union[int, str, list],
        candidate_conditioning_features: Union[str, list],
        space_partitioner: Union[str, "effector.space_partitioning.Best"],
    ):
        """The regional-fit skeleton (template method): resolve the
        partitioner once (R6), then per feature: `_precompute_global` →
        `_create_heterogeneity_function` → partition."""
        if isinstance(space_partitioner, str):
            space_partitioner = effector.space_partitioning.return_default(
                space_partitioner
            )
        if space_partitioner.min_points_per_subregion < 2:
            raise ValueError("min_points_per_subregion must be >= 2")

        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            self._precompute_global(feat)
            heter = self._create_heterogeneity_function(
                feat, space_partitioner.min_points_per_subregion
            )
            self._fit_feature(
                feat, heter, space_partitioner, candidate_conditioning_features
            )

    def _fit_feature(
        self,
        feature: int,
        heter_func: Callable,
        space_partitioner: "effector.space_partitioning.Best",
        candidate_foc: Union[str, List],
    ):
        """
        Find the subregions for a single feature (on a fresh copy of the
        partitioner: `compile` mutates it and it is stored per feature).
        """
        if feature >= self.dim:
            raise ValueError("Feature index out of bounds")
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
            self.target_name,
        )
        self.tree["feature_{}".format(feature)] = space_partitioner.fit()

        # store the partitioning object
        self.partitioners["feature_{}".format(feature)] = space_partitioner

        # update state
        self.is_fitted[feature] = True

    def refit(self, feature):
        if not self.is_fitted[feature]:
            self.fit(feature)

    def _resolve_centering(self, centering):
        """`None` means the underlying method's class default (R3)."""
        if centering is None:
            centering = resolve_method(self.method_name).cls.DEFAULT_CENTERING
        return helpers.prep_centering(centering)

    def _extra_fe_kwargs(self, active_indices: np.ndarray) -> dict:
        """Hook: method-specific constructor kwargs for a node's fe object."""
        return {}

    def _create_fe_object(self, feature, node_idx, scale_x_list):
        feature_tree = self.tree["feature_{}".format(feature)]
        if feature_tree is None:
            raise ValueError("Feature {} has no splits".format(feature))
        if not node_idx < len(feature_tree.nodes):
            raise ValueError(
                "Node {} does not exist for feature {} (tree has {} nodes)".format(
                    node_idx, feature, len(feature_tree.nodes)
                )
            )

        node = feature_tree.get_node_by_idx(node_idx)
        name = feature_tree.set_display_name(node.name, scale_x_list)
        mask = node.info["active_indices"].astype(bool)
        data = self.data[mask, :]
        feature_names = copy.deepcopy(self.feature_names)
        feature_names[feature] = name

        spec = resolve_method(self.method_name)
        kwargs = dict(
            nof_instances="all",
            feature_names=feature_names,
            target_name=self.target_name,
        )
        if spec.uses_data_effect:
            kwargs["data_effect"] = (
                self.data_effect[mask, :] if self.data_effect is not None else None
            )
        kwargs.update(self._extra_fe_kwargs(mask))

        if spec.needs_jac:
            return spec.cls(data, self.model, self.model_jac, **kwargs)
        return spec.cls(data, self.model, **kwargs)

    def _fit_node_effect(self, feature, node_idx, centering, scale_x_list=None):
        """Build the node's fe object and fit it with the *stored* fit kwargs
        (B1: eval/plot must refit with what the user chose at fit time)."""
        fe = self._create_fe_object(feature, node_idx, scale_x_list)
        fit_kwargs = copy.deepcopy(self.kwargs_fitting)
        fit_kwargs["centering"] = centering
        fe.fit(features=feature, **fit_kwargs)
        return fe

    def eval(
        self,
        feature: int,
        node_idx: int,
        xs: np.ndarray,
        heterogeneity: bool = False,
        centering: Union[None, bool, str] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        :point_right: Evaluate the regional effect for a given feature and node.

        !!! note "The `heterogeneity` argument changes the return value of the function."

            - If `heterogeneity=False`, the function returns `y`
            - If `heterogeneity=True`, the function returns a tuple `(y, h)`
              where `h` is the heterogeneity curve (`eval_heter`)

        Args:
            feature: index of the feature
            node_idx: index of the node
            xs: horizontal grid of points to evaluate on
            heterogeneity: whether to also return the heterogeneity curve

                  - if `heterogeneity=False`, the function returns `y`, a numpy array of the mean effect at grid points `xs`
                  - If `heterogeneity=True`, the function returns `(y, h)` where `h` is the heterogeneity curve at grid points `xs`

            centering: whether to center the regional effect. The following options are available:

                - If `centering` is `None`, the underlying method's class default is used (R3)
                - If `centering` is `False`, the regional effect is not centered
                - If `centering` is `True` or `zero_integral`, the regional effect is centered around the `y` axis.
                - If `centering` is `zero_start`, the regional effect starts from `y=0`.

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, h)` otherwise

        """
        self.refit(feature)
        centering = self._resolve_centering(centering)

        fe = self._fit_node_effect(feature, node_idx, centering)
        y = fe.eval(feature, xs, centering=centering)
        if heterogeneity:
            return y, fe.eval_heter(feature, xs)
        return y

    def eval_heter(self, feature: int, node_idx: int, xs: np.ndarray) -> np.ndarray:
        """:point_right: The heterogeneity curve h(xs) of the node's regional
        effect — the regional twin of the global `eval_heter` (R2).

        No centering kwarg: h is invariant to centering by construction.

        Args:
            feature: index of the feature
            node_idx: index of the node
            xs: horizontal grid of points to evaluate on, `(T,)`

        Returns:
            the heterogeneity curve `h` at the given `xs`, `(T,)`
        """
        self.refit(feature)
        fe = self._fit_node_effect(feature, node_idx, centering=False)
        return fe.eval_heter(feature, xs)

    def _plot(self, feature, node_idx, scale_x_list, plot_kwargs):
        """Fit the node's fe object with the stored fit kwargs (B1) and
        delegate to its plot — the return rule is the global one's (R7)."""
        self.refit(feature)
        plot_kwargs["centering"] = self._resolve_centering(plot_kwargs["centering"])

        fe = self._fit_node_effect(
            feature, node_idx, plot_kwargs["centering"], scale_x_list
        )
        scale_x = scale_x_list[feature] if scale_x_list is not None else None
        return fe.plot(feature=feature, scale_x=scale_x, **plot_kwargs)

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

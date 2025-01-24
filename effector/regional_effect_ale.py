import typing
from effector.regional_effect import RegionalEffectBase
from effector import helpers, utils
import numpy as np
from effector.global_effect_ale import ALE, RHALE
from tqdm import tqdm
from effector import binning_methods
import copy
from typing import Callable, Optional, Union, List


BIG_M = helpers.BIG_M


class RegionalRHALE(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        instance_effects: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = "all",
        axis_limits: Optional[np.ndarray] = None,
        feature_types: Optional[List] = None,
        cat_limit: Optional[int] = 10,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
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

        if instance_effects is None:
            if model_jac is not None:
                instance_effects = model_jac(data)
            else:
                instance_effects = utils.compute_jacobian_numerically(model, data)

        super(RegionalRHALE, self).__init__(
            "rhale",
            data,
            model,
            model_jac,
            instance_effects,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name,
        )

    def _create_heterogeneity_function(
        self, foi, binning_method, min_points, centering
    ):
        binning_method = prep_binning_method(binning_method)

        def heter(data, instance_effects=None) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            rhale = RHALE(
                data, self.model, self.model_jac, "all", None, instance_effects
            )
            try:
                rhale.fit(
                    features=foi, binning_method=binning_method, centering=centering
                )
            except:
                return BIG_M

            # heterogeneity is the accumulated std at the end of the curve
            axis_limits = helpers.axis_limits_from_data(data)
            xs = np.linspace(axis_limits[0, foi], axis_limits[1, foi], 100)
            _, z = rhale.eval(feature=foi, xs=xs, heterogeneity=True)

            # mean
            return np.mean(z)

        return heter

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        max_depth: int = 1,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
        binning_method: typing.Union[
            str,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        centering: typing.Union[bool, str] = False,
    ):
        """
        Find the Regional RHALE for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            binning_method: binning method to use
            max_depth: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            heter = self._create_heterogeneity_function(
                feat, binning_method, min_points_per_subregion, centering
            )

            self._fit_feature(
                feat,
                heter,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_depth,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )

            self.method_args["feature_" + str(feat)] = {
                "heter_pcg_drop_thres": heter_pcg_drop_thres,
                "heter_small_enough": heter_small_enough,
                "max_depth": max_depth,
                "nof_candidate_splits_for_numerical": nof_candidate_splits_for_numerical,
                "min_points_per_subregion": min_points_per_subregion,
                "candidate_conditioning_features": candidate_conditioning_features,
                "split_categorical_features": split_categorical_features,
                "binning_method": binning_method,
            }

    def plot(
        self,
        feature,
        node_idx,
        heterogeneity=False,
        centering=False,
        scale_x_list=None,
        scale_y=None,
        y_limits=None,
        dy_limits=None,
    ):

        # get data from the node
        self.refit(feature)
        re_method = self._create_fe_object(feature, node_idx=node_idx, scale_x_list=scale_x_list)
        re_method.plot(
            feature=feature,
            heterogeneity=heterogeneity,
            centering=centering,
            scale_x=scale_x_list[feature] if scale_x_list is not None else None,
            scale_y=scale_y,
            y_limits=y_limits,
            dy_limits=dy_limits,
        )



class RegionalALE(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        nof_instances: typing.Union[int, str] = "all",
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
        super(RegionalALE, self).__init__(
            "ale",
            data,
            model,
            None,
            None,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name,
        )

    def _create_heterogeneity_function(self, foi, binning_method, min_points):
        binning_method = prep_binning_method(binning_method)
        isinstance(binning_method, binning_methods.Fixed)

        def heter(data, instance_effects=None) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            ale = ALE(data, self.model, "all", self.axis_limits, instance_effects)
            try:
                ale.fit(features=foi, binning_method=binning_method, centering=False)
            except utils.AllBinsHaveAtMostOnePointError as e:
                print(f"Error: {e}")
                return BIG_M
            except Exception as e:
                print(f"Unexpected error during ALE fitting: {e}")
                return BIG_M

            axis_limits = helpers.axis_limits_from_data(data)
            xs = np.linspace(axis_limits[0, foi], axis_limits[1, foi], 100)
            _, z = ale.eval(feature=foi, xs=xs, heterogeneity=True)
            return np.mean(z)


        return heter

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        max_depth: int = 1,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
        binning_method: typing.Union[
            str, binning_methods.Fixed
        ] = binning_methods.Fixed(nof_bins=20, min_points_per_bin=0),
    ):
        """
        Find subregions by minimizing the ALE-based heterogeneity.

        Args:
            features: for which features to search for subregions

                - use `"all"`, for all features, e.g. `features="all"`
                - use an `int`, for a single feature, e.g. `features=0`
                - use a `list`, for multiple features, e.g. `features=[0, 1, 2]`

            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important

                - use a `float`, e.g. `heter_pcg_drop_thres=0.1`
                - The heterogeity drop is expressed as percentage ${(H_{\mathtt{before\_split}} - H_{\mathtt{after\_split}}) \over H_{\mathtt{before\_split}}}$

            heter_small_enough: heterogeneity threshold for a split to be considered already small enough

                - if the current split has an heterogeneity smaller than this value, it is not further split
                - use a `float`, e.g. `heter_small_enough=0.01`

            max_depth: maximum depth of the tree

            nof_candidate_splits_for_numerical: number of candidate splits for numerical features

                - use an `int`, e.g. `nof_candidate_splits_for_numerical=20`
                - The candidate splits are uniformly distributed between the minimum and maximum values of the feature
                - e.g. if range is [0, 1] and `nof_candidate_splits_for_numerical=3`, the candidate splits are [0.25, 0.5, 0.75]

            min_points_per_subregion: minimum number of points per subregion

                - use an `int`, e.g. `min_points_per_subregion=10`
                - if a subregion has less than `min_points_per_subregion` instances, it is discarded

            candidate_conditioning_features: list of features to consider as conditioning features

                - use `"all"`, for all features, e.g. `candidate_conditioning_features="all"`
                - use a `list`, for multiple features, e.g. `candidate_conditioning_features=[0, 1, 2]`
                - it means that for each feature in the `feature` list, the algorithm will consider applying a split
                conditioned on each feature in the `candidate_conditioning_features` list

            split_categorical_features: whether to find subregions for categorical features

               - It indicates whether to create a splitting tree for categorical features
               - It does not mean whether the conditioning feature can be categorical (it can be)

            binning_method:

                - If set to `"fixed"`, the ALE plot will be computed with the  default values, which are
                `20` bins with at least `0` points per bin
                - If you want to change the parameters of the method, you pass an instance of the
                class `effector.binning_methods.Fixed` with the desired parameters.
                For example: `Fixed(nof_bins=20, min_points_per_bin=0, cat_limit=10)`
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            heter = self._create_heterogeneity_function(feat, binning_method, min_points_per_subregion)

            self._fit_feature(
                feat,
                heter,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_depth,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )

            self.method_args["feature_" + str(feat)] = {
                "heter_pcg_drop_thres": heter_pcg_drop_thres,
                "heter_small_enough": heter_small_enough,
                "max_depth": max_depth,
                "nof_candidate_splits_for_numerical": nof_candidate_splits_for_numerical,
                "min_points_per_subregion": min_points_per_subregion,
                "candidate_conditioning_features": candidate_conditioning_features,
                "split_categorical_features": split_categorical_features,
                "binning_method": binning_method,
            }

    def plot(
        self,
        feature,
        node_idx,
        heterogeneity=False,
        centering=False,
        scale_x_list=None,
        scale_y=None,
        y_limits=None,
        dy_limits=None,
    ):

        # get data from the node
        self.refit(feature)
        re_method = self._create_fe_object(feature, node_idx=node_idx, scale_x_list=scale_x_list)
        re_method.plot(
            feature=feature,
            heterogeneity=heterogeneity,
            centering=centering,
            scale_x=scale_x_list[feature] if scale_x_list is not None else None,
            scale_y=scale_y,
            y_limits=y_limits,
            dy_limits=dy_limits,
        )


def prep_binning_method(method):
    assert (
        method in ["greedy", "dp", "fixed"]
        or isinstance(method, binning_methods.Fixed)
        or isinstance(method, binning_methods.DynamicProgramming)
        or isinstance(method, binning_methods.Greedy)
    )

    if method == "greedy":
        return binning_methods.Greedy()
    elif method == "dp":
        return binning_methods.DynamicProgramming()
    elif method == "fixed":
        return binning_methods.Fixed()
    else:
        return method

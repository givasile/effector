import typing
import numpy as np
from effector.regional_effect import RegionalEffectBase
from effector import helpers
from effector.global_effect_pdp import PDP, DerivativePDP
from tqdm import tqdm

BIG_M = helpers.BIG_M


class RegionalPDPBase(RegionalEffectBase):
    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        nof_instances: typing.Union[int, str] = 100,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        super(RegionalPDPBase, self).__init__(
            method_name,
            data,
            model,
            model_jac,
            None,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name)

    def _create_heterogeneity_function(self, foi, min_points, centering, nof_instances, points_for_centering):
        def heter(data) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            if self.method_name == "pdp":
                pdp = PDP(data, self.model, self.axis_limits, nof_instances=nof_instances)
            else:
                pdp = DerivativePDP(data, self.model, self.model_jac, self.axis_limits, nof_instances=nof_instances)

            try:
                pdp.fit(features=foi, centering=centering, points_for_centering=points_for_centering)
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
        features: typing.Union[int, str, list] = "all",
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        max_depth: int = 1,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
        centering: typing.Union[bool, str] = False,
        nof_instances: int = "all",
        points_for_centering: int = 100,
    ):
        """
        Find the Regional PDP for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            max_depth: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
            split_categorical_features
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            heter = self._create_heterogeneity_function(feat, min_points_per_subregion, centering, nof_instances, points_for_centering)

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

            # todo add methdod args


class RegionalPDP(RegionalPDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        nof_instances: typing.Union[int, str] = 1000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        super(RegionalPDP, self).__init__(
            "pdp",
            data,
            model,
            None,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name)


class RegionalDerivativePDP(RegionalPDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        nof_instances: typing.Union[int, str] = 1000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        super(RegionalDerivativePDP, self).__init__(
            "d-pdp",
            data,
            model,
            model_jac,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name)
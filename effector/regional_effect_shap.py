import typing
from effector.regional_effect import RegionalEffectBase
from effector import helpers
import numpy as np
from effector.global_effect_shap import SHAPDependence
from tqdm import tqdm
from typing import Callable, Optional, Union, List


class RegionalSHAP(RegionalEffectBase):
    big_m = helpers.BIG_M

    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 100,
        feature_types: Optional[List[str]] = None,
        cat_limit: Optional[int] = 10,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
    ):
        """
        Regional SHAP constructor.

        Args:
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N, )`

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            nof_instances: maximum number of instances to be used for PDP.

                - use "all", for using all instances.
                - use an `int`, for using `nof_instances` instances.

            feature_types: The feature types.

                - use `None`, to infer them from the data; whether a feature is categorical or numerical is inferred
                from whether it exceeds the `cat_limit` unique values.
                - use a list with elements `"cat"` or `"numerical"`, to specify them manually.

            cat_limit: the minimum number of unique values for a feature to be considered categorical

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `                  ["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """
        super(RegionalSHAP, self).__init__(
            "shap",
            data,
            model,
            None,
            None,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name
        )

    def _create_heterogeneity_function(self, foi, min_points, centering, points_for_centering):

        def heterogeneity_function(data) -> float:
            if data.shape[0] < min_points:
                return self.big_m

            axis_limits = helpers.axis_limits_from_data(data)
            xx = np.linspace(axis_limits[:, foi][0], axis_limits[:, foi][1], 10)

            shap = SHAPDependence(data, self.model, None, self.nof_instances)
            shap.fit(foi, centering, points_for_centering)
            _, z = shap.eval(foi, xx, heterogeneity=True)
            return np.mean(z)

        return heterogeneity_function

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
            centering: typing.Union[bool, str] = False,
            points_for_centering: int = 100,
    ):
        """
        Fit the regional SHAP.

        Args:
            features: the features to fit.
                - If set to "all", all the features will be fitted.

            heter_pcg_drop_thres: threshold for the percentage drop in heterogeneity to consider a split valid
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            max_depth: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
            split_categorical_features: whether to search for subregions in categorical features
            centering: whether to center the SHAP dependence plots before estimating the heterogeneity
            points_for_centering: number of points to use for centering
        """
        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            heter = self._create_heterogeneity_function(
                feat, min_points_per_subregion, centering, points_for_centering
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
                "centering": centering,
                "points_for_centering": points_for_centering,
            }

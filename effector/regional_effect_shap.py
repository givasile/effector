import typing

import effector
from effector.regional_effect import RegionalEffectBase
from effector import helpers
import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Union, List
from effector import binning_methods as bm
from effector import utils




class RegionalShapDP(RegionalEffectBase):
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
        self.global_shap_values = None
        super(RegionalShapDP, self).__init__(
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
            target_name,
        )

    def _create_heterogeneity_function(self, foi, min_points, binning_method):

        def heterogeneity_function(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return self.big_m

            data = self.data[active_indices.astype(bool), :]
            shap_values = self.global_shap_values[active_indices.astype(bool), :]
            shap_dp = effector.ShapDP(data, self.model, self.axis_limits, "all", shap_values=shap_values)

            try:
                shap_dp.fit(features=foi, binning_method=binning_method, centering=False)
            except utils.AllBinsHaveAtMostOnePointError as e:
                print(f"Error: {e}")
                return self.big_m
            except Exception as e:
                print(f"Unexpected error during RHALE fitting: {e}")
                return self.big_m

            mean_spline = shap_dp.feature_effect["feature_" + str(foi)]["spline_mean"]

            residuals = (shap_values[:, foi] - mean_spline(data[:, foi]))**2
            return np.mean(residuals)

        return heterogeneity_function

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.2,
        heter_small_enough: float = 0.,
        max_depth: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
        binning_method: Union[str, bm.Greedy, bm.Fixed] = "greedy",
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
        """
        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)

        for feat in tqdm(features):
            # assert global SHAP values are available
            if self.global_shap_values is None:
                global_shap_dp = effector.ShapDP(self.data, self.model, self.axis_limits, "all")
                global_shap_dp.fit(feat, centering=False, binning_method=binning_method)
                self.global_shap_values = global_shap_dp.shap_values

            heter = self._create_heterogeneity_function(feat, min_points_per_subregion, binning_method)

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

        all_arguments = locals()
        all_arguments.pop("self")

        # region splitting arguments are the first 8 arguments
        self.kwargs_subregion_detection = {k: all_arguments[k] for k in list(all_arguments.keys())[:8]}

        # fit kwargs
        self.kwargs_fitting = {"binning_method": binning_method}

    def plot(self,
             feature,
             node_idx,
             heterogeneity=False,
             centering=False,
             nof_points=30,
             scale_x_list=None,
             scale_y=None,
             nof_shap_values='all',
             show_avg_output=False,
             y_limits=None,
             only_shap_values=False
    ):
        """
        Plot the regional SHAP.

        Args:
            feature: the feature to plot
            node_idx: the index of the node to plot
            heterogeneity: whether to plot the heterogeneity
            centering: whether to center the SHAP values
            nof_points: number of points to plot
            scale_x_list: the list of scaling factors for the feature names
            scale_y: the scaling factor for the SHAP values
            nof_shap_values: number of SHAP values to plot
            show_avg_output: whether to show the average output
            y_limits: the limits of the y-axis
            only_shap_values: whether to plot only the SHAP values
        """
        kwargs = locals()
        kwargs.pop("self")
        return self._plot(kwargs)


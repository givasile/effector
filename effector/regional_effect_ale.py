import typing
from effector.regional_effect import RegionalEffectBase
from effector import helpers, utils
import numpy as np
from effector.global_effect_ale import ALE, RHALE
from tqdm import tqdm
from effector import binning_methods
from typing import Callable, Optional, Union, List


BIG_M = helpers.BIG_M


class RegionalRHALE(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        data_effect: Optional[np.ndarray] = None,
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
            data: the dataset, shape (N,D)
            model: the black-box model, Callable (N,D) -> (N,)
            model_jac: a function that returns the jacobian of the black-box model, Callable (N,D) -> (N,D)
            data_effect: the jacobian of the black-box model applied on `data`, shape (N,D)

                - if None, it is computed as `model_jac(data)`
            nof_instances : the maximum number of instances to use for the analysis. The selection is done randomly at the beginning of the analysis.

                - if "all", all instances are used
                - if an integer, `nof_instances` instances are randomly selected from the data
            nof_instances: the maximum number of instances to use for the analysis. The selection is done randomly at the beginning of the analysis.
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
            target_name: the name of the target variable
        """

        # if data_effect is None:
        #     if model_jac is not None:
        #         data_effect = model_jac(data)
        #     else:
        #         data_effect = utils.compute_jacobian_numerically(model, data)

        super(RegionalRHALE, self).__init__(
            "rhale",
            data,
            model,
            model_jac,
            data_effect,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name,
        )


    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points."""
        if self.data_effect is None and self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        elif self.data_effect is None and self.model_jac is None:
            self.data_effect = utils.compute_jacobian_numerically(self.model, self.data)

    def _create_heterogeneity_function(
        self, foi, binning_method, min_points, points_for_mean_heterogeneity
    ):
        binning_method = prep_binning_method(binning_method)

        def heter(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return BIG_M

            data = self.data[active_indices.astype(bool), :]
            if self.data_effect is not None:
                instance_effects = self.data_effect[active_indices.astype(bool), :]
            else:
                instance_effects = None
            rhale = RHALE(data, self.model, self.model_jac, "all", self.axis_limits, instance_effects)
            try:
                rhale.fit(features=foi, binning_method=binning_method, centering=False)
            except utils.AllBinsHaveAtMostOnePointError as e:
                print(f"RegionalRHALE here: At a particular split, some bins had at most one point. I reject this split. \n Error: {e}")
                return BIG_M
            except Exception as e:
                print(f"RegionalRHALE here: An unexpected error occurred. I reject this split. \n Error: {e}")
                return BIG_M

            # heterogeneity is the accumulated std at the end of the curve
            xs = np.linspace(self.axis_limits[0, foi], self.axis_limits[1, foi], points_for_mean_heterogeneity)
            _, z = rhale.eval(feature=foi, xs=xs, heterogeneity=True, centering=False)
            return np.mean(z)
        return heter

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.,
        max_depth: int = 2,
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
        points_for_mean_heterogeneity: int = 30,
    ):
        """
        Find subregions by minimizing the RHALE-based heterogeneity.

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

            binning_method (str): the binning method to use.

                - Use `"greedy"` for using the Greedy binning solution with the default parameters.
                  For custom parameters initialize a `binning_methods.Greedy` object
                - Use `"dp"` for using a Dynamic Programming binning solution with the default parameters.
                  For custom parameters initialize a `binning_methods.DynamicProgramming` object
                - Use `"fixed"` for using a Fixed binning solution with the default parameters.
                  For custom parameters initialize a `binning_methods.Fixed` object

            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity
        """
        if self.data_effect is None:
            self.compile()

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            # find global axis limits
            heter = self._create_heterogeneity_function(
                feat, binning_method, min_points_per_subregion, points_for_mean_heterogeneity
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

        all_arguments = locals()
        all_arguments.pop("self")

        # region splitting arguments are the first 8 arguments
        self.kwargs_subregion_detection = {k: all_arguments[k] for k in list(all_arguments.keys())[:8]}

        # centering, points_for_centering, use_vectorized
        self.kwargs_fitting = {k: v for k, v in all_arguments.items() if k in ["binnning_method"]}

    def plot(
        self,
        feature,
        node_idx,
        heterogeneity=True,
        centering=True,
        scale_x_list=None,
        scale_y=None,
        y_limits=None,
        dy_limits=None,
    ):

        kwargs = locals()
        kwargs.pop("self")
        self._plot(kwargs)


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
            data: the dataset, shape (N,D)
            model: the black-box model, Callable (N,D) -> (N,)
            nof_instances: the maximum number of instances to use for the analysis. The selection is done randomly at the beginning of the analysis.
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
            target_name: the name of the target variable
        """
        self.global_bin_limits = {}
        self.global_data_effect = {}
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

    def _create_heterogeneity_function(self, foi, min_points, points_for_mean_heterogeneity):
        def heter(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return BIG_M

            data_effect = self.global_data_effect["feature_" + str(foi)][active_indices.astype(bool)]
            data = self.data[active_indices.astype(bool), foi]
            bin_limits = self.global_bin_limits["feature_" + str(foi)]

            params = utils.compute_ale_params(data, data_effect, bin_limits)

            xx = np.linspace(params["limits"][0], params["limits"][-1], points_for_mean_heterogeneity)
            var = utils.apply_bin_value(x=xx, bin_limits=params["limits"], bin_value=params["bin_variance"])
            return np.mean(var)
        return heter

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.,
        max_depth: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
        binning_method: typing.Union[
            str, binning_methods.Fixed
        ] = binning_methods.Fixed(nof_bins=20, min_points_per_bin=0),
        points_for_mean_heterogeneity: int = 30
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

            binning_method: must be the Fixed binning method

                - If set to `"fixed"`, the ALE plot will be computed with the  default values, which are
                `20` bins with at least `0` points per bin
                - If you want to change the parameters of the method, you pass an instance of the
                class `effector.binning_methods.Fixed` with the desired parameters.
                For example: `Fixed(nof_bins=20, min_points_per_bin=0, cat_limit=10)`

            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            # fit global method
            global_ale = ALE(self.data, self.model, nof_instances="all", axis_limits=self.axis_limits)
            global_ale.fit(features=feat, binning_method=binning_method, centering=False)
            self.global_data_effect["feature_" + str(feat)] = global_ale.data_effect_ale["feature_" + str(feat)]
            self.global_bin_limits["feature_" + str(feat)] = global_ale.bin_limits["feature_" + str(feat)]

            # create heterogeneity function
            heter = self._create_heterogeneity_function(feat, min_points_per_subregion, points_for_mean_heterogeneity)

            # fit feature
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

        # centering, points_for_centering, use_vectorized
        self.kwargs_fitting = {k:v for k,v in all_arguments.items() if k in ["binnning_method"]}

    def plot(
        self,
        feature,
        node_idx,
        heterogeneity=True,
        centering=True,
        scale_x_list=None,
        scale_y=None,
        y_limits=None,
        dy_limits=None,
    ):
        kwargs = locals()
        kwargs.pop("self")
        self._plot(kwargs)

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

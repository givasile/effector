import typing

import effector.space_partitioning
from effector.regional_effect import RegionalEffectBase
from effector import helpers, utils
import numpy as np
from effector.global_effect_ale import ALE, RHALE
from tqdm import tqdm
from effector import axis_partitioning as ap
from typing import Callable, Optional, Union, List


BIG_M = helpers.BIG_M


class RegionalRHALE(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        data_effect: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 100_000,
        axis_limits: Optional[np.ndarray] = None,
        feature_types: Optional[List] = None,
        cat_limit: Optional[int] = 10,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ):
        """
        Initialize the Regional Effect method.

        Args:
            data: the design matrix, `ndarray` of shape `(N,D)`
            model: the black-box model, `Callable` with signature `x -> y` where:

                - `x`: `ndarray` of shape `(N, D)`
                - `y`: `ndarray` of shape `(N)`

            model_jac: the black-box model's Jacobian, `Callable` with signature `x -> dy_dx` where:

                - `x`: `ndarray` of shape `(N, D)`
                - `dy_dx`: `ndarray` of shape `(N, D)`

            data_effect: The jacobian of the `model` on the `data`

                - `None`, infers the Jacobian internally using `model_jac(data)` or numerically
                - `np.ndarray`, to provide the Jacobian directly

                !!! tip "When possible, provide the Jacobian directly"

                    Computing the jacobian on the whole dataset can be memory demanding.
                    If you have the jacobian already computed, provide it directly to the constructor.

            axis_limits: Feature effect limits along each axis

                - `None`, infers them from `data` (`min` and `max` of each feature)
                - `array` of shape `(D, 2)`, manually specify the limits for each feature.

                !!! tip "When possible, specify the axis limits manually"

                    - they help to discard outliers and improve the quality of the fit
                    - `axis_limits` define the `.plot` method's x-axis limits; manual specification leads to better visualizations

                !!! tip "Their shape is `(2, D)`, not `(D, 2)`"

                    ```python
                    axis_limits = np.array([[0, 1, -1], [1, 2, 3]])
                    ```

            nof_instances: Max instances to use

                - `"all"`, uses all `data`
                - `int`, randomly selects `int` instances from `data`

                !!! tip "`100_000` (default), is a good choice. RHALE can handle large datasets :sunglasses: :sunglasses: "

            feature_types: The feature types.

                - `None`, infers them from data; if the number of unique values is less than `cat_limit`, it is considered categorical.
                - `['cat', 'cont', ...]`, manually specify the types of the features

            cat_limit: The minimum number of unique values for a feature to be considered categorical

                - if `feature_types` is manually specified, this parameter is ignored

            feature_names: The names of the features

                - `None`, defaults to: `["x_0", "x_1", ...]`
                - `["age", "weight", ...]` to manually specify the names of the features

            target_name: The name of the target variable

                - `None`, to keep the default name: `"y"`
                - `"price"`, to manually specify the name of the target variable
        """

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

        if isinstance(binning_method, str):
            binning_method = ap.return_default(binning_method)

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
                print(np.sum(active_indices))
                return BIG_M

            # heterogeneity is the accumulated std at the end of the curve
            xs = np.linspace(self.axis_limits[0, foi], self.axis_limits[1, foi], points_for_mean_heterogeneity)
            _, z = rhale.eval(feature=foi, xs=xs, heterogeneity=True, centering=False)
            return np.mean(z)
        return heter

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        candidate_conditioning_features: typing.Union[str, list] = "all",
        space_partitioner: typing.Union[str, effector.space_partitioning.Best] = "best",
        binning_method: typing.Union[str, ap.Fixed, ap.DynamicProgramming, ap.Greedy,] = "greedy",
        points_for_mean_heterogeneity: int = 30,
    ):
        """
        Find subregions by minimizing the RHALE-based heterogeneity.

        Args:
            features: for which features to search for subregions

                - use `"all"`, for all features, e.g. `features="all"`
                - use an `int`, for a single feature, e.g. `features=0`
                - use a `list`, for multiple features, e.g. `features=[0, 1, 2]`

            candidate_conditioning_features: list of features to consider as conditioning features
            space_partitioner: the space partitioner to use
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

        if isinstance(space_partitioner, str):
            space_partitioner = effector.space_partitioning.return_default(space_partitioner)

        assert space_partitioner.min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            # find global axis limits
            heter = self._create_heterogeneity_function(
                feat, binning_method, space_partitioner.min_points_per_subregion, points_for_mean_heterogeneity
            )

            self._fit_feature(
                feat,
                heter,
                space_partitioner,
                candidate_conditioning_features,
            )

        all_arguments = locals()
        all_arguments.pop("self")

        # region splitting arguments are the first 8 arguments
        self.kwargs_subregion_detection = {k: all_arguments[k] for k in list(all_arguments.keys())[:3]}

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
        nof_instances: typing.Union[int, str] = 100_000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        """
        Initialize the Regional Effect method.

        Args:
            data: the design matrix, `ndarray` of shape `(N,D)`
            model: the black-box model, `Callable` with signature `x -> y` where:

                - `x`: `ndarray` of shape `(N, D)`
                - `y`: `ndarray` of shape `(N)`

            axis_limits: Feature effect limits along each axis

                - `None`, infers them from `data` (`min` and `max` of each feature)
                - `array` of shape `(D, 2)`, manually specify the limits for each feature.

                !!! tip "When possible, specify the axis limits manually"

                    - they help to discard outliers and improve the quality of the fit
                    - `axis_limits` define the `.plot` method's x-axis limits; manual specification leads to better visualizations

                !!! tip "Their shape is `(2, D)`, not `(D, 2)`"

                    ```python
                    axis_limits = np.array([[0, 1, -1], [1, 2, 3]])
                    ```

            nof_instances: Max instances to use

                - `"all"`, uses all `data`
                - `int`, randomly selects `int` instances from `data`

                !!! tip "`100_000` (default) is a good choice; RegionalALE can handle large datasets. :sunglasses:"

            feature_types: The feature types.

                - `None`, infers them from data; if the number of unique values is less than `cat_limit`, it is considered categorical.
                - `['cat', 'cont', ...]`, manually specify the types of the features

            cat_limit: The minimum number of unique values for a feature to be considered categorical

                - if `feature_types` is manually specified, this parameter is ignored

            feature_names: The names of the features

                - `None`, defaults to: `["x_0", "x_1", ...]`
                - `["age", "weight", ...]` to manually specify the names of the features

            target_name: The name of the target variable

                - `None`, to keep the default name: `"y"`
                - `"price"`, to manually specify the name of the target variable
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
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union[str, effector.space_partitioning.Best] = "best",
        binning_method: typing.Union[str, ap.Fixed] = "fixed",
        points_for_mean_heterogeneity: int = 30
    ):
        """
        Find subregions by minimizing the ALE-based heterogeneity.

        Args:
            features: for which features to search for subregions

                - use `"all"`, for all features, e.g. `features="all"`
                - use an `int`, for a single feature, e.g. `features=0`
                - use a `list`, for multiple features, e.g. `features=[0, 1, 2]`

            candidate_conditioning_features: list of features to consider as conditioning features
            space_partitioner: the space partitioner to use

            binning_method: must be the Fixed binning method

                - If set to `"fixed"`, the ALE plot will be computed with the  default values, which are
                `20` bins with at least `0` points per bin
                - If you want to change the parameters of the method, you pass an instance of the
                class `effector.binning_methods.Fixed` with the desired parameters.
                For example: `Fixed(nof_bins=20, min_points_per_bin=0, cat_limit=10)`

            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity
        """
        if isinstance(space_partitioner, str):
            space_partitioner = effector.space_partitioning.return_default(space_partitioner)

        assert space_partitioner.min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            # fit global method
            global_ale = ALE(self.data, self.model, nof_instances="all", axis_limits=self.axis_limits)
            global_ale.fit(features=feat, binning_method=binning_method, centering=False)
            self.global_data_effect["feature_" + str(feat)] = global_ale.data_effect_ale["feature_" + str(feat)]
            self.global_bin_limits["feature_" + str(feat)] = global_ale.bin_limits["feature_" + str(feat)]

            # create heterogeneity function
            heter = self._create_heterogeneity_function(feat, space_partitioner.min_points_per_subregion, points_for_mean_heterogeneity)

            # fit feature
            self._fit_feature(
                feat,
                heter,
                space_partitioner,
                candidate_conditioning_features,
            )

        all_arguments = locals()
        all_arguments.pop("self")

        # region splitting arguments are the first 8 arguments
        self.kwargs_subregion_detection = {k: all_arguments[k] for k in list(all_arguments.keys())[:3]}

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


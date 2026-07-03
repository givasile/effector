import typing
import warnings
from typing import Callable, List, Optional, Union

import numpy as np

import effector.space_partitioning
from effector import axis_partitioning as ap
from effector import helpers, utils
from effector.global_effect_ale import ALE, RHALE
from effector.regional_effect import RegionalEffectBase

BIG_M = helpers.BIG_M


class RegionalRHALE(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        *,
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

    def _create_heterogeneity_function(self, feature: int, min_points: int):
        binning_method = ap.return_default(self.kwargs_fitting["binning_method"])
        points_for_mean_heterogeneity = self.kwargs_subregion_detection[
            "points_for_mean_heterogeneity"
        ]

        def heter(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return BIG_M

            data = self.data[active_indices.astype(bool), :]
            if self.data_effect is not None:
                instance_effects = self.data_effect[active_indices.astype(bool), :]
            else:
                instance_effects = None
            rhale = RHALE(
                data,
                self.model,
                self.model_jac,
                data_effect=instance_effects,
                nof_instances="all",
                axis_limits=self.axis_limits,
            )
            try:
                rhale.fit(
                    features=feature, binning_method=binning_method, centering=False
                )
            except utils.AllBinsHaveAtMostOnePointError as e:
                warnings.warn(
                    f"RegionalRHALE: at a candidate split, some bins had at most "
                    f"one point; the split is rejected. Error: {e}"
                )
                return BIG_M
            except Exception as e:
                warnings.warn(
                    f"RegionalRHALE: an unexpected error occurred at a candidate "
                    f"split ({np.sum(active_indices)} active points); the split "
                    f"is rejected. Error: {e}"
                )
                return BIG_M

            # heterogeneity is the mean of the heterogeneity curve
            xs = np.linspace(
                self.axis_limits[0, feature],
                self.axis_limits[1, feature],
                points_for_mean_heterogeneity,
            )
            z = rhale.eval_heter(feature, xs)
            return np.mean(z)

        return heter

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        candidate_conditioning_features: typing.Union[str, list] = "all",
        space_partitioner: typing.Union[str, effector.space_partitioning.Best] = "best",
        binning_method: typing.Union[
            str,
            ap.Fixed,
            ap.DynamicProgramming,
            ap.Greedy,
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

        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
            "points_for_mean_heterogeneity": points_for_mean_heterogeneity,
        }
        self.kwargs_fitting = {"binning_method": binning_method}

        self._fit_loop(features, candidate_conditioning_features, space_partitioner)

    def plot(
        self,
        feature: int,
        node_idx: int,
        heterogeneity: Union[bool, str] = True,
        centering: Union[None, bool, str] = None,
        scale_x_list: Optional[list] = None,
        scale_y: Optional[dict] = None,
        y_limits: Optional[list] = None,
        dy_limits: Optional[list] = None,
        show_plot: bool = True,
    ):
        """Plot the regional RHALE effect of `feature` at node `node_idx`.

        Args:
            feature: the feature to plot
            node_idx: the index of the node to plot
            heterogeneity: whether to plot the heterogeneity (std of the bin effects)
            centering: whether to center the plot (`None` uses the class default)
            scale_x_list: list with a `{"mean": ..., "std": ...}` dict per feature, for de-normalizing the x-axes
            scale_y: `{"mean": ..., "std": ...}` dict for de-normalizing the y-axis
            y_limits: manual limits of the y-axis
            dy_limits: manual limits of the dy/dx-axis
            show_plot: if `True`, show the figure; if `False`, return `(fig, ax)`
        """
        return self._plot(
            feature,
            node_idx,
            scale_x_list,
            dict(
                heterogeneity=heterogeneity,
                centering=centering,
                scale_y=scale_y,
                y_limits=y_limits,
                dy_limits=dy_limits,
                show_plot=show_plot,
            ),
        )


class RegionalALE(RegionalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        *,
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

    def _precompute_global(self, feature: int):
        """Fit the global ALE once and keep its per-instance bin effects: the
        candidate regions re-bin those instead of refitting the model."""
        global_ale = ALE(
            self.data, self.model, nof_instances="all", axis_limits=self.axis_limits
        )
        global_ale.fit(
            features=feature,
            binning_method=self.kwargs_fitting["binning_method"],
            centering=False,
        )
        self.global_data_effect["feature_" + str(feature)] = global_ale.data_effect_ale[
            "feature_" + str(feature)
        ]
        self.global_bin_limits["feature_" + str(feature)] = global_ale.bin_limits[
            "feature_" + str(feature)
        ]

    def _create_heterogeneity_function(self, feature: int, min_points: int):
        points_for_mean_heterogeneity = self.kwargs_subregion_detection[
            "points_for_mean_heterogeneity"
        ]

        def heter(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return BIG_M

            data_effect = self.global_data_effect["feature_" + str(feature)][
                active_indices.astype(bool)
            ]
            data = self.data[active_indices.astype(bool), feature]
            bin_limits = self.global_bin_limits["feature_" + str(feature)]

            params = utils.compute_ale_params(data, data_effect, bin_limits)

            xx = np.linspace(
                params["limits"][0], params["limits"][-1], points_for_mean_heterogeneity
            )
            var = utils.apply_bin_value(
                x=xx, bin_limits=params["limits"], bin_value=params["bin_variance"]
            )
            return np.mean(var)

        return heter

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union[str, effector.space_partitioning.Best] = "best",
        binning_method: typing.Union[str, ap.Fixed] = "fixed",
        points_for_mean_heterogeneity: int = 30,
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
        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
            "points_for_mean_heterogeneity": points_for_mean_heterogeneity,
        }
        self.kwargs_fitting = {"binning_method": binning_method}

        self._fit_loop(features, candidate_conditioning_features, space_partitioner)

    def plot(
        self,
        feature: int,
        node_idx: int,
        heterogeneity: Union[bool, str] = True,
        centering: Union[None, bool, str] = None,
        scale_x_list: Optional[list] = None,
        scale_y: Optional[dict] = None,
        y_limits: Optional[list] = None,
        dy_limits: Optional[list] = None,
        show_plot: bool = True,
    ):
        """Plot the regional ALE effect of `feature` at node `node_idx`.

        Args:
            feature: the feature to plot
            node_idx: the index of the node to plot
            heterogeneity: whether to plot the heterogeneity (std of the bin effects)
            centering: whether to center the plot (`None` uses the class default)
            scale_x_list: list with a `{"mean": ..., "std": ...}` dict per feature, for de-normalizing the x-axes
            scale_y: `{"mean": ..., "std": ...}` dict for de-normalizing the y-axis
            y_limits: manual limits of the y-axis
            dy_limits: manual limits of the dy/dx-axis
            show_plot: if `True`, show the figure; if `False`, return `(fig, ax)`
        """
        return self._plot(
            feature,
            node_idx,
            scale_x_list,
            dict(
                heterogeneity=heterogeneity,
                centering=centering,
                scale_y=scale_y,
                y_limits=y_limits,
                dy_limits=dy_limits,
                show_plot=show_plot,
            ),
        )

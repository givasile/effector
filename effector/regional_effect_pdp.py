import typing

import numpy as np

from effector import helpers
from effector.global_effect_pdp import PDP, DerPDP
from effector.regional_effect import RegionalEffectBase

BIG_M = helpers.BIG_M


class RegionalPDPBase(RegionalEffectBase):
    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        self.y_ice = {}
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
            target_name,
        )

    def _create_heterogeneity_function(self, feature: int, min_points: int):
        def heter(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return BIG_M
            yy = self.y_ice["feature_" + str(feature)][active_indices.astype(bool), :]
            z = np.var(yy, axis=0)
            return np.mean(z)

        return heter


class RegionalPDP(RegionalPDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        *,
        nof_instances: typing.Union[int, str] = 10_000,
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
            model: the black-box model, `Callable` with signature `f(x) -> y` where:

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

                !!! tip "`10_000` (default), is a good balance between speed and accuracy"

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
            target_name,
        )

    def _precompute_global(self, feature: int):
        """Fit the global PDP once and keep the centered ICE table on the
        heterogeneity grid: candidate regions score row-subsets of it."""
        pdp = PDP(
            self.data, self.model, axis_limits=self.axis_limits, nof_instances="all"
        )
        pdp.fit(
            features=feature,
            centering=True,
            points_for_centering=self.kwargs_fitting["points_for_centering"],
            use_vectorized=self.kwargs_fitting["use_vectorized"],
        )

        xx = np.linspace(
            self.axis_limits[0, feature],
            self.axis_limits[1, feature],
            self.kwargs_subregion_detection["points_for_mean_heterogeneity"],
        )
        y_ice = pdp._predict(
            pdp.data, xx, feature, self.kwargs_fitting["use_vectorized"]
        )
        y_ice = (
            y_ice
            - pdp.feature_effect["feature_" + str(feature)]["norm_const"][np.newaxis, :]
        )
        self.y_ice["feature_" + str(feature)] = y_ice.T

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union[str, None] = "best",
        points_for_centering: int = 30,
        points_for_mean_heterogeneity: int = 30,
        use_vectorized: bool = True,
    ):
        """
        Find subregions by minimizing the PDP-based heterogeneity.

        Args:
            features: for which features to search for subregions

                - use `"all"`, for all features, e.g. `features="all"`
                - use an `int`, for a single feature, e.g. `features=0`
                - use a `list`, for multiple features, e.g. `features=[0, 1, 2]`

            candidate_conditioning_features: list of features to consider as conditioning features

                - use `"all"`, for all features, e.g. `candidate_conditioning_features="all"`
                - use a `list`, for multiple features, e.g. `candidate_conditioning_features=[0, 1, 2]`
                - it means that for each feature in the `feature` list, the algorithm will consider applying a split
                conditioned on each feature in the `candidate_conditioning_features` list

            space_partitioner: the method to use for partitioning the space
            points_for_centering: number of equidistant points along the feature axis used for centering ICE plots
            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity
            use_vectorized: whether to use vectorized operations for the PDP and ICE curves


        """
        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
            "points_for_mean_heterogeneity": points_for_mean_heterogeneity,
        }
        self.kwargs_fitting = {
            "points_for_centering": points_for_centering,
            "use_vectorized": use_vectorized,
        }

        self._fit_loop(features, candidate_conditioning_features, space_partitioner)

    def plot(
        self,
        feature: int,
        node_idx: int,
        heterogeneity: typing.Union[bool, str] = "ice",
        centering: typing.Union[None, bool, str] = None,
        nof_points: int = 30,
        scale_x_list: typing.Union[None, list] = None,
        scale_y: typing.Union[None, dict] = None,
        nof_ice: typing.Union[int, str] = 100,
        show_avg_output: bool = False,
        y_limits: typing.Union[None, list] = None,
        use_vectorized: bool = True,
        show_plot: bool = True,
    ):
        """Plot the regional PDP effect of `feature` at node `node_idx`.

        Args:
            feature: the feature to plot
            node_idx: the index of the node to plot
            heterogeneity: whether to plot the heterogeneity (`"ice"`, `"std"`, or `False`)
            centering: whether to center the plot (`None` uses the class default)
            nof_points: the grid size for the PDP curve
            scale_x_list: list with a `{"mean": ..., "std": ...}` dict per feature, for de-normalizing the x-axes
            scale_y: `{"mean": ..., "std": ...}` dict for de-normalizing the y-axis
            nof_ice: number of ICE curves to show
            show_avg_output: whether to show the average output of the model
            y_limits: manual limits of the y-axis
            use_vectorized: whether to use the vectorized ICE computation
            show_plot: if `True`, show the figure; if `False`, return `(fig, ax)`
        """
        return self._plot(
            feature,
            node_idx,
            scale_x_list,
            dict(
                heterogeneity=heterogeneity,
                centering=centering,
                nof_points=nof_points,
                scale_y=scale_y,
                nof_ice=nof_ice,
                show_avg_output=show_avg_output,
                y_limits=y_limits,
                use_vectorized=use_vectorized,
                show_plot=show_plot,
            ),
        )


class RegionalDerPDP(RegionalPDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Optional[callable] = None,
        *,
        nof_instances: typing.Union[int, str] = 10_000,
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

            model_jac: the black-box model's Jacobian, `Callable` with signature `x -> dy_dx` where:

                - `x`: `ndarray` of shape `(N, D)`
                - `dy_dx`: `ndarray` of shape `(N, D)`

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

                !!! tip "`10_000` (default), is a good balance between speed and accuracy"

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

        super(RegionalDerPDP, self).__init__(
            "d-pdp",
            data,
            model,
            model_jac,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name,
        )

    def _precompute_global(self, feature: int):
        """Fit the global DerPDP once and keep the d-ICE table on the
        heterogeneity grid: candidate regions score row-subsets of it."""
        pdp = DerPDP(
            self.data,
            self.model,
            self.model_jac,
            axis_limits=self.axis_limits,
            nof_instances="all",
        )
        pdp.fit(
            features=feature,
            centering=False,
            use_vectorized=self.kwargs_fitting["use_vectorized"],
        )

        xx = np.linspace(
            self.axis_limits[0, feature],
            self.axis_limits[1, feature],
            self.kwargs_subregion_detection["points_for_mean_heterogeneity"],
        )
        y_ice = pdp._predict(
            pdp.data, xx, feature, self.kwargs_fitting["use_vectorized"]
        )
        self.y_ice["feature_" + str(feature)] = y_ice.T

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union[str, None] = "best",
        points_for_mean_heterogeneity: int = 30,
        use_vectorized: bool = True,
    ):
        """
        Find subregions by minimizing the PDP-based heterogeneity.

        Args:
            features: for which features to search for subregions

                - use `"all"`, for all features, e.g. `features="all"`
                - use an `int`, for a single feature, e.g. `features=0`
                - use a `list`, for multiple features, e.g. `features=[0, 1, 2]`

            candidate_conditioning_features: list of features to consider as conditioning features

                - use `"all"`, for all features, e.g. `candidate_conditioning_features="all"`
                - use a `list`, for multiple features, e.g. `candidate_conditioning_features=[0, 1, 2]`
                - it means that for each feature in the `feature` list, the algorithm will consider applying a split
                conditioned on each feature in the `candidate_conditioning_features` list

            space_partitioner: the method to use for partitioning the space
            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity
            use_vectorized: whether to use vectorized operations for the PDP and ICE curves


        """
        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
            "points_for_mean_heterogeneity": points_for_mean_heterogeneity,
        }
        self.kwargs_fitting = {"use_vectorized": use_vectorized}

        self._fit_loop(features, candidate_conditioning_features, space_partitioner)

    def plot(
        self,
        feature: int,
        node_idx: int,
        heterogeneity: typing.Union[bool, str] = "ice",
        centering: typing.Union[None, bool, str] = None,
        nof_points: int = 30,
        scale_x_list: typing.Union[None, list] = None,
        scale_y: typing.Union[None, dict] = None,
        nof_ice: typing.Union[int, str] = 100,
        show_avg_output: bool = False,
        dy_limits: typing.Union[None, list] = None,
        use_vectorized: bool = True,
        show_plot: bool = True,
    ):
        """Plot the regional d-PDP effect of `feature` at node `node_idx`.

        Args:
            feature: the feature to plot
            node_idx: the index of the node to plot
            heterogeneity: whether to plot the heterogeneity (`"ice"`, `"std"`, or `False`)
            centering: whether to center the plot (`None` uses the class default)
            nof_points: the grid size for the d-PDP curve
            scale_x_list: list with a `{"mean": ..., "std": ...}` dict per feature, for de-normalizing the x-axes
            scale_y: `{"mean": ..., "std": ...}` dict for de-normalizing the y-axis
            nof_ice: number of d-ICE curves to show
            show_avg_output: whether to show the average output of the model
            dy_limits: manual limits of the dy/dx-axis
            use_vectorized: whether to use the vectorized ICE computation
            show_plot: if `True`, show the figure; if `False`, return `(fig, ax)`
        """
        return self._plot(
            feature,
            node_idx,
            scale_x_list,
            dict(
                heterogeneity=heterogeneity,
                centering=centering,
                nof_points=nof_points,
                scale_y=scale_y,
                nof_ice=nof_ice,
                show_avg_output=show_avg_output,
                dy_limits=dy_limits,
                use_vectorized=use_vectorized,
                show_plot=show_plot,
            ),
        )

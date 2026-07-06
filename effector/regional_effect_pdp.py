import typing

import numpy as np

from effector import helpers, ingestion
from effector.global_effect_pdp import PDP, DerPDP
from effector.regional_effect import RegionalEffectBase
from effector.space_partitioning import Best

BIG_M = helpers.BIG_M


class RegionalPDPBase(RegionalEffectBase):
    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        *,
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        schema: typing.Optional[typing.Union[ingestion.Schema, dict]] = None,
        random_state: typing.Optional[int] = 21,
    ):
        self.y_ice = {}
        self.heter_grid: dict = {}
        super(RegionalPDPBase, self).__init__(
            method_name,
            data,
            model,
            model_jac,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
        )

    def _create_heterogeneity_function(self, feature: int, min_points: int):
        is_cat = ingestion.is_categorical(self.feature_types[feature])

        def heter(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return BIG_M
            mask = active_indices.astype(bool)
            yy = self.y_ice["feature_" + str(feature)][mask, :]
            z = np.var(yy, axis=0)
            if is_cat:
                # H = freq-weighted mean over levels, frequencies within the
                # candidate region (method_semantics.md)
                levels = self.heter_grid["feature_" + str(feature)]
                col = self.data[mask, feature]
                counts = np.array(
                    [np.isclose(col, lev).sum() for lev in levels], dtype=float
                )
                if counts.sum() == 0:
                    return BIG_M
                return float(np.average(z, weights=counts))
            return float(np.mean(z))

        return heter


class RegionalPDP(RegionalPDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        *,
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        schema: typing.Optional[typing.Union[ingestion.Schema, dict]] = None,
        random_state: typing.Optional[int] = 21,
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

            schema: input metadata (R10) — an `effector.Schema` or a plain `dict`
                with any of the keys `feature_names`, `feature_types`,
                `cat_limit`, `target_name`, `scale_x_list`, `scale_y`

                - omitted fields are auto-inferred from `data` (numpy
                  heuristics) or synthesized (`["x_0", ...]`, `"y"`); to start
                  from a DataFrame use `effector.from_dataframe`
                - explicit fields always win over inference

            random_state: seed for every internal random step (e.g. `nof_instances` subsampling)

                - `int` (default: `21`), for reproducible output; two identical constructions give identical results
                - `None`, for non-deterministic behavior
        """

        super(RegionalPDP, self).__init__(
            "pdp",
            data,
            model,
            None,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
        )

    def _precompute_global(self, feature: int):
        """Fit the global PDP once and keep the centered ICE table on the
        heterogeneity grid: candidate regions score row-subsets of it."""
        pdp = PDP(
            self.data,
            self.model,
            axis_limits=self.axis_limits,
            nof_instances="all",
            schema=self._node_schema(),
            random_state=self.random_state,
        )
        pdp.fit(
            features=feature,
            centering=True,
            points_for_centering=self.kwargs_fitting["points_for_centering"],
            use_vectorized=self.kwargs_fitting["use_vectorized"],
        )

        if ingestion.is_categorical(self.feature_types[feature]):
            xx = pdp._levels(feature)
        else:
            xx = np.linspace(
                self.axis_limits[0, feature],
                self.axis_limits[1, feature],
                helpers.NOF_INTERNAL_POINTS,
            )
        self.heter_grid["feature_" + str(feature)] = xx
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
        *,
        candidate_conditioning_features: typing.Union[str, list] = "all",
        space_partitioner: typing.Union[str, Best] = "best",
        points_for_centering: int = 30,
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
            use_vectorized: whether to use vectorized operations for the PDP and ICE curves


        """
        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
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
        nof_points: int = 100,
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
        schema: typing.Optional[typing.Union[ingestion.Schema, dict]] = None,
        random_state: typing.Optional[int] = 21,
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

            schema: input metadata (R10) — an `effector.Schema` or a plain `dict`
                with any of the keys `feature_names`, `feature_types`,
                `cat_limit`, `target_name`, `scale_x_list`, `scale_y`

                - omitted fields are auto-inferred from `data` (numpy
                  heuristics) or synthesized (`["x_0", ...]`, `"y"`); to start
                  from a DataFrame use `effector.from_dataframe`
                - explicit fields always win over inference

            random_state: seed for every internal random step (e.g. `nof_instances` subsampling)

                - `int` (default: `21`), for reproducible output; two identical constructions give identical results
                - `None`, for non-deterministic behavior
        """

        super(RegionalDerPDP, self).__init__(
            "d-pdp",
            data,
            model,
            model_jac,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
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
            schema=self._node_schema(),
            random_state=self.random_state,
        )
        pdp.fit(
            features=feature,
            centering=False,
            use_vectorized=self.kwargs_fitting["use_vectorized"],
        )

        if ingestion.is_categorical(self.feature_types[feature]):
            xx = pdp._levels(feature)
        else:
            xx = np.linspace(
                self.axis_limits[0, feature],
                self.axis_limits[1, feature],
                helpers.NOF_INTERNAL_POINTS,
            )
        self.heter_grid["feature_" + str(feature)] = xx
        y_ice = pdp._predict(
            pdp.data, xx, feature, self.kwargs_fitting["use_vectorized"]
        )
        self.y_ice["feature_" + str(feature)] = y_ice.T

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        *,
        candidate_conditioning_features: typing.Union[str, list] = "all",
        space_partitioner: typing.Union[str, Best] = "best",
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
            use_vectorized: whether to use vectorized operations for the PDP and ICE curves


        """
        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
        }
        self.kwargs_fitting = {"use_vectorized": use_vectorized}

        self._fit_loop(features, candidate_conditioning_features, space_partitioner)

    def plot(
        self,
        feature: int,
        node_idx: int,
        heterogeneity: typing.Union[bool, str] = "ice",
        centering: typing.Union[None, bool, str] = None,
        nof_points: int = 100,
        scale_x_list: typing.Union[None, list] = None,
        scale_y: typing.Union[None, dict] = None,
        nof_ice: typing.Union[int, str] = 100,
        show_avg_output: bool = False,
        y_limits: typing.Union[None, list] = None,
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
            y_limits: manual limits of the y-axis (derivative units)
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

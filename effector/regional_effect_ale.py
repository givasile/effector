import typing
from typing import Callable, Optional, Union

import numpy as np

import effector.space_partitioning
from effector import axis_partitioning as ap
from effector import helpers, ingestion, utils
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
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
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

        super(RegionalRHALE, self).__init__(
            "rhale",
            data,
            model,
            model_jac,
            data_effect=data_effect,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
        )

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points."""
        if self.data_effect is None and self.model_jac is not None:
            self.data_effect = self.model_jac(self.data)
        elif self.data_effect is None and self.model_jac is None:
            self.data_effect = utils.compute_jacobian_numerically(self.model, self.data)

    def _global_fe_kwargs(self) -> dict:
        # hand the already-computed jacobian to the global RHALE so it is not
        # recomputed there (the split search re-bins it, model-free)
        return {"data_effect": self.data_effect}

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        *,
        candidate_conditioning_features: typing.Union[str, list] = "all",
        space_partitioner: typing.Union[str, effector.space_partitioning.Best] = "best",
        binning_method: typing.Union[
            str,
            ap.Fixed,
            ap.DynamicProgramming,
            ap.Agglomerative,
            ap.Quantile,
        ] = "dp",
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
                  For custom parameters initialize a `axis_partitioning.Greedy` object
                - Use `"dp"` for using a Dynamic Programming binning solution with the default parameters.
                  For custom parameters initialize a `axis_partitioning.DynamicProgramming` object
                - Use `"fixed"` for using a Fixed binning solution with the default parameters.
                  For custom parameters initialize a `axis_partitioning.Fixed` object
        """
        if self.data_effect is None:
            self.compile()

        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
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
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: typing.Optional[int] = 21,
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

        super(RegionalALE, self).__init__(
            "ale",
            data,
            model,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
        )

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        *,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union[str, effector.space_partitioning.Best] = "best",
        binning_method: typing.Union[str, ap.Fixed] = "fixed",
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
                class `effector.axis_partitioning.Fixed` with the desired parameters.
                For example: `Fixed(nof_bins=20, min_points_per_bin=0)`
        """
        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
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

import typing
import warnings
from typing import Callable, List, Optional, Union

import numpy as np

import effector
from effector import axis_partitioning as ap
from effector import helpers, utils
from effector.regional_effect import RegionalEffectBase


class RegionalShapDP(RegionalEffectBase):
    big_m = helpers.BIG_M

    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        *,
        nof_instances: Union[int, str] = 1_000,
        axis_limits: Optional[np.ndarray] = None,
        feature_types: Optional[List[str]] = None,
        cat_limit: Optional[int] = 10,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        backend: str = "shap",
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

                !!! tip "`1_000` (default), is a good balance between speed and accuracy"

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

            backend: Package to compute SHAP values

                - use `"shap"` for the `shap` package (default)
                - use `"shapiq"` for the `shapiq` package
        """
        self.global_shap_values = None
        self.backend = backend
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

    def _extra_fe_kwargs(self, active_indices: np.ndarray) -> dict:
        """A node's ShapDP reuses the region's slice of the *global* shap
        values (attributions are not recomputed within the region)."""
        return {"shap_values": self.global_shap_values[active_indices, :]}

    def _precompute_global(self, feature: int):
        """Compute the global SHAP values once; regions score slices of them."""
        if self.global_shap_values is None:
            global_shap_dp = effector.ShapDP(
                self.data,
                self.model,
                axis_limits=self.axis_limits,
                nof_instances="all",
                backend=self.backend,
            )
            global_shap_dp.fit(feature, centering=False, **self.kwargs_fitting)
            self.global_shap_values = global_shap_dp.shap_values

    def _create_heterogeneity_function(self, feature: int, min_points: int):
        binning_method = ap.return_default(self.kwargs_fitting["binning_method"])
        points_for_mean_heterogeneity = self.kwargs_subregion_detection[
            "points_for_mean_heterogeneity"
        ]

        def heterogeneity_function(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return self.big_m

            data = self.data[active_indices.astype(bool), :]
            shap_values = self.global_shap_values[active_indices.astype(bool), :]
            shap_dp = effector.ShapDP(
                data,
                self.model,
                axis_limits=self.axis_limits,
                nof_instances="all",
                shap_values=shap_values,
            )

            try:
                shap_dp.fit(
                    features=feature, binning_method=binning_method, centering=False
                )
            except utils.AllBinsHaveAtMostOnePointError as e:
                warnings.warn(
                    f"RegionalShapDP: at a candidate split, some bins had at most "
                    f"one point; the split is rejected. Error: {e}"
                )
                return self.big_m
            except Exception as e:
                warnings.warn(
                    f"RegionalShapDP: an unexpected error occurred at a candidate "
                    f"split; the split is rejected. Error: {e}"
                )
                return self.big_m

            xs = np.linspace(
                self.axis_limits[0, feature],
                self.axis_limits[1, feature],
                points_for_mean_heterogeneity,
            )
            z = shap_dp.eval_heter(feature, xs)
            return np.mean(z)

        return heterogeneity_function

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union[
            "str", effector.space_partitioning.Best
        ] = "best",
        binning_method: Union[str, ap.Greedy, ap.Fixed] = "greedy",
        budget: int = 512,
        points_for_mean_heterogeneity: int = 30,
        shap_explainer_kwargs: Optional[dict] = None,
        shap_explanation_kwargs: Optional[dict] = None,
    ):
        """
        Fit the regional SHAP.

        Args:
            features: the features to fit.
                - If set to "all", all the features will be fitted.

            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
                - If set to "all", all the features will be considered as conditioning features.

            space_partitioner: the space partitioner to use
                - If set to "greedy", the greedy space partitioner will be used.

            binning_method: the binning method to use

            budget: Budget to use for the approximation. Defaults to 512.
                - Increasing the budget improves the approximation at the cost of slower computation.
                - Decrease the budget for faster computation at the cost of approximation error.

            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity

            shap_explainer_kwargs: the keyword arguments to be passed to the `shap.Explainer` or `shapiq.Explainer` class, depending on the backend.

                ??? note "Code behind the scene"

                    See `effector.global_effect_shap._compute_shap_values` — the single place the explainer is constructed and invoked.

                ??? warning "Be careful with custom arguments"

                    For customizing `shap_explainer_kwargs` and `shap_explanation_kwargs` args,
                    check the official documentation of [`shap`](https://shap.readthedocs.io/en/latest/) and [`shapiq`](https://shapiq.readthedocs.io/en/latest/) packages.

            shap_explanation_kwargs: the keyword arguments to be passed to the `shap` or `shapiq` Explainer to compute the SHAP values.

                ??? note "Code behind the scene"

                    See `effector.global_effect_shap._compute_shap_values` — the single place the explainer is constructed and invoked.

                ??? warning "Be careful with custom arguments"

                    For customizing `shap_explainer_kwargs` and `shap_explanation_kwargs` args,
                    check the official documentation of [`shap`](https://shap.readthedocs.io/en/latest/) and [`shapiq`](https://shapiq.readthedocs.io/en/latest/) packages.

        """
        self.kwargs_subregion_detection = {
            "features": features,
            "candidate_conditioning_features": candidate_conditioning_features,
            "space_partitioner": space_partitioner,
            "points_for_mean_heterogeneity": points_for_mean_heterogeneity,
        }
        self.kwargs_fitting = {
            "binning_method": binning_method,
            "budget": budget,
            "shap_explainer_kwargs": shap_explainer_kwargs,
            "shap_explanation_kwargs": shap_explanation_kwargs,
        }

        self._fit_loop(features, candidate_conditioning_features, space_partitioner)

    def plot(
        self,
        feature: int,
        node_idx: int,
        heterogeneity: Union[bool, str] = "shap_values",
        centering: Union[None, bool, str] = None,
        nof_points: int = 30,
        scale_x_list: Optional[list] = None,
        scale_y: Optional[dict] = None,
        nof_shap_values: Union[int, str] = "all",
        show_avg_output: bool = False,
        y_limits: Optional[list] = None,
        only_shap_values: bool = False,
        show_plot: bool = True,
    ):
        """
        Plot the regional SHAP.

        Args:
            feature: the feature to plot
            node_idx: the index of the node to plot
            heterogeneity: whether to plot the heterogeneity
            centering: whether to center the SHAP values (`None` uses the class default)
            nof_points: number of points to plot
            scale_x_list: the list of scaling factors for the feature names
            scale_y: the scaling factor for the SHAP values
            nof_shap_values: number of SHAP values to plot
            show_avg_output: whether to show the average output
            y_limits: the limits of the y-axis
            only_shap_values: whether to plot only the SHAP values
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
                nof_shap_values=nof_shap_values,
                show_avg_output=show_avg_output,
                y_limits=y_limits,
                only_shap_values=only_shap_values,
                show_plot=show_plot,
            ),
        )

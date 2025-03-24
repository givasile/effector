import typing

import effector
from effector.regional_effect import RegionalEffectBase
from effector import helpers
import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Union, List
from effector import axis_partitioning as ap
from effector import utils




class RegionalShapDP(RegionalEffectBase):
    big_m = helpers.BIG_M

    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 1_000,
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

    def _create_heterogeneity_function(self, foi, min_points, binning_method):
        if isinstance(binning_method, str):
            binning_method = ap.return_default(binning_method)

        def heterogeneity_function(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return self.big_m

            data = self.data[active_indices.astype(bool), :]
            shap_values = self.global_shap_values[active_indices.astype(bool), :]
            shap_dp = effector.ShapDP(data, self.model, self.axis_limits, "all", shap_values=shap_values)

            try:
                shap_dp.fit(features=foi, binning_method=binning_method, centering=False)
            except utils.AllBinsHaveAtMostOnePointError as e:
                print(f"RegionalShapDP here: At a particular split, some bins had at most one point. I reject this split. \n Error: {e}")
                return self.big_m
            except Exception as e:
                print(f"RegionalShapDP here: An unexpected error occurred. I reject this split. \n Error: {e}")
                return self.big_m

            mean_spline = shap_dp.feature_effect["feature_" + str(foi)]["spline_mean"]

            xs = np.linspace(self.axis_limits[0, foi], self.axis_limits[1, foi], 30)
            _, z = shap_dp.eval(feature=foi, xs=xs, centering=False, heterogeneity=True)
            # residuals = (shap_values[:, foi] - mean_spline(data[:, foi]))**2
            return np.mean(z)

        return heterogeneity_function

    def fit(
        self,
        features: typing.Union[int, str, list],
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union["str", effector.space_partitioning.Best] = "best",
        binning_method: Union[str, ap.Greedy, ap.Fixed] = "greedy",
        budget: int = 512,
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

            shap_explainer_kwargs: the keyword arguments to be passed to the `shap.Explainer` or `shapiq.Explainer` class, depending on the backend.

                ??? note "Code behind the scene"
                    Check the code that is running behind the scene before customizing `shap_explainer_kwargs`.

                    ```python
                    explainer_kwargs = explainer_kwargs.copy() if explainer_kwargs else {}
                    explanation_kwargs = explanation_kwargs.copy() if explanation_kwargs else {}
                    if self.backend == "shap":
                        explainer_defaults = {"masker": data}
                        explanation_defaults = {"max_evals": budget}
                    elif self.backend == "shapiq":
                        explainer_defaults = {
                            "data": data,
                            "index": "SV",
                            "max_order": 1,
                            "approximator": "permutation",
                            "imputer": "marginal",
                        }
                        explanation_defaults = {"budget": budget}
                    else:
                        raise ValueError("`backend` should be either 'shap' or 'shapiq'")
                    explainer_kwargs = {**explainer_defaults, **explainer_kwargs}  # User args override defaults
                    explanation_kwargs = {**explanation_defaults, **explanation_kwargs}  # User args override defaults

                    if self.backend == "shap":
                        explainer = shap.Explainer(model, **explainer_kwargs)
                        explanation = explainer(data, **explanation_kwargs)
                        self.shap_values = explanation.values
                    elif self.backend == "shapiq":
                        explainer = shapiq.Explainer(model, **explainer_kwargs)
                        explanations = explainer.explain_X(data, **explanation_kwargs)
                        self.shap_values = np.stack([ex.get_n_order_values(1) for ex in explanations])
                    else:
                        raise ValueError("`backend` should be either 'shap' or 'shapiq'")
                    ```

                ??? warning "Be careful with custom arguments"

                    For customizing `shap_explainer_kwargs` and `shap_explanation_kwargs` args,
                    check the official documentation of [`shap`](https://shap.readthedocs.io/en/latest/) and [`shapiq`](https://shapiq.readthedocs.io/en/latest/) packages.

            shap_explanation_kwargs: the keyword arguments to be passed to the `shap` or `shapiq` Explainer to compute the SHAP values.

                ??? note "Code behind the scene"

                    Check the code that is running behind the scene before customizing `shap_explanation_kwargs`.

                    ```python
                    explainer_kwargs = explainer_kwargs.copy() if explainer_kwargs else {}
                    explanation_kwargs = explanation_kwargs.copy() if explanation_kwargs else {}
                    if self.backend == "shap":
                        explainer_defaults = {"masker": data}
                        explanation_defaults = {"max_evals": budget}
                    elif self.backend == "shapiq":
                        explainer_defaults = {
                            "data": data,
                            "index": "SV",
                            "max_order": 1,
                            "approximator": "permutation",
                            "imputer": "marginal",
                        }
                        explanation_defaults = {"budget": budget}
                    else:
                        raise ValueError("`backend` should be either 'shap' or 'shapiq'")
                    explainer_kwargs = {**explainer_defaults, **explainer_kwargs}  # User args override defaults
                    explanation_kwargs = {**explanation_defaults, **explanation_kwargs}  # User args override defaults

                    if self.backend == "shap":
                        explainer = shap.Explainer(model, **explainer_kwargs)
                        explanation = explainer(data, **explanation_kwargs)
                        self.shap_values = explanation.values
                    elif self.backend == "shapiq":
                        explainer = shapiq.Explainer(model, **explainer_kwargs)
                        explanations = explainer.explain_X(data, **explanation_kwargs)
                        self.shap_values = np.stack([ex.get_n_order_values(1) for ex in explanations])
                    else:
                        raise ValueError("`backend` should be either 'shap' or 'shapiq'")
                    ```

                ??? warning "Be careful with custom arguments"

                    For customizing `shap_explainer_kwargs` and `shap_explanation_kwargs` args,
                    check the official documentation of [`shap`](https://shap.readthedocs.io/en/latest/) and [`shapiq`](https://shapiq.readthedocs.io/en/latest/) packages.

        """

        if isinstance(space_partitioner, str):
            space_partitioner = effector.space_partitioning.return_default(space_partitioner)

        assert space_partitioner.min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)

        for feat in tqdm(features):
            # assert global SHAP values are available
            if self.global_shap_values is None:
                global_shap_dp = effector.ShapDP(self.data, self.model, self.axis_limits, "all", backend=self.backend)
                global_shap_dp.fit(
                    feat,
                    centering=False,
                    binning_method=binning_method,
                    budget=budget,
                    shap_explainer_kwargs=shap_explainer_kwargs,
                    shap_explanation_kwargs=shap_explanation_kwargs
                )
                self.global_shap_values = global_shap_dp.shap_values

            heter = self._create_heterogeneity_function(feat, space_partitioner.min_points_per_subregion, binning_method)

            self._fit_feature(
                feat,
                heter,
                space_partitioner,
                candidate_conditioning_features,
            )

        all_arguments = locals()
        all_arguments.pop("self")

        # region splitting arguments are the first 3 arguments
        self.kwargs_subregion_detection = {k: all_arguments[k] for k in list(all_arguments.keys())[:3]}

        # fit kwargs
        self.kwargs_fitting = {
            "binning_method": binning_method,
            "budget": budget,
            "shap_explainer_kwargs": shap_explainer_kwargs,
            "shap_explanation_kwargs": shap_explanation_kwargs
        }

    def plot(self,
             feature,
             node_idx,
             heterogeneity="shap_values",
             centering=True,
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

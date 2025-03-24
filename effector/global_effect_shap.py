import typing
from typing import Callable, List, Optional, Union, Tuple
import effector.visualization as vis
import effector.helpers as helpers
from effector.global_effect import GlobalEffectBase
import numpy as np
import shap
import shapiq
import effector.axis_partitioning as ap
import effector.utils as utils
from scipy.interpolate import interp1d


class ShapDP(GlobalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 1_000,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        shap_values: Optional[np.ndarray] = None,
        backend: str = "shap",
    ):
        """
        Constructor of the ShapDP class.

        ??? note "Definition"

            The value of a coalition of $S$ features is estimated as:
            $$
            \hat{v}(S) = {1 \over N} \sum_{i=1}^N [f(\mathbf{x}_S \cup \mathbf{x}_C^i) - f(\mathbf{x}^i) ]
            $$
            $\hat{v}(S)$ quantifies the contribution when the features in $S$ are set to $\mathbf{x}_S$.
            For all instances, we compute two outputs:

              - $f(\mathbf{x}_S \cup \mathbf{x}_C^i)$ is the output of the model when the features in $S$ are set to $\mathbf{x}_S$ and the rest of the features are left as they are
              - $f(\mathbf{x}^i)$ is the output of the model when the instance is left as is
            The average difference (over all instances) between these two outputs is the value of the coalition $S$.

            The contribution of a feature $j$ added to a coalition $S$ is estimated as:
            $$
            \hat{\Delta}_{S, j} = \hat{v}(S \cup \{j\}) - \hat{v}(S)
            $$

            The SHAP value of a feature $j$ with value $x_j$ is the average contribution of feature $j$ across all possible coalitions with a weight $w_{S, j}$:

            $$
            \hat{\phi}_j(x_j) = {1 \over N} \sum_{S \subseteq \{1, \dots, D\} \setminus \{j\}} w_{S, j} \hat{\Delta}_{S, j}
            $$

            where $w_{S, j}$ assures that the contribution of feature $j$ is the same for all coalitions of the same size. For example, there are $D-1$ ways for $x_j$ to enter a coalition of $|S| = 1$ feature, so $w_{S, j} = {1 \over D (D-1)}$ for each of them. In contrast, there is only one way for $x_j$ to enter a coaltion of $|S|=0$ (to be the first specified feature), so $w_{S, j} = {1 \over D}$.

            The SHAP Dependence Plot (SHAP-DP) is a spline $\hat{f}^{SDP}_j(x_j)$ fit to the dataset $\{(x_j^i, \hat{\phi}_j(x_j^i))\}_{i=1}^N$ using the `UnivariateSpline` function from `scipy.interpolate`.

        ??? note "Notes"

            * The required parameters are `data` and `model`. The rest are optional.
            * SHAP values are computed using either the [`shap`](https://shap.readthedocs.io/en/latest/) package (`backend="shap"`) or the [`shapiq`](https://shapiq.readthedocs.io/en/latest/) package (`backend="shapiq"`).
            * SHAP values are centered by default, i.e., the average SHAP value is subtracted from the SHAP values.
            * More details on the SHAP values can be found in the [original paper](https://arxiv.org/abs/1705.07874) and in the book [Interpreting Machine Learning Models with SHAP](https://christophmolnar.com/books/shap/)

        Args:
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N,)`

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            nof_instances: maximum number of instances to be used for SHAP estimation.

                - use `"all"`, for using all instances.
                - use an `int`, for using `nof_instances` instances.

            avg_output: The average output of the model.

                - use a `float`, to specify it manually
                - use `None`, to be inferred as `np.mean(model(data))`

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `                  ["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`

            shap_values: The SHAP values of the model

                - if shap values are already computed, they can be passed here
                - if `None`, the SHAP values will be computed using the `shap` package

            backend: Package to compute SHAP values

                - use `"shap"` for the `shap` package (default)
                - use `"shapiq"` for the `shapiq` package

        Notes:
            SHAP values are expensive to compute.
            To speed up the computation consider using a subset of the dataset.
            The `nof_instances` parameter controls the number of instances used for computing the SHAP values.
            The default value is `1_000` instances, which is a good trade-off between speed and accuracy.
        """
        self.shap_values = shap_values if shap_values is not None else None
        assert backend in ["shap", "shapiq"]
        self.backend = backend
        super(ShapDP, self).__init__(
            "SHAP DP",
            data,
            model,
            None,
            None,
            nof_instances,
            axis_limits,
            feature_names,
            target_name,
        )

    def _fit_feature(
        self,
        feature: int,
        binning_method: Union[str, ap.Greedy, ap.Fixed] = "greedy",
        centering: typing.Union[bool, str] = False,
        points_for_centering: int = 30,
        budget: int = 512,
        explainer_kwargs: Optional[dict] = None,
        explanation_kwargs: Optional[dict] = None,
    ) -> typing.Dict:

        data = self.data
        model = self.model


        if self.shap_values is None:
            # prepare arguments
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

            # actual code
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

        # extract x and y
        yy = self.shap_values[:, feature]
        xx = data[:, feature]

        if isinstance(binning_method, str):
            binning_method = ap.return_default(binning_method)

        limits = binning_method.find_limits(data[:, feature], self.shap_values[:, feature], self.axis_limits[:, feature])

        # assert bins can be computed else raise error
        assert limits is not False, (
            "Impossible to compute bins with enough points for feature with index: i="
            + str(feature + 1)
            + " and binning strategy: "
            + str(binning_method)
            + ". Change bin strategy or "
            "the parameters of the method"
        )
        # compute the bin effect
        feature_effect_dict = utils.compute_ale_params(data[:, feature], self.shap_values[:, feature], limits)
        feature_effect_dict["alg_params"] = binning_method

        # Compute bin edges and bin centers
        bin_centers = (limits[:-1] + limits[1:]) / 2

        # Create piecewise linear interpolation
        mean_spline = interp1d(bin_centers, feature_effect_dict["bin_effect"], kind='linear', fill_value="extrapolate")
        var_spline = interp1d(bin_centers, feature_effect_dict["bin_variance"], kind='linear', fill_value="extrapolate")

        # compute norm constant
        if centering == "zero_integral":
            x_norm = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], points_for_centering)
            y_norm = mean_spline(x_norm)
            norm_const = np.mean(y_norm)
        elif centering == "zero_start":
            norm_const = mean_spline(self.axis_limits[0, feature])
        else:
            norm_const = helpers.EMPTY_SYMBOL

        ret_dict = {
            "spline_mean": mean_spline,
            "spline_std": var_spline,
            "xx": xx,
            "yy": yy,
            "norm_const": norm_const,
        }
        return ret_dict

    def fit(
        self,
        features: Union[int, str, List] = "all",
        centering: Union[bool, str] = True,
        points_for_centering: Union[int, str] = 30,
        binning_method: Union[str, ap.Greedy, ap.Fixed] = "greedy",
        budget: int = 512,
        shap_explainer_kwargs: Optional[dict] = None,
        shap_explanation_kwargs: Optional[dict] = None,
    ) -> None:
        """Fit the SHAP Dependence Plot to the data.

        Notes:
            The SHAP Dependence Plot (SDP) $\hat{f}^{SDP}_j(x_j)$ is a spline fit to
            the dataset $\{(x_j^i, \hat{\phi}_j(x_j^i))\}_{i=1}^N$
            using the `UnivariateSpline` function from `scipy.interpolate`.

            The SHAP standard deviation, $\hat{\sigma}^{SDP}_j(x_j)$, is a spline fit            to the absolute value of the residuals, i.e., to the dataset $\{(x_j^i, |\hat{\phi}_j(x_j^i) - \hat{f}^{SDP}_j(x_j^i)|)\}_{i=1}^N$, using the `UnivariateSpline` function from `scipy.interpolate`.

        Args:
            features: the features to fit.
                - If set to "all", all the features will be fitted.
            centering:
                - If set to False, no centering will be applied.
                - If set to "zero_integral" or True, the integral of the feature effect will be set to zero.
                - If set to "zero_mean", the mean of the feature effect will be set to zero.

            points_for_centering: number of linspaced points along the feature axis used for centering.

                - If set to `all`, all the dataset points will be used.


            binning_method: the binning method to be used for fitting a piecewise linear function to the SHAP values.

                - If set to "greedy", the greedy binning method will be used.
                - If set to "fixed", the fixed binning method will be used.

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
        centering = helpers.prep_centering(centering)
        features = helpers.prep_features(features, self.dim)

        # new implementation
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(
                s,
                binning_method,
                centering,
                points_for_centering,
                budget,
                shap_explainer_kwargs,
                shap_explanation_kwargs
            )
            self.is_fitted[s] = True
            self.fit_args["feature_" + str(s)] = {
                "centering": centering,
                "points_for_centering": points_for_centering,
            }

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        heterogeneity: bool = True,
        centering: typing.Union[bool, str] = True,
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot

              - `np.ndarray` of shape `(T,)`
            heterogeneity: whether to return the heterogeneity measures.

                  - if `heterogeneity=False`, the function returns the mean effect at the given `xs`
                  - If `heterogeneity=True`, the function returns `(y, std)` where `y` is the mean effect and `std` is the standard deviation of the mean effect

            centering: whether to center the plot

                - If `centering` is `False`, the SHAP curve is not centered
                - If `centering` is `True` or `zero_integral`, the SHAP curve is centered around the `y` axis.
                - If `centering` is `zero_start`, the SHAP curve starts from `y=0`.

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, std, estimator_var)` otherwise
        """
        centering = helpers.prep_centering(centering)

        if self.requires_refit(feature, centering):
            self.fit(features=feature, centering=centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        yy = self.feature_effect["feature_" + str(feature)]["spline_mean"](xs)

        if centering is not False:
            norm_const = self.feature_effect["feature_" + str(feature)]["norm_const"]
            yy = yy - norm_const

        if heterogeneity:
            yy_var = self.feature_effect["feature_" + str(feature)]["spline_std"](xs)
            return yy, yy_var
        else:
            return yy

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = "shap_values",
        centering: Union[bool, str] = True,
        nof_points: int = 30,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_shap_values: Union[int, str] = "all",
        show_avg_output: bool = False,
        y_limits: Optional[List] = None,
        only_shap_values: bool = False,
        show_plot: bool = True,
    ) -> Union[Tuple, None]:
        """
        Plot the SHAP Dependence Plot (SDP) of the s-th feature.

        Args:
            feature: index of the plotted feature
            heterogeneity: whether to output the heterogeneity of the SHAP values

                - If `heterogeneity` is `False`, no heterogeneity is plotted
                - If `heterogeneity` is `True` or `"std"`, the standard deviation of the shap values is plotted
                - If `heterogeneity` is `"shap_values"`, the shap values are scattered on top of the SHAP curve

            centering: whether to center the SDP

                - If `centering` is `False`, the SHAP curve is not centered
                - If `centering` is `True` or `zero_integral`, the SHAP curve is centered around the `y` axis.
                - If `centering` is `zero_start`, the SHAP curve starts from `y=0`.

            nof_points: number of points to evaluate the SDP plot
            scale_x: dictionary with keys "mean" and "std" for scaling the x-axis
            scale_y: dictionary with keys "mean" and "std" for scaling the y-axis
            nof_shap_values: number of shap values to show on top of the SHAP curve
            show_avg_output: whether to show the average output of the model
            y_limits: limits of the y-axis
            only_shap_values: whether to plot only the shap values
            show_plot: whether to show the plot
        """
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)

        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        # get the SHAP curve
        y = self.eval(feature, x, heterogeneity=False, centering=centering)
        y_std = (
            np.sqrt(self.feature_effect["feature_" + str(feature)]["spline_std"](x))
            if heterogeneity == "std" or True
            else None
        )

        # get some SHAP values
        _, ind = helpers.prep_nof_instances(nof_shap_values, self.data.shape[0])
        yy = (
            self.feature_effect["feature_" + str(feature)]["yy"][ind]
            if heterogeneity == "shap_values"
            else None
        )
        if yy is not None and centering is not False:
            yy = yy - self.feature_effect["feature_" + str(feature)]["norm_const"]
        xx = (
            self.feature_effect["feature_" + str(feature)]["xx"][ind]
            if heterogeneity == "shap_values"
            else None
        )

        if show_avg_output:
            avg_output = helpers.prep_avg_output(
                self.data, self.model, self.avg_output, scale_y
            )
        else:
            avg_output = None

        ret = vis.plot_shap(
            x,
            y,
            xx,
            yy,
            y_std,
            feature,
            heterogeneity=heterogeneity,
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            y_limits=y_limits,
            only_shap_values=only_shap_values,
            show_plot=show_plot,
        )

        return ret

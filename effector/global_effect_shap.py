import typing
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import effector.helpers as helpers
import effector.visualization as vis
from effector import ingestion
from effector.global_effect import GlobalEffectBase

try:
    import shap
except ImportError:
    shap = None

try:
    import shapiq
except ImportError:
    shapiq = None

from scipy.interpolate import interp1d

import effector.axis_partitioning as ap
import effector.utils as utils


def _compute_shap_values(
    model,
    data,
    backend,
    budget,
    explainer_kwargs=None,
    explanation_kwargs=None,
    random_state=None,
):
    """Compute per-instance SHAP values `(N, D)` with the chosen backend.

    Defaults (user kwargs override them):
      - `shap`:   `Explainer(model, masker=data, seed=random_state)`,
        `explainer(data, max_evals=budget)`
      - `shapiq`: `Explainer(model, data=data, index="SV", max_order=1,
        approximator="permutation", imputer="marginal",
        random_state=random_state)`,
        `explainer.explain_X(data, budget=budget)`
    """
    explainer_kwargs = explainer_kwargs.copy() if explainer_kwargs else {}
    explanation_kwargs = explanation_kwargs.copy() if explanation_kwargs else {}
    if backend == "shap":
        if shap is None:
            raise ImportError(
                "The `shap` package is required for backend='shap'. "
                "Install it with `pip install effector[shap]`."
            )
        explainer_defaults = {"masker": data, "seed": random_state}
        explanation_defaults = {"max_evals": budget}
    elif backend == "shapiq":
        if shapiq is None:
            raise ImportError(
                "The `shapiq` package is required for backend='shapiq'. "
                "Install it with `pip install effector[shap]`."
            )
        explainer_defaults = {
            "data": data,
            "index": "SV",
            "max_order": 1,
            "approximator": "permutation",
            "imputer": "marginal",
            "random_state": random_state,
        }
        explanation_defaults = {"budget": budget}
    else:
        raise ValueError("`backend` should be either 'shap' or 'shapiq'")

    explainer_kwargs = {**explainer_defaults, **explainer_kwargs}
    explanation_kwargs = {**explanation_defaults, **explanation_kwargs}

    if backend == "shap":
        explainer = shap.Explainer(model, **explainer_kwargs)
        explanation = explainer(data, **explanation_kwargs)
        return explanation.values
    explainer = shapiq.Explainer(model, **explainer_kwargs)
    explanations = explainer.explain_X(data, **explanation_kwargs)
    return np.stack([ex.get_n_order_values(1) for ex in explanations])


class ShapDP(GlobalEffectBase):
    DEFAULT_CENTERING: Union[bool, str] = "zero_integral"

    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        *,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 1_000,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
        shap_values: Optional[np.ndarray] = None,
        backend: str = "shap",
        budget: int = 512,
        shap_explainer_kwargs: Optional[dict] = None,
        shap_explanation_kwargs: Optional[dict] = None,
    ):
        r"""
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

            schema: input metadata (R10) — an `effector.Schema` or a plain `dict`
                with any of the keys `feature_names`, `feature_types`,
                `cat_limit`, `target_name`, `scale_x_list`, `scale_y`

                - omitted fields are inferred from the data (DataFrame dtypes,
                  numpy heuristics) or synthesized (`["x_0", ...]`, `"y"`)
                - explicit fields always win over inference

            random_state: seed for every internal random step (`nof_instances` subsampling and the shap/shapiq explainer, unless overridden via `shap_explainer_kwargs`)

                - use an `int` (default: `21`), for reproducible output; two identical constructions give identical results
                - use `None`, for non-deterministic behavior

            shap_values: The SHAP values of the model

                - if shap values are already computed, they can be passed here
                - if `None`, the SHAP values will be computed using the `shap` package

            backend: Package to compute SHAP values

                - use `"shap"` for the `shap` package (default)
                - use `"shapiq"` for the `shapiq` package

            budget: budget for the SHAP approximation (default 512)

                - increasing the budget improves the approximation at the cost of slower computation

            shap_explainer_kwargs: keyword arguments for the `shap.Explainer` /
                `shapiq.Explainer` (depending on `backend`). The constructor's
                `random_state` is used as the backend seed (`seed=` for `shap`,
                `random_state=` for `shapiq`) unless you pass your own here.
                See `effector.global_effect_shap._compute_shap_values` — the
                single place the explainer is constructed and invoked.
            shap_explanation_kwargs: keyword arguments for computing the SHAP
                values with the chosen backend (same code path as above).

        Notes:
            SHAP values are expensive to compute.
            To speed up the computation consider using a subset of the dataset.
            The `nof_instances` parameter controls the number of instances used for computing the SHAP values.
            The default value is `1_000` instances, which is a good trade-off between speed and accuracy.
        """
        self.shap_values = shap_values if shap_values is not None else None
        if backend not in ["shap", "shapiq"]:
            raise ValueError(f"Invalid backend: {backend!r}; use 'shap' or 'shapiq'")
        self.backend = backend
        self.budget = budget
        self.shap_explainer_kwargs = shap_explainer_kwargs
        self.shap_explanation_kwargs = shap_explanation_kwargs
        super(ShapDP, self).__init__(
            "SHAP DP",
            data,
            model,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
        )

    def _fit_feature(
        self,
        feature: int,
        binning_method: Union[str, ap.Greedy, ap.Fixed] = "greedy",
    ) -> typing.Dict:
        data = self.data

        if self.shap_values is None:
            self.shap_values = _compute_shap_values(
                self.model,
                data,
                self.backend,
                self.budget,
                self.shap_explainer_kwargs,
                self.shap_explanation_kwargs,
                self.random_state,
            )

        # extract x and y
        yy = self.shap_values[:, feature]
        xx = data[:, feature]

        if isinstance(binning_method, str):
            binning_method = ap.return_default(binning_method)

        limits = binning_method.find_limits(
            data[:, feature], self.shap_values[:, feature], self.axis_limits[:, feature]
        )

        utils.raise_if_no_binning(limits, feature, binning_method)
        # compute the bin effect
        feature_effect_dict = utils.compute_ale_params(
            data[:, feature], self.shap_values[:, feature], limits
        )
        feature_effect_dict["alg_params"] = binning_method

        # Compute bin edges and bin centers
        bin_centers = (limits[:-1] + limits[1:]) / 2

        # Create piecewise linear interpolation
        mean_spline = interp1d(
            bin_centers,
            feature_effect_dict["bin_effect"],
            kind="linear",
            fill_value="extrapolate",
        )
        var_spline = interp1d(
            bin_centers,
            feature_effect_dict["bin_variance"],
            kind="linear",
            fill_value="extrapolate",
        )

        ret_dict = {
            "spline_mean": mean_spline,
            "spline_var": var_spline,
            "xx": xx,
            "yy": yy,
        }
        return ret_dict

    def _eval_unnorm(self, feature: int, x: np.ndarray, heterogeneity: bool = False):
        params = self.feature_effect["feature_" + str(feature)]
        y = params["spline_mean"](x)
        if heterogeneity:
            return y, params["spline_var"](x)
        return y

    def fit(
        self,
        features: Union[int, str, List] = "all",
        *,
        centering: Union[bool, str] = True,
        points_for_centering: int = 30,
        binning_method: Union[str, ap.Greedy, ap.Fixed] = "greedy",
    ) -> None:
        r"""Fit the SHAP Dependence Plot to the data.

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

            binning_method: the binning method to be used for fitting a piecewise linear function to the SHAP values.

                - If set to "greedy", the greedy binning method will be used.
                - If set to "fixed", the fixed binning method will be used.

        """
        self._fit_loop(
            features,
            centering,
            points_for_centering,
            binning_method=binning_method,
        )

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = "shap_values",
        centering: Union[bool, str] = True,
        nof_points: int = 100,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_shap_values: Union[int, str] = 100,
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
        centering = helpers.prep_centering(centering)
        scale_x = helpers.resolve_scale(
            scale_x, self.scale_x_list[feature] if self.scale_x_list else None
        )
        scale_y = helpers.resolve_scale(scale_y, self.scale_y)

        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        # get the SHAP curve
        y = self.eval(feature, x, centering=centering)
        y_std = (
            np.sqrt(self.feature_effect["feature_" + str(feature)]["spline_var"](x))
            if heterogeneity == "std"
            else None
        )

        # get some SHAP values
        _, ind = helpers.prep_nof_instances(
            nof_shap_values, self.data.shape[0], self.random_state
        )
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
            avg_output = helpers.prep_avg_output(self.data, self.model, None, scale_y)
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

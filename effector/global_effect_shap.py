import typing
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

import effector.helpers as helpers
import effector.visualization as vis
from effector import ingestion
from effector.global_effect import GlobalEffectBase, check_binning_scope

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
    CAT_STRATEGY = "per_level_stats"

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

                - omitted fields are auto-inferred from `data` (numpy
                  heuristics) or synthesized (`["x_0", ...]`, `"y"`); to start
                  from a DataFrame use `effector.from_dataframe`
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

    def _compute_local(self, feature: int, frame: tuple) -> dict:
        """Cache-(a) kernel: the SHAP values are the local effect — computed
        once per object by the backend (or injected via `shap_values=`),
        feature-independent. Fill the whole `(N,D)` table on first use; each
        feature's entry is a column view of it (no frame — instance-anchored)."""
        if self.shap_values is None:
            self.shap_values = _compute_shap_values(
                self.model,
                self.data,
                self.backend,
                self.budget,
                self.shap_explainer_kwargs,
                self.shap_explanation_kwargs,
                self.random_state,
            )
        return {"frame": frame, "phi": self.shap_values[:, feature]}

    def _importance(self, feature, mask):
        """R13 for SHAP: the canonical `mean(|phi_s|)` over the (masked)
        instances — `phi_s` is already the cached local effect."""
        phi = self._local[feature]["phi"][mask]
        return float(np.mean(np.abs(phi)))

    def _summarize(
        self,
        feature: int,
        mask=None,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        binning_scope: str = "global",
    ) -> typing.Dict:
        """Step 3 (pure numpy): bin/aggregate the cached SHAP column over the
        subregion `mask` (None = all) — a spline of per-bin mean/variance for
        continuous features, per-level mean/variance for discrete ones.

        `binning_scope` (masked only): `"global"` bins over the frozen global
        frame, `"effective"` packs the bins into the masked column's own
        `[min, max]` (see `fit`)."""
        yy = self._local[feature]["phi"]
        xx = self.data[:, feature]
        if mask is not None:
            yy = yy[mask]
            xx = xx[mask]

        if self._is_cat(feature):
            # per-level mean/variance of the shap values with a step lookup —
            # no spline, no order enters the math (method_semantics.md)
            levels = np.unique(xx)
            codes = utils.codes_from_levels(
                xx, levels, feature, self.feature_names[feature]
            )
            limits = np.arange(len(levels) + 1, dtype=float) - 0.5
            feature_effect_dict = utils.compute_ale_params(
                codes.astype(float), yy, limits
            )
            return {
                "bin_effect": feature_effect_dict["bin_effect"],
                "bin_variance": feature_effect_dict["bin_variance"],
                "levels": levels,
                "is_cat": True,
                "xx": xx,
                "yy": yy,
            }

        binning = (
            ap.return_default(binning_method)
            if isinstance(binning_method, str)
            else binning_method
        )
        if mask is not None and binning_scope == "effective":
            limits_range = np.asarray(self._effective_limits(feature, mask))
        else:
            limits_range = self.axis_limits[:, feature]
        limits = binning.find_limits(xx, yy, limits_range)
        utils.raise_if_no_binning(limits, feature, binning)
        feature_effect_dict = utils.compute_ale_params(xx, yy, limits)
        feature_effect_dict["alg_params"] = binning

        # Compute bin edges and bin centers, then piecewise-linear interpolation
        bin_centers = (limits[:-1] + limits[1:]) / 2
        if len(bin_centers) == 1:
            # a single bin (e.g. an unstructured φ that the adaptive binning
            # rightly refuses to split): interp1d on one knot divides by a
            # zero span and returns nan everywhere — use the constant instead
            mean_val = float(feature_effect_dict["bin_effect"][0])
            var_val = float(feature_effect_dict["bin_variance"][0])

            def mean_spline(x, _v=mean_val):
                return np.full(np.shape(x), _v)

            def var_spline(x, _v=var_val):
                return np.full(np.shape(x), _v)
        else:
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
        return {
            "spline_mean": mean_spline,
            "spline_var": var_spline,
            "xx": xx,
            "yy": yy,
        }

    def _eval_payload(
        self, feature: int, params: dict, x: np.ndarray, heterogeneity: bool = False
    ):
        if params.get("is_cat"):
            codes = utils.codes_from_levels(
                x, params["levels"], feature, self.feature_names[feature]
            )
            y = params["bin_effect"][codes]
            if heterogeneity:
                return y, params["bin_variance"][codes]
            return y
        y = params["spline_mean"](x)
        if heterogeneity:
            # variance is non-negative by definition; linear extrapolation of the
            # per-bin variance beyond the outer bin centers (fill_value=
            # "extrapolate") can dip below 0, so clamp it.
            return y, np.maximum(params["spline_var"](x), 0.0)
        return y

    def fit(
        self,
        features: Union[int, str, List] = "all",
        *,
        centering: Union[bool, str] = True,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        binning_scope: str = "global",
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

            binning_method: the binning method to be used for fitting a piecewise linear function to the SHAP values.

                - If set to "greedy", the greedy binning method will be used.
                - If set to "fixed", the fixed binning method will be used.

            binning_scope: the x-range the binner covers when a *masked*
                summary re-bins a subregion (`eval`/`eval_heter`/`plot`/
                `heter_score` with `mask=`; the regional split search)

                - `"global"` (default): the frozen global `axis_limits` — one
                  frame for every subregion, directly comparable
                - `"effective"`: the masked column's own `[min, max]` — bins
                  packed into the subregion, finer resolution

                Recorded at fit and replayed by every masked call, so the
                split search and the display always share the same scope.
                Ignored when no mask is involved.
        """
        check_binning_scope(binning_scope)
        self._fit_loop(
            features,
            centering,
            binning_method=binning_method,
            binning_scope=binning_scope,
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
        mask: Optional[np.ndarray] = None,
        feature_label: Optional[str] = None,
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
            mask: optional boolean `(N,)` selecting a subregion — plot the
                SHAP-DP *within* it (the masked φ re-binned/re-splined from the
                cached attributions, no model calls), with the x-axis windowed
                to the subregion's own interval
            feature_label: optional display name for the feature axis (e.g. a
                regional node's name), overriding `feature_names[feature]`
        """
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        centering = helpers.prep_centering(centering)
        scale_x = helpers.resolve_scale(
            scale_x, self.scale_x_list[feature] if self.scale_x_list else None
        )
        scale_y = helpers.resolve_scale(scale_y, self.scale_y)
        mask = self._prep_mask(mask)
        feature_names = list(self.feature_names)
        if feature_label is not None:
            feature_names[feature] = feature_label

        if mask is not None and not self._is_cat(feature):
            self._effective_limits(feature, mask)  # degeneracy guard

        # one path for global and masked alike (R14): pick the payload, read it
        params = self._summary(feature, mask)
        norm = (
            self._centering_const(feature, mask, centering)
            if centering is not False
            else 0.0
        )
        avg_output = self._avg_output(mask, scale_y) if show_avg_output else None

        if self._is_cat(feature):
            # the payload's frame: the levels observed within the (masked) data
            levels, labels = self._level_display(feature, params["levels"])
            y_levels = self._eval_payload(feature, params, levels) - norm
            title = "SHAP Dependence Plot (SHAP-DP)"
            if heterogeneity == "shap_values":
                yy = params["yy"] - norm
                return vis.plot_shap_categorical(
                    levels,
                    y_levels,
                    params["xx"],
                    yy,
                    feature,
                    title=title,
                    level_labels=labels,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    avg_output=avg_output,
                    feature_names=feature_names,
                    target_name=self.target_name,
                    nof_shap_values=nof_shap_values,
                    y_limits=y_limits,
                    show_plot=show_plot,
                    random_state=self.random_state,
                )
            variances = (
                self._eval_payload(feature, params, levels, heterogeneity=True)[1]
                if heterogeneity is not False
                else None
            )
            return vis.plot_categorical_effect(
                levels,
                y_levels,
                variances,
                feature,
                heterogeneity,
                title=title,
                level_labels=labels,
                scale_x=scale_x,
                scale_y=scale_y,
                avg_output=avg_output,
                feature_names=feature_names,
                target_name=self.target_name,
                y_limits=y_limits,
                show_plot=show_plot,
            )

        # continuous: the x-axis spans the (effective) interval; the cloud is
        # the (masked) φ, already filtered by _summarize — model-free
        lo, hi = self._effective_limits(feature, mask)
        x = np.linspace(lo, hi, nof_points)
        y = self._eval_payload(feature, params, x) - norm
        y_std = (
            np.sqrt(np.maximum(params["spline_var"](x), 0.0))
            if heterogeneity == "std"
            else None
        )
        _, ind = helpers.prep_nof_instances(
            nof_shap_values, len(params["yy"]), self.random_state
        )
        yy = params["yy"][ind] - norm if heterogeneity == "shap_values" else None
        xx = params["xx"][ind] if heterogeneity == "shap_values" else None

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
            feature_names=feature_names,
            target_name=self.target_name,
            y_limits=y_limits,
            only_shap_values=only_shap_values,
            show_plot=show_plot,
        )

        return ret

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
    r"""SHAP Dependence Plot: per-instance SHAP values against the feature
    value, with a curve fitted through them.

    ```python
    sdp = effector.ShapDP(X, model, nof_instances=500)
    sdp.plot("hr")   # SHAP scatter + fitted curve
    ```

    The curve $\hat{f}^{SDP}_j(x_j)$ is fit to the SHAP cloud
    $\{(x_j^i, \hat{\phi}_j(x_j^i))\}_{i=1}^N$: the axis is binned and the
    per-bin SHAP means are interpolated piecewise-linearly. The
    heterogeneity is the per-bin variance of the SHAP values.

    !!! warning "The slow one"
        SHAP values are expensive — cost grows with instances and features.
        Keep `nof_instances` modest (default `1_000`) and raise `budget`
        only if the estimate looks noisy. Requires the `shap` or `shapiq`
        package: `pip install effector[shap]`.
    """

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
        r"""Build a ShapDP explainer. SHAP values are computed lazily, on the
        first query.

        ??? note "Definition"

            The value of a coalition $S$ of features is estimated as:
            $$
            \hat{v}(S) = {1 \over N} \sum_{i=1}^N
            [f(\mathbf{x}_S \cup \mathbf{x}_C^i) - f(\mathbf{x}^i)]
            $$
            i.e. the average change in the output when the features in $S$
            are set to $\mathbf{x}_S$ and the rest are left as observed.

            The contribution of feature $j$ added to a coalition $S$ is:
            $$
            \hat{\Delta}_{S, j} = \hat{v}(S \cup \{j\}) - \hat{v}(S)
            $$

            The SHAP value of feature $j$ at value $x_j$ averages this
            contribution over all coalitions, weighted so that every
            coalition size counts equally:
            $$
            \hat{\phi}_j(x_j) = \sum_{S \subseteq \{1, \dots, D\}
            \setminus \{j\}} w_{S, j} \hat{\Delta}_{S, j}
            $$

            The SHAP-DP curve $\hat{f}^{SDP}_j(x_j)$ is fit to
            $\{(x_j^i, \hat{\phi}_j(x_j^i))\}_{i=1}^N$: the axis is split
            into bins and the per-bin SHAP means are interpolated
            piecewise-linearly (linear extrapolation beyond the outer bin
            centers). See the
            [original paper](https://arxiv.org/abs/1705.07874).

        Args:
            data: the design matrix, shape `(N, D)` — numpy only.
            model: the black-box model — a `Callable` mapping `(N, D)`
                arrays to `(N,)` predictions.
            axis_limits: per-feature plot limits, shape `(2, D)`; `None`
                (default) infers them from `data`.
            nof_instances: max instances used for SHAP estimation (default
                `1_000` — deliberately lower than other methods, SHAP is
                expensive); an `int` subsamples randomly, `"all"` keeps
                everything.
            schema: input metadata — an `effector.Schema` or a plain `dict`
                with any of `feature_names`, `feature_types`, `cat_limit`,
                `target_name`, `scale_x_list`, `scale_y`; omitted fields are
                inferred from `data`, explicit ones win. Coming from a
                DataFrame? Use `effector.from_dataframe`.
            random_state: seed for every internal random step, including the
                shap/shapiq explainer (default `21`, reproducible); `None`
                for non-deterministic behavior.
            shap_values: precomputed SHAP values, shape `(N, D)`; if given,
                the backend is never called.
            backend: `"shap"` (default) or `"shapiq"` — the package that
                computes the SHAP values.
            budget: max model evaluations per instance for the SHAP
                approximation (default `512`); higher = more accurate,
                slower.
            shap_explainer_kwargs: extra kwargs for `shap.Explainer` /
                `shapiq.Explainer` (they override the defaults, including
                the seed). See
                `effector.global_effect_shap._compute_shap_values` — the
                single place the explainer is constructed and invoked.
            shap_explanation_kwargs: extra kwargs for the explanation call
                of the chosen backend (same code path as above).
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

    def _compute_local_cont(self, feature: int, frame: tuple) -> dict:
        """Cache-(a) kernel: the SHAP values are the local effect — computed
        once per object by the backend (or injected via `shap_values=`),
        feature-independent (hence the class-level cat alias below). Fill the
        whole `(N,D)` table on first use; each feature's entry is a column
        view of it (no frame — instance-anchored)."""
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

    _compute_local_cat = _compute_local_cont

    def _importance(self, feature, mask):
        """R13 for SHAP: the canonical `mean(|phi_s|)` over the (masked)
        instances — `phi_s` is already the cached local effect."""
        phi = self._local[feature]["phi"][mask]
        return float(np.mean(np.abs(phi)))

    def _masked_phi(self, feature: int, mask):
        """(positions, φ) of the SHAP cloud restricted to `mask` (None = all)."""
        yy = self._local[feature]["phi"]
        xx = self.data[:, feature]
        if mask is not None:
            yy = yy[mask]
            xx = xx[mask]
        return xx, yy

    def _summarize_cont(
        self,
        feature: int,
        mask=None,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        binning_scope: str = "global",
    ) -> typing.Dict:
        """Summary kernel (pure numpy): bin the cached SHAP column over the
        subregion `mask` (None = all) — per-bin mean/variance of φ, read back
        by piecewise-linear interpolation between the bin centers.

        `binning_scope` (masked only): `"global"` bins over the frozen global
        frame, `"effective"` packs the bins into the masked column's own
        `[min, max]` (see `fit`)."""
        xx, yy = self._masked_phi(feature, mask)
        return self._bin_local_effects(
            feature, xx, yy, mask, binning_method, binning_scope
        )

    def _summarize_cat(
        self,
        feature: int,
        mask=None,
        binning_method: Union[
            str, ap.DynamicProgramming, ap.Agglomerative, ap.Quantile, ap.Fixed
        ] = "dp",
        binning_scope: str = "global",
    ) -> typing.Dict:
        # per-level mean/variance of the shap values with a step lookup —
        # no interpolation, no order enters the math (method_semantics.md)
        xx, yy = self._masked_phi(feature, mask)
        levels = np.unique(xx)
        codes = utils.codes_from_levels(
            xx, levels, feature, self.feature_names[feature]
        )
        limits = np.arange(len(levels) + 1, dtype=float) - 0.5
        feature_effect_dict = utils.compute_ale_params(codes.astype(float), yy, limits)
        return {
            "bin_effect": feature_effect_dict["bin_effect"],
            "bin_variance": feature_effect_dict["bin_variance"],
            "levels": levels,
        }

    def _eval_payload_cont(
        self, feature: int, params: dict, x: np.ndarray, heterogeneity: bool = False
    ):
        # piecewise-linear between the bin centers, linear extrapolation
        # beyond the outer ones (a single bin reads as a constant)
        centers = (params["limits"][:-1] + params["limits"][1:]) / 2
        y = utils.interp_linear_extrap(x, centers, params["bin_effect"])
        if heterogeneity:
            # variance is non-negative by definition; the linear extrapolation
            # beyond the outer bin centers can dip below 0, so clamp it
            var = np.maximum(
                utils.interp_linear_extrap(x, centers, params["bin_variance"]), 0.0
            )
            return y, var
        return y

    def _eval_payload_cat(
        self, feature: int, params: dict, x: np.ndarray, heterogeneity: bool = False
    ):
        codes = utils.codes_from_levels(
            x, params["levels"], feature, self.feature_names[feature]
        )
        y = params["bin_effect"][codes]
        if heterogeneity:
            return y, params["bin_variance"][codes]
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
        r"""Declare per-feature defaults and warm the caches.

        ```python
        sdp.fit("hr", binning_method="dp")
        ```

        !!! note "fit is optional — but it pays the SHAP bill"
            Any query computes what it needs lazily with these defaults; the
            first one triggers the (expensive) SHAP computation for the whole
            `(N, D)` table. `fit` lets you pay that cost upfront.

        The curve is the piecewise-linear interpolation of the per-bin SHAP
        means (linear extrapolation beyond the outer bin centers); the
        heterogeneity is the same interpolation of the per-bin SHAP
        *variances*, clamped at zero.

        Args:
            features: feature(s) to fit — index, name, list, or `"all"`.
            centering: default centering for this feature's queries —
                `False` (none), `True`/`"zero_integral"` (center around the
                y axis), or `"zero_start"` (start at `y=0`).
            binning_method: how the axis is split before fitting the curve:

                - `"dp"` (default): dynamic programming — optimal
                  variable-size bins
                - `"agglomerative"`: bottom-up merging of small bins
                  (`"greedy"` is a deprecated alias)
                - `"quantile"`: equal-frequency bins
                - `"fixed"`: equal-width bins

                For custom parameters pass an instance from
                `effector.axis_partitioning`, e.g.
                `DynamicProgramming(max_nof_bins=30)`.

            binning_scope: the x-range the binner covers when a *masked*
                summary re-bins a subregion (`eval`/`eval_heter`/`plot`/
                `heter_score` with `mask=`; the regional split search):

                - `"global"` (default): the frozen global `axis_limits` —
                  one frame for every subregion, directly comparable
                - `"effective"`: the masked column's own `[min, max]` —
                  bins packed into the subregion, finer resolution

                Recorded at fit and replayed by every masked call. Ignored
                when no mask is involved.
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
        feature: Union[int, str],
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
        rule=None,
        feature_label: Optional[str] = None,
    ) -> Union[Tuple, None]:
        """Plot the SHAP-DP of `feature`, by default with the SHAP scatter.

        ```python
        sdp.plot("hr")                       # fitted curve + SHAP scatter
        sdp.plot("hr", heterogeneity="std")  # curve ± std band
        ```

        Args:
            feature: index or name of the feature to plot.
            heterogeneity: what to draw around the fitted curve:

                - `False`: the curve only
                - `True` or `"std"`: ± one std of the SHAP values per bin
                - `"shap_values"` (default): the SHAP values scattered on
                  top of the curve

            centering: `False` (none), `True`/`"zero_integral"` (center
                around the y axis), or `"zero_start"` (start at `y=0`).
            nof_points: grid size of the x axis (default `100`).
            scale_x: `None` or `{"mean": m, "std": s}` to undo a
                standardization of the x axis for display.
            scale_y: same, for the y axis.
            nof_shap_values: how many SHAP values to scatter (default
                `100`), or `"all"`.
            show_avg_output: draw the model's average output as a
                horizontal line.
            y_limits: `(low, high)` for the y axis; `None` = automatic.
            only_shap_values: scatter the SHAP values without the fitted
                curve.
            show_plot: if `False`, return the figure and axes instead of
                showing.
            mask: boolean `(N,)` selecting a subregion — plot the SHAP-DP
                *within* it (the masked SHAP values re-binned from the
                cached attributions, no model calls), x axis windowed to the
                subregion's own interval.
            rule: sugar over `mask` — an `effector.Rule` or a rule string,
                applied to the effect's data. Mutually exclusive with `mask`.
            feature_label: display name for the feature axis (e.g. a
                regional node's name), overriding `feature_names[feature]`.
        """
        feature = self._resolve_feature(feature)
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        centering = helpers.prep_centering(centering)
        scale_x = helpers.resolve_scale(
            scale_x, self.scale_x_list[feature] if self.scale_x_list else None
        )
        scale_y = helpers.resolve_scale(scale_y, self.scale_y)
        mask = self._resolve_mask(mask, rule)
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
                # the scatter cloud comes from cache (a) — the same (masked)
                # φ the payload was summarized from, by construction
                xx, phi = self._masked_phi(feature, mask)
                yy = phi - norm
                return vis.plot_shap_categorical(
                    levels,
                    y_levels,
                    xx,
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
        # the (masked) φ from cache (a) — the same instances the payload was
        # summarized from, model-free
        lo, hi = self._effective_limits(feature, mask)
        x = np.linspace(lo, hi, nof_points)
        y = self._eval_payload(feature, params, x) - norm
        y_std = (
            np.sqrt(self._eval_payload(feature, params, x, heterogeneity=True)[1])
            if heterogeneity == "std"
            else None
        )
        col, phi = self._masked_phi(feature, mask)
        _, ind = helpers.prep_nof_instances(
            nof_shap_values, len(phi), self.random_state
        )
        yy = phi[ind] - norm if heterogeneity == "shap_values" else None
        xx = col[ind] if heterogeneity == "shap_values" else None

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

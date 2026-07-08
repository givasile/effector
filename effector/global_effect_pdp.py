import copy
import typing
from typing import Callable, List, Optional, Union

import numpy as np

import effector.helpers as helpers
import effector.utils as utils
import effector.visualization as vis
from effector import ingestion
from effector.global_effect import GlobalEffectBase


class PDPBase(GlobalEffectBase):
    DEFAULT_CENTERING: Union[bool, str] = False
    # the vis layer scales derivative plots by std only (no mean shift) — B5
    IS_DERIVATIVE: bool = False

    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        *,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
        method_name: str = "PDP",
    ):
        super(PDPBase, self).__init__(
            method_name,
            data,
            model,
            model_jac,
            nof_instances=nof_instances,
            axis_limits=axis_limits,
            schema=schema,
            random_state=random_state,
        )

    def _predict(self, data, xx, feature, use_vectorized=True):
        method = ice_vectorized if use_vectorized else ice_non_vectorized
        if self.IS_DERIVATIVE:
            y = method(self.model, self.model_jac, data, xx, feature, True)
        else:
            y = method(self.model, None, data, xx, feature, False)
        return y

    def _use_vectorized(self, feature: int) -> bool:
        return self._config(feature).get("use_vectorized", True)

    def _canonical_grid(self, feature: int) -> np.ndarray:
        """The frozen summary frame: the observed levels for a discrete
        feature, else the uniform `NOF_INTERNAL_POINTS` grid on the global
        interval. Every (b)-stage quantity (payload, heterogeneity, centering
        constants) is a function of these rows only, so position-store growth
        never changes a summary."""
        if self._is_cat(feature):
            return self._levels(feature)
        return np.linspace(
            self.axis_limits[0, feature],
            self.axis_limits[1, feature],
            helpers.NOF_INTERNAL_POINTS,
        )

    # ------------------------------------------------------------------
    # cache (a): the growing position store
    # ------------------------------------------------------------------
    def _compute_local_cont(self, feature: int, frame: tuple) -> dict:
        """Cache-(a) kernel: the (d-)ICE table on the canonical grid — one row
        per position, one column per instance `(T, N)`. The store then grows
        as `eval` asks for never-before-seen positions (missing-only).
        Type-agnostic (the canonical grid is the levels for a discrete axis),
        hence the class-level cat alias below."""
        grid = self._canonical_grid(feature)
        ice = self._predict(self.data, grid, feature, self._use_vectorized(feature))
        return {"frame": frame, "pos": grid, "ice": ice}

    _compute_local_cat = _compute_local_cont

    def _ensure_positions(self, feature: int, xs: np.ndarray) -> dict:
        """Grow the position store to cover `xs`: compute ICE columns for the
        missing positions only (the one model touch of the global eval path)
        and append them. Growth is additive — no epoch bump, no summary is
        invalidated (they read the canonical rows only)."""
        entry = self._ensure_local(feature)
        xs = np.unique(np.asarray(xs, dtype=float))
        pos = entry["pos"]
        idx = np.clip(np.searchsorted(pos, xs), 0, len(pos) - 1)
        missing = xs[pos[idx] != xs]
        if missing.size:
            new_ice = self._predict(
                self.data, missing, feature, self._use_vectorized(feature)
            )
            pos = np.concatenate([pos, missing])
            order = np.argsort(pos, kind="mergesort")
            entry["pos"] = pos[order]
            entry["ice"] = np.concatenate([entry["ice"], new_ice], axis=0)[order]
        return entry

    def _grid_rows(self, feature: int, entry: dict):
        """(grid, ice-rows) restricted to the canonical grid — the frame every
        summary is computed on, regardless of how far the store has grown."""
        grid = self._canonical_grid(feature)
        rows = np.searchsorted(entry["pos"], grid)
        return grid, entry["ice"][rows]

    # ------------------------------------------------------------------
    # the pure kernels
    # ------------------------------------------------------------------
    def _grid_ice(self, feature: int, mask):
        """The shared summary prelude: the canonical-grid rows of the cached
        (d-)ICE table, columns restricted to the subregion `mask` (None = all)."""
        grid, ice = self._grid_rows(feature, self._local[feature])
        if mask is not None:
            ice = ice[:, mask]
        return grid, ice

    def _summarize_cont(
        self, feature: int, mask=None, use_vectorized: bool = True
    ) -> dict:
        """Summary kernel (pure numpy): from the canonical-grid rows of the
        cached (d-)ICE table restricted to the subregion `mask` (None = all),
        the mean curve and the heterogeneity curve — cross-instance variance of
        the per-instance-centered ICE curves (PDP) or of the raw d-ICE curves
        (DerPDP)."""
        grid, ice = self._grid_ice(feature, mask)
        if self.IS_DERIVATIVE:
            heter = np.var(ice, axis=1)
        else:
            # each ICE curve is centered on its own mean — a per-instance
            # shift, so mask-invariant
            per_instance_norm = np.mean(ice, axis=0)
            heter = np.var(ice - per_instance_norm[np.newaxis, :], axis=1)
        return {"grid": grid, "heter": heter, "mean": np.mean(ice, axis=1)}

    def _summarize_cat(
        self, feature: int, mask=None, use_vectorized: bool = True
    ) -> dict:
        """The categorical summary: the grid is the observed levels; ICE
        centering weights each level by its *global* frequency — deliberately
        unmasked, the per-instance shift is mask-invariant."""
        grid, ice = self._grid_ice(feature, mask)
        if self.IS_DERIVATIVE:
            heter = np.var(ice, axis=1)
        else:
            _, weights = self._level_weights(feature)
            per_instance_norm = np.average(ice, axis=0, weights=weights)
            heter = np.var(ice - per_instance_norm[np.newaxis, :], axis=1)
        return {
            "grid": grid,
            "heter": heter,
            "mean": np.mean(ice, axis=1),
            "levels": grid,
        }

    def _eval_payload_cont(
        self, feature: int, params: dict, x: np.ndarray, heterogeneity: bool = False
    ):
        """Reader kernel: mean/heterogeneity off the canonical grid — exact at
        grid positions, linear interpolation between them."""
        y = np.interp(x, params["grid"], params["mean"])
        return (
            (y, np.interp(x, params["grid"], params["heter"])) if heterogeneity else y
        )

    def _eval_payload_cat(
        self, feature: int, params: dict, x: np.ndarray, heterogeneity: bool = False
    ):
        # levels are always on the canonical grid — exact lookup, no interp
        codes = utils.codes_from_levels(
            x, params["levels"], feature, self.feature_names[feature]
        )
        y = params["mean"][codes]
        return (y, params["heter"][codes]) if heterogeneity else y

    def _eval_mean(
        self, feature: int, x: np.ndarray, params: dict, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """The (d-)PDP mean effect is *exact*, not interpolated: global calls
        read (and grow) the position store; masked calls read cached columns
        when every position is present, else recompute ICE on `data[mask]`
        transiently — the one documented model-touching exception (R14)."""
        if self._is_cat(feature):
            # the base default (read the payload) is exact at levels
            return super()._eval_mean(feature, x, params, mask)
        x = np.asarray(x, dtype=float)
        if mask is None:
            entry = self._ensure_positions(feature, x)
            rows = np.searchsorted(entry["pos"], x)
            return entry["ice"][rows].mean(axis=1)
        entry = self._local[feature]
        pos = entry["pos"]
        idx = np.clip(np.searchsorted(pos, x), 0, len(pos) - 1)
        if np.all(pos[idx] == x):
            return entry["ice"][idx][:, mask].mean(axis=1)
        y_ice = self._predict(
            self.data[mask], x, feature, self._use_vectorized(feature)
        )
        return np.mean(y_ice, axis=1)

    def _compute_norm_const_cat(
        self,
        feature: int,
        method: str,
        params: dict,
        mask: Optional[np.ndarray] = None,
    ):
        """(d-)PDP overrides the base variants: its centering constant is
        *per-instance* — each ICE curve is centered on its own — so an array
        is returned instead of a scalar, derived from the canonical-grid
        columns (masked columns for a masked call), zero model calls (R14)."""
        grid, ice = self._grid_ice(feature, mask)
        # weights of the levels *within* the mask, aligned to the grid
        # (= the globally observed levels); absent levels weigh zero
        levels, weights = self._level_weights(feature, mask)
        if method == "zero_integral":
            w = np.zeros(len(grid))
            w[np.searchsorted(grid, levels)] = weights
            return np.average(ice, axis=0, weights=w)
        return ice[np.searchsorted(grid, levels[0])]

    def _compute_norm_const_cont(
        self,
        feature: int,
        method: str,
        params: dict,
        mask: Optional[np.ndarray] = None,
    ):
        grid, ice = self._grid_ice(feature, mask)
        lo, hi = self._effective_limits(feature, mask)
        if method == "zero_integral":
            xs = np.linspace(lo, hi, helpers.NOF_INTERNAL_POINTS)
            return np.mean(_interp_columns(grid, ice, xs), axis=0)
        return _interp_columns(grid, ice, np.array([lo]))[0]

    def _mean_norm_const(self, norm_const):
        # PDP's constant is per-instance (each ICE centers on its own); the
        # mean-effect shift is their average
        return np.mean(norm_const)

    def fit(
        self,
        features: Union[int, str, list] = "all",
        *,
        centering: Union[bool, str] = False,
        use_vectorized: bool = True,
    ):
        """
        Fit the Feature effect to the data.

        Notes:
            You can use `.eval` or `.plot` without calling `.fit` explicitly:
            any query computes what it needs lazily with these defaults. `fit`
            declares the config and warms the caches (R14).

        Args:
            features: the features to fit.
                - If set to "all", all the features will be fitted.

            centering: the default centering mode for this feature's queries:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.

            use_vectorized: whether to use vectorized operations for the PDP and ICE curves

        """
        self._fit_loop(features, centering, use_vectorized=use_vectorized)

    def _plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = False,
        centering: Union[bool, str] = True,
        nof_points: int = 100,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_ice: Union[int, str] = 100,
        show_avg_output: bool = False,
        y_limits: Optional[List] = None,
        use_vectorized: bool = True,
        show_plot: bool = True,
        mask: Optional[np.ndarray] = None,
        feature_label: Optional[str] = None,
    ):
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

        is_cat = self._is_cat(feature)
        if mask is None:
            # the display grid joins the position store (grown missing-only),
            # so a repeated plot costs zero model calls
            if is_cat:
                entry = self._ensure_local(feature)
                x = entry["pos"]
                yy = entry["ice"]
            else:
                x = np.linspace(
                    self.axis_limits[0, feature],
                    self.axis_limits[1, feature],
                    nof_points,
                )
                entry = self._ensure_positions(feature, x)
                rows = np.searchsorted(entry["pos"], x)
                yy = entry["ice"][rows]
        else:
            # masked branch: the canonical-grid rows' masked columns — no model
            # calls; `nof_points` does not apply. The x array is cropped to the
            # subregion's effective interval, so the figure windows itself.
            entry = self._ensure_local(feature)
            grid, ice = self._grid_rows(feature, entry)
            ice_m = ice[:, mask]
            if is_cat:
                x = grid
                yy = ice_m
            else:
                lo, hi = self._effective_limits(feature, mask)
                x = np.concatenate([[lo], grid[(grid > lo) & (grid < hi)], [hi]])
                yy = _interp_columns(grid, ice_m, x)
        if centering is not False:
            norm_consts = self._centering_const(feature, mask, centering)
            yy = yy - norm_consts[np.newaxis, :]

        avg_output = self._avg_output(mask, scale_y) if show_avg_output else None

        title = (
            "Partial Dependence Plot (PDP)"
            if self.method_name == "pdp"
            else "derivative Partial Dependence Plot (d-PDP)"
        )
        if is_cat:
            levels, labels = self._level_display(feature)
            if heterogeneity == "ice":
                return vis.plot_pdp_ice_categorical(
                    levels,
                    yy,
                    feature,
                    title=title,
                    y_pdp_label="PDP",
                    y_ice_label="ICE",
                    level_labels=labels,
                    scale_x=scale_x,
                    scale_y=scale_y,
                    avg_output=avg_output,
                    feature_names=feature_names,
                    target_name=self.target_name,
                    nof_ice=nof_ice,
                    y_limits=y_limits,
                    show_plot=show_plot,
                    random_state=self.random_state,
                )
            if heterogeneity is not False:
                params = self._summary(feature, mask)
                variances = self._eval_payload(feature, params, x, heterogeneity=True)[
                    1
                ]
            else:
                variances = None
            return vis.plot_categorical_effect(
                levels,
                yy.mean(axis=1),
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
        # R1: the method owns the compute. Derive the mean line and the
        # std/std_err band here (from the ICE table we already have — cached
        # columns, no re-eval); the plot layer only draws. The raw table is
        # handed over only for the "ice" cloud, which genuinely needs every curve.
        y_mean = yy.mean(axis=1)
        band = None
        if heterogeneity == "std":
            band = np.std(yy, axis=1)
        elif heterogeneity == "std_err":
            band = np.std(yy, axis=1) / np.sqrt(yy.shape[1])
        return vis.plot_pdp_ice(
            x,
            feature,
            y_mean=y_mean,
            band=band,
            ice=yy if heterogeneity == "ice" else None,
            title=title,
            heterogeneity=heterogeneity,
            y_pdp_label="PDP" if self.method_name == "pdp" else "d-PDP",
            y_ice_label="ICE" if self.method_name == "pdp" else "d-ICE",
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=feature_names,
            target_name=self.target_name,
            is_derivative=self.IS_DERIVATIVE,
            nof_ice=nof_ice,
            y_limits=y_limits,
            show_plot=show_plot,
            random_state=self.random_state,
        )


class PDP(PDPBase):
    # zero_integral by default (R3 single source): matches the global .plot
    # signature default and ALE/ShapDP, so global and regional plots — and
    # eval(centering=None) — all center consistently.
    DEFAULT_CENTERING: Union[bool, str] = "zero_integral"
    CAT_STRATEGY = "ice_at_levels"

    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        *,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
    ):
        r"""
        Constructor of the PDP class.

        Definition:
            PDP:
            $$
            PDP(x_s) = {1 \over N} \sum_{i=1}^N f(x_s, \mathbf{x}_c^i)
            $$

            centered-PDP:
            $$
            PDP_c(x_s) = PDP(x_s) - c, \quad c = {1 \over M} \sum_{j=1}^M PDP(x_s^j)
            $$

            ICE:
            $$
            ICE^i(x_s) = f(x_s, \mathbf{x}_c^i), \quad i=1, \dots, N
            $$

            centered-ICE:
            $$
            ICE_c^i(x_s) = ICE^i(x_s) - c_i, \quad c_i = {1 \over M} \sum_{j=1}^M ICE^i(x_s^j)
            $$

            heterogeneity function:
            $$
            h(x_s) = {1 \over N} \sum_{i=1}^N ( ICE_c^i(x_s) - PDP_c(x_s) )^2
            $$

            The heterogeneity value is:
            $$
            \mathcal{H}(x_s) = {1 \over M} \sum_{j=1}^M h(x_s^j),
            $$
            where $x_s^j$ are an equally spaced grid of points in $[x_s^{\min}, x_s^{\max}]$.

        Notes:
            The required parameters are `data` and `model`. The rest are optional.

        Args:
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N,)`

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            nof_instances: maximum number of instances to be used

                - use "all", for using all instances.
                - use an `int`, for selecting `nof_instances` instances randomly.

            schema: input metadata (R10) — an `effector.Schema` or a plain `dict`
                with any of the keys `feature_names`, `feature_types`,
                `cat_limit`, `target_name`, `scale_x_list`, `scale_y`

                - omitted fields are auto-inferred from `data` (numpy
                  heuristics) or synthesized (`["x_0", ...]`, `"y"`); to start
                  from a DataFrame use `effector.from_dataframe`
                - explicit fields always win over inference

            random_state: seed for every internal random step (e.g. `nof_instances` subsampling)

                - use an `int` (default: `21`), for reproducible output; two identical constructions give identical results
                - use `None`, for non-deterministic behavior
        """

        super(PDP, self).__init__(
            data,
            model,
            None,
            axis_limits=axis_limits,
            nof_instances=nof_instances,
            schema=schema,
            random_state=random_state,
            method_name="PDP",
        )

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = "ice",
        centering: Union[bool, str] = True,
        nof_points: int = 100,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_ice: Union[int, str] = 100,
        show_avg_output: bool = False,
        y_limits: Optional[List] = None,
        use_vectorized: bool = True,
        show_plot: bool = True,
        mask: Optional[np.ndarray] = None,
        rule=None,
        feature_label: Optional[str] = None,
    ):
        """
        Plot the feature effect.

        Parameters:
            feature: the feature to plot
            heterogeneity: whether to plot the heterogeneity

                  - `False`, plot only the mean effect
                  - `True` or `std`, plot the standard deviation of the ICE curves
                  - `ice`, also plot the ICE curves

            centering: whether to center the plot

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.

            nof_points: the grid size for the PDP plot

            scale_x: None or Dict with keys ['std', 'mean']

                - If set to None, no scaling will be applied.
                - If set to a dict, the x-axis will be scaled `x = (x + mean) * std`

            scale_y: None or Dict with keys ['std', 'mean']

                - If set to None, no scaling will be applied.
                - If set to a dict, the y-axis will be scaled `y = (y + mean) * std`

            nof_ice: number of ICE plots to show on top of the SHAP curve
            show_avg_output: whether to show the average output of the model

            y_limits: None or tuple, the limits of the y-axis

                - If set to None, the limits of the y-axis are set automatically
                - If set to a tuple, the limits are manually set

            use_vectorized: whether to use the vectorized version of the PDP computation
            mask: optional boolean `(N,)` selecting a subregion — plot the PDP/
                ICE *within* it from the cached ICE table (grid resolution;
                `nof_points` does not apply), model-free, with the x-axis
                windowed to the subregion's own interval
            rule: sugar over `mask` — an `effector.Rule` or a rule string,
                applied to the effect's data. Mutually exclusive with `mask`.
            feature_label: optional display name for the feature axis (e.g. a
                regional node's name), overriding `feature_names[feature]`
        """
        mask = self._resolve_mask(mask, rule)
        ret = self._plot(
            feature,
            heterogeneity,
            centering,
            nof_points,
            scale_x,
            scale_y,
            nof_ice,
            show_avg_output,
            y_limits,
            use_vectorized,
            show_plot,
            mask,
            feature_label,
        )

        if not show_plot:
            return ret


class DerPDP(PDPBase):
    SUPPORTED_FEATURE_TYPES = frozenset({ingestion.CONTINUOUS})
    DEFAULT_CENTERING: Union[bool, str] = False
    IS_DERIVATIVE: bool = True

    def _importance(self, feature, mask):
        """R13 for d-PDP: the mean effect is already the derivative, whose
        *dispersion* is ~0 for a locally-linear model — a poor importance. Use
        the mean **magnitude** of the derivative over the grid instead (for a
        linear model this recovers `|coefficient|`)."""
        xs = np.linspace(
            self.axis_limits[0, feature],
            self.axis_limits[1, feature],
            helpers.NOF_INTERNAL_POINTS,
        )
        mu = self.eval(feature, xs, centering=False, mask=mask)
        return float(np.mean(np.abs(mu)))

    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        *,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
    ):
        r"""
        Constructor of the DerivativePDP class.

        Definition:
            d-PDP:
            $$
            dPDP(x_s) = {1 \over N} \sum_{i=1}^N {\partial f \over \partial x_s}(x_s, \mathbf{x}_c^i)
            $$

            centered-PDP:
            $$
            dPDP_c(x_s) = dPDP(x_s) - c, \quad c = {1 \over M} \sum_{j=1}^M dPDP(x_s^j)
            $$

            ICE:
            $$
            dICE^i(x_s) = {\partial f \over \partial x_s}(x_s, \mathbf{x}_c^i), \quad i=1, \dots, N
            $$

            centered-ICE:
            $$
            dICE_c^i(x_s) = dICE^i(x_s) - c_i, \quad c_i = {1 \over M} \sum_{j=1}^M dICE^i(x_s^j)
            $$

            heterogeneity function:
            $$
            h(x_s) = {1 \over N} \sum_{i=1}^N ( dICE_c^i(x_s) - dPDP_c(x_s) )^2
            $$

            The heterogeneity value is:
            $$
            \mathcal{H}(x_s) = {1 \over M} \sum_{j=1}^M h(x_s^j),
            $$
            where $x_s^j$ are an equally spaced grid of points in $[x_s^{\min}, x_s^{\max}]$.

        Notes:
            - The required parameters are `data` and `model`. The rest are optional.
            - The `model_jac` is the Jacobian of the model. If `None`, the Jacobian will be computed numerically.

        Args:
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N, )`

            model_jac: the black-box model Jacobian. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N, D)`

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            nof_instances: maximum number of instances to be used for PDP.

                - use "all", for using all instances.
                - use an `int`, for using `nof_instances` instances.

            schema: input metadata (R10) — an `effector.Schema` or a plain `dict`
                with any of the keys `feature_names`, `feature_types`,
                `cat_limit`, `target_name`, `scale_x_list`, `scale_y`

                - omitted fields are auto-inferred from `data` (numpy
                  heuristics) or synthesized (`["x_0", ...]`, `"y"`); to start
                  from a DataFrame use `effector.from_dataframe`
                - explicit fields always win over inference

            random_state: seed for every internal random step (e.g. `nof_instances` subsampling)

                - use an `int` (default: `21`), for reproducible output; two identical constructions give identical results
                - use `None`, for non-deterministic behavior
        """

        super(DerPDP, self).__init__(
            data,
            model,
            model_jac,
            axis_limits=axis_limits,
            nof_instances=nof_instances,
            schema=schema,
            random_state=random_state,
            method_name="d-PDP",
        )

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = "ice",
        centering: Union[bool, str] = False,
        nof_points: int = 100,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_ice: Union[int, str] = 100,
        show_avg_output: bool = False,
        y_limits: Optional[List] = None,
        use_vectorized: bool = True,
        show_plot: bool = True,
        mask: Optional[np.ndarray] = None,
        rule=None,
        feature_label: Optional[str] = None,
    ):
        """
        Plot the feature effect.

        Parameters:
            feature: the feature to plot
            heterogeneity: whether to plot the heterogeneity

                  - `False`, plot only the mean effect
                  - `True` or `std`, plot the standard deviation of the ICE curves
                  - `ice`, also plot the ICE curves

            centering: whether to center the plot

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.

            nof_points: the grid size for the PDP plot

            scale_x: None or Dict with keys ['std', 'mean']

                - If set to None, no scaling will be applied.
                - If set to a dict, the x-axis will be scaled `x = (x + mean) * std`

            scale_y: None or Dict with keys ['std', 'mean']

                - If set to None, no scaling will be applied.
                - If set to a dict, the y-axis will be scaled `y = (y + mean) * std`

            nof_ice: number of ICE plots to show on top of the SHAP curve
            show_avg_output: whether to show the average output of the model

            y_limits: None or tuple, the limits of the y-axis (derivative units)

                - If set to None, the limits of the y-axis are set automatically
                - If set to a tuple, the limits are manually set

            use_vectorized: whether to use the vectorized version of the PDP computation
            show_plot: whether to show the plot
            mask: optional boolean `(N,)` selecting a subregion — plot the
                d-PDP/d-ICE *within* it from the cached d-ICE table (grid
                resolution; `nof_points` does not apply), model-free, with the
                x-axis windowed to the subregion's own interval
            rule: sugar over `mask` — an `effector.Rule` or a rule string,
                applied to the effect's data. Mutually exclusive with `mask`.
            feature_label: optional display name for the feature axis (e.g. a
                regional node's name), overriding `feature_names[feature]`
        """
        mask = self._resolve_mask(mask, rule)
        ret = self._plot(
            feature,
            heterogeneity,
            centering,
            nof_points,
            scale_x,
            scale_y,
            nof_ice,
            show_avg_output,
            y_limits,
            use_vectorized,
            show_plot,
            mask,
            feature_label,
        )

        if not show_plot:
            fig, ax = ret
            return fig, ax


def _interp_columns(grid: np.ndarray, table: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Linear interpolation of each column of `table` `(T, N)` at positions
    `xs` `(M,)` → `(M, N)`. Outside `grid` the edge rows extend flat (the
    masked path's flat-edge-extension convention)."""
    xs = np.asarray(xs, dtype=float)
    idx = np.clip(np.searchsorted(grid, xs, side="right") - 1, 0, len(grid) - 2)
    x0, x1 = grid[idx], grid[idx + 1]
    w = np.where(x1 > x0, (xs - x0) / (x1 - x0), 0.0)
    w = np.clip(w, 0.0, 1.0)
    return table[idx] * (1 - w)[:, None] + table[idx + 1] * w[:, None]


def ice_non_vectorized(
    model: callable,
    model_jac: Optional[callable],
    data: np.ndarray,
    x: np.ndarray,
    feature: int,
    return_d_ice: bool = False,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Computes the unnormalized 1-dimensional PDP, in a non-vectorized way.

    Notes:
        The non-vectorized version is slower than the vectorized one, but it requires less memory.

    Args:
        model: The black-box function (N, D) -> (N) or the Jacobian wrt the input (N, D) -> (N, D)
        model_jac: The black-box function Jacobian (N, D) -> (N, D) or None
        data: The design matrix, (N, D)
        x: positions to evaluate pdp, (T)
        feature: index of the feature of interest
        return_d_ice (bool): whether to return the derivatives wrt the input

    Returns:
        y: Array of shape (T, N) with the PDP values that correspond to `x` for each instance in the dataset

    """
    nof_instances = x.shape[0]

    y_list = []
    if return_d_ice:
        if model_jac is None:
            for k in range(nof_instances):
                x_new = copy.deepcopy(data)
                x_new[:, feature] = x[k] + 1e-6
                y_1 = model(x_new)
                x_new[:, feature] = x[k] - 1e-6
                y_2 = model(x_new)
                y = (y_1 - y_2) / (2 * 1e-6)
                y_list.append(y)
            y = np.array(y_list)
        else:
            for k in range(nof_instances):
                x_new = copy.deepcopy(data)
                x_new[:, feature] = x[k]
                y = model_jac(x_new)[:, feature]
                y_list.append(y)
            y = np.array(y_list)
    else:
        for k in range(nof_instances):
            x_new = copy.deepcopy(data)
            x_new[:, feature] = x[k]
            y = model(x_new)
            y_list.append(y)
        y = np.array(y_list)

    return y


def ice_vectorized(
    model: callable,
    model_jac: Optional[callable],
    data: np.ndarray,
    x: np.ndarray,
    feature: int,
    return_d_ice: bool = False,
) -> np.ndarray:
    """Compute ICE plots (array of shape (T, N)) for each instance in the dataset, in positions `x`.

    Notes:
        The vectorized version is faster than the non-vectorized one, but it requires more memory.
        Be careful when using it with large datasets, since it creates an internal dataset of shape (T, N, D)
        where T is the number of positions to evaluate the PDP, N is the number of instances in the dataset
        and D is the number of features.

    Examples:
        >>> # check the gradient of the PDP of a linear model
        >>> import numpy as np
        >>> model = lambda x: np.sum(x, axis=1)
        >>> data = np.random.rand(100, 10)
        >>> x = np.linspace(0.1, 1, 10)
        >>> feature = 0
        >>> y = ice_vectorized(model, None, data, x, feature, return_d_ice=False)
        >>> (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    Args:
        model: The black-box function (N, D) -> (N)
        model_jac: The black-box function Jacobian (N, D) -> (N, D) or None
        data: The design matrix, (N, D)
        x: positions to evaluate pdp, (T)
        feature: index of the feature of interest
        return_d_ice (bool): whether to ask the model to return the derivatives wrt the input

    Returns:
        y: Array of shape (T, N) with the PDP values that correspond to `x` for each instance in the dataset

    """

    nof_instances = data.shape[0]
    x_new = copy.deepcopy(data)
    x_new = np.expand_dims(x_new, axis=0)
    x_new = np.repeat(x_new, x.shape[0], axis=0)

    if return_d_ice:
        if model_jac is None:
            x_new_1 = copy.deepcopy(x_new)
            x_new_1[:, :, feature] = np.expand_dims(x, axis=-1) + 1e-6
            x_new_1 = np.reshape(
                x_new_1, (x_new_1.shape[0] * x_new_1.shape[1], x_new_1.shape[2])
            )

            x_new_2 = copy.deepcopy(x_new)
            x_new_2[:, :, feature] = np.expand_dims(x, axis=-1) - 1e-6
            x_new_2 = np.reshape(
                x_new_2, (x_new_2.shape[0] * x_new_2.shape[1], x_new_2.shape[2])
            )

            y_1 = model(x_new_1)
            y_2 = model(x_new_2)
            y = (y_1 - y_2) / (2 * 1e-6)
            y = np.reshape(y, (x.shape[0], nof_instances))
        else:
            x_new[:, :, feature] = np.expand_dims(x, axis=-1)
            x_new = np.reshape(x_new, (x_new.shape[0] * x_new.shape[1], x_new.shape[2]))
            y = model_jac(x_new)[:, feature]
            y = np.reshape(y, (x.shape[0], nof_instances))
    else:
        x_new[:, :, feature] = np.expand_dims(x, axis=-1)
        x_new = np.reshape(x_new, (x_new.shape[0] * x_new.shape[1], x_new.shape[2]))
        y = model(x_new)
        y = np.reshape(y, (x.shape[0], nof_instances))
    return y

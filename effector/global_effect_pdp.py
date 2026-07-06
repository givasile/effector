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
        if self.method_name == "pdp":
            y = method(self.model, None, data, xx, feature, False)
        else:
            y = method(self.model, self.model_jac, self.data, xx, feature, True)
        return y

    def _fit_feature(self, feature: int, use_vectorized: bool = True) -> dict:
        # the (d-)PDP stores no per-feature payload beyond the normalization
        # constant (which the base fit loop appends); ICE curves are computed
        # by the evaluation kernel
        if self._is_cat(feature):
            return {"levels": self._levels(feature), "is_cat": True}
        return {}

    def _compute_norm_const(
        self, feature: int, method: str = "zero_integral", nof_points: int = 30
    ):
        """(d-)PDP overrides the base: its normalization constant is
        *per-instance* — each ICE curve is centered on its own — so an
        `(N,)` array is stored instead of a scalar."""
        assert method in ["zero_integral", "zero_start"]
        use_vectorized = self.fit_args.get("feature_" + str(feature), {}).get(
            "use_vectorized", True
        )
        if self._is_cat(feature):
            levels, weights = self._level_weights(feature)
            if method == "zero_integral":
                y = self._predict(self.data, levels, feature, use_vectorized)
                return np.average(y, axis=0, weights=weights)
            y = self._predict(self.data, levels[:1], feature, use_vectorized)
            return y[0]
        if method == "zero_integral":
            xx = np.linspace(
                self.axis_limits[0, feature],
                self.axis_limits[1, feature],
                nof_points,
            )
            y = self._predict(self.data, xx, feature, use_vectorized)
            return np.mean(y, axis=0)
        xx = self.axis_limits[0, feature, np.newaxis]
        y = self._predict(self.data, xx, feature, use_vectorized)
        return y[0]

    def _eval_unnorm(self, feature: int, x: np.ndarray, heterogeneity: bool = False):
        """Kernel: uncentered mean (d-)ICE at `x`; with `heterogeneity`, also
        h(x) — the variance across the *per-instance centered* ICE curves for
        the PDP (levels are only comparable after centering) and across the raw
        d-ICE curves for the DerPDP (slopes are directly comparable)."""
        if self._is_cat(feature):
            # discrete features are evaluated only at levels (R10)
            utils.codes_from_levels(
                x, self._levels(feature), feature, self.feature_names[feature]
            )
        y_ice = self._predict(self.data, x, feature, use_vectorized=True)
        y_mean = np.mean(y_ice, axis=1)
        if not heterogeneity:
            return y_mean

        if self.method_name == "pdp":
            if self._is_cat(feature):
                levels, weights = self._level_weights(feature)
                per_instance_norm = np.average(
                    self._predict(self.data, levels, feature, use_vectorized=True),
                    axis=0,
                    weights=weights,
                )
            else:
                xx = np.linspace(
                    self.axis_limits[0, feature], self.axis_limits[1, feature], 30
                )
                per_instance_norm = np.mean(
                    self._predict(self.data, xx, feature, use_vectorized=True), axis=0
                )
            y_var = np.var(y_ice - per_instance_norm[np.newaxis, :], axis=1)
        else:
            y_var = np.var(y_ice, axis=1)
        return y_mean, y_var

    def fit(
        self,
        features: Union[int, str, list] = "all",
        *,
        centering: Union[bool, str] = False,
        points_for_centering: int = 30,
        use_vectorized: bool = True,
    ):
        """
        Fit the Feature effect to the data.

        Notes:
            You can use `.eval` or `.plot` without calling `.fit` explicitly.
            The only thing that `.fit` does is to compute the normalization constant for centering the PDP and ICE plots.
            This will be automatically done when calling `eval` or `plot`, so there is no need to call `fit` explicitly.

        Args:
            features: the features to fit.
                - If set to "all", all the features will be fitted.

            centering: whether to center the plot:

                - `False` means no centering
                - `True` or `zero_integral` centers around the `y` axis.
                - `zero_start` starts the plot from `y=0`.

            points_for_centering: number of linspaced points along the feature axis used for centering.
            use_vectorized: whether to use vectorized operations for the PDP and ICE curves

        """
        self._fit_loop(
            features, centering, points_for_centering, use_vectorized=use_vectorized
        )

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
    ):
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        centering = helpers.prep_centering(centering)
        scale_x = helpers.resolve_scale(
            scale_x, self.scale_x_list[feature] if self.scale_x_list else None
        )
        scale_y = helpers.resolve_scale(scale_y, self.scale_y)

        is_cat = self._is_cat(feature)
        x = (
            self._levels(feature)
            if is_cat
            else np.linspace(
                self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
            )
        )

        # the ICE table is the method's own object: computed by the kernel and
        # centered with the stored per-instance norms (payload state)
        if self.requires_refit(feature, centering):
            self._refit(feature, centering)
        yy = self._predict(self.data, x, feature, use_vectorized)
        if centering is not False:
            norm_consts = self.feature_effect["feature_" + str(feature)]["norm_const"]
            yy = yy - norm_consts[np.newaxis, :]

        if show_avg_output:
            avg_output = helpers.prep_avg_output(self.data, self.model, None, scale_y)
        else:
            avg_output = None

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
                    feature_names=self.feature_names,
                    target_name=self.target_name,
                    nof_ice=nof_ice,
                    y_limits=y_limits,
                    show_plot=show_plot,
                    random_state=self.random_state,
                )
            variances = (
                self._eval_unnorm(feature, x, heterogeneity=True)[1]
                if heterogeneity is not False
                else None
            )
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
                feature_names=self.feature_names,
                target_name=self.target_name,
                y_limits=y_limits,
                show_plot=show_plot,
            )
        return vis.plot_pdp_ice(
            x,
            feature,
            yy=yy,
            title=title,
            heterogeneity=heterogeneity,
            y_pdp_label="PDP" if self.method_name == "pdp" else "d-PDP",
            y_ice_label="ICE" if self.method_name == "pdp" else "d-ICE",
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=self.feature_names,
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
        """
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
        )

        if not show_plot:
            return ret


class DerPDP(PDPBase):
    SUPPORTED_FEATURE_TYPES = frozenset({ingestion.CONTINUOUS})
    DEFAULT_CENTERING: Union[bool, str] = False
    IS_DERIVATIVE: bool = True

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
        """
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
        )

        if not show_plot:
            fig, ax = ret
            return fig, ax


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

    Examples:
        >>> # check the gradient of the PDP of a linear model
        >>> import numpy as np
        >>> model = lambda x: np.sum(x, axis=1)
        >>> data = np.random.rand(100, 10)
        >>> x = np.linspace(0.1, 1, 10)
        >>> feature = 0
        >>> y = ice_non_vectorized(model, data, x, feature, heterogeneity=False, model_returns_jac=False)
        >>> (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])
        >>> # the derivative-ICE mean of a linear model
        >>> d_ice = ice_non_vectorized(model, model_jac, data, x, feature, return_d_ice=True)
        >>> d_ice.mean(axis=1)
        array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])


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

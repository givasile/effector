import typing
from typing import Callable, List, Optional, Union
import copy
import numpy as np
import effector.visualization as vis
import effector.helpers as helpers
from effector.global_effect import GlobalEffectBase


class PDPBase(GlobalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
        method_name: str = "PDP",
    ):
        super(PDPBase, self).__init__(
            method_name,
            data,
            model,
            model_jac,
            None,
            nof_instances,
            axis_limits,
            feature_names,
            target_name,
        )

    def _predict(self, data, xx, feature, use_vectorized=True):
        method = ice_vectorized if use_vectorized else ice_non_vectorized
        if self.method_name == "pdp":
            y = method(self.model, None, data, xx, feature, False)
        else:
            y = method(self.model, self.model_jac, self.data, xx, feature, True)
        return y

    def _fit_feature(
        self,
        feature: int,
        centering: Union[bool, str] = False,
        points_for_centering: int = 30,
        use_vectorized: bool = True,
    ) -> dict:

        data = self.data
        if centering is True or centering == "zero_integral":
            xx = np.linspace(
                self.axis_limits[0, feature],
                self.axis_limits[1, feature],
                points_for_centering,
            )
            y = self._predict(data, xx, feature, use_vectorized)
            norm_const = np.mean(y, axis=0)
            fe = {"norm_const": norm_const}
        elif centering == "zero_start":
            xx = self.axis_limits[0, feature, np.newaxis]
            y = self._predict(data, xx, feature, use_vectorized)
            fe = {"norm_const": y[0]}
        else:
            fe = {"norm_const": np.nan}
        return fe

    def fit(
        self,
        features: Union[int, str, list] = "all",
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
        centering = helpers.prep_centering(centering)
        features = helpers.prep_features(features, self.dim)

        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(
                s, centering, points_for_centering, use_vectorized
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
        heterogeneity: bool = False,
        centering: typing.Union[bool, str] = False,
        return_all: bool = False,
        use_vectorized: bool = True,
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot

              - `np.ndarray` of shape `(T, )`

            heterogeneity: whether to return the heterogeneity measures.

                  - if `heterogeneity=False`, the function returns the mean effect at the given `xs`
                  - If `heterogeneity=True`, the function returns `(y, std)` where `y` is the mean effect and `std` is the standard deviation of the mean effect

            centering: whether to center the PDP

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.

            return_all: whether to return PDP and ICE plots evaluated at `xs`

                - If `return_all=False`, the function returns the mean effect at the given `xs`
                - If `return_all=True`, the function returns a `ndarray` of shape `(T, N)` with the `N` ICE plots evaluated at `xs`

            use_vectorized: whether to use the vectorized version of the computation

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, std)` otherwise

        """
        centering = helpers.prep_centering(centering)

        if self.requires_refit(feature, centering):
            self.fit(
                features=feature, centering=centering, use_vectorized=use_vectorized
            )

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        # new implementation
        y_ice = self._predict(self.data, xs, feature, use_vectorized)
        if centering:
            norm_consts = np.expand_dims(
                self.feature_effect["feature_" + str(feature)]["norm_const"], axis=0
            )
            y_ice = y_ice - norm_consts

        y_mean = np.mean(y_ice, axis=1)

        if return_all:
            return y_ice

        if heterogeneity:
            y_var = np.var(y_ice, axis=1)
            return y_mean, y_var
        else:
            return y_mean

    def _plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = False,
        centering: Union[bool, str] = True,
        nof_points: int = 30,
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

        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        yy = self.eval(
            feature,
            x,
            heterogeneity=False,
            centering=centering,
            return_all=True,
            use_vectorized=use_vectorized,
        )

        if show_avg_output:
            avg_output = helpers.prep_avg_output(
                self.data, self.model, self.avg_output, scale_y
            )
        else:
            avg_output = None

        title = (
            "Partial Dependence Plot (PDP)"
            if self.method_name == "pdp"
            else "derivative Partial Dependence Plot (d-PDP)"
        )
        ret = vis.plot_pdp_ice(
            x,
            feature,
            yy=yy,
            title=title,
            confidence_interval=heterogeneity,
            y_pdp_label="PDP" if self.method_name == "pdp" else "d-PDP",
            y_ice_label="ICE" if self.method_name == "pdp" else "d-ICE",
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            nof_ice=nof_ice,
            y_limits=y_limits,
            show_plot=show_plot,
        )
        if not show_plot:
            fig, ax = ret
            return fig, ax


class PDP(PDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ):
        """
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

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """

        super(PDP, self).__init__(
            data,
            model,
            None,
            axis_limits,
            nof_instances,
            feature_names,
            target_name,
            method_name="PDP",
        )

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = "ice",
        centering: Union[bool, str] = True,
        nof_points: int = 30,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_ice: Union[int, str] = "all",
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
            show_plot
        )

        if not show_plot:
            return ret


class DerPDP(PDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ):
        """
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

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """

        super(DerPDP, self).__init__(
            data,
            model,
            model_jac,
            axis_limits,
            nof_instances,
            feature_names,
            target_name,
            method_name="d-PDP",
        )

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = "ice",
        centering: Union[bool, str] = False,
        nof_points: int = 30,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_ice: Union[int, str] = 100,
        show_avg_output: bool = False,
        dy_limits: Optional[List] = None,
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

            dy_limits: None or tuple, the limits of the y-axis for the derivative PDP

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
            dy_limits,
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
        >>> # check the gradient of the PDP of a linear model with heterogeneity
        >>> dpdp, _, _ = ice_non_vectorized(model, data, x, feature, heterogeneity=True, model_returns_jac=False, return_all=False, return_d_ice=True)
        >>> dpdp
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
            # TODO: needs test, something is wrong
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

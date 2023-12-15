import typing
import copy
import numpy as np
import effector.visualization as vis
import effector.helpers as helpers
from effector.global_effect import GlobalEffect
import matplotlib.pyplot as plt


class PDP(GlobalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        nof_instances: typing.Union[int, str] = 300,
        avg_output: typing.Union[None, float] = None,
        feature_names: typing.Union[None, list] = None,
        target_name: typing.Union[None, str] = None,
    ):
        """
        Constructor of the PDP class.

        Definition:
            PDP implements the following feature effect method:
            $$
            \hat{f}(x_s) = {1 \over N} \sum_{i=1}^N f(x_s, x_{-s}^{(i)})
            $$

            and the ICE plots that are plot on top of the PDP:
            $$
            \hat{f}^{(i)}(x_s) = f(x_s, x_{-s}^{(i)}), \quad i=1, \dots, N
            $$

        Notes:
            The required parameters are `data` and `model`. The rest are optional.

        Args:
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N, )`

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            nof_instances: maximum number of instances to be used for PDP.

                - use "all", for using all instances.
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
        """
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        data = data[self.indices, :]

        super(PDP, self).__init__(
            data, model, axis_limits, avg_output, feature_names, target_name
        )

    def _fit_feature(
        self,
        feature: int,
        centering: bool | str = False,
        points_for_centering: int = 100,
    ) -> typing.Dict:

        # drop points outside of limits
        self.data = self.data[self.data[:, feature] >= self.axis_limits[0, feature]]
        self.data = self.data[self.data[:, feature] <= self.axis_limits[1, feature]]
        data = self.data

        if centering is True or centering == "zero_integral":
            xx = np.linspace(
                self.axis_limits[0, feature],
                self.axis_limits[1, feature],
                points_for_centering,
            )
            y = pdp_1d_vectorized(
                self.model, data, xx, feature, False, False, True
            )
            norm_const = np.mean(y, axis=0)
            fe = {"norm_const": norm_const}
        elif centering == "zero_start":
            xx = self.axis_limits[0, feature, np.newaxis]
            y = pdp_1d_vectorized(
                self.model, data, xx, feature, False, False, True
            )
            fe = {"norm_const": y[0]}
        else:
            fe = {"norm_const": helpers.EMPTY_SYMBOL}
        return fe

    def fit(
        self,
        features: int | str | list = "all",
        centering: bool | str = True,
        points_for_centering: int = 100,
    ):
        """
        Fit the PD Plots.

        Notes:
            The only thing that `.fit` does is to compute the normalization constant for centering the
            PDP and the ICE plots, if `centering` is not `False`.
            This will be automatically done when calling `eval` or `plot`, so there is no need to call `fit` explicitly.

        Args:
            features: the features to fit.
                - If set to "all", all the features will be fitted.
            centering:
                - If set to False, no centering will be applied.
                - If set to "zero_integral" or True, the integral of the feature effect will be set to zero.
                - If set to "zero_mean", the mean of the feature effect will be set to zero.

            points_for_centering: number of linspaced points along the feature axis used for centering.

                - If set to `all`, all the dataset points will be used.

        """
        assert centering in [
            True,
            "zero_integral",
            "zero_start",
        ], "`centering` must be True, 'zero_integral' or 'zero_start'. It is meaningless to use the .fit() method with centering=False, because it will do nothing. If you don't want to apply any centering, use immediately `.plot()` or `.eval()` with centering=False."
        centering = helpers.prep_centering(centering)
        features = helpers.prep_features(features, self.dim)

        # new implementation
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(
                s, centering, points_for_centering
            )
            self.is_fitted[s] = True
            self.method_args["feature_" + str(s)] = {
                "centering": centering,
                "points_for_centering": points_for_centering,
            }

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        uncertainty: bool = False,
        centering: typing.Union[bool, str] = False,
        return_all: bool = False,
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot

              - `np.ndarray` of shape `(T,)`

            uncertainty: whether to return the uncertainty measures.

                  - if `uncertainty=False`, the function returns the mean effect at the given `xs`
                  - If `uncertainty=True`, the function returns `(y, std)` where `y` is the mean effect and `std` is the standard deviation of the mean effect

            centering: whether to center the plot

                - If `centering` is `False`, the SHAP curve is not centered
                - If `centering` is `True` or `zero_integral`, the SHAP curve is centered around the `y` axis.
                - If `centering` is `zero_start`, the SHAP curve starts from `y=0`.

            return_all: whether to return PDP and ICE plots evaluated at `xs`

                - If `return_all=False`, the function returns the mean effect at the given `xs`
                - If `return_all=True`, the function returns a `ndarray` of shape `(T, N)` with the `N` ICE plots evaluated at `xs`

        Returns:
            the mean effect `y`, if `uncertainty=False` (default) or a tuple `(y, std, estimator_var)` otherwise

        """
        centering = helpers.prep_centering(centering)

        if self.refit(feature, centering):
            self.fit(features=feature, centering=centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        # new implementation
        yy = pdp_1d_vectorized(self.model, self.data, xs, feature, False, False, True)

        if centering:
            norm_consts = np.expand_dims(
                self.feature_effect["feature_" + str(feature)]["norm_const"], axis=0
            )
            yy = yy - norm_consts

        y_pdp = np.mean(yy, axis=1)

        if return_all:
            return yy

        if uncertainty:
            std = np.std(yy, axis=1)
            return y_pdp, std, np.zeros_like(std)
        else:
            return y_pdp

    def plot(
        self,
        feature: int,
        heterogeneity: bool | str = False,
        centering: bool | str = False,
        nof_points: int = 30,
        scale_x: None | dict = None,
        scale_y: None | dict = None,
        nof_ice: int | str = "all",
        show_avg_output: bool = False,
        y_limits: None | list = None,
    ) -> None:
        """
        Plot the PDP.

        Args:
            feature: index of the plotted feature
            heterogeneity: whether to output the heterogeneity of the SHAP values

                - If `heterogeneity` is `False`, no heterogeneity is plotted
                - If `heterogeneity` is `True` or `"std"`, the standard deviation of the shap values is plotted
                - If `heterogeneity` is `ice`, the ICE plots are plotted

            centering: whether to center the SDP

                - If `centering` is `False`, the SHAP curve is not centered
                - If `centering` is `True` or `zero_integral`, the SHAP curve is centered around the `y` axis.
                - If `centering` is `zero_start`, the SHAP curve starts from `y=0`.

            nof_points: number of points to evaluate the SDP plot
            scale_x: dictionary with keys "mean" and "std" for scaling the x-axis
            scale_y: dictionary with keys "mean" and "std" for scaling the y-axis
            nof_ice: number of shap values to show on top of the SHAP curve
            show_avg_output: whether to show the average output of the model
            y_limits: limits of the y-axis
        """
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        yy = self.eval(
            feature, x, uncertainty=False, centering=centering, return_all=True
        )

        avg_output = None if not show_avg_output else self.avg_output
        title = "PDP Plot"
        vis.plot_pdp_ice(
            x,
            feature,
            yy=yy,
            title=title,
            confidence_interval=heterogeneity,
            y_pdp_label="PDP",
            y_ice_label="ICE",
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            nof_ice=nof_ice,
            y_limits=y_limits,
        )


class DerivativePDP(GlobalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        axis_limits: typing.Union[None, np.ndarray] = None,
        nof_instances: typing.Union[int, str] = 100,
        avg_output: typing.Union[None, float] = None,
        feature_names: typing.Union[None, list] = None,
        target_name: typing.Union[None, str] = None,
    ):
        """
        Initializes the DerivativePDP class.

        Notes:
            PDP implements the following feature effect method:
            $$
            \hat{f}(x_s) = {1 \over N} \sum_{i=1}^N f(x_s, x_{-s}^{(i)})
            $$

            and the ICE plots that are plot on top of the PDP:
            $$
            \hat{f}^{(i)}(x_s) = f(x_s, x_{-s}^{(i)}), \quad i=1, \dots, N
            $$


        Args:
            data: The dataset, shape (N, D)
            model: The model to be explained, (N, D) -> (N)
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            nof_instances: maximum number of instances to be used for PDP. If "all", all instances are used.
        """
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        # TODO: check if I want to keep all data for the PDP
        # TODO: and the limited data for the ICE plot
        data = data[self.indices, :]
        self.model_jac = model_jac

        super(DerivativePDP, self).__init__(
            data, model, axis_limits, avg_output, feature_names, target_name
        )

    # TODO: fix so that centering will not be always zero mean
    def _fit_feature(
        self,
        feature: int,
        centering: typing.Union[bool, str],
        nof_points_centering: int,
    ) -> typing.Dict:
        # create one XX matrix with all the points that we want to evaluate for all ICEs
        xx = np.linspace(
            self.axis_limits[0, feature],
            self.axis_limits[1, feature],
            nof_points_centering,
        )
        if self.model_jac is not None:
            y = pdp_1d_vectorized(self.model_jac, self.data, xx, feature, False, True, True)
        else:
            y = pdp_1d_vectorized(self.model, self.data, xx, feature, False, False, True, True)

        # compute the normalization constant per ice
        norm_const = np.mean(y, axis=0)

        return {"norm_const": norm_const}

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        centering: typing.Union[bool, str] = True,
        points_used_for_centering: int = 100,
    ):
        """
        Fit the PDP and the ICE plots.

        Notes:
            Practically, the only thing that `.fit` does is to compute the normalization constant for centering the
            PDP and the ICE plots, if `centering` is not `False`.
            This will be automatically done when calling `eval` or `plot`, so there is no need to call `fit` explicitly.

        Args:
            features: The features to be fitted for explanation. If `"all"`, all features are fitted.
            centering: Whether to center the PDP and the ICE plots.
            points_used_for_centering: number of points on the x-axis to evaluate the PDP and the ICE plots on for computing the normalization constant

        Notes:
            * If `centering` is `False`, the PDP and ICE plots are not centered
            * If `centering` is `True` or `"zero_integral"`, the PDP and the ICE plots are centered wrt to the `y` axis.
            * If `centering` is `"zero_start"`, the PDP and the ICE plots start from `y=0`.
        """
        centering = helpers.prep_centering(centering)
        features = helpers.prep_features(features, self.dim)

        # new implementation
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(
                s, centering, points_used_for_centering
            )
            self.is_fitted[s] = True

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        uncertainty: bool = False,
        centering: typing.Union[bool, str] = False,
        return_all: bool = False,
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Notes:
            The standard deviation of the PDP is computed as:
            $$
            todo
            $$

        Parameters:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot, (T)
            uncertainty: whether to return the uncertainty measures
            centering: whether to center the plot
            return_all: whether to return all; PDP and ICE plots evaluated at `xs`

        Returns:
            the mean effect `y`, if `uncertainty=False` (default) or a tuple `(y, std, estimator_var)` otherwise

        Notes:
            * If `centering` is `False`, the PDP is not centered
            * If `centering` is `True` or `"zero_integral"`, the PDP is centered by subtracting the mean of the PDP.
            * If `centering` is `"zero_start"`, the PDP is centered by subtracting the value of the PDP at the first point.

        Notes:
            * If `uncertainty` is `False`, the PDP returns only the mean effect `y` at the given `xs`.
            * If `uncertainty` is `True`, the PDP returns `(y, std, estimator_var)` where:
                * `y` is the mean effect
                * `std` is the standard deviation of the mean effect
                * `estimator_var` is the variance of the mean effect estimator
        """
        centering = helpers.prep_centering(centering)
        if not self.is_fitted[feature]:
            self.fit(features=feature, centering=centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]


        if self.model_jac is not None:
            yy = pdp_1d_vectorized(self.model_jac, self.data, xs, feature, False, True, True)
        else:
            yy = pdp_1d_vectorized(self.model, self.data, xs, feature, False, False, True, True)

        # # new implementation
        # yy = pdp_1d_vectorized(
        #     self.model_jac, self.data, xs, feature, False, True, True
        # )

        if centering:
            norm_consts = np.expand_dims(
                self.feature_effect["feature_" + str(feature)]["norm_const"], axis=0
            )
            yy = yy - norm_consts

        y_pdp = np.mean(yy, axis=1)

        if return_all:
            return yy

        if uncertainty:
            std = np.std(yy, axis=1)
            estimator_var = np.var(yy, axis=1)
            return y_pdp, std, estimator_var
        else:
            return y_pdp

    def plot(
        self,
        feature: int,
        confidence_interval: bool | str = False,
        centering: bool = False,
        nof_axis_points: int = 30,
        scale_x: typing.Union[None, dict] = None,
        scale_y: typing.Union[None, dict] = None,
        nof_ice: typing.Union[int, str] = "all",
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the PDP along with the ICE plots

        Args:
            feature: index of the plotted feature
            centering: whether to center the PDP
            nof_axis_points: number of points on the x-axis to evaluate the PDP and the ICE plots
            scale_x: dictionary with keys "mean" and "std" for scaling the x-axis
            scale_y: dictionary with keys "mean" and "std" for scaling the y-axis

        Notes:
            * If `centering` is `False`, the PDP and ICE plots are not centered
            * If `centering` is `True` or `"zero_integral"`, the PDP and the ICE plots are centered wrt to the `y` axis.
            * If `centering` is `"zero_start"`, the PDP and the ICE plots start from `y=0`.

        """
        confidence_interval = helpers.prep_confidence_interval(confidence_interval)

        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_axis_points
        )

        yy = self.eval(
            feature, x, uncertainty=False, centering=centering, return_all=True
        )
        title = "Derivative PDP-ICE Plot"
        fig, ax = vis.plot_pdp_ice(
            x,
            feature,
            yy=yy,
            title=title,
            confidence_interval=confidence_interval,
            y_pdp_label="d-PDP",
            y_ice_label="d-ICE",
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=None,
            feature_names=self.feature_names,
            target_name=self.target_name,
            is_derivative=True,
            nof_ice=nof_ice,
        )
        return fig, ax


def pdp_1d_non_vectorized(
    model: callable,
    data: np.ndarray,
    x: np.ndarray,
    feature: int,
    uncertainty: bool,
    is_jac: bool,
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
        >>> y = pdp_1d_vectorized(model, data, x, feature, uncertainty=False, model_returns_jac=False)
        >>> (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    Args:
        model: The black-box function (N, D) -> (N)
        data: The design matrix, (N, D)
        x: positions to evaluate pdp, (T)
        feature: index of the feature of interest
        uncertainty: whether to also compute the uncertainty of the PDP
        model_returns_jac (bool): whether the model returns the prediction (False) or the Jacobian wrt the input (True)

    Returns:
        The PDP values `y` that correspond to `x`, if uncertainty is False, `(y, std, stderr)` otherwise

    """
    nof_points = x.shape[0]
    mean_pdp = []
    sigma_pdp = []
    stderr = []
    for i in range(nof_points):
        x_new = copy.deepcopy(data)
        x_new[:, feature] = x[i]
        y = model(x_new)[:, feature] if is_jac else model(x_new)
        mean_pdp.append(np.mean(y))
        if uncertainty:
            std = np.std(y)
            sigma_pdp.append(std)
            stderr.append(std / np.sqrt(data.shape[0]))
    return (
        (np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr))
        if uncertainty
        else np.array(mean_pdp)
    )


def pdp_1d_vectorized(
    model: callable,
    data: np.ndarray,
    x: np.ndarray,
    feature: int,
    uncertainty: bool,
    model_returns_jac: bool,
    return_all: bool = False,
    ask_for_derivatives: bool = False,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Computes the unnormalized 1-dimensional PDP, in a vectorized way.

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
        >>> y = pdp_1d_vectorized(model, data, x, feature, uncertainty=False, model_returns_jac=False)
        >>> (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    Args:
        model: The black-box function (N, D) -> (N)
        data: The design matrix, (N, D)
        x: positions to evaluate pdp, (T)
        feature: index of the feature of interest
        uncertainty: whether to also compute the uncertainty of the PDP
        model_returns_jac (bool): whether the model returns the prediction (False) or the Jacobian wrt the input (True)
        ask_for_derivatives (bool): whether to ask the model to return the derivatives wrt the input
        return_all (bool): whether to return all the predictions or only the mean (and std and stderr)

    Returns:
        The PDP values `y` that correspond to `x`, if uncertainty is False, `(y, std, stderr)` otherwise

    """

    nof_instances = data.shape[0]
    x_new = copy.deepcopy(data)
    x_new = np.expand_dims(x_new, axis=0)
    x_new = np.repeat(x_new, x.shape[0], axis=0)

    if ask_for_derivatives and not model_returns_jac:
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
        y = model(x_new)[:, feature] if model_returns_jac else model(x_new)
        y = np.reshape(y, (x.shape[0], nof_instances))

    mean_pdp = np.mean(y, axis=1)
    if return_all:
        return y

    if uncertainty:
        std = np.std(y, axis=1)
        sigma_pdp = std
        stderr = std / np.sqrt(data.shape[0])
        return mean_pdp, sigma_pdp, stderr
    else:
        return mean_pdp


# TODO: check this implementation
def pdp_nd_non_vectorized(
    model: callable,
    data: np.ndarray,
    x: np.ndarray,
    features: list,
    uncertainty: bool,
    is_jac: bool,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Computes the unnormalized n-dimensional PDP, in a non-vectorized way.

    Args:
        model (callable): model to be explained
        data (np.ndarray): dataset, shape (N, D)
        x (np.ndarray): values of the features to be explained, shape (K, D)
        features (list): indices of the features to be explained
        uncertainty (bool): whether to compute the uncertainty of the PDP
        is_jac (bool): whether the model returns the prediction (False) or the Jacobian wrt the input (True)

    Returns:
        if uncertainty is False:
            np.ndarray: unnormalized n-dimensional PDP
        if uncertainty is True:
            a tuple of three np.ndarray: (unnormalized n-dimensional PDP, standard deviation, standard error)
    """
    assert len(features) == x.shape[1]

    # nof positions to evaluate the PDP
    K = x.shape[0]

    mean_pdp = []
    sigma_pdp = []
    stderr = []
    for k in range(K):
        # take all dataset
        x_new = copy.deepcopy(data)

        # set the features of all datapoints to the values of the k-th position
        for j, feat in features:
            x_new[:, feat] = x[k, j]

        # compute the prediction or the Jacobian wrt the input
        y = model(x_new)[:, features] if is_jac else model(x_new)

        # compute the mean of the prediction or the Jacobian wrt the input
        mean_pdp.append(np.mean(y))

        # if uncertaitny, compute also the std and the stderr
        if uncertainty:
            std = np.std(y)
            sigma_pdp.append(std)
            stderr.append(std / np.sqrt(data.shape[0]))
    return (
        (np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr))
        if uncertainty
        else np.array(mean_pdp)
    )


# TODO: check this implementation
def pdp_nd_vectorized(
    model: callable,
    data: np.ndarray,
    x: np.ndarray,
    features: list,
    uncertainty: bool,
    model_returns_jac: bool,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Computes the unnormalized n-dimensional PDP, in a vectorized way.

    Notes:
        The vectorized version is faster than the non-vectorized one, but it requires more memory.
        Be careful when using it with large datasets, since it creates an internal dataset of shape (T, N, D)
        where T is the number of positions to evaluate the PDP, N is the number of instances in the dataset
        and D is the number of features.

    Examples:
        >>> # check the gradient of the PDP of a linear model
        >>> import numpy as np
        >>> model = lambda x: x
        >>> data = np.random.rand(100, 10)
        >>> x = np.stack([np.linspace(0.1, 1, 10)] * 2, axis=-1)
        >>> x.shape[1]
        >>> features = [0, 1]
        >>> y = pdp_nd_vectorized(model, data, x, features, uncertainty=False, model_returns_jac=False)
        >>> (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    Args:
        model: The black-box function (N, D) -> (N)
        data: The design matrix, (N, D)
        x: positions to evaluate pdp, (T)
        features: indices of the features of interest
        uncertainty: whether to also compute the uncertainty of the PDP
        model_returns_jac (bool): whether the model returns the prediction (False) or the Jacobian wrt the input (True)

    Returns:
        The PDP values `y` that correspond to `x`, if uncertainty is False, `(y, std, stderr)` otherwise

    """

    assert len(features) == x.shape[1]
    nof_instances = data.shape[0]
    x_new = copy.deepcopy(data)
    x_new = np.expand_dims(x_new, axis=0)
    x_new = np.repeat(x_new, x.shape[0], axis=0)
    for j in range(len(features)):
        x_new[:, :, features[j]] = np.expand_dims(x[:, j], axis=-1)
    x_new = np.reshape(x_new, (x_new.shape[0] * x_new.shape[1], x_new.shape[2]))
    y = model(x_new)[:, features] if model_returns_jac else model(x_new)
    y = np.reshape(y, (x.shape[0], nof_instances))
    mean_pdp = np.mean(y, axis=1)
    if uncertainty:
        std = np.std(y, axis=1)
        sigma_pdp = std
        stderr = std / np.sqrt(data.shape[0])
        return (mean_pdp, sigma_pdp, stderr)
    else:
        return mean_pdp

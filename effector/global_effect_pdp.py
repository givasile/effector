import typing
from typing import Callable, List, Optional, Union, Tuple
import copy
import numpy as np
import effector.visualization as vis
import effector.helpers as helpers
from effector.global_effect import GlobalEffectBase
import matplotlib.pyplot as plt


class PDPBase(GlobalEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        axis_limits: Optional[np.ndarray] = None,
        avg_output: Optional[float] = None,
        nof_instances: Union[int, str] = 300,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
        method_name: str = "PDP",
    ):
        """
        Constructor of the PDPBase class.
        """

        self.model_jac = model_jac

        super(PDPBase, self).__init__(
            method_name,
            data,
            model, nof_instances, axis_limits, avg_output, feature_names, target_name
        )

    def _predict(self, data, xx, feature):
        if self.method_name == "pdp":
            y = pdp_1d_vectorized(
                self.model, data, xx, feature, False, False, True
            )
        else:
            if self.model_jac is not None:
                y = pdp_1d_vectorized(self.model_jac, self.data, xx, feature, False, True, True)
            else:
                y = pdp_1d_vectorized(self.model, self.data, xx, feature, False, False, True, True)
        return y

    def _fit_feature(
        self,
        feature: int,
        centering: Union[bool, str] = False,
        points_for_centering: int = 100,
    ) -> dict:

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
            y = self._predict(data, xx, feature)
            norm_const = np.mean(y, axis=0)
            fe = {"norm_const": norm_const}
        elif centering == "zero_start":
            xx = self.axis_limits[0, feature, np.newaxis]
            y = self._predict(data, xx, feature)
            fe = {"norm_const": y[0]}
        else:
            fe = {"norm_const": helpers.EMPTY_SYMBOL}
        return fe

    def fit(
        self,
        features: Union[int, str, list] = "all",
        centering: Union[bool, str] = True,
        points_for_centering: int = 100,
    ):
        """
        Fit the PDP or d-PDP.

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

                - If set to `"all"`, all the dataset points will be used.

        """
        centering = helpers.prep_centering(centering)
        features = helpers.prep_features(features, self.dim)

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
        heterogeneity: bool = False,
        centering: typing.Union[bool, str] = False,
        return_all: bool = False,
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

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, std)` otherwise

        """
        centering = helpers.prep_centering(centering)

        if self.refit(feature, centering):
            self.fit(features=feature, centering=centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        # new implementation
        yy = self._predict(self.data, xs, feature)

        if centering:
            norm_consts = np.expand_dims(
                self.feature_effect["feature_" + str(feature)]["norm_const"], axis=0
            )
            yy = yy - norm_consts

        y_pdp = np.mean(yy, axis=1)

        if return_all:
            return yy

        if heterogeneity:
            std = np.std(yy, axis=1)
            return y_pdp, std
        else:
            return y_pdp

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = False,
        centering: Union[bool, str] = False,
        nof_points: int = 30,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_ice: Union[int, str] = "all",
        show_avg_output: bool = False,
        y_limits: Optional[List] = None,
    ):
        """
        Plot the PDP or d-PDP.

        Args:
            feature: index of the plotted feature
            heterogeneity: whether to output the heterogeneity of the SHAP values

                - If `heterogeneity` is `False`, no heterogeneity is plotted
                - If `heterogeneity` is `True` or `"std"`, the standard deviation of the shap values is plotted
                - If `heterogeneity` is `ice`, the ICE plots are plotted

            centering: whether to center the PDP

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.

            nof_points: number of points to evaluate the SDP plot
            scale_x: dictionary with keys "mean" and "std" for scaling the x-axis
            scale_y: dictionary with keys "mean" and "std" for scaling the y-axis
            nof_ice: number of shap values to show on top of the SHAP curve
            show_avg_output: whether to show the average output of the model
            y_limits: limits of the y-axis
        """
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)
        centering = helpers.prep_centering(centering)

        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        yy = self.eval(
            feature, x, heterogeneity=False, centering=centering, return_all=True
        )

        if show_avg_output:
            avg_output = helpers.prep_avg_output(self.data, self.model, self.avg_output, scale_y)
        else:
            avg_output = None

        title = "PDP" if self.method_name == "pdp" else "d-PDP"
        vis.plot_pdp_ice(
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
        )


class PDP(PDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 300,
        avg_output: Optional[float] = None,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ):
        """
        Constructor of the PDP class.

        Definition:
            PDP is defined as:
            $$
            \hat{f}^{PDP}(x_s) = {1 \over N} \sum_{i=1}^N f(x_s, x_C^{(i)})b
            $$

            The ICE plots are:
            $$
            \hat{f}^{(i)}(x_s) = f(x_s, x_C^{(i)}), \quad i=1, \dots, N
            $$

            The heterogeneity is:
            $$
            \mathcal{H}^{PDP}(x_s) = \sqrt {{1 \over N} \sum_{i=1}^N ( \hat{f}^{(i)}(x_s) - \hat{f}^{PDP}(x_s) )^2}
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

        super(PDP, self).__init__(
            data, model, None, axis_limits, avg_output, nof_instances, feature_names, target_name, method_name="PDP"
        )


class DerivativePDP(PDPBase):
    def __init__(
            self,
            data: np.ndarray,
            model: Callable,
            model_jac: Optional[Callable] = None,
            axis_limits: Optional[np.ndarray] = None,
            nof_instances: Union[int, str] = 300,
            avg_output: Optional[float] = None,
            feature_names: Optional[List] = None,
            target_name: Optional[str] = None,
    ):
        """
        Constructor of the DerivativePDP class.

        Definition:
            d-PDP is defined as:
            $$
            \hat{f}^{d-PDP}(x_s) = {1 \over N} \sum_{i=1}^N {df \over d x_s} (x_s, x_C^i)
            $$

            The d-ICE plots are:
            $$
            \hat{f}^i(x_s) = {df \over d x_s}(x_s, x_C^i), \quad i=1, \dots, N
            $$

            The heterogeneity is:
            $$
            \mathcal{H}^{d-PDP}(x_s) = \sqrt {{1 \over N} \sum_{i=1}^N ( \hat{f}^i(x_s) - \hat{f}^{d-PDP}(x_s) )^2}
            $$

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

            avg_output: The average output of the model.

                - use a `float`, to specify it manually
                - use `None`, to be inferred as `np.mean(model(data))`

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """

        super(DerivativePDP, self).__init__(
            data, model, model_jac, axis_limits, avg_output, nof_instances, feature_names, target_name, method_name="d-PDP"
        )


def pdp_1d_non_vectorized(
    model: callable,
    data: np.ndarray,
    x: np.ndarray,
    feature: int,
    heterogeneity: bool,
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
        >>> y = pdp_1d_vectorized(model, data, x, feature, heterogeneity=False, model_returns_jac=False)
        >>> (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    Args:
        model: The black-box function (N, D) -> (N)
        data: The design matrix, (N, D)
        x: positions to evaluate pdp, (T)
        feature: index of the feature of interest
        heterogeneity: whether to also compute the heterogeneity of the PDP
        is_jac (bool): whether the model returns the prediction (False) or the Jacobian wrt the input (True)

    Returns:
        The PDP values `y` that correspond to `x`, if heterogeneity is False, `(y, std, stderr)` otherwise

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
        if heterogeneity:
            std = np.std(y)
            sigma_pdp.append(std)
            stderr.append(std / np.sqrt(data.shape[0]))
    return (
        (np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr))
        if heterogeneity
        else np.array(mean_pdp)
    )


def pdp_1d_vectorized(
    model: callable,
    data: np.ndarray,
    x: np.ndarray,
    feature: int,
    heterogeneity: bool,
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
        >>> y = pdp_1d_vectorized(model, data, x, feature, heterogeneity=False, model_returns_jac=False)
        >>> (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    Args:
        model: The black-box function (N, D) -> (N)
        data: The design matrix, (N, D)
        x: positions to evaluate pdp, (T)
        feature: index of the feature of interest
        heterogeneity: whether to also compute the heterogeneity of the PDP
        model_returns_jac (bool): whether the model returns the prediction (False) or the Jacobian wrt the input (True)
        ask_for_derivatives (bool): whether to ask the model to return the derivatives wrt the input
        return_all (bool): whether to return all the predictions or only the mean (and std and stderr)

    Returns:
        The PDP values `y` that correspond to `x`, if heterogeneity is False, `(y, std, stderr)` otherwise

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

    if heterogeneity:
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
    heterogeneity: bool,
    is_jac: bool,
) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Computes the unnormalized n-dimensional PDP, in a non-vectorized way.

    Args:
        model (callable): model to be explained
        data (np.ndarray): dataset, shape (N, D)
        x (np.ndarray): values of the features to be explained, shape (K, D)
        features (list): indices of the features to be explained
        heterogeneity (bool): whether to compute the heterogeneity of the PDP
        is_jac (bool): whether the model returns the prediction (False) or the Jacobian wrt the input (True)

    Returns:
        if heterogeneity is False:
            np.ndarray: unnormalized n-dimensional PDP
        if heterogeneity is True:
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
        if heterogeneity:
            std = np.std(y)
            sigma_pdp.append(std)
            stderr.append(std / np.sqrt(data.shape[0]))
    return (
        (np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr))
        if heterogeneity
        else np.array(mean_pdp)
    )


# TODO: check this implementation
def pdp_nd_vectorized(
    model: callable,
    data: np.ndarray,
    x: np.ndarray,
    features: list,
    heterogeneity: bool,
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
        >>> y = pdp_nd_vectorized(model, data, x, features, heterogeneity=False, model_returns_jac=False)
        >>> (y[1:] - y[:-1]) / (x[1:] - x[:-1])
        array([1., 1., 1., 1., 1., 1., 1., 1., 1.])

    Args:
        model: The black-box function (N, D) -> (N)
        data: The design matrix, (N, D)
        x: positions to evaluate pdp, (T)
        features: indices of the features of interest
        heterogeneity: whether to also compute the heterogeneity of the PDP
        model_returns_jac (bool): whether the model returns the prediction (False) or the Jacobian wrt the input (True)

    Returns:
        The PDP values `y` that correspond to `x`, if heterogeneity is False, `(y, std, stderr)` otherwise

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
    if heterogeneity:
        std = np.std(y, axis=1)
        sigma_pdp = std
        stderr = std / np.sqrt(data.shape[0])
        return (mean_pdp, sigma_pdp, stderr)
    else:
        return mean_pdp

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
        nof_instances: typing.Union[int, str] = 100,
        avg_output: typing.Union[None, float] = None,
        feature_names: typing.Union[None, list] = None,
        target_name: typing.Union[None, str] = None,
    ):
        """
        Initializes the PDPwithICE class.

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

        super(PDP, self).__init__(
            data, model, axis_limits, avg_output, feature_names, target_name
        )

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
        y = pdp_1d_vectorized(self.model, self.data, xx, feature, False, False, True)

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
            * If `centering` is `True` or `"zero-integral"`, the PDP and the ICE plots are centered wrt to the `y` axis.
            * If `centering` is `"zero-start"`, the PDP and the ICE plots start from `y=0`.
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
            * If `centering` is `True` or `"zero-integral"`, the PDP is centered by subtracting the mean of the PDP.
            * If `centering` is `"zero-start"`, the PDP is centered by subtracting the value of the PDP at the first point.

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
            estimator_var = np.var(yy, axis=1)
            return y_pdp, std, estimator_var
        else:
            return y_pdp

    def plot(
        self,
        feature: int,
        confidence_interval: typing.Union[bool, str] = False,
        centering: bool = False,
        nof_points: int = 30,
        scale_x: typing.Union[None, dict] = None,
        scale_y: typing.Union[None, dict] = None,
        nof_ice: typing.Union[int, str] = "all",
        show_avg_output: bool = False,
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the PDP along with the ICE plots

        Args:
            feature: index of the plotted feature
            centering: whether to center the PDP
            nof_points: number of points on the x-axis to evaluate the PDP and the ICE plots
            scale_x: dictionary with keys "mean" and "std" for scaling the x-axis
            scale_y: dictionary with keys "mean" and "std" for scaling the y-axis

        Notes:
            * If `centering` is `False`, the PDP and ICE plots are not centered
            * If `centering` is `True` or `"zero-integral"`, the PDP and the ICE plots are centered wrt to the `y` axis.
            * If `centering` is `"zero-start"`, the PDP and the ICE plots start from `y=0`.

        """
        confidence_interval = helpers.prep_confidence_interval(confidence_interval)
        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        yy = self.eval(
            feature, x, uncertainty=False, centering=centering, return_all=True
        )

        avg_output = None if not show_avg_output else self.avg_output
        title = "PDP-ICE Plot"
        fig, ax = vis.plot_pdp_ice_2(
            x,
            feature,
            yy=yy,
            title=title,
            confidence_interval=confidence_interval,
            y_pdp_label="PDP",
            y_ice_label="ICE",
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            nof_ice=nof_ice,
        )
        return fig, ax


class DerivativePDP(GlobalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        nof_instances: typing.Union[int, str] = 100,
        avg_output: typing.Union[None, float] = None,
        feature_names: typing.Union[None, list] = None,
        target_name: typing.Union[None, str] = None,
    ):
        """
        Initializes the PDPwithICE class.

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
        y = pdp_1d_vectorized(self.model_jac, self.data, xx, feature, False, True, True)

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
            * If `centering` is `True` or `"zero-integral"`, the PDP and the ICE plots are centered wrt to the `y` axis.
            * If `centering` is `"zero-start"`, the PDP and the ICE plots start from `y=0`.
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
            * If `centering` is `True` or `"zero-integral"`, the PDP is centered by subtracting the mean of the PDP.
            * If `centering` is `"zero-start"`, the PDP is centered by subtracting the value of the PDP at the first point.

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

        # new implementation
        yy = pdp_1d_vectorized(
            self.model_jac, self.data, xs, feature, False, True, True
        )

        if centering:
            norm_consts = np.expand_dims(
                self.feature_effect["feature_3"]["norm_const"], axis=0
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
        confidence_interval: typing.Union[bool, str] = False,
        centering: bool = False,
        nof_points: int = 30,
        scale_x: typing.Union[None, dict] = None,
        scale_y: typing.Union[None, dict] = None,
        nof_ice: typing.Union[int, str] = "all"
    ) -> tuple[plt.Figure, plt.Axes]:
        """
        Plot the PDP along with the ICE plots

        Args:
            feature: index of the plotted feature
            centering: whether to center the PDP
            nof_points: number of points on the x-axis to evaluate the PDP and the ICE plots
            scale_x: dictionary with keys "mean" and "std" for scaling the x-axis
            scale_y: dictionary with keys "mean" and "std" for scaling the y-axis

        Notes:
            * If `centering` is `False`, the PDP and ICE plots are not centered
            * If `centering` is `True` or `"zero-integral"`, the PDP and the ICE plots are centered wrt to the `y` axis.
            * If `centering` is `"zero-start"`, the PDP and the ICE plots start from `y=0`.

        """
        confidence_interval = helpers.prep_confidence_interval(confidence_interval)

        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        yy = self.eval(
            feature, x, uncertainty=False, centering=centering, return_all=True
        )
        title = "Derivative PDP-ICE Plot"
        fig, ax = vis.plot_pdp_ice_2(
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
        return_all (bool): whether to return all the predictions or only the mean (and std and stderr)

    Returns:
        The PDP values `y` that correspond to `x`, if uncertainty is False, `(y, std, stderr)` otherwise

    """

    nof_instances = data.shape[0]
    x_new = copy.deepcopy(data)
    x_new = np.expand_dims(x_new, axis=0)
    x_new = np.repeat(x_new, x.shape[0], axis=0)
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

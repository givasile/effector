import typing
import copy
import numpy as np
import effector.visualization as vis
import effector.helpers as helpers
from effector.fe_base import FeatureEffectBase


class PDP(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        nof_instances: typing.Union[int, str] = 1000,
    ):
        """
        Initializes the PDP class.

        Notes:
            PDP implements the following feature effect method:
            $$
            \hat{f}(x_s) =  {1 \over N} \sum_{i=1}^N f(x_s, x_{-s}^{(i)})
            $$

        Args:
            data: The dataset, shape (N, D)
            model: The model to be explained, (N, D) -> (N)
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            nof_instances: maximum number of instances to be used for PDP. If "all", all instances are used.
        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.nof_instances, self.indices = helpers.prep_nof_instances(nof_instances, data.shape[0])
        self.data = data[self.indices, :]
        self.D = data.shape[1]

        axis_limits = (helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits)
        super(PDP, self).__init__(axis_limits)

    def fit(self,
            features: typing.Union[int, str, list] = "all",
            centering: typing.Union[bool, str] = False,
            ) -> None:
        """
        Fit the PDP plot.

        Notes:
            Practically, the only thing that `.fit` does is to compute the normalization constant for centering the PDP plot.
            This will be automatically done when calling `eval` or `plot`, so there is no need to call `fit` explicitly.

        Args:
            features: The features to be fitted for explanation. If `"all"`, all features are fitted.
            centering: Whether to center the PDP plot.

        Notes:
            * If `centering` is `False`, the PDP is not centered
            * If `centering` is `True` or `"zero-integral"`, the PDP is centered by subtracting the mean of the PDP.
            * If `centering` is `"zero-start"`, the PDP is centered by subtracting the value of the PDP at the first point.
        """
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            self.norm_const[s] = self._compute_norm_const(s, centering) if centering is not False else self.norm_const[s]
            self.is_fitted[s] = True

    def _eval_unnorm(self,
                     feature: int,
                     x: np.ndarray,
                     uncertainty: bool = False
                     ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        # yy = predict_non_vectorized(self.model, self.data, x, feature, uncertainty, is_jac=False)
        yy = pdp_1d_vectorized(self.model, self.data, x, feature, uncertainty, model_returns_jac=False)
        return yy

    def plot(self,
             feature: int,
             uncertainty: typing.Union[bool, str] = False,
             centering: typing.Union[bool, str] = False,
             nof_points: int = 30) -> None:
        """Plot the PDP for a single feature.

        Args:
            feature: index of the plotted feature
            uncertainty: whether to plot the uncertainty
            centering: whether to center the PDP
            nof_points: number of points on the x-axis to evaluate the PDP on

        Notes:
            * If `centering` is `False`, the PDP is not centered
            * If `centering` is `True` or `"zero-integral"`, the PDP is centered by subtracting the mean of the PDP.
            * If `centering` is `"zero-start"`, the PDP is centered by subtracting the value of the PDP at the first point.

        Notes:
            * If `uncertainty` is `False`, the PDP is plotted without uncertainty (=heterogeneity)
            * If `uncertainty` is `True` or `"std"`, the PDP is plotted with the standard deviation of the PDP
            * If `uncertainty` is `"std_err"`, the PDP is plotted with the standard error of the mean of the PDP

        """
        title = "PDP: feature %d" % (feature + 1)
        uncertainty = helpers.prep_uncertainty(uncertainty)
        centering = helpers.prep_centering(centering)
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        func = self.eval
        vis.plot_1d(x, feature, func, confidence=uncertainty, centering=centering, title=title)

# TODO test the implementation
class DerivativePDP(FeatureEffectBase):
    def __init__(
            self,
            data: np.ndarray,
            model: callable,
            model_jac: callable,
            axis_limits: typing.Union[None, np.ndarray] = None,
            max_nof_instances: typing.Union[int, str] = "all",
    ):
        """
        Initializes the Derivative-PDP class.

        Notes:
            DerivativePDP implements the following feature effect method:
            $$
            \hat{f}(x_s) =  {1 \over N} \sum_{i=1}^N {d f \over d x_s}(x_s, x_{-s}^{(i)})_s
            $$

        Args:
            data: The dataset, shape (N, D)
            model: The model to be explained, (N, D) -> (N)
            model_jac: The Jacobian of the model to be explained, (N, D) -> (N, D)
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            nof_instances: maximum number of instances to be used for PDP. If "all", all instances are used.
        """

        assert data.ndim == 2

        self.nof_instances, self.indices = helpers.prep_nof_instances(max_nof_instances, data.shape[0])
        self.data = data[self.indices, :]
        self.model = model
        self.model_jac = model_jac
        self.D = self.data.shape[1]
        axis_limits = (helpers.axis_limits_from_data(self.data) if axis_limits is None else axis_limits)
        super(DerivativePDP, self).__init__(axis_limits)

    def fit(self,
            features: typing.Union[int, str, list] = "all",
            centering: typing.Union[bool, str] = True) -> None:
        """
        Fit the Derivative-PDP plot.

        Notes:
            Practically, the only thing that `.fit` does is to compute the normalization constant for centering the PDP plot.
            This will be automatically done when calling `eval` or `plot`, so there is no need to call `fit` explicitly.

        Args:
            features: The features to be fitted for explanation. If `"all"`, all features are fitted.
            centering: Whether to center the PDP plot.

        Notes:
            * If `centering` is `False`, the PDP is not centered
            * If `centering` is `True` or `"zero-integral"`, the PDP is centered by subtracting the mean of the PDP.
            * If `centering` is `"zero-start"`, the PDP is centered by subtracting the value of the PDP at the first point.
        """

        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.is_fitted[s] = True

    def _eval_unnorm(self,
                     feature: int,
                     x: np.ndarray,
                     uncertainty: bool = False
                     ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:

        # yy = predict_non_vectorized(self.model_jac, self.data, x, feature, uncertainty, is_jac=True)
        yy = pdp_1d_vectorized(self.model_jac, self.data, x, feature, uncertainty, model_returns_jac=True)
        return yy

    def plot(self,
             feature: int,
             uncertainty: typing.Union[bool, str] = False,
             nof_points: int = 30) -> None:
        title = "d-PDP for feature %d" % (feature + 1)
        uncertainty = helpers.prep_uncertainty(uncertainty)
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        func = self.eval
        vis.plot_1d(x, feature, func, confidence=uncertainty, centering=False, title=title)


class ICE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        instance: int = 0,
    ):
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        self.D = data.shape[1]
        self.instance = instance
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(ICE, self).__init__(axis_limits)

    def fit(self, features: typing.Union[int, str, list] = "all", centering: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            self.norm_const[s] = self._compute_norm_const(s, centering) if centering is not False else self.norm_const[s]
            self.is_fitted[s] = True

    def _eval_unnorm(self, feature: int, x: np.ndarray, uncertainty: bool = False) -> np.ndarray:
        if uncertainty:
            raise NotImplementedError

        i = self.instance
        xi = copy.deepcopy(self.data[i, :])
        xi_repeat = np.tile(xi, (x.shape[0], 1))
        xi_repeat[:, feature] = x
        y = self.model(xi_repeat)
        return y

    def plot(self, feature: int, centering: bool = True, nof_points: int = 30) -> None:
        # getters
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "ICE: feature %d, instance %d" % (feature + 1, self.instance)
        vis.plot_1d(x, feature, self.eval, confidence=False, centering=centering, title=title)


class dICE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        instance: int = 0,
    ):
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.model_jac = model_jac
        self.data = data
        self.D = data.shape[1]
        self.instance = instance
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(dICE, self).__init__(axis_limits)

    def fit(self, features: typing.Union[int, str, list] = "all", centering: bool = True) -> None:
        features = helpers.prep_features(features, self.dim)
        for s in features:
            if centering is not False:
                self.norm_const[s] = self._compute_norm_const(s, helpers.prep_centering(centering))
            self.is_fitted[s] = True

    def _eval_unnorm(self, feature: int, x: np.ndarray, uncertainty: bool = False) -> np.ndarray:
        if uncertainty:
            raise NotImplementedError

        i = self.instance
        xi = copy.deepcopy(self.data[i, :])
        xi_repeat = np.tile(xi, (x.shape[0], 1))
        xi_repeat[:, feature] = x
        y = self.model_jac(xi_repeat)[:, feature]
        return y

    def plot(self, feature: int, centering: bool = False, nof_points: int = 30) -> None:
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "d-ICE: feature %d, instance %d" % (feature + 1, self.instance)
        vis.plot_1d(x, feature, self.eval, confidence=False, centering=centering, title=title)


# TODO: check if I can replace ICE with PDP with just one point inside
class PDPwithICE:
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        nof_instances: typing.Union[int, str] = 100,
    ):
        # assertions
        assert data.ndim == 2

        self.model = model
        self.dim = data.shape[1]
        self.nof_instances, self.indices = helpers.prep_nof_instances(nof_instances, data.shape[0])
        self.data = data[self.indices, :]
        self.axis_limits = (helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits)

        self.y_pdp = PDP(data=self.data, model=model, axis_limits=axis_limits, nof_instances="all")
        self.y_ice = [ICE(data=self.data, model=model, axis_limits=axis_limits, instance=i) for i in range(nof_instances)]

        # boolean variable for whether a FE plot has been computed
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

    def fit(self, features: typing.Union[int, str, list], centering: typing.Union[bool, str] = True):
        centering = helpers.prep_centering(centering)
        self.y_pdp.fit(features, centering)
        [self.y_ice[i].fit(features, centering) for i in range(len(self.y_ice))]

    def plot(
        self,
        feature: int,
        centering: bool = True,
        nof_points: int = 30,
        scale_x=None,
        scale_y=None,
        savefig=None
    ) -> None:
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "PDP-ICE: feature %d" % (feature + 1)
        vis.plot_pdp_ice(x, feature, self.y_pdp, self.y_ice, title=title, centering=centering,
                         scale_x=scale_x, scale_y=scale_y, savefig=savefig)


# TODO check if I can replace dICE with DerivativePDP with just one point inside
class DerivativePDPwithICE:
    def __init__(
            self,
            data: np.ndarray,
            model: callable,
            model_jac: callable,
            axis_limits: typing.Union[None, np.ndarray] = None,
            nof_instances: typing.Union[int, str] = 100,
    ):
        # assertions
        assert data.ndim == 2

        self.model = model
        self.model_jac = model_jac
        self.nof_instances, self.indices = helpers.prep_nof_instances(nof_instances, data.shape[0])
        self.data = data[self.indices, :]
        self.axis_limits = (helpers.axis_limits_from_data(self.data) if axis_limits is None else axis_limits)
        self.y_pdp = DerivativePDP(data=data, model=model, model_jac=model_jac, axis_limits=axis_limits, max_nof_instances="all")
        self.y_dice = [dICE(data=data, model=model, model_jac=model_jac, axis_limits=axis_limits, instance=i) for i in range(nof_instances)]
        self.dim = data.shape[1]

        # boolean variable for whether a FE plot has been computed
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

    def fit(self, features: typing.Union[int, str, list]):
        self.y_pdp.fit(features)
        [self.y_dice[i].fit(features) for i in range(len(self.y_dice))]


    def plot(
            self,
            feature: int,
            nof_points: int = 30,
            scale_x=None,
            scale_y=None,
            savefig=None
    ) -> None:
        x = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points)
        title = "PDP with d-ICE: feature %d" % (feature + 1)
        vis.plot_pdp_ice(x, feature, self.y_pdp, self.y_dice, title=title, centering=False,
                         scale_x=scale_x, scale_y=scale_y, savefig=savefig)


def pdp_1d_non_vectorized(model: callable,
                          data: np.ndarray,
                          x: np.ndarray,
                          feature: int,
                          uncertainty: bool,
                          is_jac: bool
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
    return (np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr)) if uncertainty else np.array(mean_pdp)


def pdp_1d_vectorized(model: callable,
                      data: np.ndarray,
                      x: np.ndarray,
                      feature: int,
                      uncertainty: bool,
                      model_returns_jac: bool
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
    if uncertainty:
        std = np.std(y, axis=1)
        sigma_pdp = std
        stderr = std / np.sqrt(data.shape[0])
        return mean_pdp, sigma_pdp, stderr
    else:
        return mean_pdp


# TODO: check this implementation
def pdp_nd_non_vectorized(model: callable,
                          data: np.ndarray,
                          x: np.ndarray,
                          features: list,
                          uncertainty: bool,
                          is_jac: bool
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
    return (np.array(mean_pdp), np.array(sigma_pdp), np.array(stderr)) if uncertainty else np.array(mean_pdp)


# TODO: check this implementation
def pdp_nd_vectorized(model: callable,
                      data: np.ndarray,
                      x: np.ndarray,
                      features: list,
                      uncertainty: bool,
                      model_returns_jac: bool
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
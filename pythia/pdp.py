import typing
import copy
import numpy as np
import pythia.visualization as vis
import pythia.helpers as helpers
import pythia.utils_integrate as utils_integrate
from pythia.fe_base import FeatureEffectBase


class PDP(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        """

        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)
        :param axis_limits: Union[None, np.ndarray(2, D)]

        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        self.D = data.shape[1]
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(PDP, self).__init__(axis_limits)

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        return {}

    def _eval_unnorm(
        self, x: np.ndarray, s: int, uncertainty: bool = False
    ) -> np.ndarray:
        """Evaluate the unnormalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """
        if uncertainty:
            raise NotImplementedError

        y = []
        for i in range(x.shape[0]):
            data1 = copy.deepcopy(self.data)
            data1[:, s] = x[i]
            y.append(np.mean(self.model(data1)))
        return np.array(y)

    def plot(self, s: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature"""
        # getters
        x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], nof_points)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self._eval_unnorm(x, s)
        vis.plot_1d(x, y, title="PDP Monte Carlo feature %d" % (s + 1))


class PDPGroundTruth(FeatureEffectBase):
    def __init__(self, func: np.ndarray, axis_limits: np.ndarray):
        self.func = func
        super(PDPGroundTruth, self).__init__(axis_limits)

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        return {}

    def _eval_unnorm(
        self, x: np.ndarray, s: int, uncertainty: bool = False
    ) -> np.ndarray:
        """

        :param x: np.ndarray (N, D)
        :returns: np.ndarray (N,)

        """
        return self.func(x)

    def plot(self, s: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature"""
        # getters
        x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], nof_points)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self._eval_unnorm(x, s)
        vis.plot_1d(x, y, title="Ground-truth PDP for feature %d" % (s + 1))


class ICE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        instance: int = 0,
    ):
        """
        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)
        :param axis_limits: Union[None, np.ndarray(2, D)]
        :param instance: int, index of the instance
        """
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

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        return {}

    def _eval_unnorm(
        self, x: np.ndarray, s: int, uncertainty: bool = False
    ) -> np.ndarray:
        """Evaluate the unnormalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """
        if uncertainty:
            raise NotImplementedError

        i = self.instance
        xi = copy.deepcopy(self.data[i, :])
        xi_repeat = np.tile(xi, (x.shape[0], 1))
        xi_repeat[:, s] = x
        y = self.model(xi_repeat)
        return y

    def plot(self, s: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature"""
        # getters
        x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], nof_points)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self._eval_unnorm(x, s)
        vis.plot_1d(
            x, y, title="ICE for Instance %d, Feature %d" % (self.instance, s + 1)
        )


class PDPwithICE:
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        """

        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)
        :param axis_limits: Union[None, np.ndarray(2, D)]

        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        self.D = data.shape[1]
        self.axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )

        self.y_pdp = None
        self.y_ice = None

    def fit(self, features: int, normalized: bool = True, nof_points: int = 30):
        axis_limits = self.axis_limits
        X = self.data
        model = self.model
        x = np.linspace(axis_limits[0, features], axis_limits[1, features], nof_points)
        self.x = x
        # pdp
        pdp = PDP(data=X, model=model, axis_limits=axis_limits)
        if normalized:
            y_pdp = pdp.eval(x=x, feature=features, uncertainty=False)
        else:
            y_pdp = pdp._eval_unnorm(x=x, s=features, uncertainty=False)
        self.y_pdp = y_pdp

        # ice curves
        y_ice = []
        for i in range(X.shape[0]):
            ice = ICE(data=X, model=model, axis_limits=axis_limits, instance=i)
            if normalized:
                y = ice.eval(x=x, feature=features, uncertainty=False)
            else:
                y = ice._eval_unnorm(x=x, s=features, uncertainty=False)
            y_ice.append(y)
        self.y_ice = np.array(y_ice)

    def plot(
        self,
        s: int,
        scale_x=None,
        scale_y=None,
        normalized: bool = True,
        nof_points: int = 30,
        savefig=None,
    ) -> None:
        """Plot the s-th feature"""
        self.fit(s, normalized, nof_points)

        axis_limits = self.axis_limits
        vis.plot_pdp_ice(s, self.x, self.y_pdp, self.y_ice, scale_x, scale_y, savefig)

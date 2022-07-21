import typing
import copy
import numpy as np
from functools import partial
import feature_effect.visualization as viz
import feature_effect.helpers as helpers
import feature_effect.utils_integrate as utils_integrate


class PDPBase:
    empty_symbol = 1e8
    def __init__(self, dim: int, axis_limits: np.ndarray) -> None:
        """
        :param dim: int
        :param axis_limits: np.ndarray (2, D)

        """
        self.D: int = dim
        self.axis_limits: np.ndarray = axis_limits
        self.z: np.ndarray = np.ones([dim])*self.empty_symbol


    def eval_unnorm(self,):
        raise NotImplementedError


    def fit(self, features: typing.Union[str, list] = "all") -> None:
        """Compute normalization constants for asked features

        :param features: list of features to compute the normalization constant
        :returns: None

        """
        features = helpers.prep_features(features)
        for s in features:
            func = partial(self.eval_unnorm, s=s)
            start = self.axis_limits[0, s]
            stop = self.axis_limits[1, s]
            self.z[s] = utils_integrate.normalization_constant_1D(func, start, stop)


    def eval(self, x: np.ndarray, s: int) -> np.ndarray:
        """Evaluate the normalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """
        assert self.axis_limits[0, s] < self.axis_limits[1, s]

        if self.z[s] == self.empty_symbol:
            self.fit(features=s)
        y = self.eval_unnorm(x, s) - self.z[s]
        return y


    def plot(self, s: int, normalized: bool = True, nof_points: int = 30) -> None:
        """Plot the s-th feature
        """
        # getters
        x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], nof_points)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self.eval_unnorm(x, s)
        vis.plot_1d_pdp(x, y, s)



class PDP(PDPBase):
    def __init__(self, data: np.ndarray, model: callable, axis_limits: typing.Union[None, np.ndarray]=None):
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
        dim = data.shape[1]
        axis_limits = helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        super(PDP, self).__init__(dim, axis_limits)


    def eval_unnorm(self, x: np.ndarray, s: int) -> np.ndarray:
        """Evaluate the unnormalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """

        y = []
        for i in range(x.shape[0]):
            data1 = copy.deepcopy(self.data)
            data1[:, s] = x[i]
            y.append(np.mean(self.model(data1)))
        return np.array(y)


class PDPNumerical(PDPBase):
    def __init__(self, p_xc: callable, model: callable, axis_limits: np.ndarray, s: int, D: int):
        """

        :param p_xc: callable (D-1,) -> ()
        :param model: callable (N,D) -> (N,)
        :param axis_limits: np.ndarray (2, D)
        :param s: int, index of feature of interest
        :param D: int, dimensionality of the problem

        """
        super(PDPNumerical, self).__init__(D, axis_limits)

        self.model = model
        self.p_xc = p_xc
        self.s = s

    def eval_unnorm(self, x: np.ndarray, s: int) -> np.ndarray:
        """Evaluate the unnormalized PDP at positions x

        :param x: np.array (N,)
        :param s: int, index of feature of interest
        :returns: np.array (N,)

        """
        if self.D == 2:
            y = []
            for i in range(x.shape[0]):
                xs = x[i]
                c = 1 if s == 0 else 0
                start = self.axis_limits[0, c]
                stop = self.axis_limits[1, c]
                res = utils_integrate.expectation_1D(xs, self.model, self.p_xc, s, start, stop)
                y.append(res[0])
            return np.array(y)
        elif self.D == 3:
            y = []
            for i in range(x.shape[0]):
                xs = x[i]
                c = 1 if s == 0 else 0
                start = self.axis_limits[0, c]
                stop = self.axis_limits[1, c]
                res = utils_integrate.expecation_2D(xs, self.model, self.p_xc, s, self.axis_limits)
                y.append(res[0])
            return np.array(y)
        else:
            raise NotImplmentedError


class PDPGroundTruth(PDPBase):
    def __init__(self, func: np.ndarray, axis_limits: np.ndarray, D: int):
        self.func = func
        super(PDPGroundTruth, self).__init__(D, axis_limits)

    def eval_unnorm(self, x: np.ndarray, s: int) -> np.ndarray:
        """

        :param x: np.ndarray (N, D)
        :returns: np.ndarray (N,)

        """
        return self.func(x)

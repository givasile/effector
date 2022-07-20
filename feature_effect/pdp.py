import typing
import numpy as np
import scipy.integrate as integrate
import feature_effect.visualization as viz
from feature_effect import utils_integrate
import copy
import matplotlib.pyplot as plt
from functools import partial
from feature_effect import helpers

class PDPBase:
    big_M = 1e8
    def __init__(self, model, dim, axis_limits):
        self.model = model
        self.D = dim
        self.axis_limits = axis_limits

        self.z = np.ones([dim])*self.big_M


    def eval_unnorm(self,):
        raise NotImplementedError


    def fit(self, features: typing.Union[str, list] = "all"):
        # compute normalization constant for each feature
        features = helpers.prep_features(features)
        for s in features:
            func = partial(self.eval_unnorm, s=s)
            start = self.axis_limits[0, s]
            stop = self.axis_limits[1, s]
            self.z[s] = utils_integrate.normalization_constant_1D(func, start, stop)


    def eval(self, x, s):
        """Evaluate the normalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """

        # assertions
        assert self.axis_limits[0, s] < self.axis_limits[1, s]

        # getters
        start = self.axis_limits[0, s]
        stop = self.axis_limits[1, s]

        # main part
        if self.z[s] == self.big_M:
            self._normalize(s, start, stop)
        y = self.eval_unnorm(x, s) - self.z[s]
        return y


    def plot(self, s, normalized=True, step=100):
        # getters
        x = np.linspace(self.axis_limits[0, s], self.axis_limits[1, s], step)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self.eval_unnorm(x, s)
        vis.plot_1d_pdp(x, y, s)



class PDP(PDPBase):
    def __init__(self, data, model, axis_limits=None):
        """

        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)

        """

        # assertions
        assert data.ndim == 2

        # setters
        self.data = data
        self.D = data.shape[1]
        dim = data.shape[1]
        axis_limits = self._set_axis_limits() if axis_limits is None else axis_limits
        super(PDP, self).__init__(model, dim, axis_limits)


    def _set_axis_limits(self):
        axis_limits = np.zeros([2, self.D])
        for d in range(self.D):
            axis_limits[0, d] = self.data[:, d].min()
            axis_limits[1, d] = self.data[:, d].max()
        return axis_limits


    def eval_unnorm(self, x, s):
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
    def __init__(self, p_xc: typing.Callable, model, axis_limits, s, D):
        """

        :param p_xc: Callable (D,) -> ()
        :param s: int, index of feature of interest
        :param D: int, dimensionality
        """
        super(PDPNumerical, self).__init__(model, D, axis_limits)

        # add assertions
        self.p_xc = p_xc
        self.s = s

    def eval_unnorm(self, x, s):
        """Evaluate the unnormalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
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
            raise NotImplementedError

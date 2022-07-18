import typing
import numpy as np
import scipy.integrate as integrate
from feature_effect import utils_integrate
import copy
import matplotlib.pyplot as plt







class PDP:
    def __init__(self, data, model):
        """

        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)

        """
        assert data.ndim == 2

        self.data = data
        self.model = model
        self.D = data.shape[1]

        self.z = {}

    @staticmethod
    def _pdp(x, data, model, s):
        """Monte-Carlo estimation of PDP.

        :param x: np.array (N, )
        :param data:  np.array (N, D)
        :param model: Callable (N, D) -> (N,)
        :param s: index of feature of interest {0, ..., D-1}
        :returns: np.array(N, )

        """
        y = []
        for i in range(x.shape[0]):
            data1 = copy.deepcopy(data)
            data1[:, s] = x[i]
            y.append(np.mean(model(data1)))
        return np.array(y)


    def _normalize(self, s):
        """Computes normalization constant of PDP of feature

        :param s: index of feature
        :returns:

        """

        def wrapper(x):
            x = np.array([[x]])
            return self.eval_unnorm(x, s)

        start = self.data[:, s].min()
        stop = self.data[:, s].max()
        y = utils_integrate.mean_value_1D(wrapper, start, stop)[0]
        self.z["feature_" + str(s)] = y


    def eval_unnorm(self, x, s):
        """Evaluate the unnormalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """
        return self._pdp(x, self.data, self.model, s)


    def eval(self, x, s):
        """Evaluate the normalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """
        if "feature_" + str(s) not in self.z.keys():
            self._normalize(s)
        return self.eval_unnorm(x, s) - self.z["feature_" + str(s)]


    def plot(self, s, normalized=True, step=100):
        min_x = np.min(self.data[:, s])
        max_x = np.max(self.data[:, s])

        x = np.linspace(min_x, max_x, step)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self.eval_unnorm(x, s)

        plt.figure()
        plt.title("PDP for feature %d" % (s+1))
        plt.plot(x, y, "b-")
        plt.show(block=False)



class PDPNumerical:
    def __init__(self, p_xc: typing.Callable, model, D, start, stop):
        """

        :param p_xc: Callable (D,) -> ()
        :param model: Callable (N, D) -> (N)
        :param D: int
        :param start: float
        :param stop: float
        :returns:

        """
        self.p_xc = p_xc
        self.model = model
        self.D = 2
        self.start = start
        self.stop = stop

        self.z = {}


    def eval(self, x, feature):
        x1 = np.linspace(start, stop, 20)
        y = self.eval_gt_integration_unnorm(x1, feature, p_xc, start, stop)
        c = np.mean(y)
        return self.eval_gt_integration_unnorm(x, feature, p_xc, start, stop) - c


    def _normalize(self, s):
        """Computes normalization constant of PDP of feature

        :param s: index of feature
        :returns:

        """

        def wrapper(x):
            x = np.array([[x]])
            return self.eval_unnorm(x, s)

        y = utils_integrate.mean_value_1D(wrapper, self.start, self.stop)[0]
        self.z["feature_" + str(s)] = y


    def eval_unnorm(self, x, s):
        y = []
        for i in range(x.shape[0]):
            xs = x[i]
            start = self.start
            stop = self.stop
            res = utils_integrate.expectation_1D(xs, self.model, self.p_xc, s, start, stop)[0]
            y.append(res)
        return np.array(y)


    def eval(self, x, s):
        """Evaluate the normalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """
        if "feature_" + str(s) not in self.z.keys():
            self._normalize(s)
        return self.eval_unnorm(x, s) - self.z["feature_" + str(s)]


    def plot(self, s, normalized=True, step=100):
        min_x = self.start
        max_x = self.stop

        x = np.linspace(min_x, max_x, step)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self.eval_unnorm(x, s)

        plt.figure()
        plt.title("PDP for feature %d" % (s+1))
        plt.plot(x, y, "b-")
        plt.show(block=False)

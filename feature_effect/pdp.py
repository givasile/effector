import typing
import numpy as np
import scipy.integrate as integrate
from feature_effect import utils_integrate
import copy
import matplotlib.pyplot as plt



class PDPBase:
    def __init__(self, model):
        self.model = model

    def _normalize(self, s, start, stop):
        """Computes normalization constant of PDP of feature

        :param s: index of feature
        :returns:

        """

        def wrapper(x):
            x = np.array([[x]])
            return self.eval_unnorm(x, s)

        y = utils_integrate.mean_value_1D(wrapper, start, stop)[0]
        self.z["feature_" + str(s)] = y


    def eval(self, x, s):
        """Evaluate the normalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """
        start = self.start["feature_" + str(s)]
        stop = self.stop["feature_" + str(s)]
        if "feature_" + str(s) not in self.z.keys():
            self._normalize(s, start, stop)
        return self.eval_unnorm(x, s) - self.z["feature_" + str(s)]


    def plot(self, s, normalized=True, step=100):
        min_x = self.start["feature_" + str(s)]
        max_x = self.stop["feature_" + str(s)]

        x = np.linspace(min_x, max_x, step)
        if normalized:
            y = self.eval(x, s)
        else:
            y = self.eval_unnorm(x, s)

        plt.figure()
        plt.title("PDP for feature %d" % (s+1))
        plt.plot(x, y, "b-")
        plt.show(block=False)


class PDP(PDPBase):
    def __init__(self, data, model):
        """

        :param data: np.array (N, D), the design matrix
        :param model: Callable (N, D) -> (N,)

        """
        super(PDP, self).__init__(model)

        assert data.ndim == 2
        self.data = data
        self.D = data.shape[1]

        self.start = {}
        self.stop = {}
        for d in range(data.shape[1]):
            self.start["feature_" + str(d)] = data[:, d].min()
            self.stop["feature_" + str(d)] = data[:, d].max()

        self.z = {}

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

        return self._pdp(x, self.data, self.model, s)



class PDPNumerical(PDPBase):
    def __init__(self, p_xc: typing.Callable, model, s, D, start, stop):
        """

        :param p_xc: Callable (D,) -> ()
        :param s: int, index of feature of interest
        :param D: int, dimensionality
        :param start: float, xs_min
        :param stop: float, xs_max

        """
        super(PDPNumerical, self).__init__(model)

        self.p_xc = p_xc
        self.D = 2

        self.start = {"feature_" + str(s) : start}
        self.stop = {"feature_" + str(s) : stop}

        self.z = {}


    def eval_unnorm(self, x, s):
        """Evaluate the unnormalized PDP at positions x

        :param x: np.array (N,)
        :param s: index of feature of interest
        :returns: np.array (N,)

        """
        y = []
        for i in range(x.shape[0]):
            xs = x[i]
            # start = self.start["feature_" + str(s)]
            # stop = self.stop["feature_" + str(s)]
            start = -np.Inf
            stop = np.Inf
            res = utils_integrate.expectation_1D(xs, self.model, self.p_xc, s, start, stop)
            print(res[1])
            y.append(res[0])
        return np.array(y)

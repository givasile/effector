import scipy.integrate as integrate
import numpy as np


def mean_value_1D(func, start, stop):
    return integrate.quad(func, start, stop)


def expectation_1D(xs, func, p_xc, s, start=-np.inf, stop=np.inf):
    """

    :param func:  (N, D) -> (N)
    :param xs: np.float
    :param p_xc: p_xc float -> [0,1]
    :param s: index of feature of interest
    :param start: left limit of integral
    :param stop: right limit of integral
    :returns:

    """
    def func_wrap(xc):
        x = np.array([xs, xc]) if s == 0 else np.array([xc, xs])
        x = np.expand_dims(x, axis=0)
        return func(x)[0] * p_xc(xc)
    return integrate.quad(func_wrap, start, stop)

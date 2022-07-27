import matplotlib.pyplot as plt
import typing
import numpy as np
from functools import partial


class ModelBase:
    def __init__(self):
        raise NotImplementedError

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Callable (x: np.ndarray (N,D)) -> np.ndarray (N)
        """
        raise NotImplementedError

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Callable (x: np.ndarray (N,D)) -> np.ndarray (N, D)
        """
        raise NotImplementedError

    def plot(self, axis_limits: np.ndarray, nof_points: int,
             X: typing.Union[None, np.ndarray] = None) -> None:
        """Works only if model is 2D
        """
        x1 = np.linspace(axis_limits[0, 0], axis_limits[1, 0], nof_points)
        x2 = np.linspace(axis_limits[0, 1], axis_limits[1, 1], nof_points)
        XX, YY = np.meshgrid(x1, x2)
        x = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.predict(x)
        ZZ = Z.reshape([30, 30])

        plt.figure()
        plt.contourf(XX, YY, ZZ, levels=100)
        if X is not None:
            plt.plot(X[:, 0], X[:, 1], "ro")
        plt.colorbar()
        plt.show(block=True)


class LinearWithInteraction(ModelBase):
    def __init__(self, b0: float, b1: float, b2: float, b3: float):
        """f(x1, x2) = b0 + b1*x1 + b2*x2 + b3*x1*x2
        """
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def predict(self, x):
        y = self.b0 + self.b1*x[:, 0] + self.b2*x[:, 1] + self.b3*x[:, 0]*x[:, 1]
        return y

    def jacobian(self, x):
        df_dx1 = self.b1 + self.b3 * x[:, 1]
        df_dx2 = self.b2 + self.b3 * x[:, 0]
        return np.stack([df_dx1, df_dx2], axis=-1)


class SquareWithInteraction(ModelBase):
    def __init__(self, b0, b1, b2, b3):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def predict(self, x):
        y = self.b0 + self.b1*x[:, 0]**2 + self.b2*x[:, 1]**2 + self.b3*x[:, 0]*x[:, 1]
        return y

    def jacobian(self, x):
        df_dx1 = 2*self.b1*x[:, 0] + self.b3 * x[:, 1]
        df_dx2 = 2*self.b2*x[:, 1] + self.b3 * x[:, 0]
        return np.stack([df_dx1, df_dx2], axis=-1)


class PiecewiseLinear(ModelBase):
    def __init__(self, params):
        self.params = params

    @staticmethod
    def _linear_part(x, a, b, x0):
        return a + b*(x[:, 0]-x0) + x[:, 0]*x[:, 1]

    def _create_cond(self, x, i, s):
        par = self.params
        if x.ndim >= 2:
            return np.logical_and(x[:, s] >= par[i]["from"], x[:, s] <= par[i]["to"])
        elif x.ndim == 1:
            return np.logical_and(x >= par[i]["from"], x <= par[i]["to"])

    def _create_func(self, i, func):
        par = self.params
        return partial(func, a=par[i]["a"], b=par[i]["b"], x0=par[i]["from"])

    def predict(self, x):
        """f(x1, x2) = a + b*x1 + x1x2
        """
        condlist = [self._create_cond(x, i, s=0) for i in range(4)]
        funclist = [self._create_func(i, self._linear_part) for i in range(4)]

        y = np.zeros(x.shape[0])
        for i, cond in enumerate(condlist):
            y[cond] = funclist[i](x[cond, :])
        return y

    def jacobian(self, x):
        condlist = [self._create_cond(x, i, s=0) for i in range(4)]

        def df_dx1(x, a, b, x0):
            return b + x[:, 1]

        def df_dx2(x, a, b, x0):
            return x[:, 0]

        funclist1 = [self._create_func(i, df_dx1) for i in range(4)]
        funclist2 = [self._create_func(i, df_dx2) for i in range(4)]
        y1 = np.zeros(x.shape[0])
        y2 = np.zeros(x.shape[0])
        for i, cond in enumerate(condlist):
            y1[cond] = funclist1[i](x[cond, :])
            y2[cond] = funclist2[i](x[cond, :])

        return np.stack([y1,y2], axis=-1)


class LinearWithInteraction3D(ModelBase):
    def __init__(self, b0, b1, b2, b3):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def predict(self, x):
        y = self.b0 + self.b1*x[:, 0] + self.b2*x[:, 1] + self.b3*x[:, 0]*x[:, 1] + 100*x[:, 2]
        return y

    def jacobian(self, x):
        df_dx1 = self.b1 + self.b3 * x[:, 1]
        df_dx2 = self.b2 + self.b3 * x[:, 0]
        df_dx3 = np.ones([x.shape[0]])*100
        return np.stack([df_dx1, df_dx2], axis=-1)

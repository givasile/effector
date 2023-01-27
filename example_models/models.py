import matplotlib.pyplot as plt
import typing
import numpy as np
from functools import partial
import scipy.stats as sps
import copy

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
        ZZ = Z.reshape([nof_points, nof_points])

        plt.figure()
        plt.contourf(XX, YY, ZZ, levels=100)
        if X is not None:
            plt.plot(X[:, 0], X[:, 1], "ro")
        plt.colorbar()
        plt.show(block=True)


class Example1(ModelBase):
    """
    f(x1, x2) = 1 - x1 - x2    , if x1 + x2 < 1
                0              , otherwise
    """
    def __init__(self):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Callable (x: np.ndarray (N,D)) -> np.ndarray (N)
        """

        # find indices
        ind1 = x[:,0] + x[:,1] < 1

        # set values
        y = np.zeros_like(x[:,0]*x[:,1])
        y[ind1] = 1 - x[ind1,0] - x[ind1,1]
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Callable (x: np.ndarray (N,D)) -> np.ndarray (N)
        """

        # find indices
        ind1 = x[:,0] + x[:,1] < 1

        # set values
        y = np.zeros_like(x)
        y[ind1] = -1
        return y


class Example2(ModelBase):
    """
                x1 + x2          , if x1 + x2 < 0.5
    f(x1, x2) = 0.5 - x1 - x2    , if x1 + x2 >= 0.5 and x1 + x2 < 1
                0                , otherwise
    """
    def __init__(self):
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Callable (x: np.ndarray (N,D)) -> np.ndarray (N)
        """

        # find indices
        ind1 = x[:,0] + x[:,1] < .5
        ind2 = np.logical_and(x[:,0] + x[:,1] >= 0.5, x[:,0] + x[:,1] < 1)

        # set values
        y = np.zeros_like(x[:,0])
        y[ind1] = x[ind1,0] + x[ind1,1]
        y[ind2] = .5 - x[ind2,0] - x[ind2,1]
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Callable (x: np.ndarray (N,D)) -> np.ndarray (N)
        """

        # find indices
        ind1 = x[:,0] + x[:,1] < .5
        ind2 = np.logical_and(x[:,0] + x[:,1] >= 0.5, x[:,0] + x[:,1] < 1)

        # set values
        y = np.zeros_like(x)
        y[ind1] = 1
        y[ind2] = -1
        return y


class Example3(ModelBase):
    """
                x1 + x2 + x1x3          , if x1 + x2 < 0.5
    f(x1, x2) = 0.5 - x1 - x2 + x1x3    , if x1 + x2 >= 0.5 and x1 + x2 < 1
                x1 + x3                 , otherwise
    """
    def __init__(self, a1=1, a2=1, a=0):
        self.a1 = a1
        self.a2 = a2
        self.a = a

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Callable (x: np.ndarray (N,D)) -> np.ndarray (N)
        """
        x = copy.deepcopy(x)
        x[:,0] = self.a1 * x[:,0]
        x[:,1] = self.a2 * x[:,1]

        # find indices
        ind1 = x[:,0] + x[:,1] < .5
        ind2 = np.logical_and(x[:,0] + x[:,1] >= 0.5, x[:,0] + x[:,1] < 1)

        # set values
        y = np.zeros_like(x[:,0])
        y[ind1] = x[ind1,0] + x[ind1,1]
        y[ind2] = .5 - x[ind2,0] - x[ind2,1]
        y += self.a * x[:, 0]*x[:, 2]
        return y

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Callable (x: np.ndarray (N,D)) -> np.ndarray (N)
        """

        x = copy.deepcopy(x)
        x[:,0] = self.a1 * x[:,0]
        x[:,1] = self.a2 * x[:,1]

        # find indices
        ind1 = x[:,0] + x[:,1] < .5
        ind2 = np.logical_and(x[:,0] + x[:,1] >= 0.5, x[:,0] + x[:,1] < 1)

        # set values
        y = np.zeros_like(x)

        # for df/dx1
        y[ind1, 0] = self.a1
        y[ind2, 0] = -self.a1
        y[:, 0] += self.a * x[:, 2]

        # for df/dx2
        y[ind1, 1] = self.a2
        y[ind2, 1] = -self.a2

        # for df/dx3
        y[:, 2] = self.a * x[:, 0]
        return y


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
        self.nof_parts = len(params)

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
        condlist = [self._create_cond(x, i, s=0) for i in range(self.nof_parts)]
        funclist = [self._create_func(i, self._linear_part) for i in range(self.nof_parts)]

        y = np.zeros(x.shape[0])
        for i, cond in enumerate(condlist):
            y[cond] = funclist[i](x[cond, :])
        return y

    def jacobian(self, x):
        condlist = [self._create_cond(x, i, s=0) for i in range(self.nof_parts)]

        def df_dx1(x, a, b, x0):
            return b + x[:, 1]

        def df_dx2(x, a, b, x0):
            return x[:, 0]

        funclist1 = [self._create_func(i, df_dx1) for i in range(self.nof_parts)]
        funclist2 = [self._create_func(i, df_dx2) for i in range(self.nof_parts)]
        y1 = np.zeros(x.shape[0])
        y2 = np.zeros(x.shape[0])
        for i, cond in enumerate(condlist):
            y1[cond] = funclist1[i](x[cond, :])
            y2[cond] = funclist2[i](x[cond, :])

        return np.stack([y1,y2], axis=-1)


class LinearWithInteraction3D(ModelBase):
    def __init__(self, b0, b1, b2, b3, b4):
        """f(x1, x2, x3) = b0 + b1*x1 + b2*x2 + b3*x1*x2 + b4*x3
        """
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4

    def predict(self, x):
        y = self.b0 + self.b1*x[:, 0] + self.b2*x[:, 1] + self.b3*x[:, 0]*x[:, 1] + self.b4*x[:, 2]
        return y

    def jacobian(self, x):
        df_dx1 = self.b1 + self.b3 * x[:, 1]
        df_dx2 = self.b2 + self.b3 * x[:, 0]
        df_dx3 = np.ones([x.shape[0]]) * self.b4
        return np.stack([df_dx1, df_dx2, df_dx3], axis=-1)


class RepidSimpleModel:
    def __init__(self, a1=0.2, a2=-8, a3=8, a4=16):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def predict(self, x):
        y = self.a1*x[:, 0] + self.a2*x[:, 1]

        cond = x[:, 0] > 0
        y[cond] += self.a3*x[cond, 1]

        cond = x[:, 2] == 0
        y[cond] += self.a4*x[cond, 1]

        eps = np.random.normal(loc=0, scale=0.1, size=y.shape[0])
        y += eps
        return y

    def jacobian(self, x):
        y = np.stack([self.a1*np.ones(x.shape[0]), self.a2*np.ones(x.shape[0]), np.zeros(x.shape[0])], axis=-1)

        cond = x[:, 0] > 0
        y[cond, 1] += self.a3

        cond = x[:, 2] == 0
        y[cond, 1] += self.a4
        return y

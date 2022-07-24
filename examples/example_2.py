import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import scipy.stats as sps
import scipy.integrate as integrate
import pytest
from functools import partial

np.random.seed(21)

class OpaqueModel:

    def __init__(self, params):
        self.params = params
        self.s2 = 0.1

    def _linear_part(self, x, a, b, x0):
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
        par = self.params
        condlist = [self._create_cond(x, i, s=0) for i in range(4)]
        funclist = [self._create_func(i, self._linear_part) for i in range(4)]

        y = np.zeros(x.shape[0])
        for i, cond in enumerate(condlist):
            y[cond] = funclist[i](x[cond, :])
        return y

    def jacobian(self, x):
        par = self.params
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


    def ale_mu(self, x):
        def mu(x, a, b, x0):
            return b

        par = self.params
        condlist = [self._create_cond(x, i, s=0) for i in range(4)]
        funclist = [self._create_func(i, mu) for i in range(4)]

        y = np.zeros(x.shape[0])
        for i, cond in enumerate(condlist):
            y[cond] = funclist[i](x[cond])
        return y

    def ale_var(self, x):
        y = np.ones(x.shape[0]) * self.s2
        return y


    def plot(self, X=None):
        x1 = np.linspace(-.5, 1.5, 30)
        x2 = np.linspace(-.5, .5, 30)
        XX, YY = np.meshgrid(x1, x2)
        x = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self._predict(x)
        ZZ = Z.reshape([30, 30])

        plt.figure()
        plt.contourf(XX, YY, ZZ, levels=100)
        if X is not None:
            plt.plot(X[:, 0], X[:, 1], "ro")
        plt.colorbar()
        plt.show(block=True)


class GenerativeDistribution:

    def __init__(self, D, x1_min, x1_max, x2_sigma):
        self.D = D
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_sigma = x2_sigma

        self.axis_limits = np.array([[0, 1], [-4*x2_sigma, 4 * x2_sigma]]).T

    def generate(self, N):
        x1 = np.concatenate((np.array([0]),
                             np.random.uniform(0., 1., size=int(N)),
                             np.array([1])))
        x2 = np.random.normal(loc=np.zeros_like(x1), scale=self.x2_sigma)
        x = np.stack((x1, x2), axis=-1)
        return x

    def pdf_x2(self, x2):
        x2_dist = sps.norm(loc=0, scale=self.x2_sigma)
        return x2_dist.pdf(x2)





params = [{"a": 0., "b":10, "from": 0., "to": .25},
          {"a": 2.5, "b":-10., "from": .25, "to": .5},
          {"a": 0., "b": 5., "from": .5, "to": .75},
          {"a": 1.25, "b":-5., "from": .75, "to": 1}]

model = OpaqueModel(params)

D = 2
x1_min = 0
x1_max = 1
x2_sigma = .1
gen_dist = GenerativeDistribution(D, x1_min, x1_max, x2_sigma)

X = gen_dist.generate(N=10000)
X_jac = model.jacobian(X)

# DALE normal
dale = fe.DALE(data=X, model=model.predict, model_jac=model.jacobian)
dale.fit(features="all", method="fixed-size", alg_params={"nof_bins":4})
xs = np.linspace(0,1, 10000)
y = dale.eval(xs, s=0)[0]
# dale.plot(s=0)

dale_bins = fe.DALEBinsGT(model.ale_mu, model.ale_var, gen_dist.axis_limits)
dale_bins.fit(compute_z = False)
xs = np.linspace(0,1, 10000)
ys = dale_bins.eval(xs, s=0)

plt.figure()
plt.plot(xs, ys, "r", label="ground truth bins")
plt.plot(xs, y, "b--", label="samples")
plt.legend()
plt.show(block=False)

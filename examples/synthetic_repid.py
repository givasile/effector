import sys, os
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pythia
import pythia.regions as regions
import matplotlib.pyplot as plt

class RepidSimpleDist:
    """
    x1 ~ U(-1, 1)
    x2 ~ U(-1, 1)
    x3 ~ Bernoulli(0.5)
    """

    def __init__(self):
        self.D = 2
        self.axis_limits = np.array([[-1, 1], [-1, 1], [0, 1]]).T

    def generate(self, N):
        x1 = np.concatenate((np.array([-1]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x2 = np.concatenate((np.array([-1]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x3 = np.random.choice([0, 1], int(N), p=[0.5, 0.5])

        x = np.stack((x1, x2, x3), axis=-1)
        return x


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


np.random.seed(21)
dist = RepidSimpleDist()
model = RepidSimpleModel()

# generate data
X = dist.generate(N=1000)
Y = model.predict(X)

nof_levels = 2
nof_splits = 10
foi = 1
foc = "all"
features, types, positions, heterogeneity = regions.find_dICE_splits(nof_levels, nof_splits, foi, foc, X, model.predict, model.jacobian, dist.axis_limits, nof_instances='all')

# exit(0)

# plot global effect
pdp_ice = pythia.pdp.PDPwithICE(X, model.predict, dist.axis_limits)
pdp_ice.fit(features="all", normalize=True)
pdp_ice.plot(feature=foi, normalized=True)
plt.show()

# plot regional effects based on split of the first level
pdp_ice = pythia.pdp.PDPwithICE(X[X[:, features[0]] == positions[0], :], model.predict, dist.axis_limits)
pdp_ice.fit(features=foi, normalize=True)
pdp_ice.plot(feature=foi, normalized=True)
plt.show()

pdp_ice = pythia.pdp.PDPwithICE(X[X[:, features[0]] != positions[0], :], model.predict, dist.axis_limits)
pdp_ice.fit(features=foi, normalize=True)
pdp_ice.plot(feature=foi, normalized=True)
plt.show()

# plot regional effects based on both splits
ind = np.logical_and(X[:, features[0]] == positions[0], X[:, features[1]] < positions[1])
pdp_ice = pythia.pdp.PDPwithICE(X[ind, :], model.predict, dist.axis_limits)
pdp_ice.fit(features=foi, normalize=True)
pdp_ice.plot(feature=foi, normalized=True)
plt.show()

ind = np.logical_and(X[:, features[0]] != positions[0], X[:, features[1]] >= positions[1])
pdp_ice = pythia.pdp.PDPwithICE(X[ind, :], model.predict, dist.axis_limits)
pdp_ice.fit(features=foi, normalize=True)
pdp_ice.plot(feature=foi, normalized=True)
plt.show()

# ind = np.logical_and(X[:, features[0]] == positions[0], X[:, features[1]] < positions[1])
# pdp_ice = pythia.pdp.PDPwithICE(X[ind, :], model.predict, dist.axis_limits)
# pdp_ice.fit(features=foi, normalize=True)
# pdp_ice.plot(feature=foi, normalized=True)
# plt.show()

# ind = np.logical_and(X[:, features[0]] != positions[0], X[:, features[1]] >= positions[1])
# pdp_ice = pythia.pdp.PDPwithICE(X[ind, :], model.predict, dist.axis_limits)
# pdp_ice.fit(features=foi, normalize=True)
# pdp_ice.plot(feature=foi, normalized=True)
# plt.show()
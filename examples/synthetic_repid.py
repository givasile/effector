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
        y = self.a1*x[:, 0] + self.a2*x[:, 1] + x[:, 2]

        cond = x[:, 0] > 0
        y[cond] += self.a3*x[cond, 1]

        cond = x[:, 2] == 0
        y[cond] += self.a4*x[cond, 1]

        eps = np.random.normal(loc=0, scale=0.1, size=y.shape[0])
        y += eps
        return y

    def jacobian(self, x):
        y = np.stack([self.a1*np.ones(x.shape[0]), self.a2*np.ones(x.shape[0]), np.ones(x.shape[0])], axis=-1)

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

# # find regions
# nof_levels = 2
# nof_splits = 10
# foi = 1
# foc = "all"
# splits = regions.find_splits(
#     nof_levels=nof_levels,
#     nof_splits=nof_splits,
#     foi=foi,
#     foc=foc,
#     cat_limit=10,
#     data=X,
#     model=model.predict,
#     model_jac=model.jacobian,
#     criterion="ale"
# )


# plot global effect
rhale = pythia.RHALE(data=X, model=model.predict, model_jac=model.jacobian)
fixed = pythia.binning_methods.Fixed(nof_bins=10)
dp = pythia.binning_methods.DynamicProgramming(max_nof_bins=10)
greedy = pythia.binning_methods.Greedy(init_nof_bins=100, min_points_per_bin=100, discount=0.5)
rhale.fit(features="all", binning_method=greedy)
rhale.plot(feature=0, confidence_interval="std")
rhale.plot(feature=1, confidence_interval="std")
rhale.plot(feature=2, confidence_interval="std")

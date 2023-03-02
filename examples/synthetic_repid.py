import numpy as np
import pythia


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
X = dist.generate(N=10000)
Y = model.predict(X)

# PDP
pdp = pythia.PDP(X, model.predict, dist.axis_limits)
pdp.fit(features="all", normalize=True)
pdp.plot(feature=0, normalized=True, confidence_interval=None, nof_points=200)
pdp.plot(feature=1, normalized=True, confidence_interval=None, nof_points=200)
pdp.plot(feature=2, normalized=True, confidence_interval=None, nof_points=200)

dale = pythia.DALE(X, model.predict, model.jacobian, dist.axis_limits)
binning_method = pythia.binning_methods.Fixed(nof_bins=100)
dale.fit(features="all", binning_method=binning_method, normalize=True)
dale.plot(feature=0, confidence_interval="std")
dale.plot(feature=1, confidence_interval=None)
dale.plot(feature=2, confidence_interval=None)

# for feat in [0, 1, 2]:
#     pdp.plot(feature=feat, normalized=True, confidence_interval="std", nof_points=50)

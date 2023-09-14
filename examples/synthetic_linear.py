import numpy as np
import pythia


class GeneratingDist:
    """
    x1 ~ U(x1_min, x1_max)
    x2 ~ N(x1, x2_sigma)
    x3 ~ N(0, x3_sigma)
    """
    def __init__(self, x1_min, x1_max, x2_sigma, x3_sigma):
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_sigma = x2_sigma
        self.x3_sigma = x3_sigma

        self.axis_limits = np.array([[0, 1],
                                     [-4*x2_sigma, 1 + 4*x2_sigma],
                                     [-4*x3_sigma, 4*x3_sigma]]).T

    def generate(self, nof_samples):
        x1 = np.concatenate((np.array([self.x1_min]),
                             np.random.uniform(self.x1_min, self.x1_max, size=int(nof_samples - 2)),
                             np.array([self.x1_max])))
        x2 = np.random.normal(loc=x1, scale=self.x2_sigma)
        x3 = np.random.normal(loc=np.zeros_like(x1), scale=self.x3_sigma)
        x = np.stack((x1, x2, x3), axis=-1)
        return x


class Model:
    def __init__(self, b0, b1, b2, b12, b3):
        """f(x1, x2, x3) = b0 + b1*x1 + b2*x2 + b12*x1*x2 + b3*x3
        """
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b12 = b12
        self.b3 = b3

    def predict(self, x):
        y = self.b0 + self.b1*x[:, 0] + self.b2*x[:, 1] + self.b12*x[:, 0]*x[:, 1] + self.b3*x[:, 2]
        return y

    def jacobian(self, x):
        df_dx1 = self.b1 + self.b12 * x[:, 1]
        df_dx2 = self.b2 + self.b12 * x[:, 0]
        df_dx3 = np.ones([x.shape[0]]) * self.b3
        return np.stack([df_dx1, df_dx2, df_dx3], axis=-1)


np.random.seed(21)
dist = GeneratingDist(x1_min=0, x1_max=1, x2_sigma=.01, x3_sigma=1.)
model = Model(b0=0, b1=7, b2=-3, b12=0, b3=4)

# generate data
X = dist.generate(nof_samples=10000)
Y = model.predict(X)
prediction = model.predict
prediction_jac = model.jacobian

# simplest use
pdp = pythia.PDP(X, prediction)
pdp.plot(feature=0)


# PDP
pdp = pythia.PDP(X, prediction, dist.axis_limits)
pdp.fit(features="all", centering=True)
for feat in [0, 1, 2]:
    pdp.plot(feature=feat, centering=True, uncertainty="std", nof_points=50)

# # dPDP
# dpdp = pythia.pdp.dPDP(X, model.predict, model.jacobian, dist.axis_limits)
# dpdp.fit(features="all", normalize="zero_start")
# for feat in [0, 1, 2]:
#     dpdp.plot(feature=feat, normalized=True, confidence_interval="std", nof_points=50)


#
# # ICE
# ice = pythia.pdp.ICE(X, model.predict, dist.axis_limits, 0)
# ice.plot(feature=0)
#
# ice = pythia.pdp.ICE(X, model.predict, dist.axis_limits, 10)
# ice.plot(feature=0, normalized=False)


# d-ICE
# dice = pythia.pdp.dICE(X, model.predict, model.jacobian, dist.axis_limits, 0)
# dice.plot(feature=0)


# pdp_ice = pythia.pdp.PDPwithICE(X, model.predict, dist.axis_limits)
# pdp_ice.fit("all", "zero_start")
# pdp_ice.plot(0, normalized=True)
#
# pdp_dice = pythia.pdp.PDPwithdICE(X, model.predict, model.jacobian, dist.axis_limits)
# pdp_dice.fit("all", "zero_start")
# pdp_dice.plot(0, normalized=False)


# DALE
dale = pythia.RHALE(X, model.predict, model.jacobian, dist.axis_limits)
binning_method = pythia.binning_methods.Fixed(nof_bins=100)
dale.fit(features="all", binning_method=binning_method)

# plot
dale.plot(feature=0, uncertainty="std")
dale.plot(feature=1, uncertainty="std")
dale.plot(feature=2, uncertainty="std")
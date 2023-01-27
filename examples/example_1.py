import numpy as np
import pythia
import example_models.distributions as dist
import example_models.models as models

np.random.seed(21)

# define distribution and model
dist = dist.Correlated_3D_1(D=3, x1_min=0, x1_max=1, x2_sigma=.01, x3_sigma=1.)
model = models.LinearWithInteraction3D(b0=0, b1=1, b2=1, b3=30, b4=10)

# generate data
X = dist.generate(N=1000)
Y = model.predict(X)

# PDP
pdp = pythia.PDP(X, model.predict, dist.axis_limits)
pdp.fit(features="all", normalize=True)
for feat in [0, 1, 2]:
    pdp.plot(feature=feat, normalized=True, confidence_interval="std", nof_points=50)

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


pdp_ice = pythia.pdp.PDPwithICE(X, model.predict, dist.axis_limits)
pdp_ice.fit("all", "zero_start")
pdp_ice.plot(0, normalized=True)

pdp_dice = pythia.pdp.PDPwithdICE(X, model.predict, model.jacobian, dist.axis_limits)
pdp_dice.fit("all", "zero_start")
pdp_dice.plot(0, normalized=False)


# # DALE
# dale = pythia.DALE(X, model.predict, model.jacobian, dist.axis_limits)
# binning_method = pythia.binning_methods.Fixed(nof_bins=100)
# dale.fit(features="all", binning_method=binning_method)
#
# # plot
# dale.plot(feature=0, confidence_interval="std")
# dale.plot(feature=1, confidence_interval="std")
# dale.plot(feature=2, confidence_interval="std")
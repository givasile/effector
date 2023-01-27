import numpy as np
import pythia
import example_models.distributions as dist
import example_models.models as models

np.random.seed(21)

# define distribution and model
dist = dist.RepidSimpleDist()
model = models.RepidSimpleModel()

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

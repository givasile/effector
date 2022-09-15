import examples.utils as utils
path = utils.add_parent_path()
import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import example_models.distributions as dist
import example_models.models as models


# gen dist
sigma = .5
gen_dist = dist.Correlated_3D_1(D=3, x1_min=0, x1_max=1, x2_sigma=0, x3_sigma=.5)
X = gen_dist.generate(N=500)
axis_limits = gen_dist.axis_limits

# model
model = models.Example3()
# model.plot(axis_limits=axis_limits, nof_points=30)

# ALE with equal bin sizes
dale = fe.DALE(data=X,
               model=model.predict,
               model_jac=model.jacobian,
               axis_limits=axis_limits)

alg_params = {"bin_method" : "fixed", "nof_bins" : 20, "min_points_per_bin": 5}
dale.fit(features=[0], alg_params=alg_params)
y, var, stderr = dale.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
                           s=0,
                           uncertainty=True)
dale.plot(s=0, error="std")


# ALE with variable bin sizes
dale = fe.DALE(data=X,
               model=model.predict,
               model_jac=model.jacobian,
               axis_limits=axis_limits)

alg_params = {"bin_method" : "dp", "max_nof_bins" : 20, "min_points_per_bin": 10}
dale.fit(features=[0], alg_params=alg_params)
y, var, stderr = dale.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
                           s=0,
                           uncertainty=True)
dale.plot(s=0, error="std")


# # PDP
# pdp = fe.PDP(data=X,
#              model=model.predict,
#              axis_limits=axis_limits)
# y = pdp.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
#              s=0,
#              uncertainty=False)
# pdp.plot(s=0)


# PDP with ICE
pdp_ice = fe.PDPwithICE(data=X,
                        model=model.predict,
                        axis_limits=axis_limits)
pdp_ice.plot(s=0)

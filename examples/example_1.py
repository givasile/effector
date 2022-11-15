import examples.utils as utils
import numpy as np
import pythia as fe
import example_models.distributions as dist
import example_models.models as models

savefig = False
np.random.seed(21)

# gen dist
gen_dist = dist.Correlated_3D_1(D=3, x1_min=0, x1_max=1, x2_sigma=0, x3_sigma=0.5)
X = gen_dist.generate(N=200)
axis_limits = gen_dist.axis_limits

# model
model = models.Example3(a1=1, a2=1, a=0)

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
# if savefig:
#     pathname = os.path.join(path, "example_1", "dale_fixed_bins.pdf")
#     dale.plot(s=0, error="std", savefig=pathname)
# else:
#     dale.plot(s=0, error="std")


# ALE with variable bin sizes
dale = fe.DALE(data=X,
               model=model.predict,
               model_jac=model.jacobian,
               axis_limits=axis_limits)

alg_params = {"bin_method" : "dp", "max_nof_bins" : 20, "min_points_per_bin": 10}
dale.fit(features="all", alg_params=alg_params)
y, var, stderr = dale.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
                           s=0,
                           uncertainty=True)
if savefig:
    pathname = os.path.join(path, "example_1", "dale_feat_0.pdf")
    dale.plot(s=0, error="std", savefig=pathname)
    pathname = os.path.join(path, "example_1", "dale_feat_1.pdf")
    dale.plot(s=1, error="std", savefig=pathname)
    pathname = os.path.join(path, "example_1", "dale_feat_2.pdf")
    dale.plot(s=2, error="std", savefig=pathname)
else:
    dale.plot(s=0, error="std")
    dale.plot(s=1, error="std")
    dale.plot(s=2, error="std")

# PDP with ICE
pdp_ice = fe.PDPwithICE(data=X,
                        model=model.predict,
                        axis_limits=axis_limits)
if savefig:
    pathname = os.path.join(path, "example_1", "pdp_ice_feat_0.pdf")
    pdp_ice.plot(s=0, normalized=True, nof_points=300, savefig=pathname)
    pathname = os.path.join(path, "example_1", "pdp_ice_feat_1.pdf")
    pdp_ice.plot(s=1, normalized=True, nof_points=300, savefig=pathname)
    pathname = os.path.join(path, "example_1", "pdp_ice_feat_2.pdf")
    pdp_ice.plot(s=2, normalized=True, nof_points=300, savefig=pathname)
else:
    pdp_ice.plot(s=0, normalized=True, nof_points=300)
    pdp_ice.plot(s=1, normalized=True, nof_points=300)
    pdp_ice.plot(s=2, normalized=True, nof_points=300)

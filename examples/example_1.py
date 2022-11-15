import numpy as np
import pythia
import example_models.distributions as dist
import example_models.models as models

# creates 4 points
x1 = np.array([0., 0.2, 0.8, 1.])
x2 = np.random.normal(loc=0, scale=0.01, size=(int(x1.shape[0])))
x = np.stack((x1, x2), axis=-1)

y_grad = np.array([[10, 10, -10, -10],
                   [10, 10, -10, -10]]).T
axis_limits = np.stack([x.min(axis=0), x.max(axis=0)])


def mu(x):
    y = np.ones_like(x) * 10
    y[x > 0.5] = -10
    return y


def var(x):
    return np.zeros_like(x)


equi = pythia.bin_estimation.Fixed(x, y_grad, feature=0, axis_limits=axis_limits)
equi.solve(min_points=2, K=3, enforce_bin_creation=True)
print(equi.limits)
print(equi.limits_valid)
equi.plot()

equi = pythia.bin_estimation.FixedGT(mu, var, feature=0, axis_limits=axis_limits)
equi.solve(min_points=2, K=3, enforce_bin_creation=True)
print(equi.limits)
print(equi.limits_valid)
equi.plot()


dp = pythia.bin_estimation.DP(x, y_grad, feature=0, axis_limits=axis_limits)
dp.solve(min_points=2, K=20)
print(dp.limits)
print(dp.limits_valid)
dp.plot()


dp = pythia.bin_estimation.DPGT(mu, var, feature=0, axis_limits=axis_limits)
dp.solve(min_points=2, K=20)
print(dp.limits)
print(dp.limits_valid)
dp.plot()


greedy = pythia.bin_estimation.Greedy(x, y_grad, feature=0, axis_limits=axis_limits)
greedy.solve(min_points=2, n_max=20)
print(greedy.limits)
print(greedy.limits_valid)
greedy.plot()

# savefig = False
# np.random.seed(21)
#
# # gen dist
# gen_dist = dist.Correlated_3D_1(D=3, x1_min=0, x1_max=1, x2_sigma=0, x3_sigma=0.5)
# axis_limits = gen_dist.axis_limits
#
# # model
# model = models.Example3(a1=1, a2=1, a=0)
# model_pred = model.predict
# model_jac = model.jacobian
#
# # generate data
# data = gen_dist.generate(N=200)
#
# # DALE with equal bin sizes
# dale = pythia.DALE(data, model_pred, model_jac, axis_limits)
# alg_params = {"bin_method": "fixed", "nof_bins": 20, "min_points_per_bin": 5}
# alg_params = {"bin_method": "greedy", "max_nof_bins": 20, "min_points_per_bin": 5}
# alg_params = {"bin_method": "dp", "max_nof_bins": 20, "min_points_per_bin": 5}
# dale.fit(alg_params=alg_params)
# y, var, stderr = dale.eval(x=np.linspace(axis_limits[0, 0], axis_limits[1, 0], 100),
#                            s=0,
#                            uncertainty=True)
# dale.plot(s=0, error="std")

# dale = pythia.DALE(data, model, model_jac, axis_limits)

# # ALE with variable bin sizes
# dale = fe.DALE(data=X,
#                model=model.predict,
#                model_jac=model.jacobian,
#                axis_limits=axis_limits)
#
# alg_params = {"bin_method" : "dp", "max_nof_bins" : 20, "min_points_per_bin": 10}
# dale.fit(features="all", alg_params=alg_params)
# y, var, stderr = dale.eval(x=np.linspace(axis_limits[0,0], axis_limits[1,0], 100),
#                            s=0,
#                            uncertainty=True)
# if savefig:
#     pathname = os.path.join(path, "example_1", "dale_feat_0.pdf")
#     dale.plot(s=0, error="std", savefig=pathname)
#     pathname = os.path.join(path, "example_1", "dale_feat_1.pdf")
#     dale.plot(s=1, error="std", savefig=pathname)
#     pathname = os.path.join(path, "example_1", "dale_feat_2.pdf")
#     dale.plot(s=2, error="std", savefig=pathname)
# else:
#     dale.plot(s=0, error="std")
#     dale.plot(s=1, error="std")
#     dale.plot(s=2, error="std")
#
# # PDP with ICE
# pdp_ice = fe.PDPwithICE(data=X,
#                         model=model.predict,
#                         axis_limits=axis_limits)
# if savefig:
#     pathname = os.path.join(path, "example_1", "pdp_ice_feat_0.pdf")
#     pdp_ice.plot(s=0, normalized=True, nof_points=300, savefig=pathname)
#     pathname = os.path.join(path, "example_1", "pdp_ice_feat_1.pdf")
#     pdp_ice.plot(s=1, normalized=True, nof_points=300, savefig=pathname)
#     pathname = os.path.join(path, "example_1", "pdp_ice_feat_2.pdf")
#     pdp_ice.plot(s=2, normalized=True, nof_points=300, savefig=pathname)
# else:
#     pdp_ice.plot(s=0, normalized=True, nof_points=300)
#     pdp_ice.plot(s=1, normalized=True, nof_points=300)
#     pdp_ice.plot(s=2, normalized=True, nof_points=300)

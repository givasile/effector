"""
if feature effect is made from variable size piecewise-linear functions:
- fixed-size approach:
  - (a) if big-size bins, fails in high-resolution effects
  - (b) if small-size bins, suffers from instabilities in low-resolution effects

This examples shows that the effect of (b) is bigger if data points are not uniformly distributed

variable-size finds the bins correctly.
"""

import numpy as np
import examples.example_utils as utils
import feature_effect as fe
import matplotlib.pyplot as plt

# define piecewise linear function
def create_model_params():
    def find_a(params, x_start):
        params[0]["a"] = x_start
        for i, param in enumerate(params):
            if i < len(params) - 1:
                a_next = param["a"] + (param["to"] - param["from"]) * param["b"]
                params[i + 1]["a"] = a_next

    params = [{"b":10, "from": 0., "to": 20.},
              {"b":-10, "from": 20., "to": 40.},
              {"b":0. , "from": 40., "to": 100}]

    x_start = -25.
    find_a(params, x_start)
    return params


# generate samples
def generate_samples(N):
    eps = 1e-05
    x1 = np.random.uniform(0., 20, size=int(N / 3))
    x2 = np.random.uniform(20, 40, size=int(N / 3))
    x3 = np.random.uniform(60, 80, size=int(N / 10))
    x = np.expand_dims(np.concatenate((np.array([0.0]),
                                       x1,
                                       x2,
                                       np.array([41.0, 44., 50., 53, 57]),
                                       x3,
                                       np.array([87, 92]),
                                       np.array([100-eps]))), axis=-1)
    return x

example_dir = "./examples/bullet_3/"

# experiment parameters
N = 500
noise_level = 4.
K_max_fixed = 40
K_max_var = 40
min_points_per_bin = 10

# set seed
seed = 4854894
np.random.seed(seed)

# define functions
model_params = create_model_params()
model_jac = utils.create_noisy_jacobian(model_params, noise_level, seed)

# generate data and data effect
data = np.sort(generate_samples(N=N), axis=0)
model = utils.create_model(model_params, data)
y = model(data)
data_effect = model_jac(data)

# check bin creation
bin_est = fe.BinEstimatorDP(data, data_effect, feature=0, K=45)
limits = bin_est.solve_dp(min_points=min_points_per_bin)
clusters = np.squeeze(np.digitize(data, limits))

plt.figure()
plt.title("Local effect")
plt.plot(data, data_effect, "bo")
# plt.vlines(limits, ymin=np.min(data_effect), ymax=np.max(data_effect))
plt.savefig(example_dir + "im_2.png")
plt.show(block=False)

# plot data effects and gt effect
utils.plot_gt_effect(data, y, savefig=example_dir + "im_1.png")

# compute loss and mse for many different K
dale_fixed = utils.fit_multiple_K(data, model, model_jac, K_max_fixed, min_points_per_bin, method="fixed-size")
utils.plot_loss(dale_fixed, savefig=example_dir + "im_3.png")

dale0 = fe.DALE(data, model, model_jac)
dale0.fit(method="fixed-size", alg_params={"nof_bins": 4, "min_points_per_bin": min_points_per_bin})
dale0.plot(gt=model, savefig=example_dir + "im_4.png")

# check bin creation
bin_est = fe.bin_estimation.BinEstimatorDP(data, data_effect, feature=0, K=40)
limits = bin_est.solve_dp(min_points=min_points_per_bin)
clusters = np.squeeze(np.digitize(data, limits))

plt.figure()
plt.title("Local effect")
plt.plot(data, data_effect, "bo")
plt.vlines(limits, ymin=np.min(data_effect), ymax=np.max(data_effect))
plt.savefig(example_dir + "im_5.png")
plt.show(block=False)


dale_variable = utils.fit_multiple_K(data, model, model_jac, K_max_var, min_points_per_bin, method="variable-size")
utils.plot_loss(dale_variable, savefig=example_dir + "im_6.png")

dale1 = fe.DALE(data, model, model_jac)
dale1.fit(method="variable-size", alg_params={"max_nof_bins": 10, "min_points_per_bin": min_points_per_bin})
dale1.plot(gt=model, savefig=example_dir + "im_7.png")

# plot loss
# utils.plot_combined_loss(dale_fixed, dale_variable)
#
# # plot mae
# utils.plot_mse(dale_fixed, model)
# utils.plot_mse(dale_variable, model)
# utils.plot_combined_mse(dale_fixed, dale_variable, model)
# # plot best fixed solution
# best_fixed = np.nanargmin([dale.dale_params["feature_0"]["loss"] for dale in dale_fixed])
# dale_fixed[best_fixed].plot(s=0, gt=model, gt_bins=utils.create_gt_bins(model_params), block=False)
#
# # plot best variable size solution
# best_var = np.nanargmin([dale.dale_params["feature_0"]["loss"] for dale in dale_variable])
# dale_variable[best_var].plot(s=0, gt=model, gt_bins=utils.create_gt_bins(model_params), block=False)

"""Goal: show that if data non-uniformly distributed, you have to find the K that matches the piecewise linear regions.
This is not feasible in all cases. Piecewise linear regions, under normal circumstances, are have not equal length.
"""

import numpy as np
import examples.example_utils as utils

# define piecewise linear function
def f_params():
    def find_a(params, x_start):
        params[0]["a"] = x_start
        for i, param in enumerate(params):
            if i < len(params) - 1:
                a_next = param["a"] + (param["to"] - param["from"]) * param["b"]
                params[i + 1]["a"] = a_next

    params = [{"b":10, "from": 0., "to": 5.},
              {"b":-10, "from": 5., "to": 10.},
              {"b":0. , "from": 10., "to": 100}]

    x_start = -25.
    find_a(params, x_start)
    return params


# generate samples
def generate_samples(N):
    eps = 1e-05
    x1 = np.random.uniform(0., 5, size=int(N / 3))
    x2 = np.random.uniform(5, 10, size=int(N / 3))
    x3 = np.random.uniform(10, 99, size=int(N / 3))
    x = np.expand_dims(np.concatenate((np.array([0.0]), x1, x2, x3, np.array([100-eps]))), axis=-1)
    return x

# parameters
N = 5000
noise_level = 3.
K_max_fixed = 50
K_max_var = 30
min_points_per_bin = 10

# init functions
seed = 4834545
np.random.seed(seed)

model = utils.create_f1_center(f_params)
model_jac = utils.create_data_effect(f_params, noise_level, seed)
data = generate_samples(N=N)
y = model(data)
data_effect = model_jac(data)

# plot data effects and gt effect
utils.plot_gt_effect(data, y)
utils.plot_data_effect(data, data_effect)

# compute loss and mse for many different K
k_list_fixed, mse_fixed, loss_fixed, dale_fixed = utils.count_loss_mse(K_max_fixed, model, data, model, model_jac,
                                                                       min_points_per_bin, method="fixed-size")
k_list_var, mse_var, loss_var, dale_var = utils.count_loss_mse(K_max_var, model, data, model, model_jac,
                                                               min_points_per_bin, method="variable-size")

# plot
utils.plot_mse(k_list_var, mse_var, mse_fixed)
utils.plot_loss(k_list_var, loss_var, loss_fixed)

# plot best fixed solution
best_fixed = np.nanargmin(loss_fixed)
dale_fixed[best_fixed].plot(s=0,
                            gt=model,
                            gt_bins=utils.create_gt_bins(f_params),
                            block=False)

# plot best variable size solution
best_var = np.nanargmin(loss_var)
dale_var[best_var].plot(s=0,
                        gt=model,
                        gt_bins=utils.create_gt_bins(f_params),
                        block=False)

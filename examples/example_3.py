"""Goal: Check clustering
"""

import numpy as np
import examples.example_utils as utils
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe


########### DEFINITIONS #############
def f_params():
    def find_a(params, x_start):
        params[0]["a"] = x_start
        for i, param in enumerate(params):
            if i < len(params) - 1:
                a_next = param["a"] + (param["to"] - param["from"]) * param["b"]
                params[i + 1]["a"] = a_next

    params = [{"b":1000, "from": 0., "to": 1.},
              {"b":-1000, "from": 3., "to": 6.},
              {"b":1000, "from": 6., "to": 9.},
              {"b":-20. , "from": 9., "to": 70},
              {"b":20, "from": 70, "to": 100}]
    x_start = -1
    find_a(params, x_start)
    return params


def generate_samples(N):
    eps = 1e-05
    x1 = np.random.uniform(0., 9, size=int(N / 3))
    x2 = np.random.uniform(9, 60, size=int(N / 3))
    x3 = np.random.uniform(75, 90, size=int(N/10))
    x = np.expand_dims(np.concatenate((np.array([0.0]), x1, x2, x3, np.array([100-eps]))), axis=-1)
    return x


# parameters
N = 5000
noise_level = 3.
K_max_fixed = 50
K_max_variable = 30

# init functions
seed = 4834545
np.random.seed(seed)

f1_center = utils.create_model(f_params)
f1 = utils.create_f1(f_params)
f1_jac = utils.create_noisy_jacobian(f_params, noise_level, seed)
x = generate_samples(N=N)
y = f1_center(x)
data = x
data_effect = f1_jac(x)

# show gt effect
utils.plot_gt_effect(data, y)

# show data effect
utils.plot_data_effect(data, data_effect)


utils.plot_gt_effect(data, f(data))
utils.plot_data_effect(data, data_effect)
# utils.count_loss_mse_fixed(K_max_fixed, f, data, f, f_jac)

for k in np.arange(1, 11):
    dale0 = fe.DALE(data=data, model=f, model_jac=f_jac)
    dale0.fit(features=[0], k=k)
    dale0.plot(s=0, block=False)
    print(find_loss(dale0, min_points_per_bin))

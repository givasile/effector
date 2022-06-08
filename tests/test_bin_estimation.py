import pytest
import numpy as np
import examples.example_utils as example_utils
import feature_effect.utils as utils
import matplotlib.pyplot as plt
import feature_effect as fe

def gen_model_params():
    params = [{"b":0.3, "from": 0., "to": 10.},
              {"b":7., "from": 10, "to": 20},
              {"b":-1.5, "from": 20, "to": 30},
              {"b":0., "from": 30, "to": 40},
              {"b":-5., "from": 40, "to": 50},
              {"b":0.3, "from": 50, "to": 60},
              {"b":7., "from": 60, "to": 70},
              {"b":-1.5, "from": 70, "to": 80},
              {"b":0., "from": 80, "to": 90},
              {"b":-5., "from": 90, "to": 100}]
    x_start = -1
    example_utils.find_a(params, x_start)
    return params


def generate_samples(N):
    eps = 1e-05
    x = np.random.uniform(0, 100-eps, size=int(N))
    x = np.expand_dims(np.concatenate((np.array([0.0]), x, np.array([100. - eps]))), axis=-1)
    return x


# parameters
N = 10000
noise_level = 4.
K_max_fixed = 40
K_max_var = 40
min_points_per_bin = 10

# init functions
seed = 4834545
np.random.seed(seed)

# define functions
model_params = gen_model_params()
model_jac = example_utils.create_noisy_jacobian(model_params, noise_level, seed)

# generate data and data effect
data = np.sort(generate_samples(N=N), axis=0)
model = example_utils.create_model(model_params, data)
y = model(data)
data_effect = model_jac(data)


# example_utils.plot_data_effect(data, data_effect)

# check bin creation
bin_est = fe.bin_estimation.BinEstimatorDP(data, data_effect, feature=0, K=75)
limits = bin_est.solve_dp(min_points=10)
clusters = np.squeeze(np.digitize(data, limits))

plt.figure()
plt.title("Dynamic programming")
plt.plot(data, data_effect, "bo")
plt.vlines(limits, ymin=np.min(data_effect), ymax=np.max(data_effect))
plt.show(block=False)

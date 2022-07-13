import pytest
import numpy as np
import examples.example_utils as example_utils
import feature_effect.utils as utils
import matplotlib.pyplot as plt
import feature_effect as fe

def gen_model_params():
    params = [{"b":10, "from": 0., "to": .3},
              {"b":-10., "from": .3, "to": .6},
              {"b":0., "from": .6, "to": .8},
              {"b":-3., "from": .8, "to": 1}]
    x_start = 0.
    example_utils.find_a(params, x_start)
    return params


def generate_samples(N):
    eps = 1e-05
    x = np.random.uniform(0, 1-eps, size=int(N))
    x = np.concatenate((np.array([0.0]), x, np.array([1. - eps])))
    x = np.expand_dims(x, axis=-1)
    return x

# make process deterministic
seed = 4834545
np.random.seed(seed)

# create data, data_effects
N = 10000
noise_level = 2
model_jac = example_utils.create_noisy_jacobian(gen_model_params(),noise_level, seed)
data = generate_samples(N=N)
data_effect = model_jac(data)

# Bin creation - dynamic programming
bin_1 = fe.bin_estimation.BinEstimatorDP(data, data_effect, feature=0, K=30)
bin_1.plot()
bin_1.solve_dp(min_points=10)
bin_1.plot(block=False)

# bin creation - with merging
bin_2 = fe.bin_estimation.BinEstimatorGreedy(data, data_effect)
bin_2.solve(s=0, tau=10, K=100)
bin_2.plot(block=False)
# print(limits)

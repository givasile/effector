import pytest
import numpy as np
from scipy.stats import norm
import feature_effect.utils as utils
import matplotlib.pyplot as plt
import feature_effect as fe
import numdifftools

# piecewise linear function (a + bx) parameters
params = [{"a": 0., "b":10, "from": 0., "to": .3},
          {"a": 3., "b":-10., "from": .3, "to": .6},
          {"a": 0., "b": 5., "from": .6, "to": .8},
          {"a": 1., "b":-5., "from": .8, "to": 1}]


def model(x):
    ind_1 = np.logical_and(x[:, 0] >= params[0]["from"], x[:, 0] < params[0]["to"])
    ind_2 = np.logical_and(x[:, 0] >= params[1]["from"], x[:, 0] < params[1]["to"])
    ind_3 = np.logical_and(x[:, 0] >= params[2]["from"], x[:, 0] < params[2]["to"])
    ind_4 = np.logical_and(x[:, 0] >= params[3]["from"], x[:, 0] <= params[3]["to"])

    y = params[0]["b"] * x[:, 0] + params[0]["a"] + x[:, 0] * x[:, 1]
    y[ind_2] = params[1]["b"] * (x[ind_2, 0] - params[1]["from"]) + params[1]["a"] + x[ind_2,0] * x[ind_2, 1]
    y[ind_3] = params[2]["b"] * (x[ind_3, 0] - params[2]["from"]) + params[2]["a"] + x[ind_3,0] * x[ind_3, 1]
    y[ind_4] = params[3]["b"] * (x[ind_4, 0] - params[3]["from"]) + params[3]["a"] + x[ind_4,0] * x[ind_4, 1]

    return y


def model_jac(x):
    ind_1 = np.logical_and(x[:, 0] >= params[0]["from"], x[:, 0] < params[0]["to"])
    ind_2 = np.logical_and(x[:, 0] >= params[1]["from"], x[:, 0] < params[1]["to"])
    ind_3 = np.logical_and(x[:, 0] >= params[2]["from"], x[:, 0] < params[2]["to"])
    ind_4 = np.logical_and(x[:, 0] >= params[3]["from"], x[:, 0] <= params[3]["to"])

    y = np.ones_like(x)
    y[ind_1, 0] = params[0]["b"] + x[ind_1, 1]
    y[ind_2, 0] = params[1]["b"] + x[ind_2, 1]
    y[ind_3, 0] = params[2]["b"] + x[ind_3, 1]
    y[ind_4, 0] = params[3]["b"] + x[ind_4, 1]

    y[ind_1, 1] = x[ind_1, 0]
    y[ind_2, 1] = x[ind_2, 0]
    y[ind_3, 1] = x[ind_3, 0]
    y[ind_4, 1] = x[ind_4, 0]
    return y


def generate_samples(N, noise_level):
    eps = 1e-03
    stop = 5
    x1 = np.concatenate((np.array([0. + eps]),
                         np.random.uniform(0., 1., size=int(N)),
                         np.array([1. - eps]))
                        )

    x2 = np.random.normal(loc=0, scale=noise_level, size=(int(x1.shape[0])))
    x = np.stack((x1, x2), axis=-1)
    return x


# generate data
N = 100
noise_level = 0.5
x = generate_samples(N, noise_level)
y = model(x)
y_grad = model_jac(x)


# Bin creation - dynamic programming
bin_1 = fe.bin_estimation.BinEstimatorDP(x, y_grad, feature=0)
bin_1.solve(min_points=5, K=30)
bin_1.plot(block=False)

# Bin creation - with merging
bin_2 = fe.bin_estimation.BinEstimatorGreedy(x, y_grad, feature=0)
bin_2.solve(min_points=5, K=100)
bin_2.plot(block=False)
bin_2.compute_statistics()

import numpy as np
import pythia

np.random.seed(21)

# creates 4 points
x1 = np.array([0., 0.2, 0.4, 0.8, 1.])
x2 = np.random.normal(loc=0, scale=0.01, size=(int(x1.shape[0])))
x = np.stack((x1, x2), axis=-1)

y_grad = np.array([[10, 11, 10.5, -10, -11],
                   [10, 10, 10, -10, -10]]).T
axis_limits = np.stack([x.min(axis=0), x.max(axis=0)])


def mu(x):
    y = np.ones_like(x) * 10
    y[x > 0.5] = -10
    return y


def var(x):
    return np.zeros_like(x)

# Equi-sized
equi = pythia.bin_estimation.Fixed(x, y_grad, feature=0, axis_limits=axis_limits)
equi.find(nof_bins=3, min_points=2)
print(equi.limits)
equi.plot()

equi = pythia.bin_estimation.FixedGT(mu, var, feature=0, axis_limits=axis_limits)
equi.find(nof_bins=3)
print(equi.limits)
equi.plot()

# Dynamic programming
dp = pythia.bin_estimation.DP(x, y_grad, feature=0, axis_limits=axis_limits)
dp.find(min_points=2, k_max=20, discount=.3)
print(dp.limits)
dp.plot()


dp = pythia.bin_estimation.DPGT(mu, var, feature=0, axis_limits=axis_limits)
dp.find(k_max=20)
print(dp.limits)
dp.plot()

# greedy
greedy = pythia.bin_estimation.Greedy(x, y_grad, feature=0, axis_limits=axis_limits)
greedy.find(min_points=2, n_max=20)
print(greedy.limits)
greedy.plot()

greedy = pythia.bin_estimation.GreedyGT(mu, var, feature=0, axis_limits=axis_limits)
greedy.find(n_max=20)
print(greedy.limits)
greedy.plot()

import effector
import numpy as np
import timeit

np.random.seed(21)

def f(x):
    return -x[:, 0]**2 * (x[:, 1] < 0) + x[:, 0]**2 * (x[:, 1] >= 0) + np.exp(x[:, 2])

def f_jac(x):
    return np.array([
        -2 * x[:, 0] * (x[:, 1] < 0) + 2 * x[:, 0] * (x[:, 1] >= 0),
        -x[:, 0]**2 * (x[:, 1] < 0) + x[:, 0]**2 * (x[:, 1] >= 0),
        np.exp(x[:, 2])
    ]).T

N = 100_000
D = 3
M = 1_000

X = np.random.uniform(-1, 1, (N, D))

#%%
# ALE
ale = effector.RegionalALE(
    data=X,
    model=f,
    feature_names=["x1", "x2", "x3"],
    nof_instances=N,
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]),
    target_name="y"
)

ale.fit(features=0)
ale.tree_full["feature_0"]



# binning_method = effector.binning_methods.Fixed(10)
# ale.fit(features="all", binning_method=binning_method)
# y, h = ale.eval(feature=0, xs=np.linspace(-1, 1, 1000), heterogeneity=True, centering=True)
#
# import matplotlib.pyplot as plt
# plt.plot(np.linspace(-1, 1, 1000), h)
# plt.show()
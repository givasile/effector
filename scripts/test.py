import numpy as np
import effector


np.random.seed(21)

N = 1000
T = 1000

# create data, model
data = np.stack(
    [
        np.random.uniform(-1, 1, N),
        np.random.uniform(-1, 1, N),
        np.random.randint(0, 2, N)
    ],
    axis=1)

axis_limits = np.stack([[-1]*3, [1]*3], axis=0)

def model(x):
    y = np.zeros_like(x[:, 0])
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind] = 5*x[ind, 0]
    return y

def model_jac(x):
    y = np.zeros_like(x)
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind, 0] = 5
    return y

xs = np.linspace(-1, 1, T)

reg_method = effector.RegionalPDP(data, model, axis_limits=axis_limits, nof_instances="all")
reg_method.summary(0)
partitioner = reg_method.partitions["feature_" + str(0)]
# reg_eff.summary(0)
# reg_eff.plot(0, 5, heterogeneity="shap_values")

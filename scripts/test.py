import numpy as np
import effector


np.random.seed(21)

N = 1000
T = 100

X_test = np.random.uniform(-1, 1, (N, 2))
axis_limits = np.array([[-1, -1], [1, 1]])

# create data, model
y_limits = [-15, 15]
dy_limits = [-25, 25]

def predict(x):
    y = np.zeros(x.shape[0])
    ind = x[:, 1] > 0
    y[ind] = 10*x[ind, 0]
    y[~ind] = -10*x[~ind, 0]
    return y + np.random.normal(0, 1, x.shape[0])

def jacobian(x):
    J = np.zeros((x.shape[0], 2))
    ind = x[:, 1] > 0
    J[ind, 0] = 10
    J[~ind, 0] = -10
    return J


xs = np.linspace(-1, 1, T)

# global effectpppppp
pdp = effector.PDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
yy, het = pdp.eval(0, xs, heterogeneity=True)
print(het)

reg_method = effector.RegionalPDP(X_test, predict, axis_limits=axis_limits, nof_instances="all")
reg_method.summary(0)
# partitioner = reg_method.partitioners["feature_" + str(0)]


# partitioner.visualize_candidate_splits(1)
# partitioner.visualize_candidate_splits(2)

# reg_eff.summary(0)
# reg_eff.plot(0, 5, heterogeneity="shap_values")

"""Goal: show that if data non-uniformly distributed, you have to find the K that matches the piecewise linear regions.
This is not feasible in all cases. Piecewise linear regions, under normal circumstances, are have not equal length.
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import feature_effect as fe

########### DEFINITIONS #############
def f_params():
    def find_a(params, x_start):
        params[0]["a"] = x_start
        for i, param in enumerate(params):
            if i < len(params) - 1:
                a_next = param["a"] + (param["to"] - param["from"]) * param["b"]
                params[i + 1]["a"] = a_next

    params = [{"b":0.3, "from": 0., "to": 10.},
              {"b":7. , "from": 10, "to": 20},
              {"b":-1.5, "from": 20, "to": 30},
              {"b":0., "from": 30, "to": 40},
              {"b":-5., "from": 40, "to": 50},
              {"b":0.3, "from": 50, "to": 60},
              {"b":7. , "from": 60, "to": 70},
              {"b":-1.5, "from": 70, "to": 80},
              {"b":0., "from": 80, "to": 90},
              {"b":-5., "from": 90, "to": 100}]
    x_start = -1
    find_a(params, x_start)
    return params


def generate_samples(N):
    eps = 1e-05
    x1 = np.random.uniform(7, 9, size=int(N/10))
    x2 = np.random.uniform(15, 17, size=int(N/10))
    x3 = np.random.uniform(21, 22, size=int(N/10))
    x4 = np.random.uniform(28, 30, size=int(N/10))
    x5 = np.random.uniform(38, 40, size=int(N/10))
    x6 = np.random.uniform(48, 50, size=int(N/10))
    x7 = np.random.uniform(58, 60, size=int(N/10))
    x8 = np.random.uniform(68, 70, size=int(N/10))
    x9 = np.random.uniform(78, 79, size=int(N/10))
    x10 = np.random.uniform(88, 99, size=int(N/10))
    x = np.expand_dims(np.concatenate((np.array([0.0]), x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, np.array([100-eps]))), axis=-1)
    return x


def create_f1():
    params = f_params()
    limits = [param["from"] for param in params]
    limits.append(params[-1]["to"])

    def f1(x):
        """Piece-wise linear feature-effect"""
        ind = np.squeeze(np.digitize(x, limits))
        y = []
        for i, point in enumerate(x):
            res = params[ind[i] - 1]["a"] + \
                (point - params[ind[i] - 1]["from"])*params[ind[i] - 1]["b"]

            y.append(res)
        return np.array(y)
    return f1


def create_f1_center():
    f1 = create_f1()
    z = np.mean(f1(np.linspace(0.0, 99.99999, 100000)))

    def f1_center(x):
        return f1(x) - z
    return f1_center


def create_data_effect(noise_level, seed):
    params = f_params()
    limits = [param["from"] for param in params]
    limits.append(params[-1]["to"])

    def compute_data_effect(x):
        """Piece-wise linear"""

        x = np.squeeze(x)
        ind = np.squeeze(np.digitize(x, limits))
        res1 = np.array(params)[ind-1]
        y = np.array([r['b'] for r in res1])

        # add noise
        rv = norm()
        noise = rv.rvs(size=y.shape[0], random_state=seed)*noise_level
        return np.expand_dims(y+noise, -1)
    return compute_data_effect


def create_gt_bins():
    params = f_params()
    gt_bins = {}
    gt_bins["height"] = [par["b"] for par in params]
    gt_bins["limits"] = [par["from"] for par in params]
    gt_bins["limits"].append(params[-1]["to"])
    return gt_bins


################ parameters ##################
N = 10000
noise_level = 0.
K_max_fixed = 50
K_max_variable = 40

################ main part  ##################
seed = 48375
np.random.seed(seed)

f1_center = create_f1_center()
f1 = create_f1()
f1_jac = create_data_effect(noise_level, seed)
x = generate_samples(N=N)
y = f1_center(x)
data = x
data_effect = f1_jac(x)


# show data points
plt.figure()
plt.title("gt feature effect")
plt.plot(data, y, "ro")
plt.xlabel("points")
plt.ylabel("feature effect")
plt.show(block=False)

# show data effect
plt.figure()
plt.title("effect per point")
plt.plot(data, data_effect, "ro")
plt.xlabel("points")
plt.ylabel("data effect")
plt.show(block=False)

############ Evaluation ###################
def compute_mse(dale, gt_func):
    xx = np.linspace(0., 10., 1000)
    y_pred = dale.eval(xx, s=0)[0]
    y_gt = gt_func(xx)
    return np.mean(np.square(y_pred - y_gt))


def compute_loss_fixed_size(dale, s):
    big_M = 1.e+3
    points_per_bin = dale.parameters["feature_" + str(s)]["points_per_bin"]
    bin_variance_nans = dale.parameters["feature_" + str(s)]["bin_variance_nans"]

    if np.sum(points_per_bin <= 1) > 0:
        error = big_M
    else:
        error = np.sum(bin_variance_nans)*dale.parameters["feature_" + str(s)]["dx"]**2
    return error


def compute_loss_variable_size(dale, s):
    big_M = 1.e+3
    points_per_bin = dale.parameters["feature_" + str(s)]["points_per_bin"]
    bin_variance_nans = dale.parameters["feature_" + str(s)]["bin_variance_nans"]
    limits = dale.parameters["feature_" + str(s)]["limits"]
    if np.sum(points_per_bin <= 1) > 0:
        error = big_M
    else:
        dx_list = np.array([limits[i+1] - limits[i] for i in range(len(limits)-1)])
        error = np.sum(bin_variance_nans*dx_list**2)
    return error


# count loss and mse
k_list_fixed = np.arange(2, K_max_fixed+1)
mse_fixed_size = []
loss_fixed_size = []
dale_list_fixed = []
gt_func = f1_center
for k in k_list_fixed:
    dale_list_fixed.append(fe.DALE(data=x, model=f1, model_jac=f1_jac))
    dale_list_fixed[-1].fit(features=[0], k=k, method="fix-size")
    mse_fixed_size.append(compute_mse(dale_list_fixed[-1], gt_func))
    loss_fixed_size.append(compute_loss_fixed_size(dale_list_fixed[-1], s=0))


k_list_variable = np.arange(2, K_max_variable + 1)
mse_variable_size = []
loss_variable_size = []
dale_list_variable = []
for k in k_list_variable:
    print(k)
    dale_list_variable.append(fe.DALE(data=x, model=f1, model_jac=f1_jac))
    dale_list_variable[-1].fit(features=[0], k=k, method="variable-size")
    gt_func = f1_center
    mse_variable_size.append(compute_mse(dale_list_variable[-1], gt_func))
    loss_variable_size.append(compute_loss_variable_size(dale_list_variable[-1], s=0))



# visualize
plt.figure()
plt.title("MSE vs K")
plt.plot(k_list_variable, mse_variable_size, "ro-", label="variable size")
plt.plot(k_list_variable, mse_fixed_size[:K_max_variable-1], "bo-", label="fixed size")
plt.legend()
plt.show(block=False)


plt.figure()
plt.title("Loss vs K")
plt.plot(k_list_variable, loss_variable_size, "ro-", label="variable size")
plt.plot(k_list_variable, loss_fixed_size[:K_max_variable-1], "bo-", label="fixed size")
plt.legend()
plt.show(block=False)

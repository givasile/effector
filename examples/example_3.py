"""Goal: show that if data non-uniformly distributed, you dynamic programming finds the regions with that match the piecewise linear regions.

"""

import numpy as np
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
    x1 = np.random.uniform(9, 10, size=int(N/10))
    x2 = np.random.uniform(15, 17, size=int(N/10))
    x3 = np.random.uniform(21, 22, size=int(N/10))
    x4 = np.random.uniform(28, 39, size=int(N/10))
    x5 = np.random.uniform(35, 38, size=int(N/10))
    x6 = np.random.uniform(42, 44, size=int(N/10))
    x7 = np.random.uniform(57, 59, size=int(N/10))
    x8 = np.random.uniform(64, 67, size=int(N/10))
    x9 = np.random.uniform(75, 76, size=int(N/10))
    x10 = np.random.uniform(85, 87, size=int(N/10))
    x11 = np.random.uniform(95, 97, size=int(N/10))

    # x = np.random.uniform(0, 100, size=int(N))
    x = np.expand_dims(np.concatenate((np.array([0.001]), x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, np.array([99.9]))), axis=-1)
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
    z = np.mean(f1(np.linspace(0.0001, 99.99, 10000)))

    def f1_center(x):
        return f1(x) - z
    return f1_center


def create_data_effect(noise_level):
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
        np.random.seed(2443534)
        noise = np.random.normal(0, noise_level, y.shape[0])
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
N = 20000
noise_level = 10.
################ main part  ##################
f1_center = create_f1_center()
f1 = create_f1()
f1_jac = create_data_effect(noise_level)

seed = 4837571
np.random.seed(seed)

x = generate_samples(N=N)
y = f1_center(x)
data = x
# data_effect = f1_jac(x)

plt.figure()
plt.plot(x, y, "ro")
# plt.plot(x, data_effect, "bo")
plt.show(block=False)

K = 80
dale = fe.DALE(data=x, model=f1, model_jac=f1_jac)
dale.fit(features=[0], k=K, method="variable-size")
dale.plot(s=0, block=False, gt=f1_center, gt_bins=create_gt_bins())

# K = 10
# dale = fe.DALE(data=x, model=f1, model_jac=f1_jac)
# dale.fit(features=[0], k=K)
# dale.plot(s=0, block=False, gt=f1_center, gt_bins=create_gt_bins())


# ############ Evaluation ###################
# def compute_error(dale_func, gt_func):
#     xx = np.linspace(0.001, 99.9, 20)
#     y_pred = dale_func(xx, s=0)[0]
#     y_gt = gt_func(xx)
#     return np.mean(np.abs(y_pred - y_gt))

# k_list_variable = np.arange(1, 500)
# error = []
# for k in k_list_variable:
#     # dale = fe.DALE(data=x, model=f1, model_jac=f1_jac)
#     dale.fit(features=[0], k=k)
#     dale_func = dale.eval
#     gt_func = f1_center
#     error.append(compute_error(dale_func, gt_func))


# plt.figure()
# plt.plot(k_list_variable, error, "bo")
# plt.show(block=False)

# ###################
# def error_estimator(dale, s):
#     big_M = 1.e+3
#     points_per_bin = dale.parameters["feature_" + str(s)]["points_per_bin"]
#     bin_variance_nans = dale.parameters["feature_" + str(s)]["bin_variance_nans"]

#     if np.sum(points_per_bin <= 1) > 0:
#         error = big_M
#     else:
#         error = np.mean(bin_variance_nans)
#     return error

# loss_fixed_size = []
# for k in k_list_variable:
#     # dale = fe.DALE(data=x, model=f1, model_jac=f1_jac)
#     dale.fit(features=[0], k=k)
#     dale_func = dale.eval
#     gt_func = f1_center
#     loss_fixed_size.append(error_estimator(dale, s=0))

# plt.figure()
# plt.plot(k_list_variable, loss_fixed_size, "bo")
# plt.show(block=False)

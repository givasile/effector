"""Goal: show that if uniformly distributed data, the biggest K wins in fix-length bins.

The set-up
==========
* The data points are uniformly distributed in [0,100]
* The feature-effect is made from 10 piece-wise linear parts of length 10;

For different values of parameters:
* N: nof points
* noise_level: how much noise to add in the data_effect estimation

Questions to be addressed
=========================
For fixed-size bins:
* what is the best K in terms of accuracy?
* Does standard mse_fixed_size or variance estimator chooses the best K?
For variable-sized bins:
* what is the best (a) number and (b) size of bins, in terms of mse_fixed_size?
* does dynamic programming based on mse_fixed_size finds them and for which K?
* does dynamic programming based on standard mse_fixed_size finds them and for which K?

Final Goals
===========
* show that choosing the correct bin-size K is crucial
* in cases of uniformly distributed data, choosing a big K is always a good option - or even the best option
 (independently of noise level and nof points)
* show that variable-size bins, has equivalent accuracy with fixed-bin in case of fixed-size piecewise linear effect
 and our algorithm is able to find variable-sizes with equivalent accuracy in this case
* show that standard mse_fixed_size is a good estimator of the best bin size

Experiments
===========

(a) For noise_level = 0
=======================
For fixed-size bins:
* The best K are the ones that permit changing bin every 10
i.e. K = {10, 20, 30, ..., 100} or dx = {10, 10/2=5 ,10/3=3.33, ... , 10/10=1}
* but in the range of big K, all choices have good accuracy
* Standard mse_fixed_size perfectly chooses the best candidates
For Variable-size bins:
* Finds good bins based on accuracy
* Finds good bins base on standard mse_fixed_size


The conclusions above, hold independently of the number of points. In case of limited points, variable-size bins can be
"wrong" but this is due to the absence of points, not a problem of the method.


(b) For noise_level = 3.0
=======================
For fixed-size bins:
* The best K are the ones that permit changing bin every 10
i.e. K = {10, 20, 30, ..., 100} or dx = {10, 10/2=5 ,10/3=3.33, ... , 10/10=1}
* but in the range of big K, all choices have good accuracy
* Standard mse_fixed_size perfectly chooses the best candidates
For Variable-size bins:
* Finds good bins based on accuracy
* Finds good bins base on standard mse_fixed_size

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
    find_a(params, x_start)
    return params


def generate_samples(N):
    eps = 1e-05
    x = np.random.uniform(0, 100-eps, size=int(N))
    x = np.expand_dims(np.concatenate((np.array([0.0]), x, np.array([100. - eps]))), axis=-1)
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

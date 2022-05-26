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
import examples.example_utils as utils


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

# parameters
N = 10000
noise_level = 0.
K_max_fixed = 50
K_max_variable = 40

# init functions
seed = 48375
np.random.seed(seed)

f1_center = utils.create_f1_center(f_params)
f1 = utils.create_f1(f_params)
f1_jac = utils.create_data_effect(f_params, noise_level, seed)
x = generate_samples(N=N)
y = f1_center(x)
data = x
data_effect = f1_jac(x)

# show gt effect
utils.plot_gt_effect(data, y)

# show data effect
utils.plot_data_effect(data, data_effect)

# count loss and mse
fixed_params = utils.count_loss_mse_fixed(K_max_fixed, f1_center, data, f1, f1_jac)
k_list_fixed, mse_fixed_size, loss_fixed_size, dale_list_fixed = fixed_params

variable_params = utils.count_loss_mse_variable(K_max_variable, f1_center, data, f1, f1_jac)
k_list_variable, mse_variable_size, loss_variable_size, dale_list_variable = variable_params

# plot
utils.plot_mse(k_list_variable, mse_variable_size, mse_fixed_size)
utils.plot_loss(k_list_variable, loss_variable_size, loss_fixed_size)

# plot best fixed solution
best_fixed_index = np.argmin(mse_fixed_size)
dale_list_fixed[best_fixed_index].plot(s=0, gt=f1_center, gt_bins=utils.create_gt_bins(f_params))

# plot best variable size solution
best_variable_index = np.argmin(mse_variable_size)
dale_list_variable[best_variable_index].plot(s=0, gt=f1_center, gt_bins=utils.create_gt_bins(f_params))

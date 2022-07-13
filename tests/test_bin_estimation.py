import pytest
import numpy as np
import examples.example_utils as example_utils
import feature_effect.utils as utils
import matplotlib.pyplot as plt
import feature_effect as fe

def gen_model_params():
    params = [{"b":2, "from": 0., "to": .3},
              {"b":-2., "from": .3, "to": .6},
              {"b":0., "from": .6, "to": 1}]
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
N = 1000
noise_level = 1.
model_jac = example_utils.create_noisy_jacobian(gen_model_params(),
                                                noise_level,
                                                seed)
data = np.sort(generate_samples(N=N), axis=0)
data_effect = model_jac(data)



example_utils.plot_data_effect(data, data_effect)

# check bin creation
bin_est1 = fe.bin_estimation.BinEstimatorDP(data, data_effect,
                                           feature=0, K=5)
# limits = bin_est.solve_dp(min_points=10)
# clusters = np.squeeze(np.digitize(data, limits))

# plt.figure()
# plt.title("Dynamic programming")
# plt.plot(data, data_effect, "bo")
# plt.vlines(limits, ymin=np.min(data_effect), ymax=np.max(data_effect))
# plt.show(block=False)


# bin_est_2 = fe.bin_estimation.BinEstimatorGreedy(data, data_effect,
#                                                  feature=0, K=5)

# bin spliting
feature = 0
tau = 10
big_M = 1e10
K = 10
xs = data[:, feature]
dy_dxs = data_effect[:, feature]
limits = fe.utils.create_fix_size_bins(xs, K)
# bin_mu, bin_points = fe.utils.compute_bin_effect_mean(xs, dy_dxs, limits)
# bin_var, bin_est_var = fe.utils.compute_bin_effect_variance(xs, dy_dxs, limits, bin_mu)
# bin_thres = np.logical_or(np.isnan(bin_mu), bin_points < tau)

# bin merging
i = 0
merged_limits = [limits[0]]
while (i < K):
    # left limit is always the last item of merged_limits
    left_lim = merged_limits[-1]

    # choose whether to close the bin
    if i == K - 1:
        close_bin = True
    else:
        # compare the added loss from closing or keep open in a greedy manner
        _, effect_1 = fe.utils.filter_points_belong_to_bin(xs, dy_dxs, np.array([left_lim, limits[i+1]]))
        _, effect_2 = fe.utils.filter_points_belong_to_bin(xs, dy_dxs, np.array([left_lim, limits[i+2]]))
        loss_1 = np.var(effect_1) if effect_1.size > tau else big_M
        loss_2 = np.var(effect_2) if effect_2.size > tau else big_M

        if loss_1 == big_M:
            close_bin = False
        else:
            close_bin = False if (loss_2 / loss_1 <= 1.1) else True

    # apply action
    if close_bin:
        merged_limits.append(limits[i+1])
        i += 1
    else:
        i += 1


# post processing - if last bin doesn't have enough points merge

print(merged_limits)

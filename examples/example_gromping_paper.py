import examples.utils as utils
path = utils.add_parent_path()
import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import example_models.distributions as dist

savefig = False

# define distribution
D = 2
x1_min = 0
x1_max = 1
x2_sigma = 2.
gen_dist = dist.Correlated1(D, x1_min, x1_max, x2_sigma)
axis_limits = gen_dist.axis_limits





# # define model
# def model_dale(b0, b1, b2, b3, x2_sigma):

#     # dale
#     dale_mean = lambda x: b1 + b3*x
#     dale_mean_int = lambda x: b1*x + b3/2*x**2
#     dale_var = lambda x: b3**2 * x2_sigma
#     dale_var_int = lambda x: b3**2 * x2_sigma

#     dale_gt = fe.DALEGroundTruth(dale_mean, dale_mean_int, dale_var, dale_var_int, axis_limits)
#     dale_gt.fit(features=0)
#     return dale_gt


# # plot
# def plot1(x2_sigma):
#     plt.figure()
#     plt.title("ALE s2=" + str(x2_sigma))
#     xs = np.linspace(axis_limits[0,0], axis_limits[1,0], 100)
#     dale_gt = model_dale(0, 1, 1, 2, x2_sigma)
#     y, var, _ = dale_gt.eval(xs, s=0, uncertainty=True)
#     plt.plot(xs, y, "b--", label="mean ALE")
#     plt.fill_between(xs, y-np.sqrt(var), y+np.sqrt(var), alpha=0.2, label="std ALE")
#     plt.xlabel("x1")
#     # plt.ylim(-4, 4)
#     plt.ylabel("f_ALE")
#     plt.legend()
#     if savefig:
#         plt.savefig(path + "ALE_gt_s2_" + str(x2_sigma).replace(".", "_") + ".png")
#     plt.show(block=False)


# plot1(x2_sigma=0.01)
# plot1(x2_sigma=0.1)
# plot1(x2_sigma=1.)


# #
# b0 = 0
# b1 = 1
# b2 = 1
# b3 = 2
# x2_sigma = .1
# dale_mean = lambda x: b1 + b3*x
# dale_mean_int = lambda x: b1*x + b3/2*x**2
# dale_var = lambda x: b3**2 * x2_sigma
# dale_var_int = lambda x: b3**2 * x2_sigma

# axis_limits = np.array([[0,10], [0,1]]).T
# bin_gt = fe.bin_estimation.FixedSizeGT(dale_mean, dale_var, axis_limits, feature=0)

# cost_list = []
# for K in range(1, 40):
#     bin_gt.solve(min_points=2, K=K)
#     limits = bin_gt.limits
#     cost = np.mean([bin_gt.cost_of_bin(limits[i], limits[i+1]) for i in range(bin_gt.limits.shape[0]-1)])
#     cost_list.append(cost)


# nof_points = np.array([100, 200, 400, 1000])
# min_points = 25
# max_K = nof_points / min_points

# plt.figure()
# plt.title("Mean variance of ALE")
# plt.plot(range(4, 40), cost_list[3:], "bx")
# plt.hlines(b3**2*x2_sigma, 0, 40, label="unavoidable variance")
# [plt.plot([k,k], [0, 3], "--", label="max K, samples=" + str(nof_points[i])) for i, k in enumerate(max_K)]
# plt.xlabel("K")
# plt.ylabel("mean bin variance")
# plt.legend()
# if savefig:
#     plt.savefig(path + "bin_varinace.png")
# plt.show(block=False)

# # dale = fe.DALE(X, model.predict, model.jacobian)
# # dale.fit(features=0, alg_params={"bin_method": "fixed", "nof_bins":20})

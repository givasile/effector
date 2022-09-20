import examples.utils as utils
path = utils.add_parent_path()
import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import example_models.distributions as dist
import example_models.models as models


def compare(dale_gt, dale):
    limits_gt = dale_gt.feature_effect["feature_0"]["limits"]
    limits = dale.feature_effect["feature_0"]["limits"]
    ind = np.digitize(limits, limits_gt) - 1


    bins_res_err = []
    bins_mu_err = []
    bins_var_err = []
    for i in range(limits.shape[0] - 1):
        means = dale_gt.feature_effect["feature_0"]["bin_effect"][ind[i]+1:ind[i+1]]
        variances = dale_gt.feature_effect["feature_0"]["bin_variance"][ind[i]+1:ind[i+1]]

        res_err = means.var()
        bins_res_err.append(res_err)

        mu_gt = means.mean()
        mu_est = dale.feature_effect["feature_0"]["bin_effect"][i]
        bins_mu_err.append(np.abs(mu_gt - mu_est))

        var_gt = variances.mean()
        var_est = dale.feature_effect["feature_0"]["bin_variance"][i]
        bins_var_err.append(np.abs(var_gt - var_est))

    res = np.mean(bins_res_err)
    mu = np.mean(bins_mu_err)
    var = np.mean(bins_var_err)
    return res, mu, var


def measure_fixed_error(dale_gt, K_list, nof_iterations):
    # fit approximation
    rho_mean = []
    rho_std = []

    mu_mean = []
    mu_std = []

    var_mean = []
    var_std = []

    # for fixed
    for i, k in enumerate(K_list):
        rho_tmp = []
        mu_tmp = []
        var_tmp = []
        for l in range(nof_iterations):
            X = gen_dist.generate(N=500)
            dale = fe.DALE(data=X,
                           model=model.predict,
                           model_jac=model.jacobian,
                           axis_limits=axis_limits)
            alg_params = {"bin_method" : "fixed", "nof_bins" : k, "min_points_per_bin": 2}
            try:
                dale.fit(features=[0], alg_params=alg_params)
                res_err, mu_err, var_err = compare(dale_gt, dale)
                rho_tmp.append(res_err)
                mu_tmp.append(mu_err)
                var_tmp.append(var_err)
            except:
                pass

        rho_mean.append(np.mean(rho_tmp))
        rho_std.append(np.std(rho_tmp))

        mu_mean.append(np.mean(mu_tmp))
        mu_std.append(np.std(mu_tmp))

        var_mean.append(np.mean(var_tmp))
        var_std.append(np.std(var_tmp))
    return rho_mean, rho_std, mu_mean, mu_std, var_mean, var_std


def measure_auto_error(dale_gt, nof_iterations):
    rho = []
    mu = []
    var = []
    for l in range(nof_iterations):
        X = gen_dist.generate(N=500)
        dale = fe.DALE(data=X,
                       model=model.predict,
                       model_jac=model.jacobian,
                       axis_limits=axis_limits)
        alg_params = {"bin_method" : "dp", "max_nof_bins" : 20, "min_points_per_bin": 10}
        dale.fit(features=[0], alg_params=alg_params)
        res_err1, mu_err1, var_err1 = compare(dale_gt, dale)

        rho.append(res_err1)
        mu.append(mu_err1)
        var.append(var_err1)
    return np.mean(rho), np.std(rho), np.mean(mu), np.std(mu), np.mean(var), np.std(var)


savefig = True
np.random.seed(23)

# gen dist
gen_dist = dist.Correlated1(D=2, x1_min=0, x1_max=1, x2_sigma=2)
axis_limits = gen_dist.axis_limits

# model
params = [{"from": 0., "to": .2, "a": 0, "b": 10},
           {"from": .2, "to": .4, "a": 2, "b": -10},
           {"from": .4, "to": .5, "a": 0, "b": -20},
           {"from": .5, "to": .6, "a": -2, "b": 100},
          {"from": .6, "to": .9, "a": 8, "b": 5.},
          {"from": .9, "to": 1., "a": 9.5, "b": -15}]
model = models.PiecewiseLinear(params)

# fit ground-truth
X_0 = gen_dist.generate(N=1000000)
dale_gt = fe.DALE(data=X_0,
               model=model.predict,
               model_jac=model.jacobian,
               axis_limits=axis_limits)

alg_params = {"bin_method" : "fixed", "nof_bins" : 1000, "min_points_per_bin": 5}
dale_gt.fit(features=0, alg_params=alg_params)


# fit approximation
K_list = list(range(1,51))
rho_fixed_err_mean, rho_fixed_err_std, mu_fixed_err_mean, mu_fixed_err_std, var_fixed_err_mean, var_fixed_err_std = measure_fixed_error(dale_gt, K_list, nof_iterations=10)

rho_auto_err_mean, rho_auto_err_std, mu_auto_err_mean, mu_auto_err_std, var_auto_err_mean, var_auto_err_std = measure_auto_error(dale_gt, nof_iterations=10)


plt.figure()
plt.title("mu error per bin")
plt.errorbar(K_list, mu_fixed_err_mean, yerr=mu_fixed_err_std, fmt='o', label="fixed-size")
plt.plot(K_list, np.repeat(mu_auto_err_mean, len(K_list)), "r--", label="auto-bin")
plt.legend()
plt.show(block=False)

plt.figure()
plt.title("var error per bin")
plt.errorbar(K_list, var_fixed_err_mean, yerr=var_fixed_err_std, fmt='o', label="fixed-size")
plt.plot(K_list, np.repeat(var_auto_err_mean, len(K_list)), "r--", label="auto-bin")
plt.legend()
plt.show(block=False)


plt.figure()
plt.title("resolution error per bin")
plt.errorbar(K_list, rho_fixed_err_mean, yerr=rho_fixed_err_std, fmt='o', label="fixed-size")
plt.plot(K_list, np.repeat(rho_auto_err_mean, len(K_list)), "r--", label="auto-bin")
plt.legend()
plt.show(block=False)

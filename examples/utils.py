import sys
import os
import feature_effect as fe
import numpy as np
import matplotlib.pyplot as plt

def add_parent_path():
    cwd = os.getcwd()
    if os.path.split(os.getcwd())[-1] == "mdale":
        path_to_examples = os.path.join(cwd, "examples/")
        sys.path.append(cwd)
    elif os.path.split(os.getcwd())[-1] == "examples":
        path_to_examples = cwd
        sys.path.append(os.path.split(cwd)[0])
    else:
        raise OSError

    return path_to_examples


def compare(dale_gt, dale):
    """Returns the mean error (error per bin) for mu, var, res

    :param dale_gt:
    :param dale:
    :returns:

    """
    limits_gt = dale_gt.feature_effect["feature_0"]["limits"]
    limits = dale.feature_effect["feature_0"]["limits"]
    ind = np.digitize(limits, limits_gt) - 1

    rho_list = []
    mu_err_list = []
    var_err_list = []
    for i in range(limits.shape[0] - 1):
        means = dale_gt.feature_effect["feature_0"]["bin_effect"][ind[i]+1:ind[i+1]]
        variances = dale_gt.feature_effect["feature_0"]["bin_variance"][ind[i]+1:ind[i+1]]

        rho_bin = means.var()
        rho_list.append(rho_bin)

        mu_gt = means.mean()
        mu_est = dale.feature_effect["feature_0"]["bin_effect"][i]
        mu_err_list.append(np.abs(mu_gt - mu_est))

        var_gt = variances.mean()
        var_est = dale.feature_effect["feature_0"]["bin_variance"][i]
        var_err_list.append(np.abs(var_gt - var_est))

    mean_rho = np.mean(rho_list)
    mean_mu_err = np.mean(mu_err_list)
    mean_var_err = np.mean(var_err_list)
    return mean_rho, mean_mu_err, mean_var_err


def measure_fixed_error(dale_gt, gen_dist, model, axis_limits, K_list, nof_iterations, nof_points):
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
            X = gen_dist.generate(N=nof_points)
            dale = fe.DALE(data=X,
                           model=model.predict,
                           model_jac=model.jacobian,
                           axis_limits=axis_limits)
            alg_params = {"bin_method" : "fixed",
                          "nof_bins" : k,
                          "min_points_per_bin": 2}

            try:
                dale.fit(features=[0], alg_params=alg_params)
                res_err, mu_err, var_err = compare(dale_gt, dale)
                rho_tmp.append(res_err)
                mu_tmp.append(mu_err)
                var_tmp.append(var_err)
            except:
                print("aek")
                pass

        rho_mean.append(np.mean(rho_tmp))
        rho_std.append(np.std(rho_tmp))

        mu_mean.append(np.mean(mu_tmp))
        mu_std.append(np.std(mu_tmp))

        var_mean.append(np.mean(var_tmp))
        var_std.append(np.std(var_tmp))

    stats = {"rho_mean": rho_mean,
             "rho_std": rho_std,
             "mu_err_mean": mu_mean,
             "mu_err_std": mu_std,
             "var_err_mean": var_mean,
             "var_err_std": var_std}

    return stats


def measure_auto_error(dale_gt, gen_dist, model, axis_limits, nof_iterations, nof_points):
    rho = []
    mu = []
    var = []
    for l in range(nof_iterations):
        X = gen_dist.generate(N=nof_points)
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

    stats = {"rho_mean": np.mean(rho),
             "rho_std": np.std(rho),
             "mu_err_mean": np.mean(mu),
             "mu_err_std": np.std(mu),
             "var_err_mean": np.mean(var),
             "var_err_std": np.std(var)}
    return stats


def plot_fixed_vs_auto(K_list, fixed_mean, fixed_std, auto_mean, auto_std, title, savefig=None):
    plt.figure()
    plt.title(title)
    plt.plot(K_list, fixed_mean, 'x', label="fixed-bin")
    plt.plot(K_list,
             np.repeat(auto_mean, len(K_list)),
             "r--",
             label="auto-bin")
    plt.legend()
    if savefig is not None:
        plt.savefig(savefig)
    plt.show(block=False)

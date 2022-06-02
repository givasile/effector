import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import feature_effect as fe


def create_model(params):
    def create_f1(params):
        limits = [param["from"] for param in params]
        limits.append(params[-1]["to"])

        def f1(x):
            """Piece-wise linear feature-effect"""
            ind = np.squeeze(np.digitize(x, limits))
            y = []
            for i, point in enumerate(x):
                res = params[ind[i] - 1]["a"] + \
                      (point - params[ind[i] - 1]["from"]) * params[ind[i] - 1]["b"]

                y.append(res)
            return np.array(y)

        return f1

    f1 = create_f1(params)
    z = np.mean(f1(np.linspace(0.0, 99.99999, 100000)))

    def f1_center(x):
        return f1(x) - z
    return f1_center


def create_noisy_jacobian(params, noise_level, seed):
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


def create_gt_bins(params):
    gt_bins = {}
    gt_bins["height"] = [par["b"] for par in params]
    gt_bins["limits"] = [par["from"] for par in params]
    gt_bins["limits"].append(params[-1]["to"])
    return gt_bins


def compute_mse(dale, gt_func):
    xx = np.linspace(0., 99.9999, 1000)
    y_pred = dale.eval(xx, s=0)[0]
    y_gt = gt_func(xx)
    return np.sum(np.abs(y_pred - y_gt)) * .001


def fit_multiple_K(data, model, model_jac, K_max, min_points_per_bin, method):
    k_list = np.arange(1, K_max + 1)
    dale_list = []
    for k in k_list:
        dale_list.append(fe.DALE(data=data, model=model, model_jac=model_jac))
        if method == "fixed-size":
            dale_list[-1].fit(method=method, alg_params={"nof_bins": k, "min_points_per_bin": min_points_per_bin})
        elif method == "variable-size":
            dale_list[-1].fit(method=method, alg_params={"max_nof_bins": k, "min_points_per_bin": min_points_per_bin})
    return dale_list


def plot_gt_effect(points, y):
    plt.figure()
    plt.title("gt feature effect")
    plt.plot(points, y, "b--")
    plt.xlabel("points")
    plt.ylabel("feature effect")
    plt.show(block=False)


def plot_data_effect(data, data_effect):
    plt.figure()
    plt.title("effect per point")
    plt.plot(data, data_effect, "bo")
    plt.xlabel("points")
    plt.ylabel("data effect")
    plt.show(block=False)


# def plot_mse(k_list_variable, mse_variable_size, mse_fixed_size):
#     plt.figure()
#     plt.title("MSE vs K")
#     plt.plot(k_list_variable, mse_variable_size, "ro-", label="variable size")
#     plt.plot(k_list_variable, mse_fixed_size[:len(k_list_variable)], "bo-", label="fixed size")
#     plt.legend()
#     plt.show(block=False)


def plot_loss(dale_list):
    plt.figure()
    loss = [dale.dale_params["feature_0"]["loss"] for dale in dale_list]
    nof_bins = [dale.dale_params["feature_0"]["nof_bins"] for dale in dale_list]
    plt.plot(nof_bins, loss, "ro--")
    if dale_list[0].dale_params["feature_0"]["method"] == "fixed-size":
        plt.title("Loss vs number of bins (fixed-size)")
        plt.xlabel("number of bins")
    else:
        plt.title("Loss vs max number of bins (variable-size)")
        plt.xlabel("maximum number of bins")
    plt.ylabel("loss")
    plt.show(block=False)

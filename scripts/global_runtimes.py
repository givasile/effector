import matplotlib.pyplot as plt
import effector
import numpy as np
import time
import os
np.random.seed(21)


def return_predict(t):
    def predict(x):
        time.sleep(t)
        return np.sum(x, axis=1)
    return predict


def return_jacobian(t):
    def jacobian(x):
        time.sleep(t)
        return x
    return jacobian


def measure_time(
        method_name,
        features,
        model,
        N,
        M,
        D,
        repetitions,
        K,
        model_jac=None
):
    fit_time_list, eval_time_list = [], []
    X = np.random.uniform(-1, 1, (N, D))
    xx = np.linspace(-1, 1, M)
    axis_limits = np.array([[-1] * D, [1] * D])

    method_map = {
        "pdp": effector.PDP,
        "d_pdp": effector.DerPDP,
        "ale": effector.ALE,
        "rhale": effector.RHALE,
        "shap_dp": effector.ShapDP
    }

    for _ in range(repetitions):
        # general kwargs
        method_kwargs = {"data": X, "model": model, "axis_limits": axis_limits, "nof_instances": "all"}
        fit_kwargs = {"features": features, "centering": True, "points_for_centering": K}

        # specialize kwargs per method
        if method_name in ["d_pdp", "rhale"]:
            method_kwargs["model_jac"] = model_jac
        if method_name in ["rhale", "ale"]:
            fit_kwargs["binning_method"] = effector.axis_partitioning.Fixed(nof_bins=20)

        # init
        method = method_map[method_name](**method_kwargs)

        # fit
        tic = time.time()
        method.fit(**fit_kwargs)
        fit_time_list.append(time.time() - tic)

        # eval
        tic = time.time()
        for feat in features:
            eval_kwargs = {"feature": feat, "xs": xx, "centering": True, "heterogeneity": True}
            method.eval(**eval_kwargs)
        eval_time_list.append(time.time() - tic)

    return {"fit": np.mean(fit_time_list), "eval": np.mean(eval_time_list), "total": (np.mean(fit_time_list) + np.mean(eval_time_list))}


def bar_plot(xs, time_dict, methods, metric, xlabel, ylabel, savepath=None):

    bar_width = (np.max(xs) - np.min(xs)) / 40
    method_to_label = {
        "ale": "ALE",
        "rhale": "RHALE",
        "pdp": "PDP",
        "d_pdp": "d-PDP",
        "shap_dp": "SHAP DP"
    }
    plt.figure()
    # Calculate the offsets for each bar group
    offsets = np.linspace(-2*bar_width, 2*bar_width, len(methods))

    for i, method in enumerate(methods):
        label = method_to_label[method]
        plt.bar(
            xs + offsets[i],
            [tt[metric] for tt in time_dict[method]],
            label=label,
            width=bar_width
        )

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(xs, fontsize=14)
    plt.legend(fontsize=14)
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show(block=False)

# make the plot a funciton
def bar_plot_shap_dp(xs, time_dict, metric, xlabel, ylabel, savepath=None):
    bar_width = (np.max(xs) - np.min(xs)) / 40
    shap_label = "SHAP-DP"
    shap_color = "C4"  # Same distinct color for SHAP-DP

    plt.figure()
    plt.bar(
        xs,
        [tt[metric] for tt in time_dict["shap_dp"]],
        width=bar_width,
        color=shap_color,
        label=shap_label
    )

    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(xs, fontsize=14)
    plt.legend(fontsize=14)
    if savepath is not None:
        plt.savefig(savepath, dpi=300, bbox_inches='tight')
    plt.show(block=False)


def plot_shap_dp_runtime(metric, xlabel, ylabel, y_ticks, savepath):
    plt.figure()
    plt.plot(
        vec,
        [tt[metric] for tt in time_dict["shap_dp"]],
        "o-",
        label="SHAP DP",
    )
    plt.legend(fontsize=14)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(vec, fontsize=14)
    if y_ticks is not None:
        plt.yticks(y_ticks, fontsize=14)
    else:
        plt.yticks(fontsize=14)
    plt.savefig(savepath, bbox_inches='tight')
    plt.show(block=False)


savedir = "./global_runtime_figures"
if not os.path.exists(savedir):
    os.makedirs(savedir)

# measure N
t = 0.1
N = 10_000
D = 3
K = 100
M = 100
repetitions = 2
features = [0]

method_names = ["ale", "rhale", "pdp", "d_pdp"]
vec = np.array([10_000, 25_000, 50_000])
time_dict = {method_name: [] for method_name in method_names}
for N in vec:
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))

bar_plot(
    vec,
    time_dict,
    method_names,
    metric="total",
    xlabel="N",
    ylabel="time (sec)",
    savepath=os.path.join(savedir, "global_runtime_N.pdf")
)

# measure T_F
t = 0.1
N = 10_000
D = 3
K = 100
M = 100
repetitions = 2
features=[0]

method_names = ["ale", "rhale", "pdp", "d_pdp"]
time_dict = {method_name: [] for method_name in method_names}
vec = np.array([0.1, 0.5, 1.])
for t in vec:
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))

bar_plot(
    vec,
    time_dict,
    method_names,
    metric="total",
    xlabel="time (sec) to execute f on X",
    ylabel="time (sec)",
    savepath=os.path.join(savedir, "global_runtime_T.pdf")
)

# measure D
t = 0.1
N = 10_000
D = 3
K = 100
M = 100
repetitions = 2

method_names = ["ale", "rhale", "pdp", "d_pdp"]
time_dict = {method_name: [] for method_name in method_names}
vec = np.array([10, 15, 20])
for D in vec:
    features = [i for i in range(D)]
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))

bar_plot(
    vec,
    time_dict,
    method_names,
    metric="total",
    xlabel="D",
    ylabel="time (sec)",
    savepath=os.path.join(savedir, "global_runtime_D.pdf")
)

## Same analysis for SHAP DP alone
# measure N
t = 0.1
N = 10_000
D = 3
K = 100
M = 100
repetitions = 2
features = [0]

method_names = ["shap_dp"]
vec = np.array([10, 50, 100, 200])
time_dict = {method_name: [] for method_name in method_names}
for N in vec:
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(
            measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))

plot_shap_dp_runtime(
    metric="total",
    xlabel="N",
    ylabel="time (sec)",
    y_ticks=None,
    savepath=os.path.join(savedir, "global_runtime_N_shap_dp.pdf")
)

bar_plot_shap_dp(
    vec,
    time_dict,
    metric="total",
    xlabel="N",
    ylabel="time (sec)",
    savepath=os.path.join(savedir, "global_runtime_N_shap_dp_bar.pdf")
)

# measure T_F
t = 0.1
N = 50
D = 3
K = 100
M = 100
repetitions = 2
features = [0]
method_names = ["shap_dp"]
vec = np.array([0.05, 0.1, 0.15, 0.2])
time_dict = {method_name: [] for method_name in method_names}
for t in vec:
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(
            measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))

plot_shap_dp_runtime(
    metric="total",
    xlabel="time (sec) to execute f on X",
    ylabel="time (sec)",
    y_ticks=None,
    savepath=os.path.join(savedir, "global_runtime_T_shap_dp.pdf")
)

bar_plot_shap_dp(
    vec,
    time_dict,
    metric="total",
    xlabel="time (sec) to execute f on X",
    ylabel="time (sec)",
    savepath=os.path.join(savedir, "global_runtime_T_shap_dp_bar.pdf")
)

# measure D
t = 0.1
N = 50
D = 3
K = 100
M = 100
repetitions = 2
method_names = ["shap_dp"]
vec = np.array([2, 4, 6])
time_dict = {method_name: [] for method_name in method_names}
for D in vec:
    features = [i for i in range(D)]
    model = return_predict(t)
    model_jac = return_jacobian(t)
    for method_name in method_names:
        time_dict[method_name].append(
            measure_time(method_name, features, model, N, M, D, repetitions, K, model_jac=model_jac))

plot_shap_dp_runtime(
    metric="total",
    xlabel="D",
    ylabel="time (sec)",
    y_ticks=[8, 10, 12],
    savepath=os.path.join(savedir, "global_runtime_D_shap_dp.pdf")
)

bar_plot_shap_dp(
    vec,
    time_dict,
    metric="total",
    xlabel="D",
    ylabel="time (sec)",
    savepath=os.path.join(savedir, "global_runtime_D_shap_dp_bar.pdf")
)

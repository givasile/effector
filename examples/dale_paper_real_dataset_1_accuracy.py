import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tensorflow import keras
import tensorflow as tf
import random as python_random
import timeit
import tikzplotlib as tplt
matplotlib.rcParams['text.usetex'] = True
save_fig = True
import feature_effect as fe

path = "/home/givasile/projects-org/org-feature-effect/paper-acml/images/"

def create_nn(X):
    nn = keras.Sequential([keras.layers.Input(shape=[X.shape[1]]),
                           keras.layers.Dense(1024, activation='relu', use_bias=True),
                           keras.layers.Dense(512, activation='relu', use_bias=True),
                           keras.layers.Dense(256, activation='relu', use_bias=True),
                           keras.layers.Dense(128, activation='relu', use_bias=True),
                           keras.layers.Dense(64, activation='relu', use_bias=True),
                           keras.layers.Dense(32, activation='relu', use_bias=True),
                           keras.layers.Dense(1, use_bias=True)])

    nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss="mean_squared_error",
              metrics="mean_absolute_error")

    return nn


def create_model(nn):
    def model(X):
        return nn(X).numpy()[:, 0]
    return model


def create_model_grad(nn):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    def model_grad(inp):
        x_inp = tf.cast(inp, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(x_inp)
            preds = nn(x_inp)
            grads = tape.gradient(preds, x_inp)
        return grads.numpy()
    return model_grad


def scale_x(x, s):
    x = copy.deepcopy(x)
    if s == 7:
        x = x*(39 - (-8)) + (-8)
    elif s==8:
        x = x*(50 - (-16)) + (-16)
    elif s==9:
        x = x*100
    elif s==10:
        x = x*67
    return x


# systematic evaluation
def mse(X, X_small, s, K_ref, K_list):
    x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)

    # DALE
    dale = fe.DALE(data=X, model=model, model_jac=model_grad)
    dale.fit(alg_params={"nof_bins": K_ref})
    y1, _, _ = dale.eval(x, s=s)

    err_dale_mse = []
    err_dale_nmse = []
    err_dale_var = []
    for k in K_list:
        dale = fe.DALE(data=X_small, model=model, model_jac=model_grad)
        dale.fit(alg_params={"nof_bins": k})
        y2, _, _ = dale.eval(x, s=s)
        tmp = np.mean(np.square(y1 - y2))
        err_dale_mse.append(tmp)
        err_dale_var.append(y1.var())
        err_dale_nmse.append(tmp/y1.var())


    # ALE
    ale = fe.ALE(data=X, model=model)
    ale.fit(alg_params={"nof_bins": K_ref})
    y1, _, _ = ale.eval(x, s=s)

    err_ale_mse = []
    err_ale_nmse = []
    err_ale_var = []
    for k in K_list:
        ale = fe.ALE(data=X_small, model=model)
        ale.fit(alg_params={"nof_bins": k})
        y2, _, _ = ale.eval(x, s=s)
        tmp = np.mean(np.square(y1 - y2))
        err_ale_mse.append(tmp)
        err_ale_var.append(y1.var())
        err_ale_nmse.append(tmp/y1.var())

    return err_ale_nmse, err_ale_mse,err_ale_var, err_dale_nmse, err_dale_mse, err_dale_var


# comparison for hour
def comparison_systematic(s, X, X_small):
    err_ale_nmse, err_ale_mse,err_ale_var, err_dale_nmse, err_dale_mse, err_dale_var = mse(X, X_small, s=s, K_ref=200, K_list=[1,2,3,4,5,6,7,8,9,10, 15, 20, 25, 50, 100])


    # plt.figure()
    # plt.plot([25, 50, 100], err_ale_nmse, label="ALE")
    # plt.plot([25, 50, 100], err_dale_nmse, label="DALE")
    # plt.legend()
    # plt.show(block=False)
    return err_ale_nmse, err_ale_mse,err_ale_var, err_dale_nmse, err_dale_mse, err_dale_var


mapping = {"0": "\mathtt{year}",
           "1": "\mathtt{month}",
           "2": "\mathtt{hour}",
           "3": "\mathtt{holiday}",
           "4": "\mathtt{weekday}",
           "5": "\mathtt{workingday}",
           "6": "\mathtt{weather-situation}",
           "7": "\mathtt{temp}",
           "8": "\mathtt{atemp}",
           "9": "\mathtt{hum}",
           "10": "\mathtt{windspeed}"
           }


# freeze seed
np.random.seed(138232)
python_random.seed(1244343)
tf.random.set_seed(12384)

# init data
data_init = pd.read_csv('./data/Bike-Sharing-Dataset/hour.csv')
data = copy.deepcopy(data_init)
cols = ["yr", "mnth", "hr", "holiday", "weekday", "workingday",
        "weathersit", "temp", "atemp", "hum", "windspeed"]
X_df = data.loc[:, cols]
X = X_df.to_numpy()
Y_df = data.loc[:,'cnt']
Y = Y_df.to_numpy()

# init - train nn
nn = create_nn(X)
nn.fit(X, Y, epochs=20)

# freeze model
model = create_model(nn)
model_grad = create_model_grad(nn)

K = 100
s=2
ale = fe.ALE(data=X, model=model)
ale.fit(alg_params={"nof_bins": K})
# ale.plot_local_effects(s=s, K=K)
# ale.plot(s=s)

dale = fe.DALE(data=X, model=model, model_jac=model_grad)
dale.fit(alg_params={"nof_bins": K})
# dale.plot_local_effects(s=s)
# dale.plot(s=s)


# feature effect
def plot_feature(s, savefig=False):
    plt.figure()
    x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "-", color="black", label="$f_{\mathtt{DALE}}$")

    y1, _, _ = ale.eval(x, s=s)
    plt.title("Feature effect of $X_{" + mapping[str(s)] + "}$")
    plt.plot(scale_x(x, s), y1, "--", color="dodgerblue", label="$\hat{f}_{\mathtt{ALE}}$")

    plt.xlabel(r"$X_{" + mapping[str(s)] + "}$")
    plt.ylabel("$Y_{\mathtt{counts}}$")
    plt.legend()
    if savefig:
        tplt.clean_figure()
        tplt.save(savefig)

    plt.show(block=False)

# plot all
for i in range(11):
    if savefig:
        plot_feature(i, savefig=path+"bike-dataset-fe-" + str(i) + ".tex")
    else:
        plot_feature(i, savefig=False)

# # save some
# for i in [1, 2, 8, 10]:
#     plot_feature(i, savefig=path+"bike-dataset-fe-" + str(i) + ".tex")


def plot_comparison_different_K(X, s, savefig=False):
    ale = fe.ALE(data=X, model=model)
    dale = fe.DALE(data=X, model=model, model_jac=model_grad)

    # ale part
    plt.figure()
    plt.title("ALE effect $X_{\mathtt{hour}}$")
    x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)
    ale.fit(alg_params={"nof_bins": 100})
    y2, _, _ = ale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", color="red", label="K=100")
    ale.fit(alg_params={"nof_bins": 50})
    y2, _, _ = ale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=50")
    ale.fit(alg_params={"nof_bins": 25})
    y2, _, _ = ale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=25")
    # ale.fit(alg_params={"nof_bins": 10})
    # y2, _, _ = ale.eval(x, s=s)
    # plt.plot(scale_x(x, s), y2, "--", label="K=10")
    # ale.fit(alg_params={"nof_bins": 5})
    # y2, _, _ = ale.eval(x, s=s)
    # plt.plot(scale_x(x, s), y2, "--", label="K=5")
    plt.xlabel("$X_{\mathtt{hour}}$")
    plt.ylabel("$Y_{\mathtt{counts}}$")
    plt.legend()

    if savefig:
        tplt.clean_figure()
        tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/bike-dataset-ale-comparison.tex")
    plt.show(block=False)

    # dale part
    plt.figure()
    plt.title("DALE effect $X_{\mathtt{hour}}$")
    x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)
    dale.fit(alg_params={"nof_bins": 100})
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", color="red", label="K=100")
    dale.fit(alg_params={"nof_bins": 50})
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=50")
    dale.fit(alg_params={"nof_bins": 25})
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=25")
    # dale.fit(alg_params={"nof_bins": 10})
    # y2, _, _ = dale.eval(x, s=s)
    # plt.plot(scale_x(x, s), y2, "--", label="K=10")
    # dale.fit(alg_params={"nof_bins": 5})
    # y2, _, _ = dale.eval(x, s=s)
    # plt.plot(scale_x(x, s), y2, "--", label="K=5")
    plt.legend()
    plt.xlabel("$X_{\mathtt{hour}}$")
    plt.ylabel("$Y_{\mathtt{counts}}$")
    if savefig:
        tplt.clean_figure()
        tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/bike-dataset-dale-comparison.tex")
    plt.show(block=False)

savefig = True
plot_comparison_different_K(X, s=2, savefig=path)





err_dale_mse_list = []
err_dale_nmse_list = []
err_dale_var_list = []

err_ale_mse_list = []
err_ale_nmse_list = []
err_ale_var_list = []

s = 2
for pcg in [100]:
    N = int(X.shape[0] * pcg / 100)
    ind = np.arange(X.shape[0])
    np.random.shuffle(ind)
    X_small = X[ind[:N], :]

    err_ale_nmse, err_ale_mse, err_ale_var, err_dale_nmse, err_dale_mse, err_dale_var = comparison_systematic(s, X, X)

    err_ale_nmse_list.append(err_ale_nmse)
    err_ale_mse_list.append(err_ale_mse)
    err_ale_var_list.append(err_ale_var)

    err_dale_nmse_list.append(err_dale_nmse)
    err_dale_mse_list.append(err_dale_mse)
    err_dale_var_list.append(err_dale_var)


# # isolate less points
# N = 5000
# ind = np.arange(X.shape[0])
# np.random.shuffle(ind)
# X_small = X[ind[:N], :]
# Y_small = Y[ind[N:]]

# # ale_small = fe.ALE(data=X_small, model=model)
# # ale_small.fit(alg_params={"nof_bins": K})

# # dale_small = fe.DALE(data=X_small, model=model, model_jac=model_grad)
# # dale.small(alg_params={"nof_bins": K})

# plot_comparison_different_K(X_small, s=2)


# # def plot_feature(s):
# #     plt.figure()
# #     x = np.linspace(X_small[:, s].min(), X_small[:, s].max(), 1000)
# #     y2, _, _ = dale.eval(x, s=s)
# #     plt.plot(scale_x(x, s), y2, "--", color="black", label="$f_{\mathtt{DALE}}$")

# #     y1, _, _ = ale.eval(x, s=s)
# #     plt.title(r"$X_{" + mapping[str(s)] + "}$")
# #     plt.plot(scale_x(x, s), y1, "--", color="dodgerblue", label="$\hat{f}_{\mathtt{ALE}}$")

# #     plt.legend()
# #     plt.show(block=False)


# # for i in range(11):
# #     plot_feature(i)


# # def compare_DALE_ALE(X, K):
# #     ale = fe.ALE(data=X, model=model)
# #     dale = fe.DALE(data=X, model=model, model_jac=model_grad)

# #     mse_list = []
# #     nmse_list = []
# #     var_list = []
# #     for s in range(X.shape[1]):
# #         x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)

# #         # DALE
# #         dale.fit(alg_params={"nof_bins": K})
# #         y1, _, _ = dale.eval(x, s=s)

# #         # ALE
# #         ale.fit(alg_params={"nof_bins": K})
# #         y2, _, _ = ale.eval(x, s=s)

# #         mse = np.mean(np.square(y1 - y2))
# #         mse_list.append(mse)
# #         var_list.append(y2.var())
# #         nmse_list.append(mse/y2.var())
# #     return mse_list, var_list, nmse_list

# # K = 1000
# # mse_list, var_list, nmse_list = compare_DALE_ALE(X, K)

import examples.utils as utils
path = utils.add_parent_path()
import numpy as np
import pandas as pd
import random as python_random
import tensorflow as tf
from tensorflow import keras
import copy
import feature_effect as fe
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

savefig = True
def create_nn(X):
    nn = keras.Sequential([keras.layers.Input(shape=[X.shape[1]]),
                           keras.layers.Dense(256, activation='relu', use_bias=True),
                           keras.layers.Dense(128, activation='relu', use_bias=True),
                           keras.layers.Dense(32, activation='relu', use_bias=True),
                           keras.layers.Dense(1, use_bias=True)])

    nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.metrics.MeanAbsoluteError(),
                       tf.keras.metrics.MeanSquaredError()])

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


def plot_local_effects(s, X, dfdx, mu, sigma):
    plt.figure()
    plt.title("Local Effects of feature: " + cols[s])
    plt.plot(X[:, s]*sigma+ mu, dfdx[:, s], "rx")
    plt.show(block=False)


def load_prepare_data(path):
    data_init = pd.read_csv(path)
    data = copy.deepcopy(data_init)
    cols = list(data.columns)

    data = data.dropna()
    X_df = data.iloc[:, :-2]
    X = X_df.to_numpy()

    Y_df = data.iloc[:, -2]
    Y = Y_df.to_numpy()

    return X_df, Y_df, X, Y

def preprocess_data(X, Y):
    # remove points 3 std away from the mean
    tmp_mean = X.mean(0)
    tmp_std = X.std(0)
    tmp = (X - tmp_mean) / tmp_std
    ind = np.sum(np.abs(tmp) > 3, axis=1) == 0
    X = X[ind,:]
    Y = Y[ind]

    # normalize
    X_mu = X.mean(0)
    X_sigma = X.std(0)
    X_norm = (X-X_mu)/X_sigma

    Y_mu = Y.mean()
    Y_sigma = Y.std()
    Y_norm = (Y-Y_mu)/Y_sigma
    return X_norm, Y_norm, X_mu, X_sigma, Y_mu, Y_sigma

def split(X_norm, Y_norm):
   X_tr, X_te, Y_tr, Y_te =  train_test_split(X_norm, Y_norm, test_size=.2)
   return X_tr, X_te, Y_tr, Y_te

def find_axislimits(X):
    axis_limits = np.array([np.min(X, 0), np.max(X, 0)])
    axis_limits[:, 0] = [-1.7, 1.3]
    axis_limits[:, 1] = [-1.5, 1.7]
    axis_limits[:, 3] = [-1.2, 2.3]
    axis_limits[:, 4] = [-1.27, 2.5]
    axis_limits[:, 5] = [-1.25, 2.]
    axis_limits[:, 6] = [-1.3, 2.2]
    axis_limits[:, 7] = [-1.77, 2.1]
    return axis_limits


# freeze seed
np.random.seed(138232)
python_random.seed(1244343)
tf.random.set_seed(12384)

folder_name = "real_dataset_3"

# load-prepare data
parent_path = os.path.dirname(os.path.dirname(path))
datapath = os.path.join(parent_path, "data", "California-Housing", "housing.csv")
X_df, Y_df, X, Y = load_prepare_data(datapath)

# preprocess data
X_norm, Y_norm, X_mu, X_sigma, Y_mu, Y_sigma = preprocess_data(X, Y)
axis_limits = find_axislimits(X_norm)

X_tr, X_te, Y_tr, Y_te = split(X_norm, Y_norm)

# Train
nn = create_nn(X_tr)
nn.fit(X_tr, Y_tr, epochs=15)

# Eval
results = nn.evaluate(X_te, Y_te)

# freeze model
model = create_model(nn)
model_grad = create_model_grad(nn)
dfdx = model_grad(X_norm)


stats_fixed_list = []
stats_auto_list = []


# fit feature
for s in range(8):
    scale_x = {"mean": X_mu[s], "std": X_sigma[s]}
    scale_y = {"mean": Y_mu, "std": Y_sigma}

    print("Feature " + str(s))
    dale = fe.DALE(X_tr, model, model_grad, axis_limits)
    alg_params = {"bin_method" : "dp",
                  "max_nof_bins" : 20,
                  "min_points_per_bin": 30}
    dale.fit(features=s, alg_params=alg_params)
    # dale.plot(s=s)


    dale_gt = fe.DALE(X_tr, model, model_grad, axis_limits)
    alg_params = {"bin_method" : "fixed",
                  "nof_bins" : 80,
                  "min_points_per_bin": 30}
    dale_gt.fit(features=s, alg_params=alg_params)
    # dale_gt.plot(s)


    # get statistics
    K_list = list(range(1,40))
    stats_fixed = utils.measure_fixed_error_real_dataset(dale_gt,
                                                         X_tr,
                                                         model,
                                                         model_grad,
                                                         axis_limits,
                                                         K_list,
                                                         nof_iterations=30,
                                                         nof_points=1000,
                                                         feature=s)
    stats_fixed_list.append(stats_fixed)
    stats_auto = utils.measure_auto_error_real_dataset(dale_gt,
                                                       X_tr,
                                                       model,
                                                       model_grad,
                                                       axis_limits,
                                                       nof_iterations=30,
                                                       nof_points=1000,
                                                       feature=s)
    stats_auto_list.append(stats_auto)

    # plots
    path2dir = os.path.join(path, folder_name)
    savepath = os.path.join(path2dir, "compare_mu_err_feature_" + str(s) + ".pdf") if savefig else None
    utils.plot_fixed_vs_auto(K_list,
                             stats_fixed["mu_err_mean"],
                             stats_fixed["mu_err_std"],
                             stats_auto["mu_err_mean"],
                             stats_auto["mu_err_std"],
                             "mu",
                             scale_y,
                             savefig=savepath)

    savepath = os.path.join(path2dir, "compare_var_err_feature_" + str(s) + ".pdf") if savefig else None
    utils.plot_fixed_vs_auto(K_list,
                             stats_fixed["var_err_mean"],
                             stats_fixed["var_err_std"],
                             stats_auto["var_err_mean"],
                             stats_auto["var_err_std"],
                             "var",
                             scale_y,
                             savefig=savepath)

    savepath = os.path.join(path2dir, "compare_rho_feature_" + str(s) + ".pdf") if savefig else None
    utils.plot_fixed_vs_auto(K_list,
                             stats_fixed["rho_mean"],
                             stats_fixed["rho_std"],
                             stats_auto["rho_mean"],
                             stats_auto["rho_std"],
                             "rho",
                             scale_y,
                             savefig=savepath)


path2dir = os.path.join(path, folder_name)
for s in range(8):
    scale_x = {"mean": X_mu[s], "std": X_sigma[s]}
    scale_y = {"mean": Y_mu, "std": Y_sigma}

    print("Feature " + str(s))
    ind = np.random.choice(X_tr.shape[0], size=200, replace=False)
    pdp_ice = fe.PDPwithICE(X_tr[ind], model, axis_limits)
    savepath = os.path.join(path2dir, "feature_" + str(s) +  "_pdp_ice.pdf") if savefig else None
    pdp_ice.plot(s=s, scale_x=scale_x, scale_y=scale_y,
                 normalized=True, nof_points=50, savefig=savepath)

    dale = fe.DALE(X_tr, model, model_grad, axis_limits)
    alg_params = {"bin_method" : "dp",
                  "max_nof_bins" : 20,
                  "min_points_per_bin": 30}
    dale.fit(features=s, alg_params=alg_params)
    savepath = os.path.join(path2dir, "feature_" + str(s) +  "_ale_auto.pdf") if savefig else None
    dale.plot(s=s, scale_x=scale_x, scale_y=scale_y, savefig=savepath)

    # dale_gt = fe.DALE(X_tr, model, model_grad, axis_limits)
    # alg_params = {"bin_method" : "fixed",
    #               "nof_bins" : 80,
    #               "min_points_per_bin": 30}

    # dale_gt.fit(features=s, alg_params=alg_params)
    # savepath = os.path.join(path2dir, "feature_" + str(s) +  "_ale_fixed.pdf") if savefig else None
    # dale_gt.plot(s, savefig=savepath)

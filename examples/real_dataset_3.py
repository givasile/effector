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

def create_nn(X):
    nn = keras.Sequential([keras.layers.Input(shape=[X.shape[1]]),
                           # keras.layers.Dense(512, activation='relu', use_bias=True),
                           # keras.layers.Dense(256, activation='relu', use_bias=True),

                           # keras.layers.Dense(256, activation='relu', use_bias=True),

                           keras.layers.Dense(256, activation='relu', use_bias=True),                              keras.layers.Dense(128, activation='relu', use_bias=True),
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
    X_mu = X.mean(0)
    X_sigma = X.std(0)
    X_norm = (X-X_mu)/X_sigma

    Y_mu = Y.mean()
    Y_sigma = Y.std()
    Y_norm = (Y-Y_mu)/Y_sigma
    return X_norm, Y_norm, X_mu, X_sigma, Y_mu, Y_sigma


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
savefig = True


# load-prepare data
parent_path = os.path.dirname(os.path.dirname(path))
datapath = os.path.join(parent_path, "data", "California-Housing", "housing.csv")
X_df, Y_df, X, Y = load_prepare_data(datapath)

# preprocess data
X_norm, Y_norm, X_mu, X_sigma, Y_mu, Y_sigma = preprocess_data(X, Y)
axis_limits = find_axislimits(X_norm)

# train nn
nn = create_nn(X_norm)
nn.fit(X_norm, Y_norm, epochs=10)

# freeze model
model = create_model(nn)
model_grad = create_model_grad(nn)
dfdx = model_grad(X_norm)

# fit feature
for s in range(8):
    dale = fe.DALE(X_norm, model, model_grad, axis_limits)
    alg_params = {"bin_method" : "dp",
                  "max_nof_bins" : 50,
                  "min_points_per_bin": 30}
    dale.fit(features=s, alg_params=alg_params)
    dale.plot(s=s)


    dale_gt = fe.DALE(X_norm, model, model_grad, axis_limits)
    alg_params = {"bin_method" : "fixed",
                  "nof_bins" : 60,
                  "min_points_per_bin": 30}
    dale_gt.fit(features=s, alg_params=alg_params)
    dale_gt.plot(s)


    # get statistics
    K_list = list(range(1,20))
    stats_fixed = utils.measure_fixed_error_real_dataset(dale_gt,
                                                         X_norm,
                                                         model,
                                                         model_grad,
                                                         axis_limits,
                                                         K_list,
                                                         nof_iterations=10,
                                                         nof_points=1000,
                                                         feature=s)

    stats_auto = utils.measure_auto_error_real_dataset(dale_gt,
                                                       X_norm,
                                                       model,
                                                       model_grad,
                                                       axis_limits,
                                                       nof_iterations=10,
                                                       nof_points=2500,
                                                       feature=s)

    # plots
    path2dir = os.path.join(path, folder_name)
    savepath = os.path.join(path2dir, "compare_mu_err_feature_" + str(s) + ".png") if savefig else None
    utils.plot_fixed_vs_auto(K_list,
                             stats_fixed["mu_err_mean"],
                             stats_fixed["mu_err_std"],
                             stats_auto["mu_err_mean"],
                             stats_auto["mu_err_mean"],
                             "mu",
                             savefig=savepath)

    savepath = os.path.join(path2dir, "compare_var_err_feature_" + str(s) + ".png") if savefig else None
    utils.plot_fixed_vs_auto(K_list,
                             stats_fixed["var_err_mean"],
                             stats_fixed["var_err_std"],
                             stats_auto["var_err_mean"],
                             stats_auto["var_err_mean"],
                             "var",
                             savefig=savepath)

    savepath = os.path.join(path2dir, "compare_rho_feature_" + str(s) + ".png") if savefig else None
    utils.plot_fixed_vs_auto(K_list,
                             stats_fixed["rho_mean"],
                             stats_fixed["rho_std"],
                             stats_auto["rho_mean"],
                             stats_auto["rho_mean"],
                             "rho",
                             savefig=savepath)

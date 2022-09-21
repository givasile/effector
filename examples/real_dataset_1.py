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


def plot_local_effects(s, X, dfdx):
    plt.figure()
    plt.title("Local Effects of feature: " + cols[s+1])
    plt.plot(X[:, s], dfdx[:, s], "rx")
    plt.show(block=False)


# freeze seed
np.random.seed(138232)
python_random.seed(1244343)
tf.random.set_seed(12384)

# init data
parent_path = os.path.dirname(os.path.dirname(path))
data_init = os.path.join(parent_path, "data", "Bike-Sharing-Dataset", "hour.csv")
data_init = pd.read_csv(data_init)
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
dfdx = model_grad(X)


# FEATURE 9
s = 9
plot_local_effects(s, X, dfdx)

# fixed-bins
dale = fe.DALE(data=X, model=model, model_jac=model_grad)
K = 10
dale.fit(features=s,alg_params={"nof_bins": K, "min_points_per_bin":2})
dale.plot(s, error="std")

# auto-bins
K_max = 50
dale.fit(features=s, alg_params={"bin_method":"dp",
                                 "max_nof_bins": K_max,
                                 "min_points_per_bin":20})
dale.plot(s=s)


# FEATURE 8
s = 8
plot_local_effects(s, X, dfdx)

# fixed-bins
dale = fe.DALE(data=X, model=model, model_jac=model_grad)
K = 10
dale.fit(features=s,alg_params={"nof_bins": K, "min_points_per_bin":2})
dale.plot(s, error="std")

# auto-bins
K_max = 50
dale.fit(features=s, alg_params={"bin_method":"dp",
                                 "max_nof_bins": K_max,
                                 "min_points_per_bin":20})
dale.plot(s=s)


# FEATURE 4
s = 4
plot_local_effects(s, X, dfdx)

# fixed-bins
dale = fe.DALE(data=X, model=model, model_jac=model_grad)
K = 6
dale.fit(features=s,alg_params={"nof_bins": K, "min_points_per_bin":2})
dale.plot(s, error="std")

# auto-bins
K_max = 50
dale.fit(features=s, alg_params={"bin_method":"dp",
                                 "max_nof_bins": K_max,
                                 "min_points_per_bin":20})
dale.plot(s=s)


# FEATURE 2
s = 2
plot_local_effects(s, X, dfdx)

# fixed-bins
dale = fe.DALE(data=X, model=model, model_jac=model_grad)
K = 24
dale.fit(features=s,alg_params={"nof_bins": K, "min_points_per_bin":2})
dale.plot(s, error="std")

# auto-bins
K_max = 48
dale.fit(features=s, alg_params={"bin_method":"dp",
                                 "max_nof_bins": K_max,
                                 "min_points_per_bin":20})
dale.plot(s=s)

for s in range(11):
    plot_feature(s, X, ale, dale, savefig)

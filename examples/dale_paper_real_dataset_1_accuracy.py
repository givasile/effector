import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tensorflow import keras
import tensorflow as tf
import random as python_random
import timeit
matplotlib.rcParams['text.usetex'] = True
save_fig = True
import feature_effect as fe


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

# K = 100
# s = 10
# ale = fe.ALE(data=X, model=model)
# ale.fit(alg_params={"nof_bins": K})
# ale.plot_local_effects(s=s, K=K)
# ale.plot(s=s)

# dale = fe.DALE(data=X, model=model, model_jac=model_grad)
# dale.fit(alg_params={"nof_bins": K})
# dale.plot_local_effects(s=s)
# dale.plot(s=s)


# feature effect
def plot_comparison_different_K(X):
    s = 2
    ale = fe.ALE(data=X, model=model)
    dale = fe.DALE(data=X, model=model, model_jac=model_grad)

    # ale part
    plt.figure()
    plt.title("ALE different K")
    x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)
    ale.fit(alg_params={"nof_bins": 100})
    y2, _, _ = ale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", color="dodgerblue", label="K=100")
    ale.fit(alg_params={"nof_bins": 23})
    y2, _, _ = ale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=24")
    ale.fit(alg_params={"nof_bins": 11})
    y2, _, _ = ale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=12")
    ale.fit(alg_params={"nof_bins": 7})
    y2, _, _ = ale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=8")
    plt.legend()
    plt.show(block=False)

    # dale part
    plt.figure()
    plt.title("DALE different K")
    x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)
    dale.fit(alg_params={"nof_bins": 100})
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", color="black", label="K=100")
    dale.fit(alg_params={"nof_bins": 23})
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=24")
    dale.fit(alg_params={"nof_bins": 11})
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=12")
    dale.fit(alg_params={"nof_bins": 7})
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", label="K=8")
    plt.legend()
    plt.show(block=False)

plot_comparison_different_K(X)

# def plot_feature(s):
#     plt.figure()
#     x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)
#     y2, _, _ = dale.eval(x, s=s)
#     plt.plot(scale_x(x, s), y2, "--", color="black", label="$f_{\mathtt{DALE}}$")

#     y1, _, _ = ale.eval(x, s=s)
#     plt.title(r"$X_{" + mapping[str(s)] + "}$")
#     plt.plot(scale_x(x, s), y1, "--", color="dodgerblue", label="$\hat{f}_{\mathtt{ALE}}$")

#     plt.legend()
#     plt.show(block=False)


# for i in range(11):
#     plot_feature(i)


# isolate less points
K = 100
N = 1000
ind = np.arange(X.shape[0])
np.random.shuffle(ind)
X_small = X[ind[:N], :]
Y_small = Y[ind[N:]]

ale = fe.ALE(data=X_small, model=model)
ale.fit(alg_params={"nof_bins": K})

dale = fe.DALE(data=X_small, model=model, model_jac=model_grad)
dale.fit(alg_params={"nof_bins": K})

plot_comparison_different_K(X_small)

def plot_feature(s):
    plt.figure()
    x = np.linspace(X_small[:, s].min(), X_small[:, s].max(), 1000)
    y2, _, _ = dale.eval(x, s=s)
    plt.plot(scale_x(x, s), y2, "--", color="black", label="$f_{\mathtt{DALE}}$")

    y1, _, _ = ale.eval(x, s=s)
    plt.title(r"$X_{" + mapping[str(s)] + "}$")
    plt.plot(scale_x(x, s), y1, "--", color="dodgerblue", label="$\hat{f}_{\mathtt{ALE}}$")

    plt.legend()
    plt.show(block=False)


for i in range(11):
    plot_feature(i)


# systematic evaluation
def mse(X, s, dale, ale, K_ref, K_list):
    x = np.linspace(X[:, s].min(), X[:, s].max(), 1000)

    # DALE
    dale.fit(alg_params={"nof_bins": K_ref})
    y1, _, _ = dale.eval(x, s=s)

    err_dale = []
    for k in K_list:
        dale.fit(alg_params={"nof_bins": k})
        y2, _, _ = dale.eval(x, s=s)
        err_dale.append(np.mean(np.square(y1 - y2)))

    # ALE
    ale.fit(alg_params={"nof_bins": K_ref})
    y1, _, _ = ale.eval(x, s=s)

    err_ale = []
    for k in K_list:
        ale.fit(alg_params={"nof_bins": k})
        y2, _, _ = ale.eval(x, s=s)
        err_ale.append(np.mean(np.square(y1 - y2)))

    return err_ale, err_dale


# comparison for hour
def comparison_systematic(s):
    ale = fe.ALE(data=X, model=model)
    dale = fe.DALE(data=X, model=model, model_jac=model_grad)
    err_ale, err_dale = mse(X, s=s, dale=dale, ale=ale, K_ref=100,
                            K_list=np.arange(5, 41, 5))


    plt.figure()
    plt.plot(np.arange(5, 41, 5), err_ale, label="ALE")
    plt.plot(np.arange(5, 41, 5), err_dale, label="DALE")
    plt.legend()
    plt.show(block=False)


# comparison_systematic(s=1)
comparison_systematic(s=2)
# comparison_systematic(s=7)

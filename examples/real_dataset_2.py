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
                           keras.layers.Dense(256, activation='relu', use_bias=True),
                           keras.layers.Dense(256, activation='relu', use_bias=True),                                     keras.layers.Dense(256, activation='relu', use_bias=True),                                     keras.layers.Dense(256, activation='relu', use_bias=True),
                           keras.layers.Dense(128, activation='relu', use_bias=True),
                           keras.layers.Dense(32, activation='relu', use_bias=True),
                           keras.layers.Dense(1, use_bias=True)])

    nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                       tf.keras.metrics.BinaryCrossentropy()])

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


# freeze seed
np.random.seed(138232)
python_random.seed(1244343)
tf.random.set_seed(12384)

# init data
parent_path = os.path.dirname(os.path.dirname(path))
data_init = os.path.join(parent_path, "data", "Diabetes-Dataset", "diabetes.csv")
data_init = pd.read_csv(data_init)
data = copy.deepcopy(data_init)
cols = list(data.columns)

X_df = data.loc[:, cols[:-1]]
X = X_df.to_numpy()


Y_df = data.loc[:,'Outcome']
Y = Y_df.to_numpy()


mu = X.mean(0)
sigma = X.std(0)
X_norm = (X-mu)/sigma

# init - train nn
nn = create_nn(X_norm)
nn.fit(X_norm, Y, epochs=50)

# freeze model
model = create_model(nn)
model_grad = create_model_grad(nn)
dfdx = model_grad(X_norm)

# Plots all features
for s in range(len(cols)-1):
    plot_local_effects(s, X_norm, dfdx, mu[s], sigma[s])

# auto-bins
s = 1
plot_local_effects(s, X_norm, dfdx, mu[s], sigma[s])
dale = fe.DALE(data=X_norm, model=model, model_jac=model_grad)
K_max = 50
dale.fit(features=s, alg_params={"bin_method":"dp",
                                 "max_nof_bins": K_max,
                                 "min_points_per_bin":5})
dale.plot(s=s)

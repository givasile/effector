import pandas as pd
import copy
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tensorflow import keras
import tensorflow as tf
import random as python_randomy
import timeit
import tikzplotlib as tplt
matplotlib.rcParams['text.usetex'] = True
save_fig = True
import feature_effect as fe

savefig = True

np.random.seed(1232)
python_random.seed(12343)
tf.random.set_seed(1234)

# initialize the data
data_init = pd.read_csv('./data/Bike-Sharing-Dataset/hour.csv')
data = copy.deepcopy(data_init)


# part 1 -> measure time for computing
X = data.iloc[:, 2:-1].to_numpy()
Y = data.iloc[:, -1].to_numpy()

big_nn = keras.Sequential([keras.layers.Input(shape=[X.shape[1]]),
                           keras.layers.Dense(2048, activation='relu', use_bias=True),
                           keras.layers.Dense(1024, activation='relu', use_bias=True),
                           keras.layers.Dense(512, activation='relu', use_bias=True),
                           keras.layers.Dense(256, activation='relu', use_bias=True),
                           keras.layers.Dense(128, activation='relu', use_bias=True),
                           keras.layers.Dense(64, activation='relu', use_bias=True),
                           keras.layers.Dense(32, activation='relu', use_bias=True),
                           keras.layers.Dense(1, use_bias=True)])



big_nn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss="mean_squared_error",
              metrics="mean_absolute_error")


def model_grad(inp):
    """Computes the gradients of outputs w.r.t input image.

    Args:
        img_input: 4D image tensor
        top_pred_idx: Predicted label for the input image

    Returns:
        Gradients of the predictions w.r.t img_input
    """
    x_inp = tf.cast(inp, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(x_inp)
        preds = big_nn(x_inp)
    grads = tape.gradient(preds, x_inp)
    return grads.numpy()


def model(X):
    return big_nn(X).numpy()[:, 0]



time_ale = []
time_dale = []
D = np.arange(1, X.shape[1] + 1)
for nof_features in D:
    K = 100

    features = list(np.arange(nof_features))
    print(features)
    a = timeit.default_timer()
    ale = fe.ALE(data=X, model=model)
    ale.fit(features=features, alg_params={"nof_bins": K})
    time_ale.append(timeit.default_timer() - a)

    a = timeit.default_timer()
    dale = fe.DALE(data=X, model=model, model_jac=model_grad)
    dale.fit(features=features, alg_params={"nof_bins":K})
    time_dale.append(timeit.default_timer() - a)

plt.figure()
plt.plot(D, time_ale, "--o", color="dodgerblue", label="$\hat{f}_{\mathtt{ALE}}$")
plt.plot(D, time_dale, "--o", color="black", label="$f_{\mathtt{DALE}}$")
plt.xlabel("$D$")
plt.ylabel("time (seconds)")
plt.legend()
plt.title("Execution time")
if savefig:
    tplt.clean_figure()
    tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/bike-dataset-exec-time.tex")
plt.show(block=False)

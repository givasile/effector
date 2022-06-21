import numpy as np
import matplotlib.pyplot as plt
import feature_effect as fe
import timeit
import random as python_random
from tensorflow import keras
import tensorflow as tf
import tikzplotlib as tplt
savefig = True

# make experiment deterministic
np.random.seed(1232)
python_random.seed(12343)
tf.random.set_seed(1234)


def create_nn(X, nof_layers):
    nn = keras.Sequential()
    nn.add(keras.Input(shape=(X.shape[1])))
    for _ in range(nof_layers):
        nn.add(keras.layers.Dense(units=1024, activation='tanh'))
    nn.add(keras.layers.Dense(units=1, activation='tanh'))
    nn.compile()
    return nn


def create_model(nn):
    def ff(x):
        return nn(x)[:,0]
    return ff


def create_model_grad(nn):
    def ff_grad(inp):
        x_inp = tf.cast(inp, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(x_inp)
            preds = nn(x_inp)
        grads = tape.gradient(preds, x_inp)
        return grads.numpy()
    return ff_grad


def systematic_evaluation(layers_list, D_list, N_list):
    print("START")
    time_forward = []
    time_backward = []
    time_dale = []
    time_ale = []
    for jj, nof_layers in enumerate(layers_list):
        time_dale.append([])
        time_ale.append([])
        time_forward.append([])
        time_backward.append([])
        for ii, N in enumerate(N_list):
            time_dale[jj].append([])
            time_ale[jj].append([])
            time_forward[jj].append([])
            time_backward[jj].append([])
            for D in D_list:
                print("nof layers:",  nof_layers)
                print("N: ", N)
                print("D: ", D)

                # initi dataset
                X = np.random.randn(N, D).astype("float32")

                # init model
                nn = create_nn(X, nof_layers)
                model = create_model(nn)
                model_grad = create_model_grad(nn)
                tmp = model(X)
                tmp = model_grad(X)

                # init ale, dale
                dale = fe.DALE(X, model, model_grad)
                ale = fe.ALE(X, model)

                # measure exec times
                a = timeit.default_timer()
                y = model(X)
                time_forward[jj][ii].append(timeit.default_timer() - a)

                a = timeit.default_timer()
                X_der = model_grad(X)
                time_backward[jj][ii].append(timeit.default_timer() - a)

                a = timeit.default_timer()
                dale.fit(alg_params={"nof_bins":100})
                time_dale[jj][ii].append(timeit.default_timer() - a)

                a = timeit.default_timer()
                ale.fit(alg_params={"nof_bins":100})
                time_ale[jj][ii].append(timeit.default_timer() - a)
    print("END")
    return time_dale, time_ale, time_forward, time_backward


# plot 1
layers_list = [2]
N_list = [1000]
D_list = [5, 10, 15, 20, 25, 50, 100]
time_dale, time_ale, time_forward, time_backward = systematic_evaluation(layers_list, D_list, N_list)

plt.figure()
plt.plot(D_list, time_forward[0][0], "r--o", label="$f(x)$")
plt.plot(D_list, time_backward[0][0], "b--o", label="$\mathbf{J}$")
plt.plot(D_list, time_ale[0][0], "--o", color="dodgerblue", label="$\hat{f}_{\mathtt{ALE}}$")
plt.plot(D_list, time_dale[0][0], "--o", color="black", label="$\hat{f}_{\mathtt{DALE}}$")
plt.xlabel("$D$")
plt.ylabel("time (seconds)")
plt.title("DALE vs ALE: Light setup")
plt.legend()
if savefig:
    tplt.clean_figure()
    tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-1-plot-1.tex")
plt.show(block=False)


# plot 2
layers_list = [4]
N_list = [100000]
D_list = [5, 10, 15, 20, 25, 50, 100]
time_dale, time_ale, time_forward, time_backward = systematic_evaluation(layers_list, D_list, N_list)

plt.figure()
plt.plot(D_list, time_forward[0][0], "r--o", label="$f(x)$")
plt.plot(D_list, time_backward[0][0], "b--o", label="$\mathbf{J}$")
plt.plot(D_list, time_ale[0][0], "--o", color="dodgerblue", label="$\hat{f}_{\mathtt{ALE}}$")
plt.plot(D_list, time_dale[0][0], "--o", color="black", label="$\hat{f}_{\mathtt{DALE}}$")
plt.xlabel("$D$")
plt.ylabel("time (seconds)")
plt.title("DALE vs ALE: Heavy setup")
plt.legend()
if savefig:
    tplt.clean_figure()
    tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-1-plot-2.tex")
plt.show(block=False)


# plots 3 - 4
layers_list = [2]
N_list = [100, 1000, 10000, 50000, 100000]
D_list = [5, 10, 15, 20, 25, 100]
time_dale, time_ale, time_forward, time_backward = systematic_evaluation(layers_list, D_list, N_list)

mark = ['s', 'o', 'D', 'v', 'x']
col = ['r', 'g', 'b', 'y', 'purple']
N_lab = ['10^2', '10^3', '10^4', '2 x 10^4', '4 x 10^4']
plt.figure()
for ii, N in enumerate(N_list):
    plt.plot(D_list, time_ale[0][ii], linestyle='--',
             marker=mark[ii], color=col[ii],
             label="$N=" + str(N_lab[ii]) + "$")
plt.xlabel("$D$")
plt.ylabel("time (seconds)")
plt.title("Execution time ALE: $L=2$")
plt.legend()
if savefig:
    tplt.clean_figure()
    tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-1-plot-3.tex")
plt.show(block=False)

plt.figure()
for ii, N in enumerate(N_list):
    plt.plot(D_list, time_dale[0][ii], linestyle='--',
             marker=mark[ii], color=col[ii],
             label="$N=" + str(N_lab[ii]) + "$")
plt.xlabel("$D$")
plt.ylabel("time (seconds)")
plt.title("Execution time DALE: $L=2$")
plt.legend()
if savefig:
    tplt.clean_figure()
    tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-1-plot-4.tex")
plt.show(block=False)

# plot 5 - 6
layers_list = [2, 3, 5, 8, 10]
N_list = [1000]
D_list = [5, 10, 15, 20, 25, 50, 100]
time_dale, time_ale, time_forward, time_backward = systematic_evaluation(layers_list, D_list, N_list)

mark = ['s', 'o', 'D', 'v', 'x']
col = ['r', 'g', 'b', 'y', 'purple']
plt.figure()
for ii, L in enumerate(layers_list):
    plt.plot(D_list, time_ale[ii][0], linestyle='--',
             marker=mark[ii], color=col[ii],
             label="$L=" + str(L) + "$")
plt.xlabel("$D$")
plt.ylabel("time (seconds)")
plt.title("Execution time ALE: $N=10^3$")
plt.legend()
if savefig:
    tplt.clean_figure()
    tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-1-plot-5.tex")
plt.show(block=False)

plt.figure()
for ii, L in enumerate(layers_list):
    plt.plot(D_list, time_dale[ii][0], linestyle='--',
             marker=mark[ii], color=col[ii],
             label="$L=" + str(L) + "$")
plt.xlabel("$D$")
plt.ylabel("time (seconds)")
plt.title("Execution time DALE: $N=10^3$")
plt.legend()
if savefig:
    tplt.clean_figure()
    tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-1-plot-6.tex")
plt.show(block=False)

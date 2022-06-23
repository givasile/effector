import random as python_random
import copy
import keras
import numpy as np
import scipy.optimize
import importlib
import matplotlib
import tensorflow as tf
import matplotlib.pyplot as plt
import PyALE
from matplotlib import cm
import pandas as pd
import tikzplotlib as tplt
import feature_effect as fe
import matplotlib
savefig = False


np.random.seed(12532)
python_random.seed(1323)
tf.random.set_seed(12434)


def model(X: np.array) -> np.array:
    tau_2 = 1.2
    a = 7
    y = []

    diff = X[:,0] - X[:,1]
    ind1 = np.abs(diff) < tau_2
    ind2 = (diff >= tau_2)
    ind3 = (-diff >= tau_2)

    y = X[:,0]*X[:,1] + X[:,0]*X[:,2]
    if np.sum(ind2) > 0:
        y[ind2] = y[ind2] - (a*(X[ind2,0] - X[ind2,1])**2 - a*tau_2**2)
    if np.sum(ind3) > 0:
        y[ind3] = y[ind3] + (a*(X[ind3,0] - X[ind3,1])**2 - a*tau_2**2)
    return y


def model_jac(X: np.array) -> np.array:
    h= 1e-5
    y = []

    # for x1
    Xplus = copy.deepcopy(X)
    Xplus[:,0] += h
    Xminus = copy.deepcopy(X)
    Xminus[:,0] -= h
    y1 = (model(Xplus) - model(Xminus))/h/2

    # for x2
    Xplus = copy.deepcopy(X)
    Xplus[:,1] += h
    Xminus = copy.deepcopy(X)
    Xminus[:,1] -= h
    y2 = (model(Xplus) - model(Xminus))/h/2

    # for x2
    Xplus = copy.deepcopy(X)
    Xplus[:,2] += h
    Xminus = copy.deepcopy(X)
    Xminus[:,2] -= h
    y3 = (model(Xplus) - model(Xminus))/h/2

    y = np.stack((y1, y2, y3), axis=-1)
    return y


def generate_samples(N: int, samples_range) -> np.array:
    """Generates N samples

    :param N: nof samples
    :returns: (N, D)

    """
    # tmp1 = np.random.uniform(0, samples_range/4, size=int(N/2))
    # tmp2 = np.random.uniform(samples_range*3/4, samples_range, size=int(N/2))
    # x1 = np.concatenate([tmp1, tmp2])
    std = .1
    x1 = np.random.normal(1.5, std, size=int(N/5))
    x2 = np.random.normal(3., std, size=int(N/5))
    x3 = np.random.normal(5, std, size=int(N/5))
    x4 = np.random.normal(6.3, std, size=int(N/5))
    x5 = np.random.normal(8.2, std, size=int(N/5))
    # x1 = np.random.uniform(0, samples_range, size=N-2)
    x1 = np.concatenate([np.zeros(int(1)),
                         x1,
                         2*np.ones(1),
                         x2,
                         4.2*np.ones(1),
                         x3,
                         x4,
                         7.1*np.ones(1),
                         x5,
                         9.1*np.ones(1),
                         np.ones(int(1))*samples_range])
    x2 = x1 + np.random.normal(size=(x1.shape[0]))*.6
    x3 = np.random.normal(size=(x1.shape[0]))*20
    return np.stack([x1, x2, x3]).T


def plot_f(model, samples, nof_points, samples_range, savefig):
    x = np.linspace(-.1*samples_range, 1.1*samples_range, nof_points)
    y = np.linspace(-.1*samples_range, 1.1*samples_range, nof_points)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    z = model(np.concatenate([positions, np.zeros((positions.shape[0], 1))], axis=-1))
    zz = np.reshape(z, [x.shape[0], y.shape[0]])
    fig, ax = plt.subplots()
    cs = ax.contourf(xx, yy, zz, levels=400, vmin=-100, vmax=200., cmap=cm.viridis, extend='both')
    ax.plot(samples[:, 0], samples[:, 1], 'ro', label="samples")
    ax.plot(np.linspace(0, 10, 10), np.linspace(0, samples_range, 10), "r--")
    plt.title(r"$f(x_1, x_2, x_3=0)$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    if savefig:
        tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-2-f-gt.tex")
    plt.show(block=False)


def create_gt_effect(s):
    # find gt
    X_big = generate_samples(N=1000000, samples_range=samples_range)
    dale_gt = fe.DALE(data=X_big, model=model, model_jac=model_jac)
    dale_gt.fit(alg_params={"nof_bins": 5})

    def tmp(x):
        return dale_gt.eval(x, s)[0]
    return tmp

def plot_effect(K, savefig):
    plt.figure()
    plt.title("Effect Estimation: K = " + str(K))
    x = np.linspace(0,10, 1000)
    y_gt = ale_gt_0(x)

    # dale estimation
    dale = fe.DALE(data=X, model=model, model_jac=model_jac)
    dale.fit(alg_params={"nof_bins": K})
    y_dale = dale.eval(x=x, s=0)[0]

    # ale estimation
    ale = fe.ALE(data=X, model=model)
    ale.fit(alg_params={"nof_bins": K})
    y_ale = ale.eval(x=x, s=0)[0]

    plt.plot(x, y_gt, "r--", label="$f_{gt}$")
    plt.plot(x, y_ale, "--", color="dodgerblue", label="$\hat{f}_{\mathtt{ALE}}$")
    plt.plot(x, y_dale, "--", color="black", label="$f_{\mathtt{DALE}}$")
    plt.xlabel("$x_1$")
    plt.legend()
    if savefig:
        tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-2-fe-bins-" + str(K) + ".tex")
    plt.show(block=False)


savefig = True
# create gt effect
ale_gt_0 = create_gt_effect(s=0)

# generate and plot points
N = 1000
samples_range = 10
X = generate_samples(N, samples_range)
# plot_f(model=model, samples=X, nof_points=30, samples_range=samples_range, savefig=savefig)

K = 5
plot_effect(K, savefig)

K = 15
plot_effect(K, savefig)

K = 50
plot_effect(K, savefig)


def evaluate_many_K(K_list, X):
    x = np.linspace(0, 10, 1000)
    y1 = ale_gt_0(x)
    norm = y1.var()

    err_dale = []
    err_ale = []
    for k in K_list:
        dale = fe.DALE(data=X, model=model, model_jac=model_jac)
        dale.fit(alg_params={"nof_bins": k})
        y2 = dale.eval(x, s=0)[0]
        mse = np.mean((y2 - y1)**2)
        err_dale.append(mse/norm)
        ale = fe.ALE(data=X, model=model)
        ale.fit(alg_params={"nof_bins": k})
        y2 = ale.eval(x, s=0)[0]
        mse = np.mean((y2 - y1)**2)
        err_ale.append(mse/norm)

    return err_dale, err_ale

K_list = np.arange(1, 51)
dale_err, ale_err = evaluate_many_K(K_list, X)
print(np.array(ale_err) < .1)
print(np.array(dale_err) < .1)

# plot gt
N = 100
samples_range = 10
X = generate_samples(N, samples_range)
plot_f(model=model, samples=X, nof_points=30, samples_range=samples_range, savefig=savefig)

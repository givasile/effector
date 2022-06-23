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
savefig = True


np.random.seed(12532)
python_random.seed(1323)
tf.random.set_seed(12434)


def model(X: np.array) -> np.array:
    tau_2 = 0.5
    a = 10
    y = []

    diff = X[:,0] - X[:,1]
    ind1 = np.abs(diff) < tau_2
    ind2 = (diff >= tau_2)
    ind3 = (-diff >= tau_2)

    y = X[:,0]*X[:,1] # + X[:,0]**2/2
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

    y = np.stack((y1, y2), axis=-1)
    return y


def generate_samples(N: int, samples_range, noise) -> np.array:
    """Generates N samples

    :param N: nof samples
    :returns: (N, D)

    """
    std = .1
    x1 = np.random.uniform(0, samples_range, size=int(N))
    x2 = x1 + np.random.normal(size=(x1.shape[0]))*noise
    return np.stack([x1, x2]).T


def plot_f(model, samples, nof_points, samples_range, savefig):
    x = np.linspace(-.1*samples_range, 1.1*samples_range, nof_points)
    y = np.linspace(-.1*samples_range, 1.1*samples_range, nof_points)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    z = model(positions)
    zz = np.reshape(z, [x.shape[0], y.shape[0]])
    fig, ax = plt.subplots()
    cs = ax.contourf(xx, yy, zz, levels=400, vmin=-100, vmax=200., cmap=cm.viridis, extend='both')
    if samples is not None:
        ax.plot(samples[:, 0], samples[:, 1], 'ro', label="samples")
    ax.plot(np.linspace(0, 10, 10), np.linspace(0, samples_range, 10), "r--", label="samples axis")
    plt.title(r"$f(x_1, x_2)$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    if savefig:
        tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/OOD-1.tex")
    plt.show(block=False)


N = 10
samples_range = 10
noise = 0.
X = generate_samples(N, samples_range, noise)
plot_f(model=model, samples=None, nof_points=30, samples_range=samples_range, savefig=savefig)


def model(X: np.array) -> np.array:
    tau_2 = 0.5
    a = 1
    y = []

    diff = X[:,0] - X[:,1]
    ind1 = np.abs(diff) < tau_2
    ind2 = (diff >= tau_2)
    ind3 = (-diff >= tau_2)

    y = X[:,0]*X[:,1] # + X[:,0]**2/2
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

    y = np.stack((y1, y2), axis=-1)
    return y


def generate_samples(N: int, samples_range, noise) -> np.array:
    """Generates N samples

    :param N: nof samples
    :returns: (N, D)

    """
    std = .1
    x1 = np.random.uniform(0, samples_range, size=int(N))
    x2 = x1 + np.random.normal(size=(x1.shape[0]))*noise
    return np.stack([x1, x2]).T


def plot_f(model, samples, nof_points, samples_range, savefig):
    x = np.linspace(-.1*samples_range, 1.1*samples_range, nof_points)
    y = np.linspace(-.1*samples_range, 1.1*samples_range, nof_points)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()]).T
    z = model(positions)
    zz = np.reshape(z, [x.shape[0], y.shape[0]])
    fig, ax = plt.subplots()
    cs = ax.contourf(xx, yy, zz, levels=400, vmin=-100, vmax=200., cmap=cm.viridis, extend='both')
    if samples is not None:
        ax.plot(samples[:, 0], samples[:, 1], 'ro', label="samples")
    ax.plot(np.linspace(0, 10, 10), np.linspace(0, samples_range, 10), "r--")
    plt.title(r"$f(x_1, x_2)$")
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.legend()
    if savefig:
        tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/case-2-f-gt.tex")
    plt.show(block=False)


N = 10000
samples_range = 10
noise = 0.0
X = generate_samples(N, samples_range, noise)

dale_effect = []
ale_effect = []
for k in np.arange(1, 101):
    dale = fe.DALE(data=X, model=model, model_jac=model_jac)
    dale.fit(alg_params={"nof_bins": k})
    dale_effect.append(dale.dale_params["feature_0"]["bin_effect"][0])

    ale = fe.ALE(data=X, model=model)
    ale.fit(alg_params={"nof_bins": k})
    ale_effect.append(ale.ale_params["feature_0"]["bin_effect"][0])


K = 20
plt.figure()
plt.title("DALE vs ALE")
plt.plot(np.arange(1, K+1), ale_effect[:K], "--o", color="dodgerblue", label="ALE")
plt.plot(np.arange(1, K+1), dale_effect[:K], "--o", color="black", label="DALE")
plt.plot(np.arange(1, K+1), 5/np.arange(1,K+1), "--x", markersize=10, color="red", label="ground-truth")
plt.ylabel("Local effect")
plt.xlabel("number of bins $(K)$")
plt.legend()
if savefig:
    tplt.save("/home/givasile/projects-org/org-feature-effect/paper-acml/images/OOD-2.tex")
plt.show(block=False)

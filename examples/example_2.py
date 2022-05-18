import sys
import os
import feature_effect as fe
import numpy as np
import matplotlib.pyplot as plt


def generate_samples(N, seed):
    if seed is not None:
        np.random.seed(seed)

    x1 = np.random.uniform(size=N)
    x2 = np.random.uniform(size=N)
    return np.stack([x1, x2]).T


def f(x):
    t = 10.5
    return np.cos(2*np.pi*t*x[:,0]) + np.sin(2*np.pi*t*x[:,1])


def f_der(x):
    t = 10
    return np.stack([-2*np.pi*t*np.sin(2*np.pi*t*x[:,0]), 2*np.pi*t*np.cos(2*np.pi*t*x[:,1])], axis=-1)


seed = 1
N = 10
X = generate_samples(N, seed)
y = f(X)
dy = f_der(X)

dale = fe.DALE(data=X, model=f, model_jac=f_der)
dale.fit(features=[0, 1], k=80, method="variable-size")
dale.plot(s=0, block=False)


for feat in [0, 1]:
    print(dale.parameters["feature_" + str(feat)]["nof_bins"])

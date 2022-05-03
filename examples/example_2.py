import mdale.ale as ale
import mdale.dale as dale
import numpy as np
import matplotlib.pyplot as plt


def generate_samples(N, seed):
    if seed is not None:
        np.random.seed(seed)

    x1 = np.random.uniform(size=N)
    x2 = x1 + np.random.normal(size=N)
    return np.stack([x1, x2]).T


def f(x):
    return x[:,0]**2 + x[:,0]**2 * x[:,1]


def f_der(x):
    return np.stack([2*x[:,0]*(1 + x[:,0]), x[:,0]**2], axis=-1)


seed = 1
N = 20
X = generate_samples(N, seed)
y = f(X)
dy = f_der(X)


plt.figure()
plt.plot(X[:,0], X[:,1], "ro")
plt.show(block=False)


K = 100
# ALE
ale_inst = ale.ALE(points=X, f=f)
ale_inst.fit(features=[0, 1], k=K)
ale_inst.plot(s=0, block=False)

# DALE
dale_inst = dale.DALE(points=X, f=f, f_der=f_der)
dale_inst.fit(features=[0, 1], k=K)
dale_inst.plot(s=0, block=False)

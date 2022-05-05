import numpy as np
import matplotlib.pyplot as plt
import feature_effect as fe

def generate_samples(N, seed):
    if seed is not None:
        np.random.seed(seed)

    x1 = np.random.uniform(size=N)
    x2 = np.random.normal(size=N)
    return np.stack([x1, x2]).T


def f(x):
    return x[:,0]**2 + x[:,0]**2 * x[:,1]


def f_der(x):
    return np.stack([2*x[:,0]*(1 + x[:,1]), x[:,0]**2], axis=-1)


seed = 1
N = 200
X = generate_samples(N, seed)
y = f(X)
dy = f_der(X)


plt.figure()
plt.plot(X[:,0], X[:,1], "ro")
plt.show(block=False)


K = 10
# ALE
ale_inst = fe.ALE(data=X, model=f)
ale_inst.fit(features=[0, 1], k=K)
ale_inst.plot(s=0, block=False)

# DALE
dale_inst = fe.DALE(data=X, model=f, model_jac=f_der)
dale_inst.fit(features=[0, 1], k=K)
dale_inst.plot(s=0, block=False)

est = fe.Estimator(data=X, model=f, model_jac=f_der)
est.fit(features=[0, 1], method="all", nof_bins=K)
est.plot(feature=0, method="all")

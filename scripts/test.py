import effector
import numpy as np
import timeit

np.random.seed(21)

def f(x):
    return -x[:, 0]**2 * (x[:, 1] < 0) + x[:, 0]**2 * (x[:, 1] >= 0) + np.exp(x[:, 2])

def f_jac(x):
    return np.array([
        -2 * x[:, 0] * (x[:, 1] < 0) + 2 * x[:, 0] * (x[:, 1] >= 0),
        -x[:, 0]**2 * (x[:, 1] < 0) + x[:, 0]**2 * (x[:, 1] >= 0),
        np.exp(x[:, 2])
    ]).T

N = 100_000
D = 3
M = 1_000

X = np.random.uniform(-1, 1, (N, D))

#%%
# Global effect PDP - more control
pdp = effector.PDP(
    X,
    f,
    nof_instances="all"
)

tic = timeit.default_timer()
pdp.fit(
    features="all",
    centering="zero_integral",
    points_for_centering=M,
    use_vectorized=True
)
toc = timeit.default_timer()

print(f"Time: {toc - tic:.2f} s")


#%%
tic = timeit.default_timer()
pdp.fit(
    features="all",
    centering="zero_integral",
    points_for_centering=M,
    use_vectorized=False
)
toc = timeit.default_timer()

print(f"Time: {toc - tic:.2f} s")

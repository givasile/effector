import numpy as np
import time
import os
import sys

# hack to import mdale
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import effector.utils as utils


def measure_time(k, n):
    # create date
    k = int(k)
    n = int(n)

    limits = np.linspace(0, k + 1, k + 1)
    bin_effects = np.ones(k)
    xs = np.linspace(0, k, n)
    dx = 1.

    # measure time
    tic = time.time()
    predict = utils.compute_accumulated_effect(xs, limits, bin_effects, dx)
    toc = time.time()
    exec_time = toc - tic
    print("Time elapsed: %.2f, k: %d, n:%d" % (exec_time, k, n))
    return exec_time


def measure_exec_time():
    """Measure execution time for n, k in log-scale.

    The experiment shows that the method scales linear (??) in terms of both (a) data points and (b) bins

    .. math:: \mathcal{O}_f(n,k) = n + k

    """
    K = 8
    N = 8
    time_list = []
    ind = 0
    for k in np.logspace(1, K, K):
        time_list.append([])
        for n in np.logspace(1, N, N):
            time_list[ind].append(measure_time(k,n))
        ind += 1
    return np.array(time_list)


def measure_exec_time_for_k():
    """Measure execution time for k in log-scale.

    The experiment shows that the method scales linear in terms of number of bins

    .. math:: \mathcal{O}_f(k) = k

    """
    n = 1000
    K = 8
    time_list = []
    for k in np.logspace(1, K, K):
        k = int(k)
        measure_time(k, n)
    return np.array(time_list)


def measure_exec_time_for_n():
    """Measure execution time for n in log-scale.

    The experiment shows that the method scales linear in terms of number of data points

    .. math:: \mathcal{O}_f(n) = n

    """
    k = 1000
    N = 8
    time_list = []
    for n in np.logspace(1, N, N):
        n = int(n)
        measure_time(k, n)
    return np.array(time_list)


if __name__ == "__main__":
    measure_exec_time()
    measure_exec_time_for_k()
    measure_exec_time_for_n()

import random as python_random
import ale2
import utils
import numpy as np
import importlib
import pandas as pd
from alibi.explainers import ALE, plot_ale
ale2 = importlib.reload(ale2)
import tikzplotlib as tplt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True


def f_bb(x: np.array) -> np.array:
    tau = 3.
    if np.abs(x[:,1] - x[:,0]**2) <= tau:
        return x[:,0] + x[:,1] + x[:,0]*x[:,1]
    else:
        return x[:,0] + x[:,1] + x[:,0]*x[:,1] + np.e**(x[:,1]-x[:,0]**2) - np.e**tau


def f_der(x: np.array) -> np.array:
    return np.stack((x[:,1]+1, x[:,0]+1), axis=1)


def generate_samples(N: int) -> np.array:
    """Generates N samples

p    :param N: nof samples
    :returns: (N, D)

    """
    x1 = np.random.uniform(-5, 5, N)
    x2 = x1**2 + np.random.normal(size=(N))
    return np.stack([x1, x2]).T

def normalize(x):
    return x/np.expand_dims(np.linalg.norm(x, 2, axis=-1), -1)

# def ale_gt(x):
#     """TODO describe function

#     :param x:
#     :returns:

#     """
#     return x**2

for nof_bins in [3, 6, 12]:
    np.random.seed(34)
    python_random.seed(12343)
    tf.random.set_seed(1234)

    bins_x = np.linspace(-5, 5, nof_bins+1)
    bins_y = np.linspace(-5, 25, nof_bins+1)
    N = 5
    x = generate_samples(N)


    f_der(x)


    plt.figure()
    plt.plot(x[:,0], x[:, 1], 'ro', label="real samples")

    plt.vlines(bins_x[1:-1], ymin=-5, ymax=25, colors="blue", linestyles="solid")
    plt.hlines(bins_y[1:-1], xmin=-5, xmax=5, colors="blue", linestyles="solid")

    ind = np.digitize(x[:,0], bins_x)
    ind_y = np.digitize(x[:,1], bins_y)
    for i in range(x.shape[0]):
        # draw horizontal line
        plt.hlines(x[i,1], xmin=bins_x[ind[i]-1], xmax=bins_x[ind[i]],
                   linestyles='dotted', color='black')
        plt.plot(bins_x[ind[i]-1], x[i,1], 'bo')
        plt.plot(bins_x[ind[i]], x[i,1], 'bo')

        # draw vertical line
        plt.vlines(x[i,0], ymin=bins_y[ind_y[i]-1], ymax=bins_y[ind_y[i]],
                   linestyles='dotted', color='black')
        plt.plot(x[i,0], bins_y[ind_y[i]-1], 'bo')
        plt.plot(x[i,0], bins_y[ind_y[i]], 'bo')

        # compute gradient
        f_x1 = f_bb(np.array([[bins_x[ind[i]], x[i,1]]]))
        f_x0 = f_bb(np.array([[bins_x[ind[i]-1], x[i,1]]]))
        df_dx = f_x1 - f_x0

        f_y1 = f_bb(np.array([[x[i,0], bins_y[ind_y[i]]]]))
        f_y0 = f_bb(np.array([[x[i,0], bins_y[ind_y[i]-1]]]))
        df_dy = f_y1 - f_y0

        df = np.array([df_dx, df_dy]) / np.sqrt(df_dx**2 + df_dy**2)
        plt.quiver(x[i,0], x[i,1], df[0], df[1], scale=10)

    # just to add labels
    plt.quiver(x[i,0], x[i,1], df[0], df[1], scale=10, label="gradients")
    plt.plot(x[i,0], bins_y[ind_y[i]], 'bo', label="artificial samples")

    x1 = np.linspace(-5, 5, 1000)
    y1 = x1**2
    plt.plot(x1, y1, 'r--', label="generative distribution")
    plt.legend()
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("ALE")
    # tplt.save("./paper-ijcai/images/example-comparison-1-ALE.tex")
    plt.savefig("./paper-ijcai/images/example-comparison-1-ALE-" +str(nof_bins) +"-bins.png")
    plt.show(block=False)


    plt.figure()
    plt.quiver(x[:,0], x[:,1],
               normalize(f_der(x))[:,0],
               normalize(f_der(x))[:,1],
               scale=10, label="gradients")
    plt.plot(x[:,0], x[:, 1], 'ro', label="real samples")
    x1 = np.linspace(-5, 5, 1000)
    y1 = x1**2
    plt.plot(x1, y1, 'r--', label="generative distribution")

    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("DALE")
    plt.legend()
    # tplt.clean_figure()
    # tplt.save("./paper-ijcai/images/example-comparison-1-DALE.tex")
    plt.savefig("./paper-ijcai/images/example-comparison-1-DALE.png")
    plt.show(block=False)

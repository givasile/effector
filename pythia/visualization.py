import matplotlib.pyplot as plt
import numpy as np


def affine(x, transform):
     return x*transform["std"] + transform["mean"]


def scale(x, transform, square=False):
    if square:
        y = x*transform["std"]**2
    else:
        y = x*transform["std"]
    return y


def scale_bin(x, scale_x, scale_y, square=False):
    if square:
        y = x*(scale_y["std"]/scale_x["std"])**2
    else:
        y = x*scale_y["std"]/scale_x["std"]
    return y


def ale_plot(params,
             fe_func,
             feature,
             error,
             scale_x=None,
             scale_y=None,
             savefig=False):
    assert all(name in params for name in ["limits", "dx", "bin_estimator_variance", "bin_effect"])

    x = np.linspace(params["limits"][0], params["limits"][-1], 1000)
    y, std, std_err = fe_func(x, feature, True)

    # transform
    x = x if scale_x is None else affine(x, scale_x)
    y = y if scale_y is None else affine(y, scale_y)
    std = std if scale_y is None else scale(std, scale_y)
    std_err = std_err if scale_y is None else scale(std_err, scale_y)
    limits = params["limits"] if scale_x is None else affine(params["limits"], scale_x)
    dx = params["dx"] if scale_x is None else scale(params["dx"], scale_x)
    bin_variance = params["bin_variance"] if scale_y is None else scale_bin(params["bin_variance"], scale_x, scale_y, True)
    bin_effect = params["bin_effect"] if scale_y is None else scale_bin(params["bin_effect"], scale_x, scale_y)

    # PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title("ALE plot for: $x_{" + str(feature + 1) + "}$")

    # first subplot
    ale_curve(ax1, x, y, std_err, std, error=error)

    # second subplot
    ale_bins(ax2, bin_effect, bin_variance, limits, dx, error)

    ax1.set_ylabel("$y$")
    ax2.set_xlabel("$x_{%d}$" % (feature+1))
    ax2.set_ylabel("$\partial y / \partial x_{%d}$" % (feature+1))

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show(block=False)


def ale_curve(ax1, x, y, std_err, std, error=None):
    ax1.plot(x, y, "b--", label="$\hat{f}_{\mu}$")

    if error == "std":
        ax1.fill_between(x, y-std, y+std, color='red', alpha=0.2, label="$\hat{f}_{\sigma^2}$")
    elif error == "standard error":
        ax1.fill_between(x, y-2*std_err, y+2*std_err, color='red', alpha=0.6, label="standard error")
    elif error == "both":
        ax1.fill_between(x, y-std, y+std, color='red', alpha=0.2, label="std")
        ax1.fill_between(x, y-np.sqrt(std_err), y+np.sqrt(std_err), color='red', alpha=0.6, label="standard error")
    ax1.legend()


def ale_bins(ax2, bin_effects, bin_variance, limits, dx, error):
    bin_centers = (limits[:-1] + limits[1:]) / 2
    yerr = np.sqrt(bin_variance) if error else None
    ax2.bar(x=bin_centers,
            height=bin_effects,
            width=dx,
            color=(0.1, 0.1, 0.1, 0.1),
            edgecolor='blue',
            yerr=yerr,
            ecolor='red',
            label="$\hat{\mu}_k$")
    ax2.legend()


def plot_1D(x, y, title):
    plt.figure()
    plt.title(title)
    plt.plot(x, y, "b-")
    plt.show(block=False)


def plot_PDP_ICE(s, x, y_pdp, y_ice, scale_x, scale_y, savefig):
    x = x if scale_x is None else affine(x, scale_x)
    y_pdp = y_pdp if scale_y is None else affine(y_pdp, scale_y)
    y_ice = y_ice if scale_y is None else affine(y_ice, scale_y)

    plt.figure()
    plt.title("PDP-ICE: $x_{%d}$" % (s+1))
    plt.plot(x, y_ice[0,:], color="red", alpha=.1, label="$f_{\mathtt{ICE}}$")
    plt.plot(x, y_ice.T, color="red", alpha=.1)
    plt.plot(x, y_pdp, color="blue", label="$f_{\mu}$")
    plt.xlabel("$x_{%d}$" % (s+1))
    plt.ylabel("$y$")
    plt.legend()

    if savefig is not None:
        plt.savefig(savefig, bbox_inches="tight")
    plt.show(block=False)

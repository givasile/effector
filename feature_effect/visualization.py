import matplotlib.pyplot as plt
import numpy as np
from feature_effect.utils import compute_remaining_effect


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


def feature_effect_plot(params,
                        eval,
                        feature,
                        error,
                        min_points_per_bin,
                        title=None,
                        block=False,
                        gt=False,
                        gt_bins=None,
                        scale_x = None,
                        scale_y = None,
                        savefig=False):
    assert all(name in params for name in ["first_empty_bin", "limits", "dx", "is_bin_empty", "bin_estimator_variance", "bin_effect"])

    x = np.linspace(params["limits"][0], params["limits"][-1], 1000)
    y, std, estimator_var = eval(x, feature, True)
    rem_eff = compute_remaining_effect(x,
                                       params["limits"],
                                       np.sqrt(params["bin_variance"]),
                                       square=False)

    x = x if scale_x is None else affine(x, scale_x)
    y = y if scale_y is None else affine(y, scale_y)
    std = std if scale_y is None else scale(std, scale_y)
    estimator_var = estimator_var if scale_y is None else scale(estimator_var, scale_y, True)


    limits = params["limits"] if scale_x is None else affine(params["limits"], scale_x)
    dx = params["dx"] if scale_x is None else scale(params["dx"], scale_x)

    bin_variance = params["bin_variance"] if scale_y is None else scale_bin(params["bin_variance"], scale_x, scale_y, True)
    bin_effect = params["bin_effect"] if scale_y is None else scale_bin(params["bin_effect"], scale_x, scale_y)

    is_bin_empty = params["points_per_bin"] < min_points_per_bin

    bins_under_limit = np.argwhere(params["points_per_bin"] < min_points_per_bin)
    if bins_under_limit.shape[0] > 0:
        first_empty_bin = bins_under_limit[0][0]
    else:
        first_empty_bin = None


    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if title is None:
        ax1.set_title("UALE: $x_{" + str(feature + 1) + "}$")
    else:
        ax1.set_title(title)

    # first subplot
    feature_effect(ax1,
                   x,
                   y,
                   estimator_var,
                   std,
                   rem_eff,
                   first_empty_bin,
                   limits,
                   min_points_per_bin,
                   error=error, gt=gt)

    # second subplot
    effects_per_bin(ax2,
                    bin_effect,
                    bin_variance,
                    error,
                    is_bin_empty,
                    limits,
                    dx,
                    gt_bins,
                    min_points_per_bin)

    ax1.set_ylabel("$y$")

    ax2.set_xlabel("$x_{%d}$" % (feature+1))
    ax2.set_ylabel("$\partial y / \partial x_{%d}$" % (feature+1))

    if savefig:
        plt.savefig(savefig, bbox_inches="tight")

    if block is False:
        plt.show(block=False)
    else:
        plt.show()


def feature_effect(ax1, x, y, estimator_var, std, rem_eff, first_empty, limits, point_limit, error=True, gt=None):
    # first subplot
    ax1.plot(x, y, "b--", label="$\hat{f}_{\mu}$")

    if first_empty is not None:
        added_line = .3*(np.max(y) - np.min(y))
        ax1.vlines(x=[limits[first_empty], limits[first_empty+1]], ymin=np.min(y) - added_line, ymax=np.max(y) + added_line,
                   colors="red",
                   alpha=.7,
                   label="first bin with < " + str(point_limit) + " points")
    if error == "std":
        ax1.fill_between(x, y-std, y+std, color='red', alpha=0.2, label="$\hat{f}_{\sigma^2}$")
        # ax1.fill_between(x, y-rem_eff, y+rem_eff, color='blue', alpha=0.7, label="std")
    elif error == "standard error":
        ax1.fill_between(x, y-2*np.sqrt(estimator_var), y+2*np.sqrt(estimator_var), color='red', alpha=0.6, label="standard error")
        # ax1.fill_between(x, y-rem_eff, y+rem_eff, color='blue', alpha=0.7, label="std")
    elif error == "both":
        ax1.fill_between(x, y-std, y+std, color='red', alpha=0.2, label="std")
        ax1.fill_between(x, y-np.sqrt(estimator_var), y+np.sqrt(estimator_var), color='red', alpha=0.6, label="standard error")
        # ax1.fill_between(x, y-rem_eff, y+rem_eff, color='blue', alpha=0.7, label="std")


    if gt is not None:
        y = gt(x)
        ax1.plot(x, y, "m--", label="$\hat{f}^{\mathtt{GT}}_{\mu}$")
    ax1.legend()


def effects_per_bin(ax2, bin_effects, bin_variance, error, is_bin_empty, limits, dx, gt_bins=None, point_limit=10):
    # ax2.set_title("Effects per bin")
    bin_centers = (limits[:-1] + limits[1:]) / 2
    is_bin_full = ~np.array(is_bin_empty)
    if error:
        # bins with enough points
        if np.sum(is_bin_full) > 0:
            bars = ax2.bar(x=bin_centers[is_bin_full],
                           height=bin_effects[is_bin_full],
                           width=dx[is_bin_full],
                           color=(0.1, 0.1, 0.1, 0.1),
                           edgecolor='blue',
                           yerr=np.sqrt(bin_variance[is_bin_full]),
                           ecolor='red',
                           label="$\hat{\mu}_k ( N_k \geq " + str(point_limit) + ")$")
            # ax2.bar_label(bars, labels=['%.1f' % e for e in np.sqrt(bin_variance[is_bin_full])])
        # bins without enough points
        if np.sum(is_bin_empty):
            ax2.bar(x=bin_centers[is_bin_empty],
                    height=bin_effects[is_bin_empty],
                    width=dx[is_bin_empty],
                    color=(0.1, 0.1, 0.1, 0.1),
                    edgecolor='red',
                    yerr=np.sqrt(bin_variance[is_bin_empty]),
                    label="non-robust estimation (< " + str(point_limit) + " points)")
    else:
        # bins with enough points
        if np.sum(is_bin_full) > 0:
            ax2.bar(x=bin_centers[is_bin_full],
                    height = bin_effects[is_bin_full],
                    width=dx[is_bin_full],
                    color=(0.1, 0.1, 0.1, 0.1),
                    edgecolor='blue',
                    ecolor='red',
                    label="$\hat{\mu}_k ( N_k \geq " + str(point_limit) + ")$")
        if np.sum(is_bin_empty):
            ax2.bar(x=bin_centers[is_bin_empty],
                    height = bin_effects[is_bin_empty],
                    width=dx[is_bin_empty],
                    color=(0.1, 0.1, 0.1, 0.1),
                    edgecolor='red',
                    label="non-robust estimation (< " + str(point_limit) + " points)")
    if gt_bins is not None:
        lims = gt_bins["limits"]
        positions = [(lims[i] + lims[i + 1]) / 2 for i in range(len(lims) - 1)]
        dx = lims[1] - lims[0]
        ax2.bar(x=positions,
                height=gt_bins["height"],
                width=dx,
                color=(0.1, 0.1, 0.1, 0.1),
                edgecolor='m',
                label="ground truth")
    ax2.legend()


def fe_all(dale, ale, pdp, mplot, feature, ale_gt=None):
    dale_first_empty_bin = dale.parameters["feature_" + str(feature)]["first_empty_bin"]
    dale_limits = dale.parameters["feature_" + str(feature)]["limits"]

    ale_first_empty_bin = ale.parameters["feature_" + str(feature)]["first_empty_bin"]
    ale_limits = ale.parameters["feature_" + str(feature)]["limits"]

    if dale_first_empty_bin is None and ale_first_empty_bin is None:
        first_empty_bin = None
    else:
        first_empty_bin = np.min([dale_first_empty_bin, ale_first_empty_bin])

    left_lim = np.min([dale_limits[0], ale_limits[0]])
    right_lim = np.max([dale_limits[-1], ale_limits[-1]])

    plt.figure()


    # ALE gt
    x = np.linspace(left_lim - .01, right_lim + .01, 10000)
    if ale_gt is not None:
        y = ale_gt(x)
        plt.plot(x, y, "r--", label="ALE gt")

    # DALE
    y, var = dale.eval(x, feature)
    plt.plot(x, y, "b--", label="DALE")
    plt.fill_between(x, y - np.sqrt(var), y + np.sqrt(var), color='b', alpha=0.3, label="DALE std error")

    # ALE
    y, var = ale.eval(x, feature)
    plt.plot(x, y, "m--", label="ALE")
    plt.fill_between(x, y - np.sqrt(var), y + np.sqrt(var), color='r', alpha=0.3, label="ALE std error")

    # PDP
    y = pdp.eval(x, feature)
    plt.plot(x, y, "g--", label="PDP")

    # MPlot
    y = mplot.eval(x, feature, tau=.1)
    plt.plot(x, y, "y--", label="MPlot")

    if first_empty_bin is not None:
        plt.axvspan(xmin=dale_limits[first_empty_bin], xmax=x[-1], ymin=np.min(y), ymax=np.max(y), alpha=.2, color="red",
                    label="not-trusted-area")
    plt.legend()
    plt.show(block=False)


def plot_local_effects(s, xs, data_effect, limits, block):
    plt.figure()
    plt.title("Local effect for feature " + str(s+1))
    plt.plot(xs, data_effect, "bo")
    if limits is not False:
        plt.vlines(limits, ymin=np.min(data_effect), ymax=np.max(data_effect))
    plt.show(block=block)


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

import matplotlib.pyplot as plt
import numpy as np


def feature_effect_plot(params, eval, feature, error, min_points_per_bin, title=None, block=False, gt=False, gt_bins=None, savefig=False):
    assert all(name in params for name in ["first_empty_bin", "limits", "dx", "is_bin_empty", "bin_estimator_variance", "bin_effect"])
    limits = params["limits"]
    dx = params["dx"]
    bin_variance = params["bin_variance"]
    bin_effect = params["bin_effect"]

    is_bin_empty = params["points_per_bin"] < min_points_per_bin

    bins_under_limit = np.argwhere(params["points_per_bin"] < min_points_per_bin)
    if bins_under_limit.shape[0] > 0:
        first_empty_bin = bins_under_limit[0][0]
    else:
        first_empty_bin = None

    x = np.linspace(params["limits"][0], params["limits"][-1], 10000)
    y, estimator_var, var = eval(x, feature)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if title is None:
        fig.suptitle("Effect of feature %d" % (feature + 1))
    else:
        fig.suptitle(title)

    # first subplot
    feature_effect(ax1, x, y, estimator_var, var, first_empty_bin, limits, min_points_per_bin, error=error, gt=gt)

    # second subplot
    effects_per_bin(ax2, bin_effect, bin_variance, error, is_bin_empty,
                    limits, dx, gt_bins, min_points_per_bin)

    if savefig:
        plt.savefig(savefig)

    if block is False:
        plt.show(block=False)
    else:
        plt.show()


def feature_effect(ax1, x, y, estimator_var, var, first_empty, limits, point_limit, error=True, gt=None):
    # first subplot
    ax1.set_title("Plot")
    ax1.plot(x, y, "b--", label="estimation")\


    if first_empty is not None:
        added_line = .3*(np.max(y) - np.min(y))
        ax1.vlines(x=[limits[first_empty], limits[first_empty+1]], ymin=np.min(y) - added_line, ymax=np.max(y) + added_line,
                   colors="red",
                   alpha=.7,
                   label="first bin with < " + str(point_limit) + " points")
    if error == "std":
        ax1.fill_between(x, y-np.sqrt(var), y+np.sqrt(var), color='green', alpha=0.2, label="std")
    elif error == "standard error":
        ax1.fill_between(x, y-2*np.sqrt(estimator_var), y+2*np.sqrt(estimator_var), color='green', alpha=0.6, label="standard error")
    elif error == "both":
        ax1.fill_between(x, y-np.sqrt(var), y+np.sqrt(var), color='green', alpha=0.2, label="std")
        ax1.fill_between(x, y-np.sqrt(estimator_var), y+np.sqrt(estimator_var), color='green', alpha=0.6, label="standard error")

    if gt is not None:
        y = gt(x)
        ax1.plot(x, y, "m--", label="ground truth")
    ax1.legend()


def effects_per_bin(ax2, bin_effects, bin_variance, error, is_bin_empty, limits, dx, gt_bins=None, point_limit=10):
    ax2.set_title("Effects per bin")
    bin_centers = (limits[:-1] + limits[1:]) / 2
    is_bin_full = ~np.array(is_bin_empty)
    if error:
        # bins with enough points
        if np.sum(is_bin_full) > 0:
            ax2.bar(x=bin_centers[is_bin_full],
                    height=bin_effects[is_bin_full],
                    width=dx[is_bin_full],
                    color=(0.1, 0.1, 0.1, 0.1),
                    edgecolor='blue',
                    yerr=np.sqrt(bin_variance[is_bin_full]),
                    label="robust estimation (>= " + str(point_limit) + " points)")
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
                    label="robust estimation (>= " + str(point_limit) + " points)")          # bins without enough points
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

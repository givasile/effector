import matplotlib.pyplot as plt
import numpy as np


def feature_effect_plot(params, eval, feature, title=None, block=False, gt=False, gt_bins=None):
    assert all(name in params for name in ["first_empty_bin", "limits", "dx", "is_bin_empty", "bin_estimator_variance", "bin_effect"])

    first_empty_bin = params["first_empty_bin"]
    limits = params["limits"]
    dx = params["dx"]
    is_bin_empty = params["is_bin_empty"]
    bin_estimator_variance = params["bin_estimator_variance"]
    bin_effect = params["bin_effect"]

    x = np.linspace(params["limits"][0], params["limits"][-1], 10000)
    y, var = eval(x, feature)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if title is None:
        fig.suptitle("Effect of feature %d" % (feature + 1))
    else:
        fig.suptitle(title)

    # first subplot
    feature_effect(ax1, x, y, var, limits, first_empty_bin, gt=gt)

    # second subplot
    effects_per_bin(ax2, bin_effect, bin_estimator_variance, is_bin_empty, limits, dx, gt_bins)
    if block is False:
        plt.show(block=False)
    else:
        plt.show()


def feature_effect(ax1, x, y, var, limits, first_empty_bin, gt=None):
    # first subplot
    ax1.set_title("Plot")
    ax1.plot(x, y, "b--", label="feature effect")

    # first_empty = first_empty_bin
    # if first_empty is not None:
    #     ax1.axvspan(xmin=limits[first_empty], xmax=x[-1], ymin=np.min(y), ymax=np.max(y), alpha=.2,
    #                 color="red",
    #                 edgecolor=None,
    #                 label="not-trusted-area")

    ax1.fill_between(x, y-np.sqrt(var), y+np.sqrt(var), color='green', alpha=0.8, label="standard error")
    # ax1.fill_between(x, y - 2*np.sqrt(var), y + 2*np.sqrt(var), color='green', alpha=0.4)

    if gt is not None:
        y = gt(x)
        ax1.plot(x, y, "r--", label="ground truth")
    ax1.legend()


def effects_per_bin(ax2, bin_effects, bin_estimator_variance, is_bin_empty, limits, dx, gt_bins=None):
    ax2.set_title("Effects per bin")
    bin_centers = limits[:-1] + dx / 2
    is_bin_full = ~np.array(is_bin_empty)
    positions = bin_centers[is_bin_full]
    std_err = bin_estimator_variance[is_bin_full]
    data1 = bin_effects[is_bin_full]
    ax2.bar(x=positions, height=data1, width=dx, color=(0.1, 0.1, 0.1, 0.1), edgecolor='blue', yerr=std_err,
            label="bin effect")

    if gt_bins is not None:
        lims = gt_bins["limits"]
        positions = [(lims[i] + lims[i + 1]) / 2 for i in range(len(lims) - 1)]
        dx = lims[1] - lims[0]
        ax2.bar(x=positions, height=gt_bins["height"], width=dx, color=(0.1, 0.1, 0.1, 0.1), edgecolor='red',
                label="bin effect gt")

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

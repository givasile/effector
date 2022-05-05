import matplotlib.pyplot as plt
import numpy as np


def feature_effect_plot(s, x, y, var, first_empty_bin, limits, dx, is_bin_empty, bin_estimator_variance, bin_effects, block):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    fig.suptitle("Effect of feature %d" % (s + 1))

    # first subplot
    ax1.set_title("Plot")
    ax1.plot(x, y, "b-", label="feature effect")

    first_empty = first_empty_bin
    if first_empty is not None:
        ax1.axvspan(xmin=limits[first_empty], xmax=x[-1], ymin=np.min(y), ymax=np.max(y), alpha=.2, color="red",
                    label="not-trusted-area")

    ax1.fill_between(x, y-np.sqrt(var), y+np.sqrt(var), color='green', alpha=0.8, label="standard error")
    # ax1.fill_between(x, y - 2*np.sqrt(var), y + 2*np.sqrt(var), color='green', alpha=0.4)
    ax1.legend()

    # second subplot
    ax2.set_title("Effects per bin")
    bin_centers = limits[:-1] + dx / 2
    is_bin_full = ~np.array(is_bin_empty)
    positions = bin_centers[is_bin_full]
    std_err = bin_estimator_variance[is_bin_full]
    data1 = bin_effects[is_bin_full]
    ax2.bar(x=positions, height=data1, width=dx, color=(0.1, 0.1, 0.1, 0.1), edgecolor='blue', yerr=std_err,
            label="bin effect")
    ax2.legend()

    if block is False:
        plt.show(block=False)
    else:
        plt.show()

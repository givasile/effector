import matplotlib.pyplot as plt
import numpy as np
import typing
from effector import helpers


def trans_affine(x, mu, std):
    return x * std + mu


def trans_scale(x, std, square=False):
    y = x * std**2 if square else x * std
    return y


def trans_bin(x, std_x, std_y, square=False):
    y = x * (std_y / std_x) ** 2 if square else x * std_y / std_x
    return y


def ale_plot(
    ale_params: dict,
    accum_effect_func: callable,
    feature: int,
    centering: bool = True,
    error: typing.Union[None, str] = None,
    scale_x: typing.Union[None, dict] = None,
    scale_y: typing.Union[None, dict] = None,
    title: typing.Union[None, str] = None,
    avg_output: typing.Union[None, float] = None,
    feature_names: typing.Union[None, list] = None,
    target_name: typing.Union[None, str] = None,
):
    """

    Parameters
    ----------
    ale_params: Dict with ale parameters
    accum_effect_func: the accumulated effect function
    feature: which feature to plot
    centering: bool, if True, the feature is centered
    error: None or in ['std', 'stderr', 'both']
     - if 'std' the accumulated standard deviation is shown
     - if 'stderr' the accumulated standard error of the mean is shown
     - if 'both', std and stderr is plot
    scale_x: None or Dict with ['std', 'mean']
    scale_y: None or Dict with ['std', 'mean']
    savefig: False or path to store figure
    """
    # assert ale_params contains needed quantities
    assert all(
        name in ale_params
        for name in ["limits", "dx", "bin_estimator_variance", "bin_effect"]
    )

    x = np.linspace(ale_params["limits"][0], ale_params["limits"][-1], 1000)
    y, std, std_err = accum_effect_func(feature, x, True, centering)

    # transform
    x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
    y = y if scale_y is None else trans_affine(y, scale_y["mean"], scale_y["std"])
    std = std if scale_y is None else trans_scale(std, scale_y["std"])
    std_err = std_err if scale_y is None else trans_scale(std_err, scale_y["std"])
    limits = (
        ale_params["limits"]
        if scale_x is None
        else trans_affine(ale_params["limits"], scale_x["mean"], scale_x["std"])
    )
    dx = (
        ale_params["dx"]
        if scale_x is None
        else trans_scale(ale_params["dx"], scale_x["std"])
    )
    bin_variance = (
        ale_params["bin_variance"]
        if scale_y is None
        else trans_bin(ale_params["bin_variance"], scale_x["std"], scale_y["std"], True)
    )
    bin_effect = (
        ale_params["bin_effect"]
        if scale_y is None
        else trans_bin(ale_params["bin_effect"], scale_x["std"], scale_y["std"])
    )

    # PLOT
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    if title is None:
        ax1.set_title("ALE plot")
    else:
        ax1.set_title(title)

    # first subplot
    ale_curve(ax1, x, y, std_err, std, error=error, avg_output=avg_output)

    # second subplot
    ale_bins(ax2, bin_effect, bin_variance, limits, dx, error)

    ax1.set_ylabel("y") if target_name is None else ax1.set_ylabel(target_name)

    x_name = "x_%d" % (feature + 1) if feature_names is None else feature_names[feature]
    ax2.set_xlabel(x_name)
    ax2.set_ylabel("dy/dx")

    plt.show(block=False)
    return fig, ax1, ax2


def ale_curve(ax1, x, y, std_err, std, error=None, avg_output=None):
    ax1.plot(x, y, "b--", label="global effect")
    if avg_output is not None:
        ax1.axhline(y=avg_output, color="black", linestyle="--", label="avg output")

    if error == "std":
        ax1.fill_between(x, y - std, y + std, color="red", alpha=0.2, label="std")
    elif error == "stderr":
        ax1.fill_between(
            x,
            y - 2 * std_err,
            y + 2 * std_err,
            color="red",
            alpha=0.6,
            label="standard error",
        )
    elif error == "both":
        ax1.fill_between(x, y - std, y + std, color="red", alpha=0.2, label="std")
        ax1.fill_between(
            x,
            y - np.sqrt(std_err),
            y + np.sqrt(std_err),
            color="red",
            alpha=0.6,
            label="standard error",
        )
    ax1.legend()


def ale_bins(ax2, bin_effects, bin_variance, limits, dx, error):
    bin_centers = (limits[:-1] + limits[1:]) / 2
    yerr = np.sqrt(bin_variance) if error else None
    ax2.bar(
        x=bin_centers,
        height=bin_effects,
        width=dx,
        color=(0.1, 0.1, 0.1, 0.1),
        edgecolor="blue",
        yerr=yerr,
        ecolor="red",
        label="dy_dx",
    )
    ax2.legend()


def plot_pdp(
    x: np.ndarray,
    feature: int,
    eval: callable,
    confidence_interval: typing.Union[bool, str],
    centering: typing.Union[bool, str],
    y_pdp_label: str,
    title: str,
    scale_x: typing.Union[None, dict] = None,
    scale_y: typing.Union[None, dict] = None,
    feature_names: typing.Union[None, list] = None,
    target_name: typing.Union[None, str] = None,
    avg_output: typing.Union[None, float] = None,
):

    fig, ax = plt.subplots()
    ax.set_title(title)

    confidence_interval = helpers.prep_confidence_interval(confidence_interval)
    if confidence_interval is False:
        y = eval(feature, x, uncertainty=False, centering=centering)
        x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
        y = y if scale_y is None else trans_affine(y, scale_y["mean"], scale_y["std"])
        ax.plot(x, y, "b-", label=y_pdp_label)
    elif confidence_interval == "std":
        y, std, _ = eval(feature, x, uncertainty=True, centering=centering)
        x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
        y = y if scale_y is None else trans_affine(y, scale_y["mean"], scale_y["std"])
        std = std if scale_y is None else trans_scale(std, scale_y["std"])
        ax.plot(x, y, "b-", label=y_pdp_label)
        ax.fill_between(
            x, y - 2 * std, y + 2 * std, color="red", alpha=0.4, label="std"
        )
    elif confidence_interval == "std_err":
        y, std, var_estimator = eval(feature, x, uncertainty=True, centering=centering)
        x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
        y = y if scale_y is None else trans_affine(y, scale_y["mean"], scale_y["std"])
        std_err = np.sqrt(var_estimator)
        std_err = std_err if scale_y is None else trans_scale(std_err, scale_y["std"])
        ax.plot(x, y, "b-", label=y_pdp_label)
        ax.fill_between(
            x, y - 2 * std_err, y + 2 * std_err, color="red", alpha=0.4, label="std_err"
        )

    if avg_output is not None:
        ax.axhline(y=avg_output, color="black", linestyle="--", label="avg output")

    ax.legend()
    ax.set_xlabel("x_%d" % (feature + 1)) if feature_names is None else ax.set_xlabel(
        feature_names[feature]
    )
    ax.set_ylabel("y") if target_name is None else ax.set_ylabel(target_name)
    plt.show(block=False)
    return fig, ax


def plot_pdp_ice(
    x,
    feature,
    y_pdp,
    y_ice,
    title,
    centering,
    y_pdp_label,
    y_ice_label,
    scale_x=None,
    scale_y=None,
):
    centering = helpers.prep_centering(centering)
    fig, ax = plt.subplots()
    ax.set_title(title)
    y_ice_outputs = np.array(
        [y_ice[i].eval(feature, x, False, centering) for i in range(len(y_ice))]
    )
    y_pdp_output = y_pdp.eval(feature, x, False, centering)

    x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
    y_pdp_output = (
        y_pdp_output
        if scale_y is None
        else trans_affine(y_pdp_output, scale_y["mean"], scale_y["std"])
    )
    y_ice_outputs = (
        y_ice_outputs
        if scale_y is None
        else trans_affine(y_ice_outputs, scale_y["mean"], scale_y["std"])
    )

    ax.plot(x, y_ice_outputs[0, :], color="red", alpha=0.1, label=y_pdp_label)
    ax.plot(x, y_ice_outputs.T, color="red", alpha=0.1)
    ax.plot(x, y_pdp_output, color="blue", label=y_ice_label)
    ax.set_xlabel("feature %d" % (feature + 1))
    ax.set_ylabel("y")
    ax.legend()

    plt.show(block=False)
    return fig, ax


def plot_pdp_ice_2(
    x,
    feature,
    yy,
    title,
    y_pdp_label,
    y_ice_label,
    scale_x=None,
    scale_y=None,
    avg_output=None,
    feature_names: typing.Union[None, list] = None,
    target_name: typing.Union[None, str] = None,
):

    fig, ax = plt.subplots()
    ax.set_title(title)

    y_pdp_output = np.mean(yy, axis=1)
    y_ice_outputs = yy

    x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
    y_pdp_output = (
        y_pdp_output
        if scale_y is None
        else trans_affine(y_pdp_output, scale_y["mean"], scale_y["std"])
    )
    y_ice_outputs = (
        y_ice_outputs
        if scale_y is None
        else trans_affine(y_ice_outputs, scale_y["mean"], scale_y["std"])
    )

    ax.plot(x, y_ice_outputs[:, 0], color="red", alpha=0.1, label=y_ice_label)
    ax.plot(x, y_ice_outputs, color="red", alpha=0.1)
    ax.plot(x, y_pdp_output, color="blue", label=y_pdp_label)
    if avg_output is not None:
        ax.axhline(y=avg_output, color="black", linestyle="--", label="avg output")

    feature_name = "x_%d" % (feature + 1) if feature_names is None else feature_names[feature]
    ax.set_xlabel(feature_name)
    ax.set_ylabel("y") if target_name is None else ax.set_ylabel(target_name)
    ax.legend()

    plt.show(block=False)
    return fig, ax

import matplotlib.pyplot as plt
import numpy as np
import typing

import scipy.interpolate

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
    y_limits: typing.Union[None, tuple] = None,
    dy_limits: typing.Union[None, tuple] = None,
    show_only_aggregated: bool = False,
    show_plot: bool = True,
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
        for name in ["limits", "dx", "bin_effect"]
    )

    x = np.linspace(ale_params["limits"][0], ale_params["limits"][-1], 1000)
    y, std = accum_effect_func(feature, x, True, centering)

    # transform
    x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
    y = y if scale_y is None else trans_affine(y, scale_y["mean"], scale_y["std"])
    std = std if scale_y is None else trans_scale(std, scale_y["std"])
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
    if show_only_aggregated:
        fig, ax1 = plt.subplots()
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    if title is None:
        ax1.set_title("Accumulated Local Effects (ALE)")
    else:
        ax1.set_title(title)

    # first subplot
    ale_curve(ax1, x, y, avg_output=avg_output)

    # second subplot
    if not show_only_aggregated:
        ale_bins(ax2, bin_effect, bin_variance, limits, dx, error, dy_limits)

    ax1.set_ylabel("y") if target_name is None else ax1.set_ylabel(target_name)

    ax1.set_ylim(y_limits[0], y_limits[1]) if y_limits is not None else None

    x_name = "x_%d" % (feature + 1) if feature_names is None else feature_names[feature]
    if not show_only_aggregated:
        ax2.set_xlabel(x_name)
        ax2.set_ylabel("dy/dx")

    if show_plot:
        plt.show(block=False)
    else:
        if show_only_aggregated:
            return fig, ax1
        else:
            return fig, (ax1, ax2)


def ale_curve(ax1, x, y, avg_output=None):
    ax1.plot(x, y, "b--", label="average effect")
    if avg_output is not None:
        ax1.axhline(y=avg_output, color="black", linestyle="--", label="avg output")
    ax1.legend()


def ale_bins(ax2, bin_effects, bin_variance, limits, dx, error, dy_limits):
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
    ax2.set_ylim(dy_limits[0], dy_limits[1]) if dy_limits is not None else None


def plot_pdp_ice(
    x,
    feature,
    yy,
    title,
    confidence_interval,
    y_pdp_label,
    y_ice_label,
    scale_x: typing.Union[None, dict] = None,
    scale_y: typing.Union[None, dict] = None,
    avg_output: typing.Union[None, float] = None,
    feature_names: typing.Union[None, list] = None,
    target_name: typing.Union[None, str] = None,
    is_derivative: bool = False,
    nof_ice: typing.Union[str, int] = "all",
    y_limits: typing.Union[None, tuple] = None,
    show_plot: bool = True,
):

    fig, ax = plt.subplots()
    ax.set_title(title)

    y_pdp_output = np.mean(yy, axis=1)

    # choose nof_ice randomly
    if nof_ice != "all":
        if nof_ice > yy.shape[1]:
            nof_ice = yy.shape[1]

    y_ice_outputs = yy
    std = np.std(y_ice_outputs, axis=1)
    std_err = np.sqrt(np.var(y_ice_outputs, axis=1))

    # scale x-axis
    x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])

    # scale y-axis
    if scale_y is not None:
        std = trans_scale(std, scale_y["std"], square=False)
        std_err = trans_scale(std_err, scale_y["std"], square=False)
        if is_derivative:
            y_pdp_output = trans_scale(y_pdp_output, scale_y["std"], square=False)
            y_ice_outputs = trans_scale(y_ice_outputs, scale_y["std"], square=False)
        elif not is_derivative:
            y_pdp_output = trans_affine(y_pdp_output, scale_y["mean"], scale_y["std"])
            y_ice_outputs = trans_affine(y_ice_outputs, scale_y["mean"], scale_y["std"])

    # if avg_output is not None and scale_y is not None:
    #     if not is_derivative:
    #         avg_output = trans_affine(avg_output, scale_y["mean"], scale_y["std"])
    #     elif is_derivative:
    #         avg_output = trans_scale(avg_output, scale_y["std"], square=False)

    # plot
    if confidence_interval == "std":
        std = np.std(y_ice_outputs, axis=1)
        ax.fill_between(
            x,
            y_pdp_output - std,
            y_pdp_output + std,
            color="red",
            alpha=0.4,
            label="std",
        )
    elif confidence_interval == "std_err":
        ax.fill_between(
            x,
            y_pdp_output - std_err,
            y_pdp_output + std_err,
            color="red",
            alpha=0.4,
            label="std_err",
        )
    elif confidence_interval == "ice":
        y_ice_outputs = (
            y_ice_outputs
            if nof_ice == "all"
            else y_ice_outputs[
                :,
                np.random.choice(
                    range(y_ice_outputs.shape[1]), size=nof_ice, replace=False
                ),
            ]
        )
        ax.plot(x, y_ice_outputs[:, 0], color="red", alpha=0.1, label=y_ice_label)
        ax.plot(x, y_ice_outputs, color="red", alpha=0.1)

    ax.plot(x, y_pdp_output, "b-", label=y_pdp_label)

    if avg_output is not None:
        ax.axhline(y=avg_output, color="black", linestyle="--", label="avg output")

    feature_name = (
        "x_%d" % (feature + 1) if feature_names is None else feature_names[feature]
    )
    ax.set_xlabel(feature_name)
    if is_derivative:
        ax.set_ylabel("dy/dx")
    else:
        ax.set_ylabel("y") if target_name is None else ax.set_ylabel(target_name)
    ax.legend()
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    if show_plot:
        plt.show(block=False)
    else:
        return fig, ax



def plot_shap(
    x: np.ndarray,
    y: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    y_std,
    feature: int,
    heterogeneity: str,
    scale_x: typing.Union[None, dict] = None,
    scale_y: typing.Union[None, dict] = None,
    avg_output: typing.Union[None, float] = None,
    feature_names: typing.Union[None, list] = None,
    target_name: typing.Union[None, str] = None,
    y_limits: typing.Union[None, tuple] = None,
    only_shap_values: bool = False,
    show_plot: bool = True,
):

    fig, ax = plt.subplots()
    ax.set_title("SHAP Dependence Plot")

    # scale x-axis
    x = x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])
    if xx is not None:
        xx = (
            xx if scale_x is None else trans_affine(xx, scale_x["mean"], scale_x["std"])
        )

    # scale y-axis
    if scale_y is not None:
        y_std = trans_scale(y_std, scale_y["std"], square=False)
        y = trans_affine(y, scale_y["mean"], scale_y["std"])
        if yy is not None:
            yy = trans_affine(yy, scale_y["mean"], scale_y["std"])
        # if avg_output is not None:
        #     avg_output = trans_scale(avg_output, scale_y["std"], square=False)

    # plot
    if heterogeneity == "std":
        ax.fill_between(
            x,
            y - y_std,
            y + y_std,
            color="red",
            alpha=0.4,
            label="std",
        )
    elif heterogeneity == "shap_values":
        ax.plot(xx[0], yy[0], "rx", alpha=0.5, label="SHAP values")
        ax.plot(xx, yy, "rx", alpha=0.5)

    if not only_shap_values:
        ax.plot(x, y, "b-", label="SHAP-DP")

    if avg_output is not None:
        ax.axhline(y=avg_output, color="black", linestyle="--", label="avg output")

    feature_name = (
        "x_%d" % (feature + 1) if feature_names is None else feature_names[feature]
    )
    ax.set_xlabel(feature_name)
    ax.set_ylabel("y") if target_name is None else ax.set_ylabel(target_name)
    ax.legend()
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])

    if show_plot:
        plt.show(block=False)
    else:
        return fig, ax

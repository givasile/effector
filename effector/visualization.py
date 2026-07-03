"""The plot layer: pure matplotlib, zero computation (R1 — every curve/band
drawn here is handed in by the caller, already evaluated and centered).

Heterogeneity-option names are normalized once, by
``helpers.prep_confidence_interval``: ``True`` always means ``"std"``;
per-method extras are ``"std_err"``/``"ice"`` (PDP family) and
``"shap_values"`` (ShapDP).

Every function returns ``(fig, ax)`` when ``show_plot=False`` and ``None``
otherwise (R7).
"""

import typing

import matplotlib.pyplot as plt
import numpy as np


def trans_affine(x, mu, std):
    return x * std + mu


def trans_scale(x, std, square=False):
    y = x * std**2 if square else x * std
    return y


def trans_bin(x, std_x, std_y, square=False):
    y = x * (std_y / std_x) ** 2 if square else x * std_y / std_x
    return y


def _scale_x(x, scale_x):
    return x if scale_x is None else trans_affine(x, scale_x["mean"], scale_x["std"])


def _scale_y(y, scale_y, is_derivative=False):
    """Level curves transform affinely; derivative curves scale by
    std_y/std_x-style factors only — adding the mean would shift dy/dx by the
    output's mean (B5)."""
    if scale_y is None:
        return y
    if is_derivative:
        return trans_scale(y, scale_y["std"])
    return trans_affine(y, scale_y["mean"], scale_y["std"])


def _feature_label(feature, feature_names):
    # 0-based fallback, matching API indices and helpers.get_feature_names
    return "x_%d" % feature if feature_names is None else feature_names[feature]


def _add_avg_output(ax, avg_output):
    if avg_output is not None:
        ax.axhline(y=avg_output, color="black", linestyle="--", label="avg output")


def _decorate_ax(ax, xlabel=None, ylabel=None, y_limits=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    ax.legend()
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])


def _finalize(fig, ax, show_plot):
    """The one exit point (R7): show and return None, or hand back (fig, ax)."""
    if show_plot:
        plt.show(block=False)
        return None
    return fig, ax


def ale_plot(
    x: np.ndarray,
    y: np.ndarray,
    bin_effect: np.ndarray,
    bin_variance: np.ndarray,
    limits: np.ndarray,
    dx: np.ndarray,
    feature: int,
    heterogeneity: typing.Union[bool, str] = False,
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
    """Draw the (RH)ALE mean effect (top axis) and the bin-effects bar plot
    (bottom axis, with sqrt(bin_variance) error bars if `heterogeneity`).

    `x`/`y` is the mean-effect curve; `bin_effect`/`bin_variance`/`limits`/`dx`
    is the stored bin payload.
    """
    x = _scale_x(x, scale_x)
    y = _scale_y(y, scale_y)
    limits = _scale_x(limits, scale_x)
    dx = dx if scale_x is None else trans_scale(dx, scale_x["std"])
    bin_variance = (
        bin_variance
        if scale_y is None
        else trans_bin(bin_variance, scale_x["std"], scale_y["std"], True)
    )
    bin_effect = (
        bin_effect
        if scale_y is None
        else trans_bin(bin_effect, scale_x["std"], scale_y["std"])
    )

    if show_only_aggregated:
        fig, ax1 = plt.subplots()
        axes = ax1
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        axes = (ax1, ax2)

    ax1.set_title("Accumulated Local Effects (ALE)" if title is None else title)
    ax1.plot(x, y, "b--", label="average effect")
    _add_avg_output(ax1, avg_output)

    x_name = _feature_label(feature, feature_names)
    y_name = "y" if target_name is None else target_name
    _decorate_ax(
        ax1,
        xlabel=x_name if show_only_aggregated else None,
        ylabel=y_name,
        y_limits=y_limits,
    )

    if not show_only_aggregated:
        bin_centers = (limits[:-1] + limits[1:]) / 2
        yerr = np.sqrt(bin_variance) if heterogeneity else None
        ax2.bar(
            x=bin_centers,
            height=bin_effect,
            width=dx,
            color=(0.1, 0.1, 0.1, 0.1),
            edgecolor="blue",
            yerr=yerr,
            ecolor="red",
            label="dy_dx",
        )
        _decorate_ax(ax2, xlabel=x_name, ylabel="dy/dx", y_limits=dy_limits)

    return _finalize(fig, axes, show_plot)


def plot_pdp_ice(
    x,
    feature,
    yy,
    title,
    heterogeneity,
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
    """Draw the mean of the ICE table `yy` (shape `(T, N)`) plus the requested
    heterogeneity: a std or standard-error band, or the ICE curves themselves."""
    x = _scale_x(x, scale_x)
    yy = _scale_y(yy, scale_y, is_derivative)
    y_mean = np.mean(yy, axis=1)

    fig, ax = plt.subplots()
    ax.set_title(title)

    if heterogeneity == "std":
        std = np.std(yy, axis=1)
        ax.fill_between(
            x, y_mean - std, y_mean + std, color="red", alpha=0.4, label="std"
        )
    elif heterogeneity == "std_err":
        std_err = np.std(yy, axis=1) / np.sqrt(yy.shape[1])
        ax.fill_between(
            x,
            y_mean - std_err,
            y_mean + std_err,
            color="red",
            alpha=0.4,
            label="std_err",
        )
    elif heterogeneity == "ice":
        yy_show = yy
        if nof_ice != "all" and nof_ice < yy.shape[1]:
            ind = np.random.choice(yy.shape[1], size=nof_ice, replace=False)
            yy_show = yy[:, ind]
        ax.plot(x, yy_show[:, 0], color="red", alpha=0.1, label=y_ice_label)
        ax.plot(x, yy_show, color="red", alpha=0.1)

    ax.plot(x, y_mean, "b-", label=y_pdp_label)
    _add_avg_output(ax, avg_output)

    y_name = "dy/dx" if is_derivative else ("y" if target_name is None else target_name)
    _decorate_ax(
        ax,
        xlabel=_feature_label(feature, feature_names),
        ylabel=y_name,
        y_limits=y_limits,
    )
    return _finalize(fig, ax, show_plot)


def plot_effect_comparison(
    x: np.ndarray,
    feature: int,
    curves: dict,
    scale_x: typing.Union[None, dict] = None,
    scale_y: typing.Union[None, dict] = None,
    avg_output: typing.Union[None, float] = None,
    feature_names: typing.Union[None, list] = None,
    target_name: typing.Union[None, str] = None,
    y_limits: typing.Union[None, tuple] = None,
    title: typing.Union[None, str] = None,
    show_plot: bool = True,
):
    """Overlay the mean effect of several methods for one feature on a single
    axis. `curves` maps `{method_label: y}`, each `y` of shape `(T,)`."""
    fig, ax = plt.subplots()
    ax.set_title("Feature Effect Comparison" if title is None else title)

    x = _scale_x(x, scale_x)
    for label, y in curves.items():
        ax.plot(x, _scale_y(y, scale_y), label=label)
    _add_avg_output(ax, avg_output)

    _decorate_ax(
        ax,
        xlabel=_feature_label(feature, feature_names),
        ylabel="y" if target_name is None else target_name,
        y_limits=y_limits,
    )
    return _finalize(fig, ax, show_plot)


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
    """Draw the SHAP-DP spline `x`/`y` plus the requested heterogeneity: a std
    band (`y_std`) or the shap-value cloud `xx`/`yy`."""
    fig, ax = plt.subplots()
    ax.set_title("SHAP Dependence Plot")

    x = _scale_x(x, scale_x)
    y = _scale_y(y, scale_y)
    if xx is not None:
        xx = _scale_x(xx, scale_x)
    if yy is not None:
        yy = _scale_y(yy, scale_y)
    if y_std is not None and scale_y is not None:
        y_std = trans_scale(y_std, scale_y["std"])

    if heterogeneity == "std":
        ax.fill_between(x, y - y_std, y + y_std, color="red", alpha=0.4, label="std")
    elif heterogeneity == "shap_values":
        ax.plot(xx[0], yy[0], "rx", alpha=0.5, label="SHAP values")
        ax.plot(xx, yy, "rx", alpha=0.5)

    if not only_shap_values:
        ax.plot(x, y, "b-", label="SHAP-DP")
    _add_avg_output(ax, avg_output)

    _decorate_ax(
        ax,
        xlabel=_feature_label(feature, feature_names),
        ylabel="y" if target_name is None else target_name,
        y_limits=y_limits,
    )
    return _finalize(fig, ax, show_plot)

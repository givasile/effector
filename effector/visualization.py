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

import effector.theme as theme


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
        ax.axhline(
            y=avg_output, color=theme.active().AVG, linestyle="--", label="avg output"
        )


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

    t = theme.active()
    ax1.set_title("Accumulated Local Effects (ALE)" if title is None else title)
    ax1.plot(x, y, color=t.MEAN, linestyle="--", label="average effect")
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
            color=t.BAR_FACE_MUTED,
            edgecolor=t.BAR_EDGE_ACCENT,
            yerr=yerr,
            ecolor=t.ERROR,
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
    random_state: typing.Union[None, int] = None,
):
    """Draw the mean of the ICE table `yy` (shape `(T, N)`) plus the requested
    heterogeneity: a std or standard-error band, or the ICE curves themselves."""
    x = _scale_x(x, scale_x)
    yy = _scale_y(yy, scale_y, is_derivative)
    y_mean = np.mean(yy, axis=1)

    fig, ax = plt.subplots()
    t = theme.active()
    ax.set_title(title)

    if heterogeneity == "std":
        std = np.std(yy, axis=1)
        ax.fill_between(
            x, y_mean - std, y_mean + std, color=t.BAND, alpha=t.BAND_ALPHA, label="std"
        )
    elif heterogeneity == "std_err":
        std_err = np.std(yy, axis=1) / np.sqrt(yy.shape[1])
        ax.fill_between(
            x,
            y_mean - std_err,
            y_mean + std_err,
            color=t.BAND,
            alpha=t.BAND_ALPHA,
            label="std_err",
        )
    elif heterogeneity == "ice":
        yy_show = yy
        if nof_ice != "all" and nof_ice < yy.shape[1]:
            ind = np.random.default_rng(random_state).choice(
                yy.shape[1], size=nof_ice, replace=False
            )
            yy_show = yy[:, ind]
        ax.plot(
            x,
            yy_show[:, 0],
            color=t.CLOUD,
            alpha=t.CLOUD_ALPHA,
            linewidth=t.CLOUD_LW,
            label=y_ice_label,
        )
        ax.plot(x, yy_show, color=t.CLOUD, alpha=t.CLOUD_ALPHA, linewidth=t.CLOUD_LW)

    ax.plot(x, y_mean, color=t.MEAN, linestyle="-", label=y_pdp_label)
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
    discrete: bool = False,
    level_labels: typing.Union[None, list] = None,
    show_plot: bool = True,
):
    """Overlay the mean effect of several methods for one feature on a single
    axis. `curves` maps `{method_label: y}`, each `y` of shape `(T,)`.

    `discrete=True` (categorical feature): the effects are per-level values, so
    each method is drawn as a marker series at the level positions (joined by a
    thin line) and the x-axis shows the level ticks/labels instead of a
    continuous grid."""
    fig, ax = plt.subplots()
    t = theme.active()
    ax.set_title("Feature Effect Comparison" if title is None else title)

    positions = np.asarray(x, dtype=float)
    x_scaled = _scale_x(positions, scale_x)
    for i, (label, y) in enumerate(curves.items()):
        color = t.CAT[i % len(t.CAT)]
        if discrete:
            ax.plot(
                x_scaled,
                _scale_y(y, scale_y),
                marker="o",
                markersize=5,
                linewidth=1.2,
                color=color,
                label=label,
            )
        else:
            ax.plot(x_scaled, _scale_y(y, scale_y), color=color, label=label)
    if discrete:
        _categorical_axis(ax, positions, level_labels, scale_x)
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
    t = theme.active()
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
        ax.fill_between(
            x, y - y_std, y + y_std, color=t.BAND, alpha=t.BAND_ALPHA, label="std"
        )
    elif heterogeneity == "shap_values":
        ax.plot(
            xx[0],
            yy[0],
            color=t.CLOUD,
            marker="x",
            linestyle="none",
            alpha=t.SHAP_MARKER_ALPHA,
            label="SHAP values",
        )
        ax.plot(
            xx,
            yy,
            color=t.CLOUD,
            marker="x",
            linestyle="none",
            alpha=t.SHAP_MARKER_ALPHA,
        )

    if not only_shap_values:
        ax.plot(x, y, color=t.MEAN, linestyle="-", label="SHAP-DP")
    _add_avg_output(ax, avg_output)

    _decorate_ax(
        ax,
        xlabel=_feature_label(feature, feature_names),
        ylabel="y" if target_name is None else target_name,
        y_limits=y_limits,
    )
    return _finalize(fig, ax, show_plot)


def _categorical_axis(ax, positions, level_labels, scale_x):
    """Ticks at the level positions; explicit labels replace numeric ticks
    when the levels carry names (nominal / encoded categories)."""
    ticks = _scale_x(np.asarray(positions, dtype=float), scale_x)
    ax.set_xticks(ticks)
    if level_labels is not None:
        ax.set_xticklabels(level_labels)


def _bar_width(positions):
    positions = np.asarray(positions, dtype=float)
    return 0.6 * np.min(np.diff(positions)) if len(positions) > 1 else 0.6


def plot_categorical_effect(
    positions,
    heights,
    variances,
    feature,
    heterogeneity,
    title,
    level_labels=None,
    scale_x=None,
    scale_y=None,
    avg_output=None,
    feature_names=None,
    target_name=None,
    y_limits=None,
    connect_line=False,
    show_plot=True,
):
    """Bars at the level positions with heterogeneity whiskers = sqrt(h(v_k)).

    The categorical analogue of the mean-effect curve (method_semantics.md):
    used by PDP (per-level means), (RH)ALE (accumulated per-level values) and
    ShapDP (per-level shap means). Pure drawing — heights/variances arrive
    evaluated and centered.

    `connect_line=True` overlays a line through the bar tops — only meaningful
    for (RH)ALE, where the bars are an accumulation and the slope between two
    bars is the per-transition step the method measures (the values are a
    cumulative sum of adjacent-level changes). Off for PDP/ShapDP, whose
    per-level bars are independent.
    """
    fig, ax = plt.subplots()
    t = theme.active()
    ax.set_title(title)

    x = _scale_x(np.asarray(positions, dtype=float), scale_x)
    y = _scale_y(np.asarray(heights, dtype=float), scale_y)
    width = _bar_width(x)

    yerr = None
    if heterogeneity is not False and variances is not None:
        std = np.sqrt(np.asarray(variances, dtype=float))
        yerr = std * scale_y["std"] if scale_y is not None else std

    ax.bar(
        x,
        y,
        width=width,
        color=t.BAR_FACE,
        edgecolor=t.BAR_EDGE,
        linewidth=0.6,
        yerr=yerr,
        ecolor=t.ERROR,
        capsize=4,
        label="mean effect",
    )
    if connect_line:
        # accumulation path: slope between bars = the per-transition step
        ax.plot(
            x,
            y,
            color=t.CONNECT,
            linewidth=1.4,
            marker="o",
            markersize=4,
            zorder=3,
            label="accumulated (steps)",
        )
    _categorical_axis(ax, positions, level_labels, scale_x)
    _add_avg_output(ax, avg_output)
    _decorate_ax(
        ax,
        xlabel=_feature_label(feature, feature_names),
        ylabel=target_name,
        y_limits=y_limits,
    )
    return _finalize(fig, ax, show_plot)


def plot_pdp_ice_categorical(
    positions,
    yy,
    feature,
    title,
    y_pdp_label="PDP",
    y_ice_label="ICE",
    level_labels=None,
    scale_x=None,
    scale_y=None,
    avg_output=None,
    feature_names=None,
    target_name=None,
    nof_ice=100,
    y_limits=None,
    show_plot=True,
    random_state=21,
):
    """Bars for the per-level mean + jittered per-instance ICE dots.

    `yy` is the (K, N) ICE table evaluated at the K levels. Jitter and the
    ICE subsample are seeded (`random_state`) — determinism is contractual
    (R8)."""
    fig, ax = plt.subplots()
    t = theme.active()
    ax.set_title(title)

    yy = np.asarray(yy, dtype=float)
    x = _scale_x(np.asarray(positions, dtype=float), scale_x)
    y_mean = _scale_y(yy.mean(axis=1), scale_y)
    width = _bar_width(x)

    rng = np.random.default_rng(random_state)
    n = yy.shape[1]
    if nof_ice != "all" and int(nof_ice) < n:
        cols = rng.choice(n, int(nof_ice), replace=False)
    else:
        cols = np.arange(n)
    jitter = rng.uniform(-0.25 * width, 0.25 * width, size=(len(x), len(cols)))
    xx = x[:, np.newaxis] + jitter
    y_dots = _scale_y(yy[:, cols], scale_y)
    ax.plot(
        xx.ravel(),
        y_dots.ravel(),
        marker=".",
        linestyle="none",
        color=t.CLOUD,
        alpha=t.DOT_ALPHA,
        markersize=3,
        label=y_ice_label,
    )

    ax.bar(
        x,
        y_mean,
        width=width,
        color=t.BAR_FACE,
        edgecolor=t.BAR_EDGE,
        linewidth=0.6,
        alpha=0.8,
        label=y_pdp_label,
    )
    _categorical_axis(ax, positions, level_labels, scale_x)
    _add_avg_output(ax, avg_output)
    _decorate_ax(
        ax,
        xlabel=_feature_label(feature, feature_names),
        ylabel=target_name,
        y_limits=y_limits,
    )
    return _finalize(fig, ax, show_plot)


def plot_shap_categorical(
    positions,
    heights,
    xx,
    yy,
    feature,
    title,
    level_labels=None,
    scale_x=None,
    scale_y=None,
    avg_output=None,
    feature_names=None,
    target_name=None,
    nof_shap_values=100,
    y_limits=None,
    show_plot=True,
    random_state=21,
):
    """Bars for the per-level shap mean + the jittered shap cloud."""
    fig, ax = plt.subplots()
    t = theme.active()
    ax.set_title(title)

    x = _scale_x(np.asarray(positions, dtype=float), scale_x)
    y_mean = _scale_y(np.asarray(heights, dtype=float), scale_y)
    width = _bar_width(x)

    rng = np.random.default_rng(random_state)
    n = len(yy)
    if nof_shap_values != "all" and int(nof_shap_values) < n:
        keep = rng.choice(n, int(nof_shap_values), replace=False)
    else:
        keep = np.arange(n)
    jitter = rng.uniform(-0.25 * width, 0.25 * width, size=len(keep))
    ax.plot(
        _scale_x(np.asarray(xx, dtype=float)[keep], scale_x) + jitter,
        _scale_y(np.asarray(yy, dtype=float)[keep], scale_y),
        marker=".",
        linestyle="none",
        color=t.CLOUD,
        alpha=t.DOT_ALPHA,
        markersize=3,
        label="shap values",
    )

    ax.bar(
        x,
        y_mean,
        width=width,
        color=t.BAR_FACE,
        edgecolor=t.BAR_EDGE,
        linewidth=0.6,
        alpha=0.8,
        label="mean shap per level",
    )
    _categorical_axis(ax, positions, level_labels, scale_x)
    _add_avg_output(ax, avg_output)
    _decorate_ax(
        ax,
        xlabel=_feature_label(feature, feature_names),
        ylabel=target_name,
        y_limits=y_limits,
    )
    return _finalize(fig, ax, show_plot)

"""The plot layer: pure matplotlib, zero computation (R1 — every curve/band
drawn here is handed in by the caller, already evaluated and centered).

Two deliberate exceptions to "zero computation": the user-facing gatherers
`compare` and `plot_triage` take fitted *effect objects*, query their public
verbs (`eval`, `importance`, `heter_score`), and then draw — every private
drawing function below stays value-in.

Heterogeneity-option names are normalized once, by
``helpers.prep_confidence_interval``: ``True`` always means ``"std"``;
per-method extras are ``"std_err"``/``"ice"`` (PDP family) and
``"shap_values"`` (ShapDP).

Every function returns ``(fig, ax)`` when ``show_plot=False`` and ``None``
otherwise (R7).
"""

import typing
import warnings

import matplotlib.pyplot as plt
import numpy as np

import effector.helpers as helpers
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
    x_limits: typing.Union[None, tuple] = None,
):
    """Draw the (RH)ALE mean effect (top axis) and the bin-effects bar plot
    (bottom axis, with sqrt(bin_variance) error bars if `heterogeneity`).

    `x`/`y` is the mean-effect curve; `bin_effect`/`bin_variance`/`limits`/`dx`
    is the stored bin payload. `x_limits` (raw feature units) windows the
    shared x-axis — the masked/subregion zoom.
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

    if x_limits is not None:
        # window the (shared) x-axis to the caller's interval, e.g. a
        # subregion's effective range
        ax1.set_xlim(*_scale_x(np.asarray(x_limits, dtype=float), scale_x))

    return _finalize(fig, axes, show_plot)


def plot_pdp_ice(
    x,
    feature,
    y_mean,
    title,
    heterogeneity,
    y_pdp_label,
    y_ice_label,
    band=None,
    ice=None,
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
    """Draw the pre-computed PDP mean curve `y_mean` (shape `(T,)`) plus the
    requested heterogeneity (R1 — nothing is computed here): a std/std-err
    `band` (shape `(T,)`), or the raw ICE table `ice` (shape `(T, N)`) as a
    curve cloud. A band is a spread, so it scales by `scale_y["std"]` only."""
    x = _scale_x(x, scale_x)
    y_mean = _scale_y(y_mean, scale_y, is_derivative)

    fig, ax = plt.subplots()
    t = theme.active()
    ax.set_title(title)

    if heterogeneity in ("std", "std_err"):
        b = band if scale_y is None else trans_scale(band, scale_y["std"])
        ax.fill_between(
            x,
            y_mean - b,
            y_mean + b,
            color=t.BAND,
            alpha=t.BAND_ALPHA,
            label=heterogeneity,
        )
    elif heterogeneity == "ice":
        yy_show = _scale_y(ice, scale_y, is_derivative)
        if nof_ice != "all" and nof_ice < yy_show.shape[1]:
            ind = np.random.default_rng(random_state).choice(
                yy_show.shape[1], size=nof_ice, replace=False
            )
            yy_show = yy_show[:, ind]
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
    """Draw the SHAP-DP curve `x`/`y` plus the requested heterogeneity: a std
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


# ---------------------------------------------------------------------------
# user-facing gatherers: take fitted effect objects, query public verbs, draw
# ---------------------------------------------------------------------------


def _method_labels(effects) -> list:
    """Display names per effect via the method registry, deduped `PDP (2)`."""
    from effector import method_registry  # lazy: one-way dep guard

    labels, seen = [], {}
    for e in effects:
        base = method_registry.resolve(e.method_name).display_name
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    return labels


def compare(
    *effects,
    feature: typing.Union[int, str],
    labels: typing.Union[None, list] = None,
    centering: typing.Union[bool, str] = True,
    nof_points: int = 100,
    scale_x: typing.Union[None, dict] = None,
    scale_y: typing.Union[None, dict] = None,
    y_limits: typing.Union[None, tuple] = None,
    title: typing.Union[None, str] = None,
    show_plot: bool = True,
):
    """Overlay the mean effect of several *fitted* effect objects on one feature.

    ```python
    effector.compare(pdp, ale, rhale, feature="hr")
    effector.compare(pdp_a, pdp_b, feature="hr", labels=["model A", "model B"])
    ```

    The cross-examination verb above the engines: you hold the effect objects
    (different methods, or even different models over the same columns);
    `compare` queries each one's `eval` on a shared grid and overlays the
    curves — one theme color per effect. It computes nothing itself and
    stores nothing.

    !!! warning "Always centered"
        A comparison is only meaningful for centered effects (each method
        uses a different reference level), so `centering=False` is coerced to
        `"zero_integral"` with a warning.

    !!! note "Derivative units don't mix"
        `DerPDP` effects (dy/dx) can only be compared with each other — never
        on the same axis as the output-unit methods
        (`PDP`/`ALE`/`RHALE`/`ShapDP`).

    The single-model shortcut with the same look is
    `effector.FeatureEffect(data, model).plot(feature, methods=[...])`, which
    builds its own engines; `compare` overlays engines you already have.

    Args:
        *effects: two or more fitted effect objects (`PDP`/`ALE`/`RHALE`/
            `ShapDP`/`DerPDP`) over data with the same columns.
        feature: index or name of the feature to compare on.
        labels: one legend label per effect; defaults to the method display
            names, deduped (`"PDP"`, `"PDP (2)"`, ...).
        centering: how to center — `True`/`"zero_integral"` (around the y
            axis) or `"zero_start"` (each curve starts at `y=0`).
        nof_points: size of the shared grid (continuous features); the grid
            spans the intersection of the effects' axis limits.
        scale_x: `None` or a `{"mean", "std"}` dict for the x axis; defaults
            to the first effect's schema scaling.
        scale_y: same, for the y axis.
        y_limits: `None` or tuple, manual y-axis limits.
        title: figure title.
        show_plot: if `True`, show the figure and return `None`; if `False`,
            return `(fig, ax)`.

    Raises:
        ValueError: fewer than two effects; effects disagree on columns,
            feature resolution, categorical status, or observed level sets;
            derivative and output units are mixed; or the axis intervals do
            not overlap.
    """
    if len(effects) < 2:
        raise ValueError(
            f"compare needs at least two effect objects, got {len(effects)}"
        )

    dims = {e.dim for e in effects}
    if len(dims) != 1:
        raise ValueError(
            f"compare needs effects over data with the same columns; "
            f"got dims {sorted(dims)}"
        )
    idxs = [e._resolve_feature(feature) for e in effects]
    if len(set(idxs)) != 1:
        raise ValueError(
            f"feature {feature!r} resolves to different columns across the "
            f"effects: {idxs}; align the schemas or pass an index"
        )
    f = idxs[0]

    is_cat = {bool(e._is_cat(f)) for e in effects}
    if len(is_cat) != 1:
        raise ValueError(
            f"the effects disagree on whether feature {feature!r} is "
            "categorical; align the schemas"
        )
    discrete = is_cat.pop()

    is_der = [e.method_name == "d-pdp" for e in effects]
    if any(is_der) and not all(is_der):
        raise ValueError(
            "cannot mix derivative-unit effects (DerPDP, dy/dx) with "
            "level-unit effects (PDP/ALE/RHALE/ShapDP) on one axis"
        )

    centering = helpers.prep_centering(centering)
    if centering is False:
        warnings.warn(
            "Comparing methods without centering is not meaningful (each "
            "method uses a different reference level). Using "
            "centering='zero_integral'.",
            stacklevel=2,
        )
        centering = "zero_integral"

    first = effects[0]
    if discrete:
        level_sets = [tuple(np.unique(e.data[:, f]).tolist()) for e in effects]
        if len(set(level_sets)) != 1:
            raise ValueError(
                f"the effects observe different level sets for feature "
                f"{feature!r}: {sorted(set(level_sets))}"
            )
        xs, level_labels = first._level_display(f)
        xs = np.asarray(xs, dtype=float)
    else:
        lo = max(e.axis_limits[0, f] for e in effects)
        hi = min(e.axis_limits[1, f] for e in effects)
        if not lo < hi:
            raise ValueError(
                f"the effects' axis intervals for feature {feature!r} do not "
                f"overlap (intersection [{lo}, {hi}])"
            )
        xs = np.linspace(lo, hi, nof_points)
        level_labels = None

    if labels is None:
        labels = _method_labels(effects)
    elif len(labels) != len(effects):
        raise ValueError(
            f"got {len(labels)} labels for {len(effects)} effects"
        )

    curves = {
        label: e.eval(f, xs, centering=centering)
        for label, e in zip(labels, effects)
    }

    scale_x = helpers.resolve_scale(
        scale_x, first.scale_x_list[f] if first.scale_x_list else None
    )
    scale_y = helpers.resolve_scale(scale_y, first.scale_y)

    return plot_effect_comparison(
        xs,
        f,
        curves,
        scale_x=scale_x,
        scale_y=scale_y,
        avg_output=None,  # possibly different models: no shared baseline
        feature_names=first.feature_names,
        target_name="dy/dx" if all(is_der) else first.target_name,
        y_limits=y_limits,
        title=title,
        discrete=discrete,
        level_labels=level_labels,
        show_plot=show_plot,
    )


def triage_scatter(
    points,
    *,
    arrows=None,
    threshold=False,
    thr_label="heterogeneity threshold",
    unit="",
    title=None,
    show_plot=False,
):
    """Draw a triage plane from precomputed scalars — no effect object needed.

    The value-in twin of `plot_triage`: `Report` (unbound) and
    `CALM.plot_triage` hand in stamped numbers, this draws them.

    Args:
        points: `[(name, importance, heterogeneity)]`, one entry per feature.
        arrows: optional `{name: ((x0, y0), (x1, y1))}` — one arrow per
            feature, e.g. global point -> weighted-mean regional point.
        threshold: heterogeneity line — a float draws it, `None`/`False`
            nothing.
        thr_label: legend label of the threshold line.
        unit: axis-label suffix, e.g. ``" (cnt units)"``.
        title: figure title.
        show_plot: if `True`, show and return `None`; else `(fig, ax)`.
    """
    t = theme.active()
    fig, ax = plt.subplots()
    ax.set_title("Feature triage" if title is None else title)
    ax.scatter(
        [p[1] for p in points],
        [p[2] for p in points],
        color=t.MEAN,
        zorder=3,
    )
    for name, x, y in points:
        ax.annotate(
            name,
            (x, y),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize="small",
        )
    if arrows:
        for i, (name, (start, end)) in enumerate(arrows.items()):
            color = t.CAT[i % len(t.CAT)]
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="->", color=color, linewidth=1.2, alpha=0.9
                ),
            )
            ax.scatter(
                [start[0]],
                [start[1]],
                facecolors="none",
                edgecolors=color,
                zorder=3,
                label=f"{name} before regions",
            )
    if threshold is not None and threshold is not False:
        ax.axhline(
            float(threshold),
            color=t.AVG,
            linestyle="--",
            linewidth=1.0,
            label=thr_label,
        )
    _decorate_ax(ax, xlabel="importance" + unit, ylabel="heterogeneity" + unit)
    fig.tight_layout()
    return _finalize(fig, ax, show_plot)


def plot_triage(
    effect,
    partitions: typing.Union[None, dict] = None,
    threshold: typing.Union[None, bool, float] = None,
    features: typing.Union[str, list] = "all",
    title: typing.Union[None, str] = None,
    show_plot: bool = True,
):
    """The triage plane of one fitted effect: importance (x) against heterogeneity (y), one labeled point per feature.

    ```python
    effector.plot_triage(pdp)                       # survey every feature
    parts = pdp.find_regions(features="heterogeneous")
    effector.plot_triage(pdp, partitions=parts)     # before/after arrows
    ```

    Read it as a to-do list: bottom-left is unimportant and honest, the
    bottom-right features are important and fully described by their mean
    effect, and the top-right corner — important *and* heterogeneous — is
    where the mean effect hides something and `find_regions` should look.

    !!! tip "The before/after story"
        With `partitions`, an arrow runs from every partitioned feature's
        global point to each of its leaf points (the leaf's
        `importance`/`heter_score` under its rule, computed model-free from
        the caches). Leaves of a good partition land right and down — more
        decisive, less heterogeneous.

    Args:
        effect: a fitted effect object (`PDP`/`RHALE`/...); its `importance`
            and `heter_score` are queried per feature.
        partitions: optional `{feature_name_or_index: Partition}` — exactly
            what `effect.find_regions(features=...)` returns. Root-only
            partitions are skipped.
        threshold: the heterogeneity threshold line. `None` (default) draws
            the median heterogeneity of the plotted features — the same
            convention `effector.explain` and `find_regions
            (features="heterogeneous")` use; a float draws that value;
            `False` draws nothing.
        features: which features to plot — `"all"` or a list of
            indices/names. Feature types the method does not support are
            skipped with one `UserWarning`.
        title: figure title.
        show_plot: if `True`, show the figure and return `None`; if `False`,
            return `(fig, ax)`.
    """
    if isinstance(features, str):
        if features != "all":
            raise ValueError(
                f"Invalid features argument: {features!r}; use 'all' or a "
                "list of indices/names"
            )
        candidates = list(range(effect.dim))
    else:
        candidates = [effect._resolve_feature(f) for f in features]

    plotted, skipped = [], []
    for f in candidates:
        try:
            effect._check_feature_type_supported(f)
            plotted.append(f)
        except ValueError:
            skipped.append(effect.feature_names[f])
    if skipped:
        warnings.warn(
            f"plot_triage skipped feature(s) {skipped} — this method does "
            f"not support their feature type.",
            UserWarning,
            stacklevel=2,
        )
    if not plotted:
        raise ValueError("plot_triage: no supported features to plot")

    imp = {f: effect.importance(f) for f in plotted}
    het = {f: effect.heter_score(f) for f in plotted}

    fig, ax = plt.subplots()
    t = theme.active()
    ax.set_title("Feature triage" if title is None else title)

    ax.scatter(
        [imp[f] for f in plotted],
        [het[f] for f in plotted],
        color=t.MEAN,
        zorder=3,
        label="global effect",
    )
    for f in plotted:
        ax.annotate(
            effect.feature_names[f],
            (imp[f], het[f]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize="small",
        )

    if threshold is None:
        threshold = float(np.median([het[f] for f in plotted]))
        thr_label = "heterogeneity threshold (median)"
    else:
        thr_label = "heterogeneity threshold"
    if threshold is not False:
        ax.axhline(
            threshold,
            color=t.AVG,
            linestyle="--",
            linewidth=1.0,
            label=thr_label,
        )

    if partitions:
        cat_cycle = t.CAT
        for i, (key, partition) in enumerate(partitions.items()):
            f = effect._resolve_feature(key)
            leaves = partition.leaves
            if len(partition) <= 1:
                continue  # root-only: nothing was found
            color = cat_cycle[i % len(cat_cycle)]
            start = (imp[f], het[f])
            first_leaf = True
            for leaf in leaves:
                end = (
                    effect.importance(f, rule=leaf.rule),
                    effect.heter_score(f, rule=leaf.rule),
                )
                ax.annotate(
                    "",
                    xy=end,
                    xytext=start,
                    arrowprops=dict(
                        arrowstyle="->", color=color, linewidth=1.2, alpha=0.9
                    ),
                )
                ax.scatter(
                    [end[0]],
                    [end[1]],
                    facecolors="none",
                    edgecolors=color,
                    zorder=3,
                    label=(
                        f"{effect.feature_names[f]} leaves" if first_leaf else None
                    ),
                )
                first_leaf = False

    # both axes are std-type quantities in the target's units (units contract)
    unit = f" ({effect.target_name} units)"
    _decorate_ax(ax, xlabel="importance" + unit, ylabel="heterogeneity" + unit)
    return _finalize(fig, ax, show_plot)

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

import textwrap
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


def _wrap_text(s, width=48, max_lines=2):
    """Wrap to at most `max_lines` lines; ellipsize when truncated."""
    lines = textwrap.wrap(str(s), width=width, break_long_words=False)
    if not lines:
        return str(s)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] += " …"
    return "\n".join(lines)


def _set_title(ax, title, tag=None):
    """The title carries the information (feature / rule), left-aligned and
    wrapped; the constant context (method · scope) is a muted corner tag."""
    if title is not None:
        ax.set_title(_wrap_text(title), loc="left")
    if tag is not None:
        ax.text(
            1.0,
            1.02,
            tag,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.5,
            color=theme.active().TAG,
        )


def _ref_line(
    ax, *, y=None, x=None, label, tag=None, color=None, linestyle="-", lw=1.0
):
    """A reference hairline (threshold / avg output / zero): the artist keeps
    its `label` (tests look lines up by it) but is excluded from the legend —
    `_decorate_ax` filters everything registered here. `tag` draws a small
    inline right-edge (or top-edge) text instead of a legend entry, in blended
    coordinates so a later `set_xlim`/`set_ylim` never strands it."""
    color = theme.active().REF if color is None else color
    labels = getattr(ax, "_effector_ref_labels", None)
    if labels is None:
        labels = set()
        ax._effector_ref_labels = labels
    labels.add(label)
    t = theme.active()
    if y is not None:
        ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=lw, label=label)
        if tag is not None:
            ax.text(
                0.995,
                y,
                tag,
                transform=ax.get_yaxis_transform(),
                ha="right",
                va="bottom",
                fontsize=7,
                color=t.TAG,
            )
    if x is not None:
        ax.axvline(x=x, color=color, linestyle=linestyle, linewidth=lw, label=label)
        if tag is not None:
            ax.text(
                x,
                0.99,
                tag,
                transform=ax.get_xaxis_transform(),
                ha="left",
                va="top",
                fontsize=7,
                color=t.TAG,
            )


def _add_avg_output(ax, avg_output):
    if avg_output is not None:
        _ref_line(
            ax,
            y=avg_output,
            label="avg output",
            tag="avg output",
            color=theme.active().AVG,
            linestyle="--",
        )


def _decorate_ax(ax, xlabel=None, ylabel=None, y_limits=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    # legend discipline: reference lines carry inline tags instead of legend
    # entries, and a single-series axis needs no legend box at all
    ref = getattr(ax, "_effector_ref_labels", ())
    handles, labels = ax.get_legend_handles_labels()
    kept = [(h, lab) for h, lab in zip(handles, labels) if lab not in ref]
    if len(kept) >= 2:
        ax.legend([h for h, _ in kept], [lab for _, lab in kept])
    if y_limits is not None:
        ax.set_ylim(y_limits[0], y_limits[1])


def _repel_labels(ax, fig, pts, *, avoid=None, max_labels=20, fontsize=8):
    """Greedy point-label placement: try a ring of candidate offsets, keep the
    first that collides with nothing placed so far (labels, ticks, `avoid`
    extents); draw a hairline leader when the label lands away from its point.
    `pts` is `[(name, x, y)]`; at most `max_labels` labels are drawn, ranked
    by x descending (importance) — the rest stay unlabeled points."""
    t = theme.active()
    fig.canvas.draw()
    placed = list(avoid or [])
    placed += [
        lbl.get_window_extent() for lbl in ax.get_xticklabels() + ax.get_yticklabels()
    ]
    ranked = sorted(pts, key=lambda p: -(p[1] + p[2]))[:max_labels]
    ink = plt.rcParams.get("text.color", "black")
    fig_bb = fig.get_window_extent()
    for name, x, y in sorted(ranked, key=lambda p: -p[2]):
        for dx, dy in [
            (7, 5),
            (7, -11),
            (-7, 5),
            (-7, -11),
            (7, 16),
            (-7, 16),
            (7, -22),
            (0, 24),
            (0, -30),
        ]:
            ha = "left" if dx > 0 else ("center" if dx == 0 else "right")
            txt = ax.annotate(
                name,
                (x, y),
                textcoords="offset points",
                xytext=(dx, dy),
                fontsize=fontsize,
                ha=ha,
                color=ink,
                zorder=4,
            )
            fig.canvas.draw()
            bb = txt.get_window_extent()
            clipped = bb.x1 > fig_bb.x1 - 2 or bb.x0 < fig_bb.x0 + 2
            if clipped or any(bb.overlaps(p) for p in placed):
                txt.remove()
                continue
            placed.append(bb.expanded(1.05, 1.15))
            if abs(dx) + abs(dy) > 20:  # displaced: hairline leader
                end = ax.transData.inverted().transform(
                    (bb.x0 - 2 if dx > 0 else bb.x1 + 2, (bb.y0 + bb.y1) / 2)
                )
                ax.plot([x, end[0]], [y, end[1]], lw=0.6, color=t.REF, zorder=2)
            break


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
    tag: typing.Union[None, str] = None,
):
    """Draw the (RH)ALE mean effect (top axis) and the bin-slope bar plot
    (bottom axis, with sqrt(bin_variance) error bars if `heterogeneity`).

    `x`/`y` is the mean-effect curve; `bin_effect`/`bin_variance`/`limits`/`dx`
    is the stored bin payload. `x_limits` (raw feature units) windows the
    shared x-axis — the masked/subregion zoom. `tag` is the muted corner
    context tag ("RHALE · global").
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
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.12},
        )
        axes = (ax1, ax2)

    t = theme.active()
    x_name = _feature_label(feature, feature_names)
    _set_title(ax1, x_name if title is None else title, tag)
    ax1.plot(x, y, color=t.MEAN, linestyle="-", linewidth=2.0, label="average effect")
    _add_avg_output(ax1, avg_output)

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
            color=t.MEAN,
            alpha=0.30,
            edgecolor=t.MEAN,
            linewidth=0.8,
            yerr=yerr,
            ecolor=t.ERROR,
            error_kw={"lw": 0.9},
            capsize=2,
            label="bin slope ± heterogeneity" if heterogeneity else "bin slope",
        )
        _ref_line(ax2, y=0, label="zero")
        ax2.set_xlabel(x_name)
        ax2.set_ylabel("Δy/Δx")
        # the sanctioned single-entry legend: the panel's series needs naming
        handles, labels = ax2.get_legend_handles_labels()
        ref = getattr(ax2, "_effector_ref_labels", ())
        kept = [(h, lab) for h, lab in zip(handles, labels) if lab not in ref]
        ax2.legend([h for h, _ in kept], [lab for _, lab in kept], fontsize=8)
        if dy_limits is not None:
            ax2.set_ylim(dy_limits[0], dy_limits[1])

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
    tag: typing.Union[None, str] = None,
):
    """Draw the pre-computed PDP mean curve `y_mean` (shape `(T,)`) plus the
    requested heterogeneity (R1 — nothing is computed here): a std/std-err
    `band` (shape `(T,)`), or the raw ICE table `ice` (shape `(T, N)`) as a
    curve cloud. A band is a spread, so it scales by `scale_y["std"]` only."""
    x = _scale_x(x, scale_x)
    y_mean = _scale_y(y_mean, scale_y, is_derivative)

    fig, ax = plt.subplots()
    t = theme.active()
    _set_title(ax, title, tag)

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
    level_kind: str = "nominal",
):
    """Overlay the mean effect of several methods for one feature on a single
    axis. `curves` maps `{method_label: y}`, each `y` of shape `(T,)`.

    `discrete=True` (categorical feature): the effects are per-level values,
    so each method is drawn as a marker series at the level positions and the
    x-axis shows the level ticks/labels instead of a continuous grid. A thin
    connecting line joins the markers only for `level_kind="ordinal"` — for
    nominal levels a line would imply an interpolation between unordered
    categories."""
    fig, ax = plt.subplots()
    t = theme.active()
    _set_title(ax, _feature_label(feature, feature_names) if title is None else title)

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
                linewidth=1.2 if level_kind == "ordinal" else 0,
                linestyle="-" if level_kind == "ordinal" else "none",
                color=color,
                label=label,
            )
        else:
            ax.plot(x_scaled, _scale_y(y, scale_y), color=color, label=label)
    if discrete:
        _categorical_axis(ax, x_scaled, level_labels, level_kind)
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
    title: typing.Union[None, str] = None,
    tag: typing.Union[None, str] = None,
):
    """Draw the SHAP-DP curve `x`/`y` plus the requested heterogeneity: a std
    band (`y_std`) or the shap-value cloud `xx`/`yy`."""
    fig, ax = plt.subplots()
    t = theme.active()
    _set_title(
        ax, _feature_label(feature, feature_names) if title is None else title, tag
    )

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
            marker=".",
            markersize=3.5,
            linestyle="none",
            alpha=t.SHAP_MARKER_ALPHA,
            label="SHAP values",
        )
        ax.plot(
            xx,
            yy,
            color=t.CLOUD,
            marker=".",
            markersize=3.5,
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


def _level_layout(heights, positions, level_kind="nominal", sort=None):
    """Display order and draw positions for a categorical axis.

    Nominal levels sort by effect value and draw at ranks `0..K-1` (a rank is
    display geometry — `scale_x` must never touch it); ordinal levels keep
    their natural order at their level values (`scale_x` applies).

    Returns `(order, draw_positions, use_scale_x)`.
    """
    positions = np.asarray(positions, dtype=float)
    k = len(positions)
    if sort is None:
        sort = level_kind == "nominal"
    order = np.argsort(np.asarray(heights, dtype=float)) if sort else np.arange(k)
    if level_kind == "nominal":
        return order, np.arange(k, dtype=float), False
    return order, positions[order], True


def _categorical_axis(ax, positions, level_labels, level_kind="nominal"):
    """The one tick-policy spot (C4). `positions` are display coordinates
    (ranks or scaled level values). Unnamed levels get raw-unit integer-ish
    labels; named levels are wrapped (≤12 chars/line) and rotated by
    crowding (0/30/45/60). Never label every level past 8 when the labels
    are numeric or the levels ordered — majors thin to ≤8 with a minor tick
    marking each level. Deterministic in K and label lengths (R8)."""
    positions = np.asarray(positions, dtype=float)
    k = len(positions)
    generated = level_labels is None
    if generated:
        near_int = np.allclose(positions, np.round(positions), atol=1e-6)
        level_labels = [f"{v:.0f}" if near_int else f"{v:g}" for v in positions]
        rot = 0
    else:
        level_labels = [str(lab) for lab in level_labels]
        longest = max(len(lab) for lab in level_labels)
        if longest <= 6 and k <= 6:
            rot = 0
        elif k <= 8 and longest <= 16:
            rot = 30
        elif k <= 16:
            rot = 45
        else:
            rot = 60
        level_labels = [_wrap_text(lab, width=12, max_lines=2) for lab in level_labels]
    # thinning drops labels, so it is reserved for levels whose identity
    # survives it: numbers, or named-but-ordered levels
    step = int(np.ceil(k / 8)) if (generated or level_kind == "ordinal") else 1
    ax.set_xticks(positions[::step])
    ax.set_xticklabels(
        level_labels[::step], rotation=rot, ha="right" if rot else "center"
    )
    if step > 1:
        ax.set_xticks(positions, minor=True)


def _level_counts(ax, positions, counts):
    """Muted per-level sample sizes along the top of the axes. Past ~10
    levels the labels would collide into noise, so dense axes skip them —
    the counts stay available programmatically (`_level_weights`)."""
    positions = np.asarray(positions, dtype=float)
    if len(positions) > 10:
        return
    t = theme.active()
    for x, n in zip(positions, counts):
        ax.text(
            x,
            0.985,
            f"n={int(n):,}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=6.5,
            color=t.TAG,
        )


def _bar_width(positions):
    positions = np.asarray(positions, dtype=float)
    return 0.6 * np.min(np.diff(positions)) if len(positions) > 1 else 0.6


def plot_categorical_effect(
    positions,
    heights,
    variances,
    feature,
    heterogeneity,
    title=None,
    level_labels=None,
    scale_x=None,
    scale_y=None,
    avg_output=None,
    feature_names=None,
    target_name=None,
    y_limits=None,
    connect_line=False,
    show_plot=True,
    tag=None,
    level_kind="nominal",
    sort=None,
    level_counts=None,
):
    """Bars at the level positions with heterogeneity whiskers = sqrt(h(v_k)).

    The categorical analogue of the mean-effect curve (method_semantics.md):
    used by PDP (per-level means), (RH)ALE (accumulated per-level values) and
    ShapDP (per-level shap means). Pure drawing — heights/variances arrive
    evaluated and centered.

    Nominal levels sort by effect value and draw at ranks; ordinal levels keep
    their natural order (`_level_layout`). `connect_line=True` overlays a line
    through the bar tops — only meaningful for (RH)ALE on *ordinal* features,
    where the bars are an accumulation and the slope between two bars is the
    per-transition step the method measures. `level_counts` writes a muted
    per-level `n=…` along the top.
    """
    fig, ax = plt.subplots()
    t = theme.active()
    _set_title(
        ax, _feature_label(feature, feature_names) if title is None else title, tag
    )

    heights = np.asarray(heights, dtype=float)
    order, draw_pos, use_scale = _level_layout(heights, positions, level_kind, sort)
    x = _scale_x(draw_pos, scale_x) if use_scale else draw_pos
    y = _scale_y(heights, scale_y)[order]
    labels = [level_labels[i] for i in order] if level_labels is not None else None
    width = _bar_width(x)

    yerr = None
    if heterogeneity is not False and variances is not None:
        std = np.sqrt(np.asarray(variances, dtype=float))[order]
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
        capsize=3,
        error_kw={"lw": 1.0},
        label="mean effect ± heterogeneity" if yerr is not None else "mean effect",
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
    _categorical_axis(ax, x, labels, level_kind)
    if level_counts is not None:
        _level_counts(ax, x, np.asarray(level_counts)[order])
    _add_avg_output(ax, avg_output)
    _decorate_ax(
        ax,
        xlabel=_feature_label(feature, feature_names),
        ylabel=target_name,
        y_limits=y_limits,
    )
    if yerr is not None and ax.get_legend() is None:
        # whiskers need naming even when the bars are the only series — the
        # sanctioned single-entry legend (like the ALE dy/dx panel)
        handles, lbls = ax.get_legend_handles_labels()
        ref = getattr(ax, "_effector_ref_labels", ())
        kept = [(h, lab) for h, lab in zip(handles, lbls) if lab not in ref]
        if kept:
            ax.legend([kept[0][0]], [kept[0][1]], fontsize=8)
    return _finalize(fig, ax, show_plot)


def plot_pdp_ice_categorical(
    positions,
    yy,
    feature,
    title=None,
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
    tag=None,
    level_kind="nominal",
    sort=None,
    level_counts=None,
):
    """Bars for the per-level mean + jittered per-instance ICE dots.

    `yy` is the (K, N) ICE table evaluated at the K levels. Jitter and the
    ICE subsample are seeded (`random_state`) — determinism is contractual
    (R8). Nominal levels sort by the per-level mean and draw at ranks."""
    fig, ax = plt.subplots()
    t = theme.active()
    _set_title(
        ax, _feature_label(feature, feature_names) if title is None else title, tag
    )

    yy = np.asarray(yy, dtype=float)
    means = yy.mean(axis=1)
    order, draw_pos, use_scale = _level_layout(means, positions, level_kind, sort)
    yy = yy[order]
    x = _scale_x(draw_pos, scale_x) if use_scale else draw_pos
    labels = [level_labels[i] for i in order] if level_labels is not None else None
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
    _categorical_axis(ax, x, labels, level_kind)
    if level_counts is not None:
        _level_counts(ax, x, np.asarray(level_counts)[order])
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
    title=None,
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
    tag=None,
    level_kind="nominal",
    sort=None,
    level_counts=None,
):
    """Bars for the per-level shap mean + the jittered shap cloud.

    Nominal levels sort by the per-level mean and draw at ranks; the raw
    cloud's x values (level values) are remapped to the display positions
    before jittering (seeding unchanged — R8)."""
    fig, ax = plt.subplots()
    t = theme.active()
    _set_title(
        ax, _feature_label(feature, feature_names) if title is None else title, tag
    )

    heights = np.asarray(heights, dtype=float)
    levels = np.asarray(positions, dtype=float)
    order, draw_pos, use_scale = _level_layout(heights, positions, level_kind, sort)
    x = _scale_x(draw_pos, scale_x) if use_scale else draw_pos
    labels = [level_labels[i] for i in order] if level_labels is not None else None
    y_mean = _scale_y(heights, scale_y)[order]
    width = _bar_width(x)

    rng = np.random.default_rng(random_state)
    n = len(yy)
    if nof_shap_values != "all" and int(nof_shap_values) < n:
        keep = rng.choice(n, int(nof_shap_values), replace=False)
    else:
        keep = np.arange(n)
    jitter = rng.uniform(-0.25 * width, 0.25 * width, size=len(keep))
    # map each raw cloud value (a level value) to its display position
    inv = np.argsort(order)
    cloud_levels = np.searchsorted(levels, np.asarray(xx, dtype=float)[keep])
    ax.plot(
        x[inv[cloud_levels]] + jitter,
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
    _categorical_axis(ax, x, labels, level_kind)
    if level_counts is not None:
        _level_counts(ax, x, np.asarray(level_counts)[order])
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
        raise ValueError(f"got {len(labels)} labels for {len(effects)} effects")

    curves = {
        label: e.eval(f, xs, centering=centering) for label, e in zip(labels, effects)
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
        level_kind=first.feature_types[f] if discrete else "nominal",
    )


def _draw_triage(fig, ax, points, arrows, threshold, thr_label, unit, title):
    """The one triage look (shared by `triage_scatter` and `plot_triage`):
    labeled global points, one mild-accent arrow per accepted split ending in
    an open circle (a single "regional effects" legend entry), the threshold
    as a solid hairline with an inline tag, and greedy label repel with
    hairline leaders — capped at the top 20 features. `arrows` is a list of
    `((x0, y0), (x1, y1))` pairs."""
    t = theme.active()
    _set_title(ax, "Feature triage" if title is None else title)
    ax.margins(0.10)
    ax.scatter(
        [p[1] for p in points],
        [p[2] for p in points],
        s=42,
        color=t.MEAN,
        zorder=3,
        edgecolor=plt.rcParams.get("axes.facecolor", "#ffffff"),
        linewidth=1.2,
        label="global effects",
    )
    first = True
    for start, end in arrows or []:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(
                arrowstyle="->",
                color=t.ARROW,
                linewidth=1.1,
                alpha=0.9,
                shrinkA=4,
                shrinkB=4,
            ),
        )
        ax.scatter(
            [end[0]],
            [end[1]],
            facecolors="none",
            edgecolors=t.ARROW,
            s=38,
            linewidth=1.2,
            zorder=3,
            label="regional effects" if first else None,
        )
        first = False
    if threshold is not None and threshold is not False:
        _ref_line(
            ax,
            y=float(threshold),
            label=thr_label,
            tag="median heterogeneity" if "median" in thr_label else thr_label,
        )
    _decorate_ax(ax, xlabel="importance" + unit, ylabel="heterogeneity" + unit)
    fig.canvas.draw()
    avoid = []
    legend = ax.get_legend()
    if legend is not None:
        avoid.append(legend.get_window_extent())
    _repel_labels(ax, fig, points, avoid=avoid)


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
        thr_label: label of the threshold line (kept on the artist; shown as
            an inline tag, not a legend entry).
        unit: axis-label suffix, e.g. ``" (cnt units)"``.
        title: figure title.
        show_plot: if `True`, show and return `None`; else `(fig, ax)`.
    """
    fig, ax = plt.subplots()
    _draw_triage(
        fig,
        ax,
        points,
        list(arrows.values()) if arrows else [],
        threshold,
        thr_label,
        unit,
        title,
    )
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

    if threshold is None:
        threshold = float(np.median([het[f] for f in plotted]))
        thr_label = "heterogeneity threshold (median)"
    else:
        thr_label = "heterogeneity threshold"

    arrows = []
    if partitions:
        for key, partition in partitions.items():
            f = effect._resolve_feature(key)
            if len(partition) <= 1:
                continue  # root-only: nothing was found
            start = (imp[f], het[f])
            for leaf in partition.leaves:
                arrows.append(
                    (
                        start,
                        (
                            effect.importance(f, rule=leaf.rule),
                            effect.heter_score(f, rule=leaf.rule),
                        ),
                    )
                )

    fig, ax = plt.subplots()
    points = [(effect.feature_names[f], imp[f], het[f]) for f in plotted]
    # both axes are std-type quantities in the target's units (units contract)
    unit = f" ({effect.target_name} units)"
    _draw_triage(fig, ax, points, arrows, threshold, thr_label, unit, title)
    return _finalize(fig, ax, show_plot)

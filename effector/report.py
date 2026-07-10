"""One-click explanation report (design goal (b)).

`effector.explain(data, model, ...)` runs the whole pipeline — fit → rank by
importance (R13) → mean-effect curves for the top-k → `find_regions` on the
heterogeneous ones (R12) → explained-variance surrogates — and returns a
`Report`, a serializable **value** (R12: values, not state). Every model call
happens through the single `effect.fit(...)` plus one prediction pass for
`f̂(X)`; importance, heter_score, curves, find_regions, and the surrogate R²s
are all model-free afterwards.

`Report` binds a reference to its producing effect only for the lazy re-plot
sugar (mirrors `Partition._bind`); `to_dict()`/`from_dict()` are the
serialization boundary and round-trip without an effect.

`to_html()` renders the report as the analyst pipeline reads, in one
self-contained file: the triage plane first (where to look), then one section
per feature — global effect, then regional — and the before/after triage last.
"""

from __future__ import annotations

import base64
import contextlib
import html as _htmlmod
import io
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from effector import explained_variance as _ev
from effector import helpers, method_registry
from effector.partition import Partition


@contextlib.contextmanager
def _offscreen_figures():
    """Build figures without ever putting them on screen.

    `show_plot=False` only suppresses the explicit `plt.show()`. Under
    `plt.ion()` (any GUI session, and what `scripts/api_playground.py` sets up)
    pyplot paints a figure the moment it is created, so a report that builds
    dozens of intermediate figures would flash dozens of windows before
    `_img` base64-encodes and closes them. Interactive mode off for the
    duration; the caller's setting is restored, so their own `plot()` calls
    keep popping up as before.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    was_interactive = matplotlib.is_interactive()
    plt.ioff()
    try:
        yield
    finally:
        if was_interactive:
            plt.ion()


@dataclass
class FeatureReport:
    """The per-feature slice of a `Report` — plain values, no live effect.

    Attributes:
        feature: feature index.
        name: feature display name.
        importance: the feature's importance score.
        heter_score: the feature's scalar heterogeneity.
        xs: evaluation grid, `(T,)`.
        y: mean-effect curve at `xs`, `(T,)`.
        h: heterogeneity curve at `xs`, `(T,)`.
        partition: `Partition.to_dict()` when `find_regions` ran, else `None`.
    """

    feature: int
    name: str
    importance: float
    heter_score: float
    xs: np.ndarray
    y: np.ndarray  # mean effect
    h: np.ndarray  # heterogeneity curve
    partition: Optional[dict] = None  # Partition.to_dict() if find_regions ran

    def to_dict(self):
        """Serialize to a plain JSON-able dict (arrays become lists)."""
        return {
            "feature": self.feature,
            "name": self.name,
            "importance": self.importance,
            "heter_score": self.heter_score,
            "xs": np.asarray(self.xs).tolist(),
            "y": np.asarray(self.y).tolist(),
            "h": np.asarray(self.h).tolist(),
            "partition": self.partition,
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild a `FeatureReport` from `to_dict()` output."""
        return cls(
            feature=d["feature"],
            name=d["name"],
            importance=d["importance"],
            heter_score=d["heter_score"],
            xs=np.asarray(d["xs"]),
            y=np.asarray(d["y"]),
            h=np.asarray(d["h"]),
            partition=d["partition"],
        )


@dataclass
class Report:
    """An importance-ranked, serializable explanation of a model — a value.

    ```python
    report = effector.explain(X, model)
    report.show()                     # ranked table + partition trees
    report.to_html("report.html")     # self-contained page
    ```

    `features` holds one `FeatureReport` per reported feature, importance
    descending. `overview` holds the cheap scalars — importance and
    heterogeneity — for **every** supported feature (superset of `features`),
    so the triage plane renders even on unbound reports. `explained_variance`
    holds the label-free surrogate R² payload (`effector.explained_variance`)
    — plain floats, so it too renders unbound; `None` for derivative-scale
    methods. A report produced by `explain` is bound to its fitted effect,
    which `to_html` uses for the per-leaf regional plots and the before/after
    triage arrows; everything else works from the stored values alone.
    """

    method_name: str
    feature_names: list
    target_name: str
    features: List[FeatureReport]
    config: dict = field(default_factory=dict)
    overview: List[dict] = field(default_factory=list)
    explained_variance: Optional[dict] = None

    def __post_init__(self):
        self._effect = None
        if not self.overview:
            # old dicts / hand-built reports: the reported features stand in
            self.overview = [
                {
                    "feature": fr.feature,
                    "name": fr.name,
                    "importance": fr.importance,
                    "heter_score": fr.heter_score,
                    "reported": True,
                }
                for fr in self.features
            ]

    def _bind(self, effect):
        self._effect = effect
        return self

    def _require_effect(self):
        if self._effect is None:
            raise RuntimeError(
                "This Report is not bound to an effect (e.g. it was rebuilt from "
                "to_dict()); live plotting is unavailable — use the stored values "
                "or to_html()."
            )
        return self._effect

    # -- terminal summary ------------------------------------------------------
    def _ev_headline(self):
        """The explained-variance one-liner, or `None` when the section is
        absent (derivative-scale method / degenerate model output)."""
        ev = self.explained_variance
        if not ev:
            return None
        line = f"global effects reproduce {ev['gam_r2']:.1%} of the model's variance"
        if ev["gains"]:
            line += f"; with subregions, {ev['regional_r2']:.1%}"
        return line

    def show(self):
        """Print the ranked feature table, the explained-variance summary,
        then each multi-region partition tree.

        Columns: feature, importance, heterogeneity, #regions. Works on
        unbound reports (rebuilt via `from_dict`) too.
        """
        title = method_registry.resolve(self.method_name).display_name
        print(f"\n{title} report — target: {self.target_name}")
        print("=" * 60)
        print(f"{'feature':<24}{'importance':>12}{'heter':>10}{'#regions':>10}")
        print("-" * 60)
        for fr in self.features:
            nregions = len(fr.partition["regions"]) if fr.partition else 1
            print(
                f"{fr.name:<24}{fr.importance:>12.4f}{fr.heter_score:>10.4f}"
                f"{nregions:>10d}"
            )
        print("=" * 60)
        headline = self._ev_headline()
        if headline:
            print(headline)
            for g in self.explained_variance["gains"]:
                print(
                    f"  splitting {g['name']} (on {g['on']}) recovers "
                    f"{g['delta_r2'] * 100:+.1f} pts"
                )
        for fr in self.features:
            if fr.partition is not None and len(fr.partition["regions"]) > 1:
                Partition.from_dict(fr.partition).show()

    # -- overview figures (R7 return rule) --------------------------------------
    def _barh(self, names, vals, xlabel, title, threshold=None):
        from effector import theme
        import matplotlib.pyplot as plt

        t = theme.active()
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.5))
        ax.barh(range(len(names)), vals, color=t.BAR_FACE)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # most important on top
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        if threshold is not None:
            ax.axvline(
                threshold,
                color=t.AVG,
                linestyle="--",
                linewidth=1.0,
                label="heterogeneity threshold",
            )
            ax.legend()
        fig.tight_layout()
        return fig, ax

    def plot_importance(self, show_plot=True):
        """Horizontal bar chart of feature importance, most important on top.

        Covers every feature in `overview` (all supported features), not just
        the reported top-k.

        Args:
            show_plot: `True` (default) displays the figure and returns
                `None`; `False` returns `(fig, ax)` instead.

        Returns:
            `None`, or `(fig, ax)` when `show_plot=False`.
        """
        import matplotlib.pyplot as plt

        fig, ax = self._barh(
            [o["name"] for o in self.overview],
            [o["importance"] for o in self.overview],
            xlabel=f"importance ({self.target_name} units)",
            title=(
                f"{method_registry.resolve(self.method_name).display_name} — "
                f"feature importance"
            ),
        )
        if show_plot:
            plt.show(block=False)
            return None
        return fig, ax

    def _heterogeneity_fig(self):
        return self._barh(
            [o["name"] for o in self.overview],
            [o["heter_score"] for o in self.overview],
            xlabel=f"heterogeneity ({self.target_name} units)",
            title=(
                f"{method_registry.resolve(self.method_name).display_name} — "
                f"heterogeneity"
            ),
            threshold=self.config.get("heter_threshold"),
        )

    def _triage_fig(self, partitions=None, title=None):
        """The triage plane — `effector.plot_triage` when bound, the stored
        scalars otherwise (same style, no arrows)."""
        thr = self.config.get("heter_threshold")
        if self._effect is not None:
            from effector.visualization import plot_triage

            return plot_triage(
                self._effect,
                partitions=partitions,
                threshold=thr,
                features=[o["feature"] for o in self.overview],
                title=title,
                show_plot=False,
            )
        from effector import theme
        from effector.visualization import _decorate_ax
        import matplotlib.pyplot as plt

        t = theme.active()
        fig, ax = plt.subplots()
        ax.set_title("Feature triage" if title is None else title)
        ax.scatter(
            [o["importance"] for o in self.overview],
            [o["heter_score"] for o in self.overview],
            color=t.MEAN,
            zorder=3,
            label="global effect",
        )
        for o in self.overview:
            ax.annotate(
                o["name"],
                (o["importance"], o["heter_score"]),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize="small",
            )
        if thr is not None:
            ax.axhline(
                thr,
                color=t.AVG,
                linestyle="--",
                linewidth=1.0,
                label="heterogeneity threshold",
            )
        unit = f" ({self.target_name} units)"
        _decorate_ax(ax, xlabel="importance" + unit, ylabel="heterogeneity" + unit)
        fig.tight_layout()
        return fig, ax

    # -- self-contained HTML page ---------------------------------------------
    def to_html(self, path=None, share_y="across"):
        """Render the report as one self-contained HTML page — the pipeline in reading order.

        The page mirrors the line-by-line analysis: **overview first** (the
        triage plane over all supported features, a clickable ranked table,
        and collapsible importance/heterogeneity bars), then **one section per
        reported feature** (global effect, partition tree, per-leaf regional
        plots), and the **before/after triage** last — arrows from each
        partitioned feature's global point to its leaves. Every figure is a
        base64 PNG; navigation, click-to-zoom, and collapsing are inline
        vanilla JS/CSS — no external assets, one file, one click.

        For at-a-glance comparison, every panel of every effect plot — the
        effect axis, and (RH)ALE's `dy/dx` axis below it — shares a y range
        with the same panel of the other plots, either across all features
        (`share_y="across"`) or within each one (`"within"`). The x range is
        always shared per feature (its global plot and its leaves). Every
        figure is drawn with the method's own default heterogeneity view — the
        `dy/dx` bars for the (RH)ALE family, the (d-)ICE cloud for the PDP
        family, the SHAP scatter for SHAP-DP.

        !!! note "Unbound reports"
            A report rebuilt with `from_dict` renders everything from stored
            values; the per-leaf regional plots and the triage arrows are
            skipped (they need the live effect) — leaf statistics are shown
            in a table instead.

        Args:
            path: optional file path to write the page to.
            share_y: scope of the shared y (and `dy/dx`) range —
                `"across"` (default) unions it over all features, `"within"`
                over each feature's own global plot and leaves.

        !!! note "No windows"
            The figures are built off-screen and closed once encoded, so
            rendering a report never pops up a plot — even in an interactive
            session where `plot()` normally does.

        Returns:
            the HTML string when `path` is `None`; `None` after writing to
            `path` — so an interactive shell doesn't echo a megabyte of
            markup.
        """
        with _offscreen_figures():
            html = self._render_html(share_y)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(html)
            return None
        return html

    def _render_html(self, share_y="across"):
        """Build the page — every figure created here is closed by `_img`."""
        esc = _htmlmod.escape
        title = method_registry.resolve(self.method_name).display_name
        bound = self._effect is not None

        # effect figures (global + leaves) are encoded LAST, after their axes
        # are harmonized — shared y across the report, shared x per feature —
        # so every effect plot compares at a glance. Triage/bar figures keep
        # their own scales (their y means something else) and encode inline.
        deferred = []  # (fig_ax, alt, section_key)

        def _defer(fig_ax, alt, section):
            deferred.append((fig_ax, alt, section))
            return ("__FIG__", len(deferred) - 1)

        # rebuild each multi-region partition once (bound when possible)
        parts = {}
        for fr in self.features:
            if fr.partition is not None and len(fr.partition["regions"]) > 1:
                p = Partition.from_dict(fr.partition)
                if bound:
                    p = p.bind(self._effect)
                parts[fr.feature] = p

        out = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>{esc(title)} report</title>",
            "<style>",
            _CSS,
            "</style></head><body>",
        ]

        # sticky nav — the pipeline's table of contents
        nav = ["<a href='#overview'>Overview</a>"]
        nav += [
            f"<a href='#feat-{fr.feature}'>{esc(fr.name)}</a>" for fr in self.features
        ]
        nav.append("<a href='#after'>After regions</a>")
        out.append("<nav>" + "".join(nav) + "</nav><main>")

        # header
        out.append(f"<h1>{esc(title)} report</h1>")
        out.append(
            f"<p class='caption'>target <b>{esc(self.target_name)}</b> · "
            f"{len(self.feature_names)} features · "
            f"{len(self.features)} reported</p>"
        )
        chips = []
        for key in (
            "method",
            "top_k",
            "heter_threshold",
            "finder",
            "nof_instances",
            "random_state",
        ):
            if key in self.config:
                val = self.config[key]
                val = f"{val:.4f}" if isinstance(val, float) else str(val)
                chips.append(f"<span class='chip'>{esc(key)} <b>{esc(val)}</b></span>")
        if chips:
            out.append("<div class='chips'>" + "".join(chips) + "</div>")

        # -- 1 · overview ------------------------------------------------------
        out.append("<section id='overview'><h2>1 · Overview — where to look</h2>")
        out.append(
            "<p class='caption'>Each point is a feature: importance (x) against "
            "heterogeneity (y). Bottom-left is ignorable; bottom-right is "
            "important and fully described by its mean effect; the top-right "
            "corner — important <i>and</i> heterogeneous — is where the mean "
            "hides something and <code>find_regions</code> looks for "
            "subregions.</p>"
        )
        out.append(self._img(self._triage_fig(), alt="feature triage"))
        ev = self.explained_variance
        if ev:
            sentence = (
                "An additive surrogate read off these global curves reproduces "
                f"<b>{ev['gam_r2']:.1%}</b> of the model's predicted variance"
            )
            if ev["gains"]:
                sentence += (
                    "; with the subregions of §3, "
                    f"<b>{ev['regional_r2']:.1%}</b>"
                )
            out.append(f"<p class='caption'>{sentence}.</p>")
        out.append(
            "<table><tr><th>#</th><th>feature</th><th>importance</th>"
            "<th>heterogeneity</th><th>#regions</th><th>regional analysis</th></tr>"
        )
        reported = {fr.feature: fr for fr in self.features}
        for rank, o in enumerate(self.overview, 1):
            fr = reported.get(o["feature"])
            if fr is None:
                out.append(
                    f"<tr class='dim'><td>{rank}</td><td>{esc(o['name'])}</td>"
                    f"<td>{o['importance']:.4f}</td><td>{o['heter_score']:.4f}</td>"
                    f"<td>·</td><td>not reported (beyond top_k)</td></tr>"
                )
                continue
            if fr.partition is None:
                nregions, note = 1, "below threshold — skipped"
            elif fr.feature not in parts:
                nregions, note = 1, "searched — no split passed"
            else:
                nregions = len(fr.partition["regions"])
                note = f"split into {len(parts[fr.feature].leaves)} regions"
            out.append(
                f"<tr data-href='feat-{fr.feature}'><td>{rank}</td>"
                f"<td>{esc(fr.name)}</td><td>{fr.importance:.4f}</td>"
                f"<td>{fr.heter_score:.4f}</td><td>{nregions}</td>"
                f"<td>{note} →</td></tr>"
            )
        out.append("</table>")
        out.append(
            "<details><summary>Bar views — importance and heterogeneity</summary>"
        )
        out.append(self._img(self.plot_importance(show_plot=False), alt="importance"))
        out.append(self._img(self._heterogeneity_fig(), alt="heterogeneity"))
        out.append("</details></section>")

        # -- 2 · per-feature analysis -------------------------------------------
        for rank, fr in enumerate(self.features, 1):
            part = parts.get(fr.feature)
            out.append(f"<section id='feat-{fr.feature}'>")
            out.append(f"<h2>2.{rank} · {esc(fr.name)}</h2>")
            out.append(
                "<div class='chips'>"
                f"<span class='chip'>importance <b>{fr.importance:.4f}</b></span>"
                f"<span class='chip'>heterogeneity <b>{fr.heter_score:.4f}</b></span>"
                f"<span class='chip'>regions <b>{len(part.leaves) if part else 1}</b>"
                "</span></div>"
            )
            out.append("<h3>Global effect</h3>")
            out.append(
                _defer(
                    self._global_fig(fr),
                    alt=f"{fr.name} global effect",
                    section=fr.feature,
                )
            )
            out.append("<h3>Regional effects</h3>")
            if fr.partition is None:
                out.append(
                    "<p class='note'>Heterogeneity below the threshold — the mean "
                    "effect tells the whole story; <code>find_regions</code> was "
                    "skipped.</p>"
                )
            elif part is None:
                out.append(
                    "<p class='note'><code>find_regions</code> searched but no "
                    "split passed the heterogeneity-drop threshold — the effect "
                    "is heterogeneous, yet no candidate rule explains it.</p>"
                )
            else:
                out.append(
                    "<details open><summary>Partition tree</summary>"
                    f"<pre>{esc(self._partition_text(part))}</pre></details>"
                )
                root_h = part[0].heterogeneity
                if bound:
                    out.append("<div class='grid'>")
                    for leaf in part.leaves:
                        fig = part.plot(leaf.idx, show_plot=False)
                        drop = (
                            f" · −{(1 - leaf.heterogeneity / root_h) * 100:.0f}% "
                            "vs global"
                            if root_h
                            else ""
                        )
                        out.append("<figure>")
                        out.append(
                            _defer(fig, alt=part.label(leaf.idx), section=fr.feature)
                        )
                        out.append(
                            f"<figcaption>{esc(part.label(leaf.idx))} · "
                            f"heterogeneity {leaf.heterogeneity:.4f}{drop} · "
                            f"n={leaf.nof_instances:,}</figcaption></figure>"
                        )
                    out.append("</div>")
                else:
                    out.append(
                        "<p class='note'>Regional plots need the live effect — "
                        "this report was rebuilt from stored values; leaf "
                        "statistics below.</p>"
                    )
                    out.append(
                        "<table><tr><th>region</th><th>heterogeneity</th>"
                        "<th>drop vs global</th><th>n</th></tr>"
                    )
                    for leaf in part.leaves:
                        drop = (
                            f"−{(1 - leaf.heterogeneity / root_h) * 100:.0f}%"
                            if root_h
                            else "—"
                        )
                        out.append(
                            f"<tr><td>{esc(part.label(leaf.idx))}</td>"
                            f"<td>{leaf.heterogeneity:.4f}</td><td>{drop}</td>"
                            f"<td>{leaf.nof_instances:,}</td></tr>"
                        )
                    out.append("</table>")
            out.append("</section>")

        # -- 3 · triage after regions -------------------------------------------
        out.append("<section id='after'><h2>3 · Triage — after regions</h2>")
        if bound and parts:
            out.append(
                "<p class='caption'>The before/after picture: an arrow runs from "
                "each partitioned feature's global point to each of its leaves. "
                "Leaves of a good partition land right and down — more decisive, "
                "less heterogeneous.</p>"
            )
            out.append(
                self._img(
                    self._triage_fig(
                        partitions=parts, title="Feature triage — after find_regions"
                    ),
                    alt="triage after regions",
                )
            )
        else:
            if parts:
                out.append(
                    "<p class='note'>Before/after arrows need the live effect — "
                    "this report was rebuilt from stored values; the global "
                    "plane is repeated below.</p>"
                )
            else:
                out.append(
                    "<p class='note'>No feature was partitioned — nothing moved; "
                    "the plane is unchanged from the overview.</p>"
                )
            out.append(self._img(self._triage_fig(), alt="feature triage"))
        if ev and ev["gains"]:
            out.append(
                "<p class='caption'>What each partition buys — the gain in "
                "explained variance from applying that feature's subregions "
                "alone, on top of the global curves. Gains need not sum to "
                "the combined figure: partitions sharing interaction variance "
                "each recover part of the same pot, so the combined figure "
                "greedily applies only the splits that still improve it.</p>"
            )
            out.append(
                "<table><tr><th>split</th><th>regions</th>"
                "<th>explained-variance gain</th></tr>"
            )
            for g in ev["gains"]:
                out.append(
                    f"<tr><td>{esc(g['name'])} (on {esc(g['on'])})</td>"
                    f"<td>{g['n_regions']}</td>"
                    f"<td>{g['delta_r2'] * 100:+.1f} pts</td></tr>"
                )
            used = [g["name"] for g in ev["gains"] if g.get("in_combined")]
            label = (
                "subregions combined"
                if len(used) == len(ev["gains"])
                else f"subregions combined (using {esc(', '.join(used))})"
            )
            out.append(
                f"<tr><td><b>{label}</b></td><td>·</td>"
                f"<td><b>{ev['regional_r2']:.1%}</b> "
                f"({(ev['regional_r2'] - ev['gam_r2']) * 100:+.1f} pts vs "
                f"global's {ev['gam_r2']:.1%})</td></tr>"
            )
            out.append("</table>")
        out.append("</section>")

        out.append(
            "<footer>generated by <b>effector</b> · "
            "<code>effector.explain(...)</code> → <code>report.to_html()</code>"
            "</footer></main>"
        )
        out.append(_TAIL)
        self._harmonize_axes(deferred, share_y)
        return "".join(
            self._img(deferred[item[1]][0], alt=deferred[item[1]][1])
            if isinstance(item, tuple)
            else item
            for item in out
        )

    @staticmethod
    def _harmonize_axes(entries, share_y="across"):
        """Unify the drawn axis ranges of effect figures for at-a-glance
        comparison. Panels are matched by position, so a figure's n-th panel
        shares its y with every other figure's n-th panel — the effect curve
        with the effect curves, (RH)ALE's `dy/dx` bars with the `dy/dx` bars.
        `share_y="across"` unions over all entries; `"within"` only over a
        `section` (a feature — its global plot and its leaves). x is always
        shared per `section`; across features it would be meaningless.

        Post-hoc on the drawn axes, so one layer covers every method — ICE
        clouds, categorical bars, two-panel (RH)ALE — without touching their
        plot internals, and without the `dy_limits` kwarg that only the ALE
        family accepts.

        Args:
            entries: list of `(fig_ax, alt, section)` — `fig_ax` a figure or
                `(fig, ax)` tuple, `section` any hashable group key.
            share_y: `"across"` features (default) or `"within"` each.
        """
        if share_y not in ("across", "within"):
            raise ValueError("share_y must be 'across' or 'within'")

        def _unify(groups, get, set_):
            for axs in groups.values():
                lims = [get(a) for a in axs]
                lo, hi = min(l[0] for l in lims), max(l[1] for l in lims)
                for a in axs:
                    set_(a, lo, hi)

        ygroups, xgroups = {}, {}
        for fig_ax, _, section in entries:
            fig = fig_ax[0] if isinstance(fig_ax, tuple) else fig_ax
            key = section if share_y == "within" else None
            for panel, ax in enumerate(fig.axes):
                ygroups.setdefault((panel, key), []).append(ax)
            # (RH)ALE's panels are created with sharex, so axes[0] carries both
            xgroups.setdefault(section, []).append(fig.axes[0])

        _unify(ygroups, lambda a: a.get_ylim(), lambda a, lo, hi: a.set_ylim(lo, hi))
        _unify(xgroups, lambda a: a.get_xlim(), lambda a, lo, hi: a.set_xlim(lo, hi))

    def _global_fig(self, fr):
        """The feature's global-effect figure — the live `effect.plot` (exactly
        what the analyst runs, with the method's own heterogeneity view and
        categorical handling) when bound, the stored curves otherwise."""
        if self._effect is not None:
            return self._effect.plot(fr.feature, show_plot=False)
        return self._effect_fig(fr)

    def _effect_fig(self, fr):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4))
        band = np.sqrt(np.clip(fr.h, 0, None))
        ax.plot(fr.xs, fr.y, color="#4C78A8", label="mean effect")
        ax.fill_between(
            fr.xs,
            fr.y - band,
            fr.y + band,
            alpha=0.2,
            color="#4C78A8",
            label="± std",
        )
        ax.set_xlabel(fr.name)
        ax.set_ylabel(self.target_name)
        ax.legend()
        fig.tight_layout()
        return fig, ax

    @staticmethod
    def _partition_text(part):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            part.show()
        return buf.getvalue()

    @staticmethod
    def _img(fig_ax, alt=""):
        import matplotlib.pyplot as plt

        fig = fig_ax[0] if isinstance(fig_ax, tuple) else fig_ax
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return (
            f"<img class='zoomable' loading='lazy' "
            f"alt='{_htmlmod.escape(alt, quote=True)}' "
            f"src='data:image/png;base64,{b64}'/>"
        )

    # -- serialization ---------------------------------------------------------
    def to_dict(self):
        """Serialize to a plain JSON-able dict.

        !!! note "Values only"
            Curves, scores, and partition rules are serialized — never the
            model, the data, or the fitted effect.

        Returns:
            a dict that `from_dict` round-trips.
        """
        return {
            "method_name": self.method_name,
            "feature_names": list(self.feature_names),
            "target_name": self.target_name,
            "config": self.config,
            "overview": [dict(o) for o in self.overview],
            "features": [fr.to_dict() for fr in self.features],
            "explained_variance": self.explained_variance,
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild a `Report` from `to_dict()` output.

        The result is unbound: `show`, `plot_importance`, and `to_html` all
        work from the stored values; `to_html` skips only the per-leaf
        regional plots and the triage arrows, which need the live effect.

        Args:
            d: a dict produced by `to_dict()`.

        Returns:
            an unbound `Report`.
        """
        return cls(
            method_name=d["method_name"],
            feature_names=list(d["feature_names"]),
            target_name=d["target_name"],
            features=[FeatureReport.from_dict(x) for x in d["features"]],
            config=d.get("config", {}),
            overview=[dict(o) for o in d.get("overview", [])],
            explained_variance=d.get("explained_variance"),
        )


# page chrome for `to_html` — inline only, no external assets
_CSS = """
:root{--ink:#1a1a1a;--muted:#666;--line:#ddd;--bg:#fff;--card:#fafafa;--accent:#4C78A8}
*{box-sizing:border-box}
body{font-family:system-ui,-apple-system,'Segoe UI',Arial,sans-serif;margin:0;color:var(--ink);background:var(--bg);line-height:1.5}
main{max-width:1000px;margin:0 auto;padding:0 1.5rem 4rem}
nav{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.95);border-bottom:1px solid var(--line);padding:.6rem 1.5rem;display:flex;gap:1.1rem;flex-wrap:wrap;font-size:.9rem}
nav a{color:var(--accent);text-decoration:none;font-weight:500}
nav a:hover{text-decoration:underline}
h1{font-size:1.6rem;margin:1.5rem 0 .25rem}
h2{font-size:1.25rem;margin:0 0 1rem;padding-bottom:.4rem;border-bottom:2px solid var(--accent)}
h3{margin:1.5rem 0 .5rem;color:var(--muted);text-transform:uppercase;letter-spacing:.05em;font-size:.78rem}
section{margin:2.75rem 0;scroll-margin-top:3.5rem}
.chips{display:flex;gap:.5rem;flex-wrap:wrap;margin:.5rem 0 1rem}
.chip{background:var(--card);border:1px solid var(--line);border-radius:999px;padding:.15rem .7rem;font-size:.8rem;color:var(--muted)}
.chip b{color:var(--ink);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:.9rem;font-variant-numeric:tabular-nums;margin:1rem 0}
td,th{border-bottom:1px solid var(--line);padding:6px 10px;text-align:right}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.03em}
th:nth-child(2),td:nth-child(2),th:last-child,td:last-child{text-align:left}
tr[data-href]{cursor:pointer}
tr[data-href]:hover{background:var(--card)}
tr.dim{color:var(--muted)}
pre{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:1rem;overflow-x:auto;font-size:.8rem;line-height:1.45}
code{background:var(--card);border-radius:4px;padding:.05rem .3rem;font-size:.85em}
figure{margin:1rem 0;text-align:center}
figcaption{font-size:.85rem;color:var(--muted);margin-top:.4rem}
img{max-width:100%;height:auto}
img.zoomable{cursor:zoom-in;border:1px solid var(--line);border-radius:6px;background:#fff}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1rem;align-items:start}
.grid figure{margin:0}
.caption{color:var(--muted);font-size:.9rem}
.note{color:var(--muted);font-style:italic;font-size:.9rem}
details{margin:1rem 0}
summary{cursor:pointer;font-weight:600;color:var(--accent)}
footer{color:var(--muted);font-size:.8rem;border-top:1px solid var(--line);padding-top:1rem;margin-top:3rem}
#lightbox{display:none;position:fixed;inset:0;background:rgba(0,0,0,.85);z-index:100;cursor:zoom-out;align-items:center;justify-content:center;padding:2rem}
#lightbox.on{display:flex}
#lightbox img{max-width:95vw;max-height:95vh;background:#fff;border-radius:6px}
#top{display:none;position:fixed;right:1.25rem;bottom:1.25rem;z-index:50;border:1px solid var(--line);background:var(--bg);color:var(--accent);border-radius:999px;width:2.5rem;height:2.5rem;font-size:1.1rem;cursor:pointer}
#top.on{display:block}
"""

_TAIL = """
<div id='lightbox'><img alt=''/></div>
<button id='top' title='back to top'>&#8593;</button>
<script>
var lb=document.getElementById('lightbox'),lbi=lb.querySelector('img');
document.querySelectorAll('img.zoomable').forEach(function(im){
  im.addEventListener('click',function(){lbi.src=im.src;lb.classList.add('on');});
});
lb.addEventListener('click',function(){lb.classList.remove('on');});
document.addEventListener('keydown',function(e){if(e.key==='Escape')lb.classList.remove('on');});
document.querySelectorAll('tr[data-href]').forEach(function(r){
  r.addEventListener('click',function(){location.hash=r.dataset.href;});
});
var topBtn=document.getElementById('top');
window.addEventListener('scroll',function(){topBtn.classList.toggle('on',window.scrollY>600);});
topBtn.addEventListener('click',function(){window.scrollTo({top:0,behavior:'smooth'});});
</script></body></html>
"""


def explain(
    data,
    model,
    model_jac=None,
    *,
    schema=None,
    method="pdp",
    top_k=5,
    heter_threshold=None,
    finder="best",
    candidate_conditioning_features="all",
    nof_instances=10_000,
    random_state=21,
) -> Report:
    """Run the whole explanation pipeline and return a `Report` — a value.

    ```python
    report = effector.explain(X, model, method="pdp", top_k=5)
    report.show()                     # ranked table + partition trees
    report.to_html("report.html")     # self-contained page
    ```

    Fits the chosen `method` once, ranks features by importance, and for the
    top-`top_k` computes the mean-effect and heterogeneity curves; features
    heterogeneous enough (`heter_score >= heter_threshold`) also get a
    `find_regions` search.

    !!! note "One model touch"
        All model calls happen through the single `fit`, plus one prediction
        pass for the explained-variance denominator (`f̂(X)`, cached on the
        effect); everything after — importances, curves, `find_regions`, the
        surrogate R²s — is model-free, so the call count does not grow with
        `top_k`.

    Args:
        data: `(N, D)` numpy design matrix.
        model: callable `(N, D) -> (N,)` — the black box.
        model_jac: callable `(N, D) -> (N, D)` Jacobian; required by
            derivative-based methods (`"rhale"`, `"derpdp"`).
        schema: optional feature schema (names/types/categories).
        method: effect method — `"pdp"` (default), `"derpdp"`, `"ale"`,
            `"rhale"`, or `"shapdp"` (aliases accepted).
        top_k: how many top-importance features to report.
        heter_threshold: minimum `heter_score` to trigger `find_regions`;
            `None` (default) uses the median across the ranked features.
        finder: region finder — `"best"` (default), `"best_level_wise"`, or
            a configured finder instance.
        candidate_conditioning_features: features allowed to define splits
            (`"all"` or a list).
        nof_instances: subsample size the effect is built on.
        random_state: seed for the subsample.

    Returns:
        a `Report` bound to the fitted effect — `FeatureReport`s in
        importance-descending order, partitions stored as dicts, an
        `overview` (importance + heterogeneity) over every supported feature,
        and the `explained_variance` summary (surrogate R², per-split gains).
    """
    spec = method_registry.resolve(method)
    ctor_args = (model, model_jac) if spec.needs_jac else (model,)
    effect = spec.cls(
        data,
        *ctor_args,
        schema=schema,
        nof_instances=nof_instances,
        random_state=random_state,
    )

    # fit only the features this method can explain (single model touch)
    supported = [
        f
        for f in range(effect.dim)
        if effect.feature_types[f] in spec.supported_feature_types
    ]
    effect.fit(features=supported)

    imp = effect.importances()
    order = [
        int(f)
        for f in np.argsort(-np.nan_to_num(imp, nan=-np.inf))
        if int(f) in set(supported) and not np.isnan(imp[f])
    ]
    ranked = order[:top_k]

    # evaluate every reported surface through the model-free masked path (an
    # all-ones mask ≡ unmasked by M1); PDP/DerPDP are model-free only ON their
    # cache grid, so continuous features use that grid, discrete ones the levels.
    # Net effect: after importances() computed the local effects once, the whole
    # report is model-free — the count does not grow with top_k.
    mask_all = np.ones(effect.data.shape[0], dtype=bool)

    def _grid(f):
        if effect._is_cat(f):
            return np.unique(effect.data[:, f])
        return np.linspace(
            effect.axis_limits[0, f],
            effect.axis_limits[1, f],
            helpers.NOF_INTERNAL_POINTS,
        )

    # cheap scalars for the WHOLE plane (the triage view), not just the top-k
    hs_all = {f: float(effect.heter_score(f, mask=mask_all)) for f in order}
    if heter_threshold is None:
        hs_ranked = [hs_all[f] for f in ranked]
        thr = float(np.median(hs_ranked)) if hs_ranked else 0.0
    else:
        thr = heter_threshold

    features = []
    live_parts = {}  # feature -> bound multi-leaf Partition, ranked order
    for f in ranked:
        xs = _grid(f)
        y = effect.eval(f, xs, mask=mask_all)
        y = y[0] if isinstance(y, tuple) else y
        h = effect.eval_heter(f, xs, mask=mask_all)
        hs = hs_all[f]
        part_obj = (
            effect.find_regions(
                f,
                finder=finder,
                candidate_conditioning_features=candidate_conditioning_features,
            )
            if hs >= thr
            else None
        )
        if part_obj is not None and len(part_obj.leaves) > 1:
            live_parts[f] = part_obj
        features.append(
            FeatureReport(
                feature=f,
                name=effect.feature_names[f],
                importance=float(imp[f]),
                heter_score=hs,
                xs=xs,
                y=np.asarray(y),
                h=np.asarray(h),
                partition=part_obj.to_dict() if part_obj is not None else None,
            )
        )

    # the explained-variance summary: surrogates read off the caches above,
    # scored against f̂(X) — the pipeline's one extra model call (cached on
    # the effect, reused by plots)
    ev = _ev.summarize(effect, live_parts, features=supported)

    report = Report(
        method_name=method_registry.canonical(method),
        feature_names=list(effect.feature_names),
        target_name=effect.target_name,
        features=features,
        config={
            "method": method_registry.canonical(method),
            "top_k": top_k,
            "heter_threshold": thr,
            "finder": finder if isinstance(finder, str) else type(finder).__name__,
            "nof_instances": nof_instances,
            "random_state": random_state,
        },
        overview=[
            {
                "feature": f,
                "name": effect.feature_names[f],
                "importance": float(imp[f]),
                "heter_score": hs_all[f],
                "reported": f in set(ranked),
            }
            for f in order
        ],
        explained_variance=ev,
    )
    report._bind(effect)
    headline = report._ev_headline()
    if headline:
        print(f"[effector] {headline}")
    return report

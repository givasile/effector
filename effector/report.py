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
    report.show()                     # ledger + ranked table + partition trees
    report.to_html("report.html")     # self-contained page
    ```

    `features` holds one `FeatureReport` per plotted feature, final-CALM
    importance descending (split features at the instance-weighted mean of
    their subregions). `overview` holds the cheap global scalars —
    importance and heterogeneity — for **every** supported feature (superset
    of `features`), so the triage plane renders even on unbound reports.
    `explained_variance` holds the serialized CALM chain
    (`CalmSequence.to_dict()`): the flat decision-sequence keys (`gam_r2`,
    `regional_r2`, `min_gain`, `stages`, `skipped`) plus one serialized
    snapshot per accepted split under `calms` — plain floats, so it too
    renders unbound; `None` for derivative-scale methods. A report produced
    by `explain` is bound to its fitted effect, which `to_html` uses for the
    per-leaf regional plots; everything else works from the stored values
    alone.
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
        if ev["stages"]:
            line += f"; with subregions, {ev['regional_r2']:.1%}"
        return line

    def show(self):
        """Print the report in reading order: the explained-variance decision
        sequence first, then the ranked feature table (final-CALM values —
        split features at the instance-weighted mean of their subregions),
        then each accepted partition tree.

        Works on unbound reports (rebuilt via `from_dict`) too.
        """
        title = method_registry.resolve(self.method_name).display_name
        print(f"\n{title} report — target: {self.target_name}")
        print("=" * 60)
        ev = self.explained_variance
        accepted = {st["feature"] for st in ev["stages"]} if ev else None
        if self._ev_headline():
            print(f"explained variance: global effects (GAM) {ev['gam_r2']:.1%}")
            for st in ev["stages"]:
                heter = (
                    f", heter {st['heter_before']:.3f}→{st['heter_after']:.3f}"
                    if st["heter_before"] is not None
                    else ""
                )
                print(
                    f"  + split {st['name']} (on {st['on']}) → {st['cum_r2']:.1%}"
                    f" ({st['delta_r2'] * 100:+.1f} pts{heter})"
                )
            for sk in ev["skipped"]:
                why = (
                    "redundant (variance already explained)"
                    if sk["reason"] == "redundant"
                    else f"below the {ev['min_gain'] * 100:.1f}-pt threshold"
                )
                print(
                    f"  rejected: {sk['name']} (on {sk['on']}) — "
                    f"{sk['delta_r2'] * 100:+.1f} pts, {why}"
                )
            print("-" * 60)
        print(f"{'feature':<24}{'importance':>12}{'heter':>10}{'#regions':>10}")
        print("-" * 60)
        for fr in self.features:
            in_calm = (
                fr.partition is not None
                and len(fr.partition["regions"]) > 1
                and (accepted is None or fr.feature in accepted)
            )
            nregions = len(fr.partition["regions"]) if in_calm else 1
            print(
                f"{fr.name:<24}{fr.importance:>12.4f}{fr.heter_score:>10.4f}"
                f"{nregions:>10d}"
            )
        print("=" * 60)
        cov = self.config.get("coverage_achieved")
        if cov is not None:
            print(
                f"the plotted features carry {cov:.0%} of the total "
                "importance mass"
            )
        for fr in self.features:
            if (
                fr.partition is not None
                and len(fr.partition["regions"]) > 1
                and (accepted is None or fr.feature in accepted)
            ):
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

    def _triage_fig(self, title=None):
        """The triage plane from the stored scalars — global points for every
        supported feature, plus one arrow per accepted split from its global
        point to its weighted-mean point in the final CALM. Renders the same
        page bound or unbound (the live per-leaf view is
        `effector.plot_triage`)."""
        from effector.visualization import triage_scatter

        thr = self.config.get("heter_threshold")
        pts = [
            (o["name"], o["importance"], o["heter_score"]) for o in self.overview
        ]
        by_feature = {o["feature"]: o for o in self.overview}
        arrows = {}
        ev = self.explained_variance
        final = (ev or {}).get("calms", [])[-1] if (ev or {}).get("calms") else None
        if final is not None:
            for st in ev["stages"]:
                o = by_feature.get(st["feature"])
                if o is None:
                    continue
                arrows[o["name"]] = (
                    (o["importance"], o["heter_score"]),
                    (
                        float(final["importances"][str(st["feature"])]),
                        float(final["heter_scores"][str(st["feature"])]),
                    ),
                )
        return triage_scatter(
            pts,
            arrows=arrows,
            threshold=thr if thr is not None else False,
            unit=f" ({self.target_name} units)",
            title="Feature triage" if title is None else title,
            show_plot=False,
        )

    def _ev_ledger_fig(self):
        """The explained-variance ledger — one 0–100% bar of `Var(f̂)`: what
        reading the global plots buys (the GAM share), what each kept split
        adds (decision order), and what stays unexplained. The reading
        protocol in a single glance: which regional plots are worth the time.
        """
        from effector import theme
        import matplotlib.pyplot as plt

        t = theme.active()
        ev = self.explained_variance
        segs = [("global effects", ev["gam_r2"], t.CAT[0], f"{ev['gam_r2']:.0%}")]
        for i, st in enumerate(ev["stages"]):
            segs.append(
                (
                    f"+ {st['name']} regions",
                    st["delta_r2"],
                    t.CAT[(i + 1) % len(t.CAT)],
                    f"{st['delta_r2'] * 100:+.1f} pts",
                )
            )
        rest = max(1.0 - ev["regional_r2"], 0.0)

        def _seg_ink(hexcolor):
            r, g, b = (int(hexcolor[i : i + 2], 16) / 255 for i in (1, 3, 5))
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            return "#ffffff" if lum < 0.55 else "#1a1a1a"

        fig, ax = plt.subplots(figsize=(7.4, 1.6))
        surface = plt.rcParams.get("axes.facecolor", "#ffffff")
        ink = plt.rcParams.get("axes.labelcolor", "#333333")
        left, stagger = 0.0, 0
        for label, width, color, value in segs:
            if width <= 0:
                continue
            ax.barh(
                0, width, left=left, height=0.5, color=color,
                edgecolor=surface, linewidth=2,
            )
            if width >= 0.14:
                ax.text(
                    left + width / 2, 0, f"{label}\n{value}",
                    ha="center", va="center", fontsize=8,
                    color=_seg_ink(color), fontweight="bold",
                )
            else:
                # narrow segment: label above the bar, staggered to dodge
                # its neighbor
                y = 0.55 + 0.4 * (stagger % 2)
                stagger += 1
                ax.annotate(
                    f"{label} · {value}",
                    xy=(left + width / 2, 0.27),
                    xytext=(left + width / 2, y),
                    ha="center", va="bottom", fontsize=7, color=ink,
                    arrowprops={"arrowstyle": "-", "lw": 0.6, "color": ink},
                )
            left += width
        if rest > 0:
            ax.barh(
                0, rest, left=left, height=0.5,
                color=t.BAR_FACE_MUTED, edgecolor=surface, linewidth=2,
            )
            if rest >= 0.14:
                ax.text(
                    left + rest / 2, 0, f"unexplained\n{rest:.0%}",
                    ha="center", va="center", fontsize=8, color=ink,
                )
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.55, 1.45)
        ax.set_yticks([])
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_title(
            "Explained variance — what reading each layer buys", fontsize=9
        )
        for spine in ("left", "top", "right"):
            ax.spines[spine].set_visible(False)
        fig.tight_layout()
        return fig, ax

    # -- self-contained HTML page ---------------------------------------------
    def to_html(self, path=None, share_y="across"):
        """Render the report as one self-contained HTML page — the pipeline in reading order.

        The page reads top-down as the analysis: **the overview first** —
        the explained-variance ledger bar (the headline: what the global
        read buys, what each accepted split adds, what stays unexplained),
        the triage plane over all supported features with one arrow per
        accepted split (global point → weighted-mean point), the clickable
        ranked table with the achieved importance coverage, and the
        decision-sequence table. Then **the regional analysis — the final
        CALM**: one section per plotted feature in descending final-CALM
        importance; an accepted split renders as a group of per-leaf
        regional plots in rule order, everything else as its global curve
        (with a one-line pointer when a found split was rejected). Last,
        **the global baseline**: the split features' global curves alone —
        the counterfactual read without regions. Every figure is a base64
        PNG; navigation, click-to-zoom, and collapsing are inline vanilla
        JS/CSS — no external assets, one file, one click.

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
            values — including the ledger and the triage arrows; only the
            per-leaf regional plots need the live effect, and leaf
            statistics are shown in a table instead.

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

        # splits the decision sequence skipped (redundant / below min gain):
        # their §2 section keeps a one-line pointer instead of the regional
        # plots — the variance they would explain is already read elsewhere
        ev = self.explained_variance
        demoted = {s["feature"]: s for s in (ev["skipped"] if ev else [])}
        # the splits the sequence accepted — the final CALM's split features
        accepted = {st["feature"] for st in ev["stages"]} if ev else set()

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
        if ev and accepted:
            nav.append("<a href='#baseline'>Global baseline</a>")
        out.append("<nav>" + "".join(nav) + "</nav><main>")

        # header
        out.append(f"<h1>{esc(title)} report</h1>")
        out.append(
            f"<p class='caption'>target <b>{esc(self.target_name)}</b> · "
            f"{len(self.feature_names)} features · "
            f"{len(self.features)} plotted</p>"
        )
        chips = []
        for key in (
            "method",
            "top_k",
            "coverage",
            "heter_threshold",
            "min_r2_gain",
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
        if ev:
            sentence = (
                "An additive surrogate read off the global curves reproduces "
                f"<b>{ev['gam_r2']:.1%}</b> of the model's predicted variance"
            )
            if ev["stages"]:
                sentence += (
                    "; adding the regional plots kept by the decision "
                    f"sequence, <b>{ev['regional_r2']:.1%}</b>"
                )
            out.append(f"<p class='caption'>{sentence}.</p>")
            if ev["gam_r2"] > 0:
                out.append(
                    self._img(
                        self._ev_ledger_fig(), alt="explained-variance ledger"
                    )
                )
            else:
                out.append(
                    "<p class='note'>The ledger bar is omitted: the global "
                    "R² is not positive — correlated features can make "
                    "additive curves double-count shared variance, so the "
                    "global read does not reproduce the model here.</p>"
                )
        out.append(
            "<p class='caption'>Each point is a feature: importance (x) against "
            "heterogeneity (y). Bottom-left is ignorable; bottom-right is "
            "important and fully described by its mean effect; the top-right "
            "corner — important <i>and</i> heterogeneous — is where the mean "
            "hides something. An arrow marks each split the decision sequence "
            "accepted: from the feature's global point to its weighted-mean "
            "point across the subregions.</p>"
        )
        out.append(self._img(self._triage_fig(), alt="feature triage"))
        out.append(
            "<table><tr><th>#</th><th>feature</th><th>importance</th>"
            "<th>heterogeneity</th><th>#regions</th><th>regional analysis</th></tr>"
        )
        reported = {fr.feature: fr for fr in self.features}
        for rank, o in enumerate(self.overview, 1):
            fr = reported.get(o["feature"])
            if fr is None:
                imp_v = o.get("calm_importance", o["importance"])
                out.append(
                    f"<tr class='dim'><td>{rank}</td><td>{esc(o['name'])}</td>"
                    f"<td>{imp_v:.4f}</td><td>{o['heter_score']:.4f}</td>"
                    f"<td>·</td><td>not plotted (below the coverage cut)</td></tr>"
                )
                continue
            if fr.partition is None:
                nregions, note = 1, "below the search threshold"
            elif fr.feature not in parts:
                nregions, note = 1, "searched — no split passed"
            elif ev is None or fr.feature in accepted:
                nregions = len(fr.partition["regions"])
                note = f"split into {len(parts[fr.feature].leaves)} regions"
            else:
                nregions = len(fr.partition["regions"])
                note = "split found — rejected by the decision sequence"
            out.append(
                f"<tr data-href='feat-{fr.feature}'><td>{rank}</td>"
                f"<td>{esc(fr.name)}</td><td>{fr.importance:.4f}</td>"
                f"<td>{fr.heter_score:.4f}</td><td>{nregions}</td>"
                f"<td>{note} →</td></tr>"
            )
        out.append("</table>")
        cov = self.config.get("coverage_achieved")
        if cov is not None:
            out.append(
                f"<p class='caption'>The plotted features carry <b>{cov:.0%}</b> "
                "of the total importance mass "
                f"(target {self.config.get('coverage', 0):.0%}, "
                f"ceiling top_k = {self.config.get('top_k')}).</p>"
            )
        if ev and (ev["stages"] or ev["skipped"]):
            out.append(
                "<p class='caption'>The decision sequence. Starting from the "
                "global curves, each round applies the split with the largest "
                "explained-variance gain, measured <i>on top of the splits "
                "above it</i>, and stops when no remaining split adds at "
                f"least {ev['min_gain'] * 100:.1f} pts. A real split (its "
                "heterogeneity does drop) can still add nothing — or even "
                "hurt, by double-counting — when its variance is already "
                "explained by an earlier split.</p>"
            )
            out.append(
                "<table><tr><th>step</th><th>regions</th>"
                "<th>heterogeneity</th><th>explained variance</th></tr>"
            )
            out.append(
                "<tr><td>global effects (GAM)</td><td>·</td><td>·</td>"
                f"<td><b>{ev['gam_r2']:.1%}</b></td></tr>"
            )

            def _heter_cell(entry):
                if entry.get("heter_before") is None:
                    return "·"
                return (
                    f"{entry['heter_before']:.3f} → {entry['heter_after']:.3f}"
                )

            for st in ev["stages"]:
                out.append(
                    f"<tr><td>+ split {esc(st['name'])} "
                    f"(on {esc(st['on'])})</td>"
                    f"<td>{st['n_regions']}</td><td>{_heter_cell(st)}</td>"
                    f"<td>{st['delta_r2'] * 100:+.1f} pts → "
                    f"<b>{st['cum_r2']:.1%}</b></td></tr>"
                )
            for sk in ev["skipped"]:
                why = (
                    "redundant (variance already explained)"
                    if sk["reason"] == "redundant"
                    else f"below the {ev['min_gain'] * 100:.1f}-pt threshold"
                )
                out.append(
                    f"<tr class='dim'><td>rejected · {esc(sk['name'])} "
                    f"(on {esc(sk['on'])})</td>"
                    f"<td>{sk['n_regions']}</td><td>{_heter_cell(sk)}</td>"
                    f"<td>{sk['delta_r2'] * 100:+.1f} pts — {why}</td></tr>"
                )
            out.append("</table>")
        out.append(
            "<details><summary>Bar views — global importance and "
            "heterogeneity</summary>"
        )
        out.append(self._img(self.plot_importance(show_plot=False), alt="importance"))
        out.append(self._img(self._heterogeneity_fig(), alt="heterogeneity"))
        out.append("</details></section>")

        # -- 2 · regional analysis — the final CALM -------------------------------
        def _partition_block(fr, part):
            """Tree + per-leaf plots (bound) or leaf statistics (unbound)."""
            blk = [
                "<details open><summary>Partition tree</summary>"
                f"<pre>{esc(self._partition_text(part))}</pre></details>"
            ]
            root_h = part[0].heterogeneity
            if bound:
                blk.append("<div class='grid'>")
                for leaf in part.leaves:
                    fig = part.plot(leaf.idx, show_plot=False)
                    drop = (
                        f" · −{(1 - leaf.heterogeneity / root_h) * 100:.0f}% "
                        "vs global"
                        if root_h
                        else ""
                    )
                    blk.append("<figure>")
                    blk.append(
                        _defer(fig, alt=part.label(leaf.idx), section=fr.feature)
                    )
                    blk.append(
                        f"<figcaption>{esc(part.label(leaf.idx))} · "
                        f"heterogeneity {leaf.heterogeneity:.4f}{drop} · "
                        f"n={leaf.nof_instances:,}</figcaption></figure>"
                    )
                blk.append("</div>")
            else:
                blk.append(
                    "<p class='note'>Regional plots need the live effect — "
                    "this report was rebuilt from stored values; leaf "
                    "statistics below.</p>"
                )
                blk.append(
                    "<table><tr><th>region</th><th>heterogeneity</th>"
                    "<th>drop vs global</th><th>n</th></tr>"
                )
                for leaf in part.leaves:
                    drop = (
                        f"−{(1 - leaf.heterogeneity / root_h) * 100:.0f}%"
                        if root_h
                        else "—"
                    )
                    blk.append(
                        f"<tr><td>{esc(part.label(leaf.idx))}</td>"
                        f"<td>{leaf.heterogeneity:.4f}</td><td>{drop}</td>"
                        f"<td>{leaf.nof_instances:,}</td></tr>"
                    )
                blk.append("</table>")
            return blk

        out.append("<section><h2>2 · Regional analysis — the final CALM</h2>")
        if ev:
            out.append(
                "<p class='caption'>The selected snapshot: global effects "
                "everywhere except the accepted splits. Features in "
                "descending importance — a split feature enters as one group "
                "at the instance-weighted mean of its subregions. The split "
                "features' global counterparts are in the baseline section "
                "at the end.</p>"
            )
        out.append("</section>")
        for rank, fr in enumerate(self.features, 1):
            part = parts.get(fr.feature)
            in_calm = ev is not None and fr.feature in accepted and part is not None
            out.append(f"<section id='feat-{fr.feature}'>")
            out.append(f"<h2>2.{rank} · {esc(fr.name)}</h2>")
            nregions = len(part.leaves) if part and (in_calm or ev is None) else 1
            out.append(
                "<div class='chips'>"
                f"<span class='chip'>importance <b>{fr.importance:.4f}</b></span>"
                f"<span class='chip'>heterogeneity <b>{fr.heter_score:.4f}</b></span>"
                f"<span class='chip'>regions <b>{nregions}</b>"
                "</span></div>"
            )
            if in_calm:
                st = next(
                    (s for s in ev["stages"] if s["feature"] == fr.feature), None
                )
                if st is not None:
                    out.append(
                        f"<p class='caption'>Split on <b>{esc(st['on'])}</b> "
                        f"into {st['n_regions']} regions — worth "
                        f"<b>{st['delta_r2'] * 100:+.1f} pts</b> of explained "
                        "variance on top of the splits above it; importance "
                        "and heterogeneity here are the instance-weighted "
                        "means over the subregions. The global counterpart "
                        "is in the <a href='#baseline'>baseline</a>.</p>"
                    )
                out.extend(_partition_block(fr, part))
                out.append("</section>")
                continue
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
            elif fr.feature in demoted:
                sk = demoted[fr.feature]
                drop = (
                    f" (heterogeneity {sk['heter_before']:.3f} → "
                    f"{sk['heter_after']:.3f})"
                    if sk.get("heter_before") is not None
                    else ""
                )
                why = (
                    "adds no explained variance beyond the splits kept there — "
                    "the same variance is already read elsewhere"
                    if sk["reason"] == "redundant"
                    else f"adds only {sk['delta_r2'] * 100:+.1f} pts, below the "
                    f"{ev['min_gain'] * 100:.1f}-pt threshold"
                )
                out.append(
                    f"<p class='note'>A split on <b>{esc(sk['on'])}</b> into "
                    f"{sk['n_regions']} regions was found{drop}, but the "
                    f"decision sequence skips it: it {why}. The regional "
                    "plots are omitted; reproduce them with "
                    "<code>find_regions</code>.</p>"
                )
            else:
                # no chain to arbitrate (derivative-scale method): render the
                # found partition feature-major, as the search proposed it
                out.extend(_partition_block(fr, part))
            out.append("</section>")

        # -- 3 · global baseline — the counterfactual ---------------------------
        if ev and accepted:
            out.append(
                "<section id='baseline'>"
                "<h2>3 · Global baseline — without regions</h2>"
            )
            out.append(
                "<p class='caption'>What you would believe about the split "
                "features without the regional analysis: their global mean "
                "effects, with the heterogeneity the accepted splits just "
                "explained still hiding inside the band. Compare with their "
                "subregions in the regional analysis above.</p>"
            )
            by_feature = {o["feature"]: o for o in self.overview}
            for fr in self.features:
                if fr.feature not in accepted:
                    continue
                out.append(f"<h3>{esc(fr.name)}</h3>")
                o = by_feature.get(fr.feature)
                if o is not None:
                    out.append(
                        "<div class='chips'>"
                        "<span class='chip'>global importance "
                        f"<b>{o['importance']:.4f}</b></span>"
                        "<span class='chip'>global heterogeneity "
                        f"<b>{o['heter_score']:.4f}</b></span></div>"
                    )
                out.append(
                    _defer(
                        self._global_fig(fr),
                        alt=f"{fr.name} global effect (baseline)",
                        section=fr.feature,
                    )
                )
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
            "schema_version": 1,
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
    coverage=0.8,
    heter_threshold=None,
    min_r2_gain=0.01,
    finder="best",
    candidate_conditioning_features="all",
    nof_instances=10_000,
    random_state=21,
) -> Report:
    """Run the whole explanation pipeline and return a `Report` — a value.

    ```python
    report = effector.explain(X, model, method="pdp")
    report.show()                     # ledger + ranked table + partition trees
    report.to_html("report.html")     # self-contained page
    ```

    Fits the chosen `method` once, searches regions on every heterogeneous
    feature (`heter_score >= heter_threshold`), greedily selects which splits
    earn their complexity (`select_regions` — the CALM chain), and plots the
    features that carry the analysis: descending final-CALM importance until
    `coverage` of the total importance mass is shown, never more than
    `top_k`, plus every accepted split feature.

    !!! note "Search wide, display by coverage"
        The region search and the R² selection run over all heterogeneous
        features — a feature outside the display cut can still carry the
        biggest explained-variance gain. `coverage`/`top_k` only trim the
        curve plots; the triage plane and the ranked table always cover every
        supported feature.

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
        top_k: hard ceiling on how many features get curve plots.
        coverage: stop plotting once the shown features carry this share of
            the total importance mass (final-CALM importances, share-of-sum
            over the supported features; default 0.8). The report states the
            achieved share.
        heter_threshold: minimum `heter_score` to enter the region search;
            `None` (default) uses the median across the supported features —
            the same convention as `find_regions(features="heterogeneous")`.
        min_r2_gain: smallest explained-variance marginal (fraction of
            `Var(f̂)`, default 0.01 = 1 pt) a split must add — on top of the
            splits already applied — to earn a snapshot in the CALM chain
            and its regional plots; splits below it are skipped as redundant
            or below-threshold.
        finder: region finder — `"best"` (default), `"best_level_wise"`, or
            a configured finder instance.
        candidate_conditioning_features: features allowed to define splits
            (`"all"` or a list).
        nof_instances: subsample size the effect is built on.
        random_state: seed for the subsample.

    Returns:
        a `Report` bound to the fitted effect — `FeatureReport`s in
        final-CALM-importance-descending order, partitions stored as dicts,
        an `overview` (global importance + heterogeneity) over every supported
        feature, and `explained_variance` holding the serialized CALM chain
        (`CalmSequence.to_dict()`: GAM R², per-stage marginal gains, skipped
        splits, one snapshot per accepted split).
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
    return _explain_effect(
        effect,
        top_k=top_k,
        coverage=coverage,
        heter_threshold=heter_threshold,
        min_r2_gain=min_r2_gain,
        finder=finder,
        candidate_conditioning_features=candidate_conditioning_features,
    )


def _explain_effect(
    effect,
    *,
    top_k=5,
    coverage=0.8,
    heter_threshold=None,
    min_r2_gain=0.01,
    finder="best",
    candidate_conditioning_features="all",
) -> Report:
    """The pipeline shared by `effector.explain` and `effect.explain()`.

    Warms the local-effect caches for the supported features — respecting
    any fit config already declared on the engine (missing features are
    computed with the defaults) — then ranks, searches wide, selects the
    CALM chain, and packages the `Report`.
    """
    method = method_registry.name_of(type(effect))
    supported = []
    for f in range(effect.dim):
        try:
            effect._check_feature_type_supported(f)
        except ValueError:
            continue
        supported.append(f)
        effect._ensure_local(f)  # the single model touch, per feature

    imp = effect.importances()
    order = [
        int(f)
        for f in np.argsort(-np.nan_to_num(imp, nan=-np.inf))
        if int(f) in set(supported) and not np.isnan(imp[f])
    ]

    # evaluate every reported surface through the model-free masked path (an
    # all-ones mask ≡ unmasked by M1); PDP/DerPDP are model-free only ON their
    # cache grid, so continuous features use that grid, discrete ones the levels.
    # Net effect: after importances() computed the local effects once, the whole
    # report is model-free — the count does not grow with top_k.
    mask_all = np.ones(effect.data.shape[0], dtype=bool)

    # cheap scalars for the WHOLE plane (the triage view)
    hs_all = {f: float(effect.heter_score(f, mask=mask_all)) for f in order}
    if heter_threshold is None:
        # the find_regions(features="heterogeneous") convention: the median
        thr = float(np.median(list(hs_all.values()))) if hs_all else 0.0
    else:
        thr = heter_threshold

    # search wide: every heterogeneous feature proposes a candidate partition
    searched = {
        f: effect.find_regions(
            f,
            finder=finder,
            candidate_conditioning_features=candidate_conditioning_features,
        )
        for f in order
        if hs_all[f] >= thr
    }
    live_parts = {f: p for f, p in searched.items() if len(p.leaves) > 1}

    # greedy R2 selection -> the CALM chain; model-free beyond the one
    # prediction pass select_regions makes for the variance denominator
    chain = None
    if _ev.supports(effect):
        try:
            chain = effect.select_regions(
                partitions=live_parts, min_r2_gain=min_r2_gain
            )
        except ValueError:  # Var == 0: constant model, nothing to explain
            chain = None

    # rank by the FINAL snapshot: weighted-mean importance for split features
    final_imp = chain.final.importances() if chain is not None else imp
    final_het = chain.final.heter_scores() if chain is not None else None
    calm_imp = {f: float(final_imp[f]) for f in order}
    order.sort(key=lambda f: -calm_imp[f])
    accepted = set(chain.final.features) if chain is not None else set()

    # display cut: descending importance until `coverage` of the mass is
    # shown, never more than top_k -- plus every accepted split feature
    total_mass = sum(calm_imp.values())
    shown = []
    cum = 0.0
    for f in order:
        if len(shown) >= top_k:
            break
        shown.append(f)
        cum += calm_imp[f]
        if total_mass > 0 and cum / total_mass >= coverage:
            break
    displayed = sorted(set(shown) | accepted, key=lambda f: -calm_imp[f])
    achieved = (
        sum(calm_imp[f] for f in displayed) / total_mass if total_mass > 0 else 1.0
    )

    features = []
    for f in displayed:
        xs = effect.grid(f)
        y = effect.eval(f, xs, mask=mask_all)
        y = y[0] if isinstance(y, tuple) else y
        h = effect.eval_heter(f, xs, mask=mask_all)
        part_obj = searched.get(f)
        features.append(
            FeatureReport(
                feature=f,
                name=effect.feature_names[f],
                importance=calm_imp[f],
                heter_score=(
                    float(final_het[f]) if final_het is not None else hs_all[f]
                ),
                xs=xs,
                y=np.asarray(y),
                h=np.asarray(h),
                partition=part_obj.to_dict() if part_obj is not None else None,
            )
        )

    # the serialized CALM chain; None = no output-scale surrogate (DerPDP)
    ev = chain.to_dict() if chain is not None else None

    report = Report(
        method_name=method_registry.canonical(method),
        feature_names=list(effect.feature_names),
        target_name=effect.target_name,
        features=features,
        config={
            "method": method_registry.canonical(method),
            "top_k": top_k,
            "coverage": coverage,
            "coverage_achieved": float(achieved),
            "heter_threshold": thr,
            "min_r2_gain": min_r2_gain,
            "finder": finder if isinstance(finder, str) else type(finder).__name__,
            "nof_instances": effect.nof_instances,
            "random_state": effect.random_state,
        },
        overview=[
            {
                "feature": f,
                "name": effect.feature_names[f],
                "importance": float(imp[f]),
                "heter_score": hs_all[f],
                "calm_importance": calm_imp[f],
                "reported": f in set(displayed),
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

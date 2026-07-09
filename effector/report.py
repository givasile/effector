"""One-click explanation report (design goal (b)).

`effector.explain(data, model, ...)` runs the whole pipeline — fit → rank by
importance (R13) → mean-effect curves for the top-k → `find_regions` on the
heterogeneous ones (R12) — and returns a `Report`, a serializable **value**
(R12: values, not state). Every model call happens through the single
`effect.fit(...)`; importance, heter_score, curves, and find_regions are all
model-free afterwards.

`Report` binds a reference to its producing effect only for the lazy re-plot
sugar (mirrors `Partition._bind`); `to_dict()`/`from_dict()` are the
serialization boundary and round-trip without an effect.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from effector import helpers, method_registry
from effector.partition import Partition


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
    descending. A report produced by `explain` is bound to its fitted effect,
    which `to_html` uses for per-leaf regional plots; everything else works
    from the stored values alone.
    """

    method_name: str
    feature_names: list
    target_name: str
    features: List[FeatureReport]
    config: dict = field(default_factory=dict)

    def __post_init__(self):
        self._effect = None

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
    def show(self):
        """Print the ranked feature table, then each multi-region partition tree.

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
        for fr in self.features:
            if fr.partition is not None and len(fr.partition["regions"]) > 1:
                Partition.from_dict(fr.partition).show()

    # -- importance bar chart (R7 return rule) ---------------------------------
    def plot_importance(self, show_plot=True):
        """Horizontal bar chart of feature importance, most important on top.

        Args:
            show_plot: `True` (default) displays the figure and returns
                `None`; `False` returns `(fig, ax)` instead.

        Returns:
            `None`, or `(fig, ax)` when `show_plot=False`.
        """
        import matplotlib.pyplot as plt

        names = [fr.name for fr in self.features]
        vals = [fr.importance for fr in self.features]
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(names) + 1.5))
        ax.barh(range(len(names)), vals, color="#4C78A8")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()  # most important on top
        ax.set_xlabel("importance")
        ax.set_title(
            f"{method_registry.resolve(self.method_name).display_name} — "
            f"feature importance"
        )
        fig.tight_layout()
        if show_plot:
            plt.show(block=False)
            return None
        return fig, ax

    # -- self-contained HTML page ---------------------------------------------
    def to_html(self, path=None):
        """Render a self-contained HTML page — every figure inlined as a base64 PNG.

        The page holds the importance chart, per-feature mean-effect curves
        with ± std bands (from the stored arrays), the partition tables and
        trees, and — when the report is bound to its effect — one regional
        plot per partition leaf. No external assets.

        !!! note "Unbound reports"
            A report rebuilt with `from_dict` renders everything from stored
            values; only the per-leaf regional plots are skipped (they need
            the live effect).

        Args:
            path: optional file path to also write the page to.

        Returns:
            the HTML string.
        """
        title = method_registry.resolve(self.method_name).display_name
        parts = [
            "<!doctype html><html><head><meta charset='utf-8'>",
            f"<title>{title} report</title>",
            "<style>body{font-family:system-ui,Arial,sans-serif;margin:2rem;"
            "max-width:900px}h1,h2{color:#222}table{border-collapse:collapse}"
            "td,th{border:1px solid #ccc;padding:4px 10px;text-align:right}"
            "th:first-child,td:first-child{text-align:left}"
            "pre{background:#f6f6f6;padding:1rem;overflow-x:auto}"
            "img{max-width:100%}</style></head><body>",
            f"<h1>{title} report</h1>",
            f"<p><b>target:</b> {self.target_name} &nbsp; "
            f"<b>features:</b> {len(self.feature_names)}</p>",
            "<h2>Feature importance</h2>",
            self._img(self.plot_importance(show_plot=False)),
            "<table><tr><th>feature</th><th>importance</th><th>heterogeneity</th>"
            "<th>#regions</th></tr>",
        ]
        for fr in self.features:
            nregions = len(fr.partition["regions"]) if fr.partition else 1
            parts.append(
                f"<tr><td>{fr.name}</td><td>{fr.importance:.4f}</td>"
                f"<td>{fr.heter_score:.4f}</td><td>{nregions}</td></tr>"
            )
        parts.append("</table>")

        for fr in self.features:
            parts.append(f"<h2>{fr.name}</h2>")
            parts.append(
                f"<p>importance {fr.importance:.4f} &nbsp; "
                f"heterogeneity {fr.heter_score:.4f}</p>"
            )
            parts.append(self._img(self._effect_fig(fr)))
            if fr.partition is not None and len(fr.partition["regions"]) > 1:
                part = Partition.from_dict(fr.partition)
                if self._effect is not None:
                    part = part.bind(self._effect)
                parts.append(f"<pre>{self._partition_text(part)}</pre>")
                if self._effect is not None:
                    for r in part.leaves:
                        fig = self._effect.plot(
                            fr.feature, mask=part.mask(r.idx), show_plot=False
                        )
                        parts.append(self._img(fig))
        parts.append("</body></html>")
        html = "".join(parts)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(html)
        return html

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
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            part.show()
        return buf.getvalue()

    @staticmethod
    def _img(fig_ax):
        import matplotlib.pyplot as plt

        fig = fig_ax[0] if isinstance(fig_ax, tuple) else fig_ax
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
        plt.close(fig)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"<img src='data:image/png;base64,{b64}'/>"

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
            "features": [fr.to_dict() for fr in self.features],
        }

    @classmethod
    def from_dict(cls, d):
        """Rebuild a `Report` from `to_dict()` output.

        The result is unbound: `show`, `plot_importance`, and `to_html` all
        work from the stored values; `to_html` skips only the per-leaf
        regional plots, which need the live effect.

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
        )


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
        All model calls happen through the single `fit`; everything after —
        importances, curves, `find_regions` — is model-free, so the call
        count does not grow with `top_k`.

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
        importance-descending order, partitions stored as dicts.
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
    ranked = [
        int(f)
        for f in np.argsort(-np.nan_to_num(imp, nan=-np.inf))
        if not np.isnan(imp[f])
    ][:top_k]

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

    hs_map = {f: float(effect.heter_score(f, mask=mask_all)) for f in ranked}
    if heter_threshold is None:
        thr = float(np.median(list(hs_map.values()))) if hs_map else 0.0
    else:
        thr = heter_threshold

    features = []
    for f in ranked:
        xs = _grid(f)
        y = effect.eval(f, xs, mask=mask_all)
        y = y[0] if isinstance(y, tuple) else y
        h = effect.eval_heter(f, xs, mask=mask_all)
        hs = hs_map[f]
        part = (
            effect.find_regions(
                f,
                finder=finder,
                candidate_conditioning_features=candidate_conditioning_features,
            ).to_dict()
            if hs >= thr
            else None
        )
        features.append(
            FeatureReport(
                feature=f,
                name=effect.feature_names[f],
                importance=float(imp[f]),
                heter_score=hs,
                xs=xs,
                y=np.asarray(y),
                h=np.asarray(h),
                partition=part,
            )
        )

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
    )
    return report._bind(effect)

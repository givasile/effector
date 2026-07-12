"""Tests for the one-click report: effector.explain -> Report.

Pins: importance-ranked output (top feature is the known-important one), the
pipeline is one fit's worth of model calls (stable as top_k grows), to_dict /
from_dict round-trip without an effect, to_html is self-contained (base64, no
external refs) and works unbound, and find_regions fires on heterogeneous
features.
"""

import numpy as np
import pytest

import effector
from effector.report import Report
from tests.conftest import (
    CountingModel,
    gated_model,
    linear_model,
    make_global_data,
    make_regional_data,
)


def test_explain_ranks_known_important_feature_first():
    data = make_global_data(n=1500)
    rep = effector.explain(data, linear_model, method="pdp", nof_instances="all")
    # linear model COEF=[2,-3,0.5] -> importance ~ |coef|, so feature 1 is top
    assert rep.features[0].feature == 1
    imps = [fr.importance for fr in rep.features]
    assert imps == sorted(imps, reverse=True)  # descending


def test_explain_is_one_fit_worth_of_model_calls():
    data = make_global_data(n=800)
    m1 = CountingModel(linear_model)
    effector.explain(data, m1, method="pdp", top_k=1, nof_instances="all")
    m3 = CountingModel(linear_model)
    effector.explain(data, m3, method="pdp", top_k=3, nof_instances="all")
    # top_k only adds model-free work (importance/eval/find_regions), so the
    # model-call count must not grow with it
    assert m1.n_calls == m3.n_calls


def test_report_roundtrips_without_effect():
    data = make_global_data(n=800)
    rep = effector.explain(data, linear_model, method="pdp", nof_instances="all")
    rep2 = Report.from_dict(rep.to_dict())
    assert [fr.name for fr in rep2.features] == [fr.name for fr in rep.features]
    assert rep2.features[0].feature == rep.features[0].feature
    np.testing.assert_allclose(rep2.features[0].xs, rep.features[0].xs)
    # text/serialized surfaces work with no bound effect
    rep2.show()
    rep2.plot_importance(show_plot=False)
    # live plotting requires an effect
    with pytest.raises(RuntimeError):
        rep2._require_effect()


def test_to_html_is_self_contained(tmp_path):
    data = make_global_data(n=800)
    rep = effector.explain(data, linear_model, method="pdp", nof_instances="all")
    out = tmp_path / "report.html"
    # with a path the file is the deliverable — return None, so interactive
    # shells don't echo a megabyte of markup
    assert rep.to_html(out) is None
    html = out.read_text()
    assert "data:image/png;base64" in html  # inlined figures
    assert "http://" not in html and "https://" not in html  # no external assets
    for name in rep.feature_names:
        assert name in html


def test_to_html_works_unbound(tmp_path):
    data = make_global_data(n=800)
    rep = effector.explain(data, linear_model, method="pdp", nof_instances="all")
    rep2 = Report.from_dict(rep.to_dict())  # unbound
    html = rep2.to_html()  # draws curves from stored xs/y/h, skips region plots
    assert "data:image/png;base64" in html
    # the full page structure renders from stored values alone
    assert "id='overview'" in html and "Regional analysis" in html


def test_explain_overview_covers_all_supported_features():
    data = make_global_data(n=800)
    rep = effector.explain(
        data, linear_model, method="pdp", top_k=1, nof_instances="all"
    )
    # top_k truncates `features`, never the overview
    assert len(rep.features) == 1
    assert len(rep.overview) == data.shape[1]
    assert sum(o["reported"] for o in rep.overview) == 1
    imps = [o["importance"] for o in rep.overview]
    assert imps == sorted(imps, reverse=True)  # importance-descending
    # round-trips through the serialization boundary
    rep2 = Report.from_dict(rep.to_dict())
    assert rep2.overview == rep.overview


def test_to_html_reads_like_the_pipeline():
    data = make_regional_data(n=800)
    rep = effector.explain(data, gated_model, method="pdp", nof_instances="all")
    html = rep.to_html()
    # section order mirrors the reading order: ledger-first overview ->
    # regional analysis (the final CALM) -> global baseline (counterfactual)
    assert (
        html.index("id='overview'")
        < html.index("id='feat-")
        < html.index("id='baseline'")
    )
    for fr in rep.features:
        assert f"id='feat-{fr.feature}'" in html
    # triage + ledger plus at least one figure per feature, all zoomable
    assert html.count("class='zoomable'") >= len(rep.features) + 2
    # inline chrome only — nav, lightbox, script — no external assets
    assert "<nav>" in html and "lightbox" in html and "<script>" in html
    assert "http://" not in html and "https://" not in html


def test_to_html_demotes_rejected_splits():
    # a split the decision sequence rejected keeps its section but trades the
    # partition tree + regional plots for a one-line pointer
    data = make_regional_data(n=800)
    rep = effector.explain(
        data, gated_model, method="pdp", nof_instances="all", coverage=1.0
    )
    ev = rep.explained_variance
    if not ev["skipped"]:
        # force one deterministically (greedy semantics are pinned elsewhere)
        st = ev["stages"].pop()
        ev["regional_r2"] -= st["delta_r2"]
        st.pop("cum_r2")
        ev["skipped"].append({**st, "reason": "redundant"})
    html = rep.to_html()
    feat = ev["skipped"][0]["feature"]
    section = html.split(f"id='feat-{feat}'")[1].split("</section>")[0]
    assert "skips it" in section  # the demotion note
    assert "Partition tree" not in section
    assert "rejected by the decision sequence" in html  # the table marker
    # kept stages still get the full regional treatment
    for st in ev["stages"]:
        kept = html.split(f"id='feat-{st['feature']}'")[1].split("</section>")[0]
        assert "Partition tree" in kept


def test_harmonize_axes_shares_y_globally_and_x_per_section():
    import matplotlib.pyplot as plt

    def fig_with(xlim, ylim):
        fig, ax = plt.subplots()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        return fig, ax

    a = fig_with((0, 1), (-1, 1))  # section "f0"
    b = fig_with((0.2, 0.8), (-5, 0.5))  # section "f0" (a leaf, windowed x)
    c = fig_with((10, 20), (0, 3))  # section "f1"
    Report._harmonize_axes([(a, "", "f0"), (b, "", "f0"), (c, "", "f1")])
    # y is shared across the WHOLE report
    for fig, _ in (a, b, c):
        assert fig.axes[0].get_ylim() == (-5.0, 3.0)
    # x is shared only within a section
    assert a[0].axes[0].get_xlim() == (0.0, 1.0)
    assert b[0].axes[0].get_xlim() == (0.0, 1.0)
    assert c[0].axes[0].get_xlim() == (10.0, 20.0)
    for fig, _ in (a, b, c):
        plt.close(fig)


def _two_panel(xlim, ylim, dylim):
    """A two-axes (RH)ALE-shaped figure: effect panel over a dy/dx panel."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_xlim(*xlim)
    ax1.set_ylim(*ylim)
    ax2.set_ylim(*dylim)
    return fig, (ax1, ax2)


def test_harmonize_axes_shares_the_dy_panel_of_two_panel_figures():
    import matplotlib.pyplot as plt

    a = _two_panel((0, 1), (-1, 1), (-2, 2))  # section "f0"
    b = _two_panel((0.2, 0.8), (-5, 0.5), (-0.5, 8))  # section "f0" (a leaf)
    c = _two_panel((10, 20), (0, 3), (0, 1))  # section "f1"
    entries = [(a, "", "f0"), (b, "", "f0"), (c, "", "f1")]
    Report._harmonize_axes(entries)
    for fig, _ in (a, b, c):
        assert fig.axes[0].get_ylim() == (-5.0, 3.0)  # effect panel, as before
        assert fig.axes[1].get_ylim() == (-2.0, 8.0)  # dy/dx panel, now too
    for fig, _ in (a, b, c):
        plt.close(fig)


def test_harmonize_axes_share_y_within_scopes_every_panel_to_its_section():
    import matplotlib.pyplot as plt

    a = _two_panel((0, 1), (-1, 1), (-2, 2))  # section "f0"
    b = _two_panel((0.2, 0.8), (-5, 0.5), (-0.5, 8))  # section "f0" (a leaf)
    c = _two_panel((10, 20), (0, 3), (0, 1))  # section "f1"
    entries = [(a, "", "f0"), (b, "", "f0"), (c, "", "f1")]
    Report._harmonize_axes(entries, share_y="within")
    # each panel shares a range inside a section...
    for fig, _ in (a, b):
        assert fig.axes[0].get_ylim() == (-5.0, 1.0)
        assert fig.axes[1].get_ylim() == (-2.0, 8.0)
    # ...and keeps its own between sections
    assert c[0].axes[0].get_ylim() == (0.0, 3.0)
    assert c[0].axes[1].get_ylim() == (0.0, 1.0)
    # x is per-section either way
    assert a[0].axes[0].get_xlim() == b[0].axes[0].get_xlim() == (0.0, 1.0)
    assert c[0].axes[0].get_xlim() == (10.0, 20.0)
    for fig, _ in (a, b, c):
        plt.close(fig)


def test_harmonize_axes_rejects_an_unknown_share_y():
    with pytest.raises(ValueError, match="share_y"):
        Report._harmonize_axes([], share_y="everywhere")


def test_to_html_pops_up_no_figures_in_an_interactive_session(monkeypatch):
    """`show_plot=False` is not enough: under `plt.ion()` pyplot paints a figure
    the moment it is created. to_html must build its figures off-screen, leak
    none, and leave the caller's interactive mode as it found it."""
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backend_bases import FigureCanvasBase

    painted = []
    orig = FigureCanvasBase.draw_idle
    monkeypatch.setattr(
        FigureCanvasBase,
        "draw_idle",
        lambda self, *a, **k: (painted.append(id(self.figure)), orig(self, *a, **k))[1],
    )
    plt.ion()
    try:
        data = make_global_data(n=200)
        rep = effector.explain(
            data, gated_model, method="pdp", top_k=2, nof_instances="all"
        )
        painted.clear()
        rep.to_html()
        assert not painted, f"{len(set(painted))} figures were drawn on screen"
        assert not plt.get_fignums(), "to_html leaked open figures"
        assert matplotlib.is_interactive(), "interactive mode was not restored"

        # ...while a direct plot() in the same session still shows, as ever
        painted.clear()
        rep._effect.plot(0)
        assert painted
    finally:
        plt.ioff()
        plt.close("all")


@pytest.mark.parametrize("method", ["pdp", "derpdp", "ale", "rhale", "shapdp"])
def test_report_figures_use_the_methods_default_heterogeneity_view(method):
    """Report figures are the plain `effect.plot(feature)` — no override — so
    each family draws its own view: ICE for pdp-based, dy/dx bars for the
    (RH)ALE family, the SHAP scatter for SHAP-DP."""
    import matplotlib.pyplot as plt

    data = make_global_data(n=200)
    kw = {"nof_instances": 100} if method == "shapdp" else {"nof_instances": "all"}
    rep = effector.explain(data, linear_model, method=method, top_k=1, **kw)
    fig, _ = rep._global_fig(rep.features[0])
    effect_ax = fig.axes[0]

    labels = {ln.get_label() for ln in effect_ax.lines}
    if method in ("pdp", "derpdp"):
        # the ICE cloud: one line per instance, far more than the mean curve
        assert len(effect_ax.lines) > 10
        assert ("ICE" if method == "pdp" else "d-ICE") in labels
    elif method == "shapdp":
        assert "SHAP values" in labels
    else:
        # (RH)ALE: mean effect alone on top, dy/dx bars (± std) below
        assert len(fig.axes) == 2
        assert not effect_ax.collections, "no band on the (RH)ALE effect panel"
        bars = fig.axes[1].containers
        assert any(getattr(c, "has_yerr", False) for c in bars), (
            "the dy/dx bars carry the std whiskers"
        )
    plt.close(fig)


def test_explain_finds_regions_on_heterogeneous_feature():
    # gated model: features 0 and 1 have high heterogeneity -> find_regions fires
    data = make_regional_data(n=800)
    rep = effector.explain(data, gated_model, method="pdp", nof_instances="all")
    assert any(fr.partition is not None for fr in rep.features)
    # a fired partition is a real tree (root + children)
    fired = [fr for fr in rep.features if fr.partition is not None]
    assert any(len(fr.partition["regions"]) > 1 for fr in fired)
    # stored partitions use the v2 (rule-based, mask-free) schema
    for fr in fired:
        assert fr.partition["schema_version"] == 2
        for region in fr.partition["regions"]:
            assert "rule" in region and "mask" not in region


def test_display_cut_stops_at_coverage_and_respects_ceiling():
    data = make_global_data(n=800)
    # COEF=[2,-3,0.5]: the top-2 features carry ~91% of the importance mass,
    # so the default coverage=0.8 stops the plots after two of them
    rep = effector.explain(data, linear_model, method="pdp", nof_instances="all")
    assert len(rep.features) == 2
    assert rep.config["coverage_achieved"] >= 0.8
    # coverage=1.0 lifts the cut to every supported feature (up to top_k)...
    rep_all = effector.explain(
        data, linear_model, method="pdp", nof_instances="all", coverage=1.0
    )
    assert len(rep_all.features) == data.shape[1]
    # ...and the top_k ceiling binds regardless of coverage
    rep_one = effector.explain(
        data, linear_model, method="pdp", nof_instances="all", top_k=1, coverage=1.0
    )
    assert len(rep_one.features) == 1


def test_accepted_split_features_are_always_plotted():
    # a coverage cut that keeps only the top feature must still display every
    # split the decision sequence accepted — the ledger references them
    data = make_regional_data(n=800)
    rep = effector.explain(
        data,
        gated_model,
        method="pdp",
        nof_instances="all",
        coverage=0.01,
        top_k=1,
    )
    ev = rep.explained_variance
    displayed = {fr.feature for fr in rep.features}
    for st in ev["stages"]:
        assert st["feature"] in displayed


def test_report_importances_are_final_calm_values():
    # the ranked table speaks the final snapshot's language: a split feature's
    # importance is the instance-weighted mean over its subregions
    data = make_regional_data(n=800)
    rep = effector.explain(data, gated_model, method="pdp", nof_instances="all")
    final = rep.explained_variance["calms"][-1]
    for fr in rep.features:
        assert fr.importance == pytest.approx(
            float(final["importances"][str(fr.feature)])
        )
        assert fr.heter_score == pytest.approx(
            float(final["heter_scores"][str(fr.feature)])
        )


@pytest.mark.parametrize("method", ["pdp", "ale", "rhale", "shapdp"])
def test_explain_across_methods(method):
    data = make_global_data(n=600)
    kw = {"nof_instances": "all", "top_k": 2}
    if method == "shapdp":
        kw["nof_instances"] = 100  # keep shap cheap
    rep = effector.explain(data, linear_model, method=method, **kw)
    assert isinstance(rep, Report)
    assert len(rep.features) == 2
    assert rep.features[0].importance >= rep.features[1].importance

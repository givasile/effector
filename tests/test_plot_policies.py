"""Pins for the plot-redesign policies: tick thinning, legend discipline,
title/tag placement, nominal sorting, ordinal-only connect line, and the
theme-scoped report rendering."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import effector


@pytest.fixture
def cat_effect():
    rng = np.random.default_rng(7)
    N = 600
    X = np.column_stack(
        [
            rng.uniform(-1, 1, N),
            rng.integers(0, 3, N).astype(float),  # nominal, 3 levels
            rng.integers(0, 24, N).astype(float),  # ordinal, 24 levels
        ]
    )

    def model(x):
        lv = np.array([0.5, -0.9, 0.2])  # non-monotone in level order
        return x[:, 0] + lv[x[:, 1].astype(int)] + 0.05 * x[:, 2]

    schema = effector.Schema(
        feature_names=["cont", "nom", "ordi"],
        feature_types=["continuous", "nominal", "ordinal"],
        category_names=[None, ["b", "g", "r"], None],
        target_name="y",
    )
    return effector.PDP(X, model, schema=schema)


def test_ordinal_ticks_thin_to_eight_majors_with_minors(cat_effect):
    fig, ax = cat_effect.plot(2, show_plot=False)
    majors = [t for t in ax.get_xticklabels() if t.get_text()]
    assert len(majors) <= 8
    # every level keeps a tick: majors + (deduped) minors cover all 24
    assert len(ax.get_xticks()) + len(ax.get_xticks(minor=True)) == 24
    # raw integer-ish labels, never 2-decimal values
    assert all("." not in t.get_text() for t in majors)
    plt.close(fig)


def test_nominal_levels_sort_by_effect_value(cat_effect):
    fig, ax = cat_effect.plot(1, show_plot=False)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    # effect values: b=+0.5, g=-0.9, r=+0.2 -> ascending g, r, b
    assert labels == ["g", "r", "b"]
    plt.close(fig)


def test_avg_output_line_present_but_not_in_legend(cat_effect):
    fig, ax = cat_effect.plot(
        0, heterogeneity="ice", show_avg_output=True, show_plot=False
    )
    assert [ln for ln in ax.get_lines() if ln.get_label() == "avg output"]
    legend_texts = {t.get_text() for t in ax.get_legend().get_texts()}
    assert "avg output" not in legend_texts
    assert {"PDP", "ICE"} <= legend_texts
    plt.close(fig)


def test_single_series_axis_has_no_legend(cat_effect):
    fig, ax = cat_effect.plot(0, heterogeneity=False, show_plot=False)
    assert ax.get_legend() is None
    plt.close(fig)


def test_title_is_feature_and_tag_is_method_scope(cat_effect):
    fig, ax = cat_effect.plot(0, show_plot=False)
    assert ax.get_title(loc="left") == "cont"
    texts = {t.get_text() for t in ax.texts}
    assert "PDP · global" in texts
    plt.close(fig)


def test_ale_connect_line_only_for_ordinal():
    rng = np.random.default_rng(3)
    N = 500
    X = np.column_stack(
        [
            rng.integers(0, 3, N).astype(float),
            rng.integers(0, 5, N).astype(float),
        ]
    )
    model = lambda x: (x[:, 0] == 1) * 1.0 + 0.3 * x[:, 1]
    schema = effector.Schema(feature_types=["nominal", "ordinal"], target_name="y")
    ale = effector.ALE(X, model, schema=schema)
    fig, ax = ale.plot(1, show_plot=False)  # ordinal: accumulation path shown
    assert [ln for ln in ax.get_lines() if ln.get_label() == "accumulated (steps)"]
    plt.close(fig)
    fig, ax = ale.plot(0, show_plot=False)  # nominal: no line between ranks
    assert not [ln for ln in ax.get_lines() if ln.get_label() == "accumulated (steps)"]
    plt.close(fig)


def test_ale_dy_panel_labels():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (400, 2))
    model = lambda x: x[:, 0] ** 2 + x[:, 1]
    ale = effector.ALE(X, model)
    fig, (ax1, ax2) = ale.plot(0, heterogeneity=True, show_plot=False)
    assert ax2.get_ylabel() == "Δy/Δx"
    texts = {t.get_text() for t in ax2.get_legend().get_texts()}
    assert "bin slope ± heterogeneity" in texts and "dy_dx" not in texts
    # the mean line is solid now
    mean = [ln for ln in ax1.get_lines() if ln.get_label() == "average effect"][0]
    assert mean.get_linestyle() == "-"
    plt.close(fig)


def test_to_html_renders_in_active_theme_without_leaking(cat_effect):
    before = dict(matplotlib.rcParams)
    effector.set_theme("default")  # ambient chrome = stock matplotlib
    try:
        report = cat_effect.explain(top_k=2)
        html = report.to_html()
        assert "data:image/png;base64" in html
        # rendering under the house theme must not mutate ambient rcParams
        assert matplotlib.rcParams["figure.facecolor"] == "white"
    finally:
        matplotlib.rcParams.update(before)


def test_report_summary_round_trips(cat_effect):
    y = cat_effect.model(cat_effect.data) + 0.1
    report = cat_effect.explain(y=y, top_k=2)
    s = report.summary
    assert s["n_instances"] == cat_effect.data.shape[0]
    assert s["score_kind"] == "R²" and 0.0 < s["score"] <= 1.0
    rebuilt = effector.Report.from_dict(report.to_dict())
    assert rebuilt.summary == s

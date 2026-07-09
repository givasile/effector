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
    rep.to_html(out)
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
    assert "id='overview'" in html and "id='after'" in html


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
    # section order mirrors the analyst pipeline:
    # overview triage -> per-feature (global + regional) -> closing triage
    assert (
        html.index("id='overview'")
        < html.index("id='feat-")
        < html.index("id='after'")
    )
    for fr in rep.features:
        assert f"id='feat-{fr.feature}'" in html
    # opening + closing triage plus one global figure per feature, all zoomable
    assert html.count("class='zoomable'") >= len(rep.features) + 2
    # inline chrome only — nav, lightbox, script — no external assets
    assert "<nav>" in html and "lightbox" in html and "<script>" in html
    assert "http://" not in html and "https://" not in html


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

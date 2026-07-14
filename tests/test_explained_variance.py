"""Explained variance (the variance ledger's report-only surface).

Analytic ground truth under feature independence (verified against
closed-form Sobol decompositions): PDP `importance_j² = V_j`, and the GAM
surrogate's R² is the first-order share `ΣV_j / Var(f)`. A split converts
interaction variance into region-conditional first-order variance, so on a
switch model the regional surrogate reaches R² = 1. Offsets are fitted
jointly on region-indicator dummies (single partition ≡ per-leaf means); the
combined figure greedily selects partitions, since overlapping partitions
double-count interaction deviations in the curves themselves.

Budget: `explain` = one fit + exactly one prediction pass (`f̂(X)`); the
surrogates themselves are model-free (summary-payload reads only).
"""

import numpy as np
import pytest

import effector
from effector import explained_variance as ev
from effector.report import Report
from tests.conftest import (
    CountingModel,
    analytic_shap_values,
    budget,
    gated_model,
    linear_model,
    make_global_data,
    make_regional_data,
)

# f = a*x1 + b*x2 + c*x1*x2 + d*x3, x ~ U(-1,1) iid: V_j = coef²/3, V_12 = c²/9
A, B, C, D = 1.0, 0.5, 2.0, 0.8
V1, V2, V3, V12 = A**2 / 3, B**2 / 3, D**2 / 3, C**2 / 9


def interaction_model(x):
    return A * x[:, 0] + B * x[:, 1] + C * x[:, 0] * x[:, 1] + D * x[:, 2]


def switch_model(x):
    # f = x1 * (3 if x3 > 0 else 1) + x2 — all interaction variance is
    # convertible by the x3-split (W1 = 4/3, W2 = 1/3, W13 = 1/3)
    return x[:, 0] * np.where(x[:, 2] > 0, 3.0, 1.0) + x[:, 1]


def make_uniform(n=6000, d=3, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (n, d))


def fitted_pdp(data, model):
    m = effector.PDP(data, model, nof_instances="all")
    m.fit(features="all")
    return m


# ---------------------------------------------------------------------------
# the surrogate against closed-form Sobol shares
# ---------------------------------------------------------------------------


def test_gam_r2_matches_the_analytic_first_order_share():
    data = make_uniform()
    m = fitted_pdp(data, interaction_model)
    r2 = ev.surrogate_r2(m, interaction_model(data), {}, [0, 1, 2])
    np.testing.assert_allclose(r2, (V1 + V2 + V3) / (V1 + V2 + V3 + V12), atol=0.02)


def test_switch_model_split_converts_interaction_to_first_order():
    data = make_uniform()
    m = fitted_pdp(data, switch_model)
    fx = switch_model(data)
    part = effector.Partition.from_rules(
        [f"{m.feature_names[2]} > 0", f"{m.feature_names[2]} <= 0"],
        effect=m,
        feature=0,
    )
    gam = ev.surrogate_r2(m, fx, {}, [0, 1, 2])
    regional = ev.surrogate_r2(m, fx, {0: part}, [0, 1, 2])
    np.testing.assert_allclose(gam, 5 / 6, atol=0.02)  # (W1+W2)/(W1+W2+W13)
    assert regional >= 0.995  # the split leaves no interaction behind


def test_two_overlapping_partitions_joint_offsets_and_greedy_combined():
    # f = x0*s + x1*s + 2s with s = sign(x2): both partitions split on the
    # SAME rule -> duplicate indicator columns (rank-deficient design) and a
    # shared mean shift (2s) that naive per-region means would count twice.
    def f(x):
        s = np.sign(x[:, 2])
        return x[:, 0] * s + x[:, 1] * s + 2 * s

    data = make_uniform()
    m = fitted_pdp(data, f)
    fx = f(data)
    rules = [f"{m.feature_names[2]} > 0", f"{m.feature_names[2]} <= 0"]
    p0 = effector.Partition.from_rules(rules, effect=m, feature=0)
    p1 = effector.Partition.from_rules(rules, effect=m, feature=1)

    # 0.99, not ~1: x2's global curve is the step 2·sign(x2), which the
    # 30-point payload interpolation smooths around 0 (the one documented
    # approximation of the payload read)
    both = ev.surrogate_r2(m, fx, {0: p0, 1: p1}, [0, 1, 2])
    assert both >= 0.99  # lstsq min-norm handles the rank deficiency

    # a single partition conditions one interaction term, and its leaf-mean
    # offsets absorb the 2s shift: only x1*s (Var = 1/3) is left over
    total = 2 / 3 + 4  # Var(f) = 1/3 + 1/3 + 4
    single = ev.surrogate_r2(m, fx, {0: p0}, [0, 1, 2])
    np.testing.assert_allclose(single, 1 - (1 / 3) / total, atol=0.02)

    # summarize: the decision sequence applies both (the second still adds
    # more than min_gain)
    s = ev.summarize(m, {0: p0, 1: p1}, [0, 1, 2])
    assert s["regional_r2"] >= 0.99
    assert len(s["stages"]) == 2 and not s["skipped"]
    # the first stage's marginal IS its solo gain (nothing applied before it)
    np.testing.assert_allclose(
        s["stages"][0]["delta_r2"], s["stages"][0]["solo_delta_r2"], atol=1e-12
    )
    # sequential marginals sum exactly to the headline shift
    np.testing.assert_allclose(
        sum(st["delta_r2"] for st in s["stages"]),
        s["regional_r2"] - s["gam_r2"],
        atol=1e-12,
    )


def test_greedy_combined_drops_a_redundant_partition():
    # gated model: either partition alone reaches ~1.0; applying both blindly
    # double-counts the interaction in the curves and scores WORSE — greedy
    # selection must keep one and drop the other
    data = make_regional_data(n=800)
    m = fitted_pdp(data, gated_model)
    p0 = m.find_regions(0)
    p2 = m.find_regions(2)
    s = ev.summarize(m, {0: p0, 2: p2}, [0, 1, 2])
    assert len(s["stages"]) == 1 and len(s["skipped"]) == 1
    assert s["skipped"][0]["reason"] == "redundant"
    # the running figure is exactly baseline + the kept stage's marginal
    np.testing.assert_allclose(
        s["regional_r2"], s["gam_r2"] + s["stages"][0]["delta_r2"]
    )
    # the redundant split still records a real solo gain — that contrast
    # (solo > 0, marginal ~ 0) is the two-claimants-one-pot signature
    assert s["skipped"][0]["solo_delta_r2"] > 0.1
    assert s["skipped"][0]["delta_r2"] <= 1e-9


def test_min_gain_threshold_gates_the_stages():
    # the switch split is worth ~1/6 of Var(f): a stage at the default
    # threshold, skipped as below_threshold when min_gain exceeds it
    data = make_uniform()
    m = fitted_pdp(data, switch_model)
    part = effector.Partition.from_rules(
        [f"{m.feature_names[2]} > 0", f"{m.feature_names[2]} <= 0"],
        effect=m,
        feature=0,
    )
    s = ev.summarize(m, {0: part}, [0, 1, 2])
    assert len(s["stages"]) == 1 and not s["skipped"]

    s_hi = ev.summarize(m, {0: part}, [0, 1, 2], min_gain=0.5)
    assert not s_hi["stages"]
    assert s_hi["skipped"][0]["reason"] == "below_threshold"
    assert s_hi["regional_r2"] == s_hi["gam_r2"]
    # the static heterogeneity pair rides along even for hand-built rules —
    # and the switch split genuinely simplifies the regional curves
    sk = s_hi["skipped"][0]
    assert sk["heter_after"] < sk["heter_before"]


def test_degenerate_feature_inside_a_region_contributes_zero():
    # x2 is binary; conditioning on x2 == 0 makes it constant inside the
    # region -> the summarizer raises ValueError -> contribution 0, R² finite
    data = make_regional_data(n=800)
    m = fitted_pdp(data, gated_model)
    fx = gated_model(data)
    part = effector.Partition.from_rules(
        [f"{m.feature_names[2]} == 0", f"{m.feature_names[2]} == 1"],
        effect=m,
        feature=2,
    )
    r2 = ev.surrogate_r2(m, fx, {2: part}, [0, 1, 2])
    assert np.isfinite(r2)


@pytest.mark.parametrize("name", ["pdp", "ale", "rhale", "shapdp"])
def test_gam_r2_is_near_one_for_an_additive_model(name):
    data = make_global_data(n=1000)
    kwargs = {"nof_instances": "all"}
    if name == "pdp":
        m = effector.PDP(data, linear_model, **kwargs)
    elif name == "ale":
        m = effector.ALE(data, linear_model, **kwargs)
    elif name == "rhale":
        from tests.conftest import linear_model_jac

        m = effector.RHALE(data, linear_model, model_jac=linear_model_jac, **kwargs)
    else:
        m = effector.ShapDP(
            data, linear_model, shap_values=analytic_shap_values(data), **kwargs
        )
    m.fit(features="all")
    r2 = ev.surrogate_r2(m, linear_model(data), {}, [0, 1, 2])
    assert r2 >= 0.98


# ---------------------------------------------------------------------------
# the report surface
# ---------------------------------------------------------------------------


def test_explain_stores_and_prints_explained_variance(capsys):
    data = make_regional_data(n=800)
    rep = effector.explain(data, gated_model, method="pdp", nof_instances="all")
    assert "of the model's variance" in capsys.readouterr().out
    s = rep.explained_variance
    assert s is not None and s["stages"]
    assert s["regional_r2"] > s["gam_r2"]
    assert set(s["stages"][0]) == {
        "feature",
        "name",
        "on",
        "n_regions",
        "delta_r2",
        "cum_r2",
        "solo_delta_r2",
        "heter_before",
        "heter_after",
    }


def test_explained_variance_roundtrips_and_renders_unbound():
    data = make_regional_data(n=800)
    rep = effector.explain(data, gated_model, method="pdp", nof_instances="all")
    rep2 = Report.from_dict(rep.to_dict())
    assert rep2.explained_variance == rep.explained_variance
    rep2.show()  # runs unbound
    html = rep2.to_html()
    assert "predicted variance" in html
    # the decision-sequence ledger renders from stored values alone
    assert "decision" in html and "global effects (GAM)" in html
    assert "explained-variance ledger" in html  # the 0-100% bar figure


def test_explain_budget_is_one_fit_plus_one_prediction_pass():
    data = make_regional_data(n=800)
    counting = CountingModel(gated_model)
    rep = effector.explain(data, counting, method="pdp", nof_instances="all")
    total = counting.n_calls

    # a manual replica of explain's model-touching work: fit + f̂(X)
    replica = CountingModel(gated_model)
    m = effector.PDP(data, replica, nof_instances="all")
    m.fit(features="all")
    m.importances()  # warms PDP's lazy ICE table — the single fit touch
    replica(data)  # the f̂(X) pass
    assert total == replica.n_calls

    # re-running the whole summary on the fitted effect costs zero calls
    effect = rep._effect
    parts = {
        fr.feature: effector.partition.Partition.from_dict(fr.partition).bind(effect)
        for fr in rep.features
        if fr.partition is not None and len(fr.partition["regions"]) > 1
    }
    with budget(counting, expected=0):
        again = ev.summarize(effect, parts, [0, 1, 2])
    np.testing.assert_allclose(again["gam_r2"], rep.explained_variance["gam_r2"])


def test_derpdp_report_skips_the_section(capsys):
    from tests.conftest import linear_model_jac

    data = make_global_data(n=400)
    rep = effector.explain(
        data,
        linear_model,
        model_jac=linear_model_jac,
        method="derpdp",
        nof_instances="all",
    )
    assert rep.explained_variance is None
    assert "of the model's variance" not in capsys.readouterr().out
    rep.show()
    html = rep.to_html()
    assert "predicted variance" not in html


def test_column_vector_model_output_does_not_corrupt_r2():
    # a raw keras-style forward returns (N,1); without raveling, `fx - g`
    # broadcasts to (N,N) and yields a garbage (typically negative) R²
    data = make_regional_data(n=800)
    m = fitted_pdp(data, gated_model)
    fx = gated_model(data)
    expected = ev.surrogate_r2(m, fx, {}, [0, 1, 2])
    np.testing.assert_allclose(ev.surrogate_r2(m, fx[:, None], {}, [0, 1, 2]), expected)
    m._y_pred = fx[:, None]  # e.g. stamped by a plot's _avg_output
    s = ev.summarize(m, {}, [0, 1, 2])
    np.testing.assert_allclose(s["gam_r2"], expected)


def test_shapdp_importance_is_the_inherited_base_flavor():
    # the mean(|φ|) override is gone: identity pin + the shared closed form
    assert "_importance" not in effector.ShapDP.__dict__
    data = make_global_data(n=2000)
    m = effector.ShapDP(
        data,
        linear_model,
        shap_values=analytic_shap_values(data),
        nof_instances="all",
    )
    m.fit(features="all")
    from tests.conftest import COEF

    for f in range(3):
        expected = abs(COEF[f]) * np.std(data[:, f])
        np.testing.assert_allclose(m.importance(f), expected, rtol=5e-2)

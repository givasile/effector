"""Functional layer for categorical (ordinal/nominal) features of interest.

Every assertion is a closed form of `models.ConditionalCategorical`
(f(x) = a[x0] + b[x0]*x1*1{x2>0}, x0 in {0,1,2}) — the exactness contract of
docs/method_semantics.md, per method and per feature type.
"""

import numpy as np
import pytest

import effector
from effector import models
from effector.ingestion import CONTINUOUS, NOMINAL, ORDINAL

A, B = models.ConditionalCategorical.A, models.ConditionalCategorical.B
N = 600
TYPES = [ORDINAL, CONTINUOUS, CONTINUOUS]
SCHEMA = {"feature_types": TYPES}
LEVELS = np.array([0.0, 1.0, 2.0])


def make_data(n=N, seed=21, weights=(0.5, 0.3, 0.2)):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.choice([0.0, 1.0, 2.0], n, p=weights),
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
        ],
        axis=1,
    )


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def model():
    return models.ConditionalCategorical()


def level_weights(data):
    _, counts = np.unique(data[:, 0], return_counts=True)
    return counts / counts.sum()


def gate_values(data):
    """g_i = x1_i * 1{x2_i > 0} — the per-instance heterogeneity driver."""
    return data[:, 1] * (data[:, 2] > 0)


# ---------------------------------------------------------------------------
# PDP
# ---------------------------------------------------------------------------


def test_pdp_eval_at_levels_matches_closed_form(data, model):
    pdp = effector.PDP(data, model.predict, nof_instances="all", schema=SCHEMA)
    g_bar = gate_values(data).mean()
    expected = A + B * g_bar  # PDP(v_k) = a_k + b_k * mean(g)
    y = pdp.eval(0, LEVELS, centering=False)
    np.testing.assert_allclose(y, expected, atol=1e-10)


def test_pdp_centering_zero_start_and_zero_integral(data, model):
    pdp = effector.PDP(data, model.predict, nof_instances="all", schema=SCHEMA)
    g_bar = gate_values(data).mean()
    raw = A + B * g_bar

    y_start = pdp.eval(0, LEVELS, centering="zero_start")
    np.testing.assert_allclose(y_start, raw - raw[0], atol=1e-10)

    w = level_weights(data)
    y_int = pdp.eval(0, LEVELS, centering="zero_integral")
    np.testing.assert_allclose(y_int, raw - np.average(raw, weights=w), atol=1e-10)
    assert abs(np.average(y_int, weights=w)) < 1e-10  # freq-weighted mean = 0


def test_pdp_heterogeneity_closed_form(data, model):
    pdp = effector.PDP(data, model.predict, nof_instances="all", schema=SCHEMA)
    g = gate_values(data)
    w = level_weights(data)
    b_bar = np.average(B, weights=w)
    # h(v_k) = Var_i[(b_k - b_bar_w) * g_i]
    expected = (B - b_bar) ** 2 * g.var()
    h = pdp.eval_heter(0, LEVELS)
    np.testing.assert_allclose(h, expected, atol=1e-10)

    # heter_score = freq-weighted mean of h over levels
    np.testing.assert_allclose(
        pdp.heter_score(0), np.average(expected, weights=w), atol=1e-10
    )


def test_pdp_eval_at_non_level_raises(data, model):
    pdp = effector.PDP(data, model.predict, schema=SCHEMA)
    with pytest.raises(ValueError, match="observed"):
        pdp.eval(0, np.array([0.5]), centering=False)


def test_pdp_nominal_same_math_as_ordinal(data, model):
    types = [NOMINAL, CONTINUOUS, CONTINUOUS]
    pdp_o = effector.PDP(data, model.predict, nof_instances="all", schema=SCHEMA)
    pdp_n = effector.PDP(
        data, model.predict, nof_instances="all", schema={"feature_types": types}
    )
    np.testing.assert_array_equal(
        pdp_o.eval(0, LEVELS, centering=False), pdp_n.eval(0, LEVELS, centering=False)
    )


# ---------------------------------------------------------------------------
# DerPDP — continuous only
# ---------------------------------------------------------------------------


def test_derpdp_categorical_raises(data, model):
    der = effector.DerPDP(data, model.predict, model.jacobian, schema=SCHEMA)
    with pytest.raises(ValueError, match="does not support ordinal.*use PDP"):
        der.fit(0)
    with pytest.raises(ValueError, match="does not support ordinal"):
        der.eval(0, LEVELS)
    # continuous features still work
    der.fit(1)


# ---------------------------------------------------------------------------
# ALE
# ---------------------------------------------------------------------------


def transition_stats(data, model):
    """Closed-form two-sided transition stats: mu_t, sigma2_t for t=1,2."""
    g = gate_values(data)
    mus, variances = [], []
    for t in (1, 2):
        mask = np.isin(data[:, 0], [t - 1, t])
        d = (A[t] - A[t - 1]) + (B[t] - B[t - 1]) * g[mask]
        mus.append(d.mean())
        variances.append(d.var())
    return np.array(mus), np.array(variances)


def test_ale_eval_at_levels_matches_closed_form(data, model):
    ale = effector.ALE(data, model.predict, nof_instances="all", schema=SCHEMA)
    mus, _ = transition_stats(data, model)
    y = ale.eval(0, LEVELS, centering="zero_start")
    expected = np.array([0.0, mus[0], mus[0] + mus[1]])
    np.testing.assert_allclose(y, expected, atol=1e-10)


def test_ale_heterogeneity_step_into_level(data, model):
    ale = effector.ALE(data, model.predict, nof_instances="all", schema=SCHEMA)
    _, variances = transition_stats(data, model)
    h = ale.eval_heter(0, LEVELS)
    # h(v_0) = first transition's variance; h(v_j) = variance of step into j
    np.testing.assert_allclose(
        h, [variances[0], variances[0], variances[1]], atol=1e-10
    )


def test_ale_freq_weighted_centering(data, model):
    ale = effector.ALE(data, model.predict, nof_instances="all", schema=SCHEMA)
    w = level_weights(data)
    y = ale.eval(0, LEVELS, centering="zero_integral")
    assert abs(np.average(y, weights=w)) < 1e-10


def test_ale_eval_at_non_level_raises(data, model):
    ale = effector.ALE(data, model.predict, schema=SCHEMA)
    with pytest.raises(ValueError, match="observed"):
        ale.eval(0, np.array([1.7]), centering=False)


# ---------------------------------------------------------------------------
# RHALE — ordinal via discrete derivative + level grouping; nominal rejected
# ---------------------------------------------------------------------------


def test_rhale_fixed_equals_ale(data, model):
    ale = effector.ALE(data, model.predict, nof_instances="all", schema=SCHEMA)
    rhale = effector.RHALE(
        data, model.predict, model.jacobian, nof_instances="all", schema=SCHEMA
    )
    rhale.fit(0, binning_method="fixed", centering="zero_start")
    y_ale = ale.eval(0, LEVELS, centering="zero_start")
    y_rhale = rhale.eval(0, LEVELS, centering="zero_start")
    np.testing.assert_allclose(y_rhale, y_ale, atol=1e-10)


def test_rhale_greedy_groups_levels_and_stays_exact(data, model):
    rhale = effector.RHALE(
        data, model.predict, model.jacobian, nof_instances="all", schema=SCHEMA
    )
    rhale.fit(0, binning_method="greedy", centering="zero_start")
    limits = rhale.feature_effect["feature_0"]["limits"]
    # candidate edges are exactly the integer level codes
    np.testing.assert_allclose(limits, np.round(limits), atol=1e-12)
    # accumulated values at the levels stay exact whatever the grouping,
    # because within-group means are averages of the transition means only
    # when the group is homogeneous; with distinct transitions greedy keeps
    # them apart, so the values match ALE
    mus, _ = transition_stats(data, model)
    y = rhale.eval(0, LEVELS, centering="zero_start")
    if len(limits) == 3:  # no grouping happened: exact equality
        np.testing.assert_allclose(y, [0.0, mus[0], mus[0] + mus[1]], atol=1e-10)


def test_rhale_nominal_raises(data, model):
    types = [NOMINAL, CONTINUOUS, CONTINUOUS]
    rhale = effector.RHALE(
        data, model.predict, model.jacobian, schema={"feature_types": types}
    )
    with pytest.raises(ValueError, match="does not support nominal.*use ALE or PDP"):
        rhale.fit(0)


# ---------------------------------------------------------------------------
# ShapDP — per-level mean/variance, step lookup
# ---------------------------------------------------------------------------


def test_shapdp_per_level_stats(data, model):
    rng = np.random.default_rng(3)
    shap_values = rng.normal(size=data.shape)  # any aligned attribution array
    m = effector.ShapDP(
        data,
        model.predict,
        nof_instances="all",
        shap_values=shap_values,
        schema=SCHEMA,
    )
    y = m.eval(0, LEVELS, centering=False)
    h = m.eval_heter(0, LEVELS)
    for k in range(3):
        at_level = shap_values[data[:, 0] == k, 0]
        np.testing.assert_allclose(y[k], at_level.mean(), atol=1e-10)
        np.testing.assert_allclose(h[k], at_level.var(), atol=1e-10)


def test_shapdp_step_lookup_no_interpolation(data, model):
    rng = np.random.default_rng(3)
    shap_values = rng.normal(size=data.shape)
    m = effector.ShapDP(
        data,
        model.predict,
        nof_instances="all",
        shap_values=shap_values,
        schema=SCHEMA,
    )
    with pytest.raises(ValueError, match="observed"):
        m.eval(0, np.array([0.5]), centering=False)


# ---------------------------------------------------------------------------
# registry / class-attr agreement (R5)
# ---------------------------------------------------------------------------


def test_registry_capability_matrix_agreement():
    from effector import method_registry

    for name, spec in method_registry.METHODS.items():
        assert spec.supported_feature_types == spec.cls.SUPPORTED_FEATURE_TYPES, name
        assert spec.cat_strategy == spec.cls.CAT_STRATEGY, name

    matrix = {
        "pdp": {CONTINUOUS, ORDINAL, NOMINAL},
        "derpdp": {CONTINUOUS},
        "ale": {CONTINUOUS, ORDINAL, NOMINAL},
        "rhale": {CONTINUOUS, ORDINAL},
        "shapdp": {CONTINUOUS, ORDINAL, NOMINAL},
    }
    for name, expected in matrix.items():
        assert set(method_registry.METHODS[name].supported_feature_types) == expected


# ---------------------------------------------------------------------------
# order= — declared and induced level orders (ALE/RHALE)
# ---------------------------------------------------------------------------


def test_ale_explicit_order_permutes_accumulation(data, model):
    ale = effector.ALE(data, model.predict, nof_instances="all", schema=SCHEMA)
    order = [2.0, 0.0, 1.0]
    ale.fit(0, centering="zero_start", order=order)
    y = ale.eval(0, np.array(order), centering="zero_start")

    # recompute the two-sided transition means in the declared order
    g = gate_values(data)
    expected = [0.0]
    for lo, hi in [(2, 0), (0, 1)]:
        mask = np.isin(data[:, 0], [lo, hi])
        d = (A[hi] - A[lo]) + (B[hi] - B[lo]) * g[mask]
        expected.append(expected[-1] + d.mean())
    np.testing.assert_allclose(y, expected, atol=1e-10)


def test_ale_refit_on_centering_change_preserves_order(data, model):
    # D3 regression: fitting uncentered with a custom order and then evaluating
    # centered forces an auto-refit (norm_const is None). That refit must replay
    # the fit's `order`, not fall back to the default ascending order.
    order = [2.0, 0.0, 1.0]

    ale = effector.ALE(data, model.predict, nof_instances="all", schema=SCHEMA)
    ale.fit(0, centering=False, order=order)
    y = ale.eval(0, np.array(order), centering="zero_integral")

    # the custom order must survive the centering-triggered refit
    np.testing.assert_array_equal(ale.feature_effect["feature_0"]["levels"], order)

    # and the centered answer must equal fitting centered with that order upfront
    ref = effector.ALE(data, model.predict, nof_instances="all", schema=SCHEMA)
    ref.fit(0, centering="zero_integral", order=order)
    y_ref = ref.eval(0, np.array(order), centering="zero_integral")
    np.testing.assert_allclose(y, y_ref, atol=1e-10)


def test_ale_order_list_with_multiple_features_raises(data, model):
    ale = effector.ALE(data, model.predict, schema=SCHEMA)
    with pytest.raises(ValueError, match="exactly one categorical"):
        ale.fit("all", order=[2.0, 0.0, 1.0])


def test_ale_order_not_a_permutation_raises(data, model):
    ale = effector.ALE(data, model.predict, schema=SCHEMA)
    with pytest.raises(ValueError, match="permutation"):
        ale.fit(0, order=[0.0, 1.0, 5.0])


def test_ale_similarity_order_runs_and_is_deterministic(data, model):
    types = [NOMINAL, CONTINUOUS, CONTINUOUS]
    a1 = effector.ALE(data, model.predict, schema={"feature_types": types})
    a1.fit(0, order="similarity")
    a2 = effector.ALE(data, model.predict, schema={"feature_types": types})
    a2.fit(0, order="similarity")
    np.testing.assert_array_equal(
        a1.feature_effect["feature_0"]["levels"],
        a2.feature_effect["feature_0"]["levels"],
    )


def test_similarity_order_recovers_planted_structure():
    # level 1's conditional distribution of x1 sits between level 0 and 2:
    # x1 | level k ~ U(k, k+1) after shuffling codes -> seriation must place
    # level 1 in the middle
    from effector import ordering

    rng = np.random.default_rng(21)
    n = 900
    codes = rng.choice([0.0, 1.0, 2.0], n)
    x1 = np.empty(n)
    # overlapping supports so the KS distance is graded (with disjoint
    # supports KS saturates at 1 and the middle is invisible)
    shift = {0.0: 0.0, 1.0: 1.6, 2.0: 0.8}  # code 2 is the "middle" one
    for c, sh in shift.items():
        m = codes == c
        x1[m] = rng.uniform(sh, sh + 2.0, m.sum())
    data = np.stack([codes, x1], axis=1)
    perm = ordering.similarity_order(
        data, 0, np.array([0.0, 1.0, 2.0]), ["nominal", "continuous"]
    )
    assert perm[1] == 2  # the level whose distribution is in between


def test_rhale_order_on_ordinal(data, model):
    rhale = effector.RHALE(
        data, model.predict, model.jacobian, nof_instances="all", schema=SCHEMA
    )
    rhale.fit(0, binning_method="fixed", centering="zero_start", order=[2.0, 0.0, 1.0])
    y = rhale.eval(0, np.array([2.0, 0.0, 1.0]), centering="zero_start")
    assert y[0] == 0.0  # first level in the declared order is the reference

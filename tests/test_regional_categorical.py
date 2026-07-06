"""Regional effects with a categorical feature of interest — the payoff of the
three-way taxonomy: "for which subgroups is the weekday effect stable?"

Ground truth: f = a[x0] + b[x0]*1{x2>0} with ordinal x0. The heterogeneity of
x0's per-level effect is driven ONLY by the x2 gate: h(v_k) =
(b_k - b_bar)^2 * Var(1{x2>0}). Splitting on x2 at ~0 zeroes the heterogeneity
in BOTH children exactly; splitting anywhere else achieves nothing — a
one-split tree with a unique answer.
"""

import numpy as np
import pytest

import effector
from effector import models
from tests.test_functional_categorical import LEVELS, SCHEMA, make_data

matplotlib_ok = pytest.importorskip("matplotlib")

A, B = models.ConditionalCategorical.A, models.ConditionalCategorical.B


def gated_cat_model(x):
    codes = x[:, 0].astype(int)
    return A[codes] + B[codes] * (x[:, 2] > 0)


def gated_cat_jac(x):
    return np.zeros_like(x)


@pytest.fixture(scope="module")
def data():
    return make_data(n=1000)


@pytest.fixture(scope="module")
def model():
    return models.ConditionalCategorical()


def _fit(reg):
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=1))
    return reg


def _assert_gate_found(reg):
    nodes = reg.tree["feature_0"].nodes
    names = " ".join(n.name for n in nodes)
    assert len(nodes) == 3, names  # root + two children
    assert "x_2" in names, names  # the gate is found
    # both children: zero heterogeneity at every level
    for idx in (1, 2):
        h = reg.eval_heter(0, idx, LEVELS)
        np.testing.assert_allclose(h, 0.0, atol=1e-10)


def test_regional_pdp_on_categorical_foi_finds_the_gate(data):
    reg = _fit(effector.RegionalPDP(data, gated_cat_model, schema=SCHEMA))
    # root heterogeneity is the closed form (b_k - b_bar_w)^2 Var(1{x2>0})
    w = np.unique(data[:, 0], return_counts=True)[1] / len(data)
    gate_var = (data[:, 2] > 0).var()
    expected_root = (B - np.average(B, weights=w)) ** 2 * gate_var
    np.testing.assert_allclose(reg.eval_heter(0, 0, LEVELS), expected_root, atol=1e-10)

    _assert_gate_found(reg)

    # per-node eval keeps the eval-at-levels contract
    y = reg.eval(0, 1, LEVELS, centering="zero_start")
    assert y.shape == (3,)
    with pytest.raises(ValueError, match="observed"):
        reg.eval(0, 1, np.array([0.5]))


def test_regional_ale_on_categorical_foi_finds_the_gate(data):
    reg = _fit(effector.RegionalALE(data, gated_cat_model, schema=SCHEMA))
    _assert_gate_found(reg)


def test_regional_rhale_on_ordinal_foi(data):
    reg = effector.RegionalRHALE(data, gated_cat_model, gated_cat_jac, schema=SCHEMA)
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=1))
    _assert_gate_found(reg)


def test_regional_derpdp_on_categorical_foi_raises(data, model):
    reg = effector.RegionalDerPDP(data, model.predict, model.jacobian, schema=SCHEMA)
    with pytest.raises(ValueError, match="does not support ordinal"):
        reg.fit(0)


def test_regional_summary_and_plot_smoke(data, model):
    reg = _fit(effector.RegionalPDP(data, gated_cat_model, schema=SCHEMA))
    reg.summary(0)
    fig, ax = reg.plot(0, 1, heterogeneity="ice", show_plot=False)
    assert fig is not None


# ---------------------------------------------------------------------------
# plot content: bars equal eval (R1 extended to the categorical branch)
# ---------------------------------------------------------------------------


def test_pdp_cat_bar_heights_equal_eval(data, model):
    pdp = effector.PDP(data, model.predict, nof_instances="all", schema=SCHEMA)
    fig, ax = pdp.plot(0, heterogeneity="std", centering=True, show_plot=False)
    heights = [patch.get_height() for patch in ax.patches]
    expected = pdp.eval(0, LEVELS, centering=True)
    np.testing.assert_allclose(heights, expected, atol=1e-10)


def test_ale_cat_bar_heights_equal_eval(data, model):
    ale = effector.ALE(data, model.predict, nof_instances="all", schema=SCHEMA)
    fig, ax = ale.plot(0, centering=True, show_plot=False)
    heights = [patch.get_height() for patch in ax.patches]
    expected = ale.eval(0, LEVELS, centering=True)
    np.testing.assert_allclose(heights, expected, atol=1e-10)


def test_pdp_cat_ice_plot_returns_fig_ax(data, model):
    pdp = effector.PDP(data, model.predict, schema=SCHEMA)
    ret = pdp.plot(0, heterogeneity="ice", show_plot=False)
    assert ret is not None
    fig, ax = ret
    assert len(ax.patches) == 3  # one bar per level


def test_shapdp_cat_plot_smoke(data, model):
    rng = np.random.default_rng(3)
    m = effector.ShapDP(
        data,
        model.predict,
        nof_instances="all",
        shap_values=rng.normal(size=data.shape),
        schema=SCHEMA,
    )
    fig, ax = m.plot(0, heterogeneity="shap_values", show_plot=False)
    assert len(ax.patches) == 3
    fig, ax = m.plot(0, heterogeneity="std", show_plot=False)
    assert len(ax.patches) == 3


def test_nominal_plot_uses_level_labels(model):
    import pandas as pd

    n = 400
    rng = np.random.default_rng(21)
    df = pd.DataFrame(
        {
            "color": pd.Categorical(rng.choice(["r", "g", "b"], n)),
            "num": rng.uniform(-1, 1, n),
        }
    )

    def df_model(d):
        return d["color"].cat.codes.to_numpy().astype(float) + d["num"].to_numpy()

    pdp = effector.PDP(df, df_model)
    fig, ax = pdp.plot(0, heterogeneity="std", show_plot=False)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["b", "g", "r"]  # pandas sorts categories alphabetically


def test_regional_capability_matrix_enforced_at_fit():
    # regression: the capability matrix must be enforced at regional fit, not
    # only later at plot — otherwise fit/summary run on an unsupported FOI and
    # only plot raises (RHALE on nominal was inconsistent this way).
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [rng.integers(0, 3, 800).astype(float), rng.uniform(-1, 1, 800)]
    )
    f = lambda z: z[:, 0] * (z[:, 1] > 0)
    jac = lambda z: np.zeros_like(z)
    schema = {"feature_types": ["nominal", "continuous"]}
    part = effector.space_partitioning.Best(max_depth=2)

    with pytest.raises(ValueError, match="rhale does not support nominal"):
        effector.RegionalRHALE(X, f, model_jac=jac, schema=schema).fit(
            0, space_partitioner=part
        )
    with pytest.raises(ValueError, match="d-pdp does not support nominal"):
        effector.RegionalDerPDP(X, f, model_jac=jac, schema=schema).fit(
            0, space_partitioner=part
        )
    # supported methods still fit on the same nominal FOI
    effector.RegionalPDP(X, f, schema=schema).fit(0, space_partitioner=part)
    effector.RegionalALE(X, f, schema=schema).fit(0, space_partitioner=part)

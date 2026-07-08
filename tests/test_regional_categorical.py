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


def _find(fx):
    """Fit feature 0 globally, then search for the region tree (one split)."""
    fx.fit(0)
    return fx.find_regions(0, finder=effector.space_partitioning.Best(max_depth=1))


def _assert_gate_found(part):
    regions = list(part)
    names = " ".join(r.name for r in regions)
    assert len(part) == 3, names  # root + two children
    assert "x_2" in names, names  # the gate is found
    # both children: zero heterogeneity at every level
    for idx in (1, 2):
        h = part.eval_heter(idx, LEVELS)
        np.testing.assert_allclose(h, 0.0, atol=1e-10)


def test_regional_pdp_on_categorical_foi_finds_the_gate(data):
    fx = effector.PDP(data, gated_cat_model, nof_instances="all", schema=SCHEMA)
    part = _find(fx)
    # root heterogeneity is the closed form (b_k - b_bar_w)^2 Var(1{x2>0})
    w = np.unique(data[:, 0], return_counts=True)[1] / len(data)
    gate_var = (data[:, 2] > 0).var()
    expected_root = (B - np.average(B, weights=w)) ** 2 * gate_var
    np.testing.assert_allclose(part.eval_heter(0, LEVELS), expected_root, atol=1e-10)

    _assert_gate_found(part)

    # per-region eval keeps the eval-at-levels contract
    y = part.eval(1, LEVELS, centering="zero_start")
    assert y.shape == (3,)
    with pytest.raises(ValueError, match="observed"):
        part.eval(1, np.array([0.5]))


def test_regional_ale_on_categorical_foi_finds_the_gate(data):
    fx = effector.ALE(data, gated_cat_model, nof_instances="all", schema=SCHEMA)
    _assert_gate_found(_find(fx))


def test_regional_rhale_on_ordinal_foi(data):
    fx = effector.RHALE(
        data,
        gated_cat_model,
        model_jac=gated_cat_jac,
        nof_instances="all",
        schema=SCHEMA,
    )
    _assert_gate_found(_find(fx))


def test_regional_derpdp_on_categorical_foi_raises(data, model):
    # the capability matrix rejects an ordinal FOI for d-PDP at fit time
    fx = effector.DerPDP(data, model.predict, model_jac=model.jacobian, schema=SCHEMA)
    with pytest.raises(ValueError, match="does not support ordinal"):
        fx.fit(0)


def test_regional_summary_and_plot_smoke(data, model):
    fx = effector.PDP(data, gated_cat_model, nof_instances="all", schema=SCHEMA)
    part = _find(fx)
    part.show()
    fig, ax = part.plot(1, heterogeneity="ice", show_plot=False)
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


def test_nominal_plot_uses_level_labels():
    import pandas as pd

    n = 400
    rng = np.random.default_rng(21)
    df = pd.DataFrame(
        {
            "color": pd.Categorical(rng.choice(["r", "g", "b"], n)),
            "num": rng.uniform(-1, 1, n),
        }
    )
    # numpy-only: convert once; the schema carries the "b"/"g"/"r" labels
    X, schema = effector.from_dataframe(df)

    def np_model(A):  # color code + num, on the encoded matrix
        return A[:, 0] + A[:, 1]

    pdp = effector.PDP(X, np_model, schema=schema)
    fig, ax = pdp.plot(0, heterogeneity="std", show_plot=False)
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert labels == ["b", "g", "r"]  # pandas sorts categories alphabetically


def test_regional_capability_matrix_enforced_at_fit():
    # regression: the capability matrix must be enforced on the unsupported FOI,
    # not only later at plot — otherwise fit/summary run on an unsupported FOI
    # and only plot raises (RHALE on nominal was inconsistent this way). For the
    # methods that reject a nominal FOI the guard fires already at fit.
    rng = np.random.default_rng(0)
    X = np.column_stack(
        [rng.integers(0, 3, 800).astype(float), rng.uniform(-1, 1, 800)]
    )
    f = lambda z: z[:, 0] * (z[:, 1] > 0)
    jac = lambda z: np.zeros_like(z)
    schema = {"feature_types": ["nominal", "continuous"]}
    finder = effector.space_partitioning.Best(max_depth=2)

    with pytest.raises(ValueError, match="rhale does not support nominal"):
        effector.RHALE(X, f, model_jac=jac, schema=schema).fit(0)
    with pytest.raises(ValueError, match="d-pdp does not support nominal"):
        effector.DerPDP(X, f, model_jac=jac, schema=schema).fit(0)

    # supported methods still fit AND find regions on the same nominal FOI
    fx = effector.PDP(X, f, nof_instances="all", schema=schema)
    fx.fit(0)
    fx.find_regions(0, finder=finder)
    fx = effector.ALE(X, f, nof_instances="all", schema=schema)
    fx.fit(0)
    fx.find_regions(0, finder=finder)

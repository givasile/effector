"""Contract layer, reproducibility (PLAN III §6.1).

Every public effect-class constructor takes ``random_state`` (default ``21``):
two identical constructions give identical ``eval``/``fit``/``plot`` output,
``None`` opts into fresh randomness, and no effect-class code touches the
global ``np.random`` state.  Parametrized over the 5 global and 5 regional classes on
the shared tiny models (see conftest); ``nof_instances`` is always set below
the dataset size so the subsampling draw is actually exercised.
"""

import numpy as np
import pytest

import effector
from tests.conftest import (
    GLOBAL_NAMES,
    REGIONAL_NAMES,
    analytic_shap_values,
    eval_mean,
    make_global,
    make_regional,
)

XS = np.linspace(-0.8, 0.8, 40)
NOF = 50  # < N_GLOBAL=200, forces the subsampling draw


def params(names):
    return [pytest.param(name, id=name) for name in names]


def make_global_subsampled(name, data, **kwargs):
    """A global object on a forced subsample, with row-aligned shap values.

    conftest's ``make_global`` setdefaults ``shap_values`` computed on the
    *full* data; under subsampling that is row-misaligned with ``m.data``
    (a latent pre-existing inconsistency, out of scope here), so recompute
    them on the kept rows — D1 guarantees both constructions keep the same.
    """
    m = make_global(name, data, nof_instances=NOF, **kwargs)
    if name == "shapdp":
        m.shap_values = analytic_shap_values(m.data)
    return m


# ---------------------------------------------------------------------------
# D1 — same seed -> same subsample (construction is deterministic)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params(GLOBAL_NAMES))
def test_d1_construction_deterministic(name, global_data):
    m1 = make_global_subsampled(name, global_data)
    m2 = make_global_subsampled(name, global_data)
    assert np.array_equal(m1.indices, m2.indices)
    assert np.array_equal(m1.data, m2.data)


# ---------------------------------------------------------------------------
# D2 — same seed -> identical eval and eval_heter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params(GLOBAL_NAMES))
def test_d2_eval_deterministic(name, global_data):
    m1 = make_global_subsampled(name, global_data)
    m2 = make_global_subsampled(name, global_data)
    np.testing.assert_allclose(eval_mean(m1, 0, XS), eval_mean(m2, 0, XS), atol=1e-12)
    np.testing.assert_allclose(m1.eval_heter(0, XS), m2.eval_heter(0, XS), atol=1e-12)


# ---------------------------------------------------------------------------
# D3 — different seeds -> different subsamples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params(GLOBAL_NAMES))
def test_d3_different_seeds_differ(name, global_data):
    m1 = make_global_subsampled(name, global_data)  # default random_state=21
    m2 = make_global_subsampled(name, global_data, random_state=1234)
    assert not np.array_equal(m1.data, m2.data)


# ---------------------------------------------------------------------------
# D4 — random_state=None still works (no determinism assert: that draw is
# genuinely random, and two None runs *may* collide on tiny data)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params(GLOBAL_NAMES))
def test_d4_none_seed_works(name, global_data):
    m = make_global_subsampled(name, global_data, random_state=None)
    assert m.data.shape[0] == NOF
    y = eval_mean(m, 0, XS)
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# D5 — regional: same seed -> identical tree and node effects
# ---------------------------------------------------------------------------


def fit_regional_subsampled(name, data):
    """Fit feature 0 with forced subsampling and *no* explicit seed kwargs:
    the constructor's default random_state must carry determinism end-to-end
    (for regional_shapdp, all the way into the shap backend)."""
    if name == "regional_shapdp":
        reg = make_regional(name, data[:50], nof_instances=30)
        reg.fit(
            0,
            space_partitioner=effector.space_partitioning.Best(max_depth=2),
            budget=128,
        )
        return reg
    reg = make_regional(name, data, nof_instances=300)
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=2))
    return reg


@pytest.mark.parametrize("name", params(REGIONAL_NAMES))
def test_d5_regional_fit_deterministic(name, regional_data):
    reg1 = fit_regional_subsampled(name, regional_data)
    reg2 = fit_regional_subsampled(name, regional_data)
    assert np.array_equal(reg1.data, reg2.data)
    tree1, tree2 = reg1.tree["feature_0"], reg2.tree["feature_0"]
    assert len(tree1.nodes) == len(tree2.nodes)
    for node_idx in range(len(tree1.nodes)):
        np.testing.assert_allclose(
            reg1.eval(0, node_idx, XS), reg2.eval(0, node_idx, XS), atol=1e-12
        )


# ---------------------------------------------------------------------------
# D6 — plot-time ICE subsampling is seeded too (visualization threading)
# ---------------------------------------------------------------------------


def _ice_ydata(m):
    fig, ax = m.plot(0, heterogeneity="ice", nof_ice=10, show_plot=False)
    return [line.get_ydata() for line in ax.get_lines()]


def test_d6_pdp_ice_plot_deterministic(global_data):
    m1 = make_global_subsampled("pdp", global_data)
    m2 = make_global_subsampled("pdp", global_data)
    for y1, y2 in zip(_ice_ydata(m1), _ice_ydata(m2), strict=True):
        np.testing.assert_allclose(y1, y2, atol=1e-12)


# ---------------------------------------------------------------------------
# D7 — helpers: the single choke point behaves as specified
# ---------------------------------------------------------------------------


def test_d7_prep_nof_instances_seeded():
    n1, ind1 = effector.helpers.prep_nof_instances(50, 200, random_state=0)
    n2, ind2 = effector.helpers.prep_nof_instances(50, 200, random_state=0)
    _, ind3 = effector.helpers.prep_nof_instances(50, 200, random_state=1)
    assert n1 == n2 == 50
    assert np.array_equal(ind1, ind2)
    assert not np.array_equal(ind1, ind3)


def test_d7_global_np_random_untouched(global_data):
    """Constructing effect objects must not consume or reseed global np.random."""
    np.random.seed(0)
    expected = np.random.get_state()[1].copy()
    np.random.seed(0)
    for name in GLOBAL_NAMES:
        make_global_subsampled(name, global_data)
    assert np.array_equal(np.random.get_state()[1], expected)

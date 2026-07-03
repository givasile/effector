"""Contract layer, regional classes (PLAN II §3.1: RC1–RC5).

Parametrized over the 5 regional classes on the gated-linear model from the
functional anchor (one obvious split; see conftest).  Same convention as
test_contract_global: green = rule holds today, ``xfail(strict=True)`` = the
refactor must make it true.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import effector
import effector.axis_partitioning as ap
from tests.conftest import make_regional_data

XS = np.linspace(-0.9, 0.9, 30)


# ---------------------------------------------------------------------------
# RC1 — fit produces a tree; summary runs
# ---------------------------------------------------------------------------


def test_rc1_tree_and_summary(fitted_regional, capsys):
    name, reg = fitted_regional
    tree = reg.tree["feature_0"]
    assert tree is not None
    root = tree.get_root()
    assert root.idx == 0
    assert root.info["weight"] == 1.0
    reg.summary(features=0)
    assert "Feature 0" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# RC2 — eval works for every node index and returns the C1 shapes
# ---------------------------------------------------------------------------


def test_rc2_eval_every_node(fitted_regional):
    name, reg = fitted_regional
    tree = reg.tree["feature_0"]
    for node_idx in range(len(tree.nodes)):
        y = reg.eval(0, node_idx, XS, centering=False)
        assert isinstance(y, np.ndarray)
        assert y.shape == XS.shape
        assert np.all(np.isfinite(y))


def test_rc2_invalid_node_idx_raises(fitted_regional):
    name, reg = fitted_regional
    tree = reg.tree["feature_0"]
    with pytest.raises((AssertionError, ValueError, IndexError)):
        reg.eval(0, len(tree.nodes), XS)


def test_rc2_eval_heter_root_node(fitted_regional):
    """New surface: the regional twin of eval_heter, on the root node."""
    name, reg = fitted_regional
    h = reg.eval_heter(0, 0, XS)
    assert isinstance(h, np.ndarray)
    assert h.shape == XS.shape
    assert np.all(h >= 0)


# ---------------------------------------------------------------------------
# RC3 — plot contract (R7)
# ---------------------------------------------------------------------------


def test_rc3_plot_smoke(fitted_regional):
    """Today's surface: plot always shows (Agg swallows it) and returns None."""
    name, reg = fitted_regional
    reg.plot(feature=0, node_idx=0)


def test_rc3_plot_returns_fig_ax(fitted_regional):
    name, reg = fitted_regional
    ret = reg.plot(feature=0, node_idx=0, show_plot=False)
    assert isinstance(ret, tuple) and len(ret) == 2
    assert isinstance(ret[0], plt.Figure)


# ---------------------------------------------------------------------------
# RC4 — fit-kwargs propagation (B1): the binning method given to fit must be
# the one eval's fe-object is fitted with
# ---------------------------------------------------------------------------


def _quad_model(x):
    return x[:, 0] ** 2


def _quad_jac(x):
    jac = np.zeros_like(x)
    jac[:, 0] = 2 * x[:, 0]
    return jac


def test_rc4_fit_kwargs_propagate_to_eval():
    rng = np.random.default_rng(21)
    data = rng.uniform(-1, 1, size=(500, 2))

    reg = effector.RegionalRHALE(data, _quad_model, model_jac=_quad_jac)
    reg.fit(
        0,
        binning_method=ap.Fixed(nof_bins=7),
        space_partitioner=effector.space_partitioning.Best(max_depth=1),
    )
    y_reg = reg.eval(0, 0, XS, centering=False)

    twin = effector.RHALE(data, _quad_model, model_jac=_quad_jac, nof_instances="all")
    twin.fit(features=0, binning_method=ap.Fixed(nof_bins=7), centering=False)
    y_twin = twin.eval(0, XS, centering=False)

    np.testing.assert_allclose(y_reg, y_twin, atol=1e-8)


# ---------------------------------------------------------------------------
# RC5 — string arguments for the space partitioner (B3 / R6)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rc5_data():
    return make_regional_data(n=300)


def _fit_pdp_with_partitioner(data, partitioner):
    reg = effector.RegionalPDP(data, _gated, nof_instances="all")
    reg.fit(0, space_partitioner=partitioner)
    return reg


def _gated(x):
    y = np.zeros_like(x[:, 0])
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind] = 5 * x[ind, 0]
    return y


def test_rc5_best_works(rc5_data):
    reg = _fit_pdp_with_partitioner(rc5_data, "best")
    assert reg.tree["feature_0"] is not None


def test_rc5_best_level_wise_works(rc5_data):
    reg = _fit_pdp_with_partitioner(rc5_data, "best_level_wise")
    assert reg.tree["feature_0"] is not None


@pytest.mark.parametrize("junk", ["cart", "worst", ""])
def test_rc5_junk_raises(rc5_data, junk):
    with pytest.raises((ValueError, AssertionError)):
        _fit_pdp_with_partitioner(rc5_data, junk)

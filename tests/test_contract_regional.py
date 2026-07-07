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
from tests.conftest import (
    REGIONAL_NAMES,
    CountingModel,
    gated_model,
    gated_model_jac,
    make_regional_data,
)

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


# ---------------------------------------------------------------------------
# RC6 — the split search is model-free: the whole point of the local-effects
# lifecycle. The global effect is fit once (its local effects computed once),
# and every candidate split re-scores cached subsets. So the number of model
# (and jacobian) calls during fit must NOT grow with how many candidate splits
# are evaluated — the old per-candidate object rebuilds scaled with grid size.
# ---------------------------------------------------------------------------


def _fit_counted(name, data, grid):
    model = CountingModel(gated_model)
    jac = CountingModel(gated_model_jac)
    part = effector.space_partitioning.Best(
        max_depth=2, numerical_features_grid_size=grid
    )
    if name == "regional_pdp":
        reg = effector.RegionalPDP(data, model)
    elif name == "regional_derpdp":
        reg = effector.RegionalDerPDP(data, model, model_jac=jac)
    elif name == "regional_ale":
        reg = effector.RegionalALE(data, model)
    elif name == "regional_rhale":
        reg = effector.RegionalRHALE(data, model, model_jac=jac)
    elif name == "regional_shapdp":
        # inject attributions: the split search never touches the model (0 calls)
        shap_values = np.random.RandomState(0).normal(size=data.shape)
        reg = effector.RegionalShapDP(data, model, shap_values=shap_values)
    else:
        raise ValueError(name)
    reg.fit(0, space_partitioner=part)
    return model.n_calls + jac.n_calls


@pytest.mark.parametrize("name", REGIONAL_NAMES)
def test_rc6_split_search_is_model_free(name):
    data = make_regional_data()
    # 5 vs 40 candidate positions per conditioning feature at the root — an ~8x
    # difference in candidate splits scored
    n_small = _fit_counted(name, data, grid=5)
    n_large = _fit_counted(name, data, grid=40)
    assert n_small == n_large, (
        f"{name}: model calls grew with the candidate count "
        f"({n_small} at grid=5 vs {n_large} at grid=40) — the split search is "
        f"re-querying the model instead of re-scoring cached local effects"
    )
    # a one-time precompute is a small constant; per-candidate work would be
    # >= the number of candidates (>= grid)
    assert n_large < 40, f"{name}: {n_large} model calls looks like per-candidate work"


# ---------------------------------------------------------------------------
# RC7 — one truth (regional ≡ masked global): the heterogeneity number that
# CHOSE each split is byte-identical to the one the node's surfaces report —
# the tree summary, heter_score(mask) and the plotted band share one source.
# ---------------------------------------------------------------------------


def test_rc7_tree_heterogeneity_equals_masked_heter_score(fitted_regional):
    name, reg = fitted_regional
    fe = reg._global_fe
    tree = reg.tree["feature_0"]
    for node in tree.nodes:
        mask = node.info["active_indices"].astype(bool)
        np.testing.assert_allclose(
            node.info["heterogeneity"],
            fe.heter_score(0, mask=mask),
            atol=1e-10,
            err_msg=f"{name}, node {node.idx}",
        )


# ---------------------------------------------------------------------------
# RC8 — the constitution on the node surfaces: after fit, eval/eval_heter/plot
# on any node are masked re-summaries of the cached local effects — ZERO model
# calls. (The one exception, PDP eval at off-grid xs, is pinned in the masked
# contract layer, test_contract_masked.py::test_m2.)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", REGIONAL_NAMES)
def test_rc8_node_surfaces_are_model_free(name):
    from effector import helpers

    data = make_regional_data()
    model = CountingModel(gated_model)
    jac = CountingModel(gated_model_jac)
    if name == "regional_pdp":
        reg = effector.RegionalPDP(data, model)
    elif name == "regional_derpdp":
        reg = effector.RegionalDerPDP(data, model, model_jac=jac)
    elif name == "regional_ale":
        reg = effector.RegionalALE(data, model)
    elif name == "regional_rhale":
        reg = effector.RegionalRHALE(data, model, model_jac=jac)
    elif name == "regional_shapdp":
        # structured (gate-aware) attributions so the fitted tree is real; the
        # small noise keeps DP binning off the single-bin/zero-variance corner
        # where ShapDP's spline goes nan (pre-existing kernel wart)
        rng = np.random.default_rng(0)
        shap_values = rng.normal(0, 0.05, data.shape)
        ind = np.logical_and(data[:, 1] > 0, data[:, 2] == 0)
        shap_values[ind, 0] += 5 * data[ind, 0]
        reg = effector.RegionalShapDP(data, model, shap_values=shap_values)
    else:
        raise ValueError(name)
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=2))

    n0 = model.n_calls + jac.n_calls
    # xs on the (d-)PDP cache grid so no method takes its exact-retouch path
    grid = np.linspace(
        reg.axis_limits[0, 0], reg.axis_limits[1, 0], helpers.NOF_INTERNAL_POINTS
    )
    tree = reg.tree["feature_0"]
    for node_idx in range(len(tree.nodes)):
        reg.eval(0, node_idx, grid, centering=True)
        reg.eval_heter(0, node_idx, grid)
        reg.plot(feature=0, node_idx=node_idx, show_plot=False)
    plt.close("all")
    assert model.n_calls + jac.n_calls == n0, (
        f"{name}: node eval/plot re-queried the model "
        f"({model.n_calls + jac.n_calls - n0} extra calls)"
    )

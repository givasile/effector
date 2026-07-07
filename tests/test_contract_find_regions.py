"""Contract layer, find_regions / Partition (RC1–RC9).

Regional questions are asked via `GlobalEffectBase.find_regions(feature) ->
Partition` (design contract R12: partitions are values, not stored state). These
port RC1–RC8 from the deleted test_contract_regional.py onto the new API and add
RC9 (values-not-state). Parametrized over the 5 global classes on the gated model
(one obvious split; see conftest).
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import effector
import effector.axis_partitioning as ap
from effector.partition import Partition
from tests.conftest import (
    CountingModel,
    GLOBAL_NAMES,
    gated_model,
    gated_model_jac,
    make_regional_data,
)

XS = np.linspace(-0.9, 0.9, 30)


# ---------------------------------------------------------------------------
# RC1 — find_regions returns a Partition; show() runs
# ---------------------------------------------------------------------------


def test_rc1_partition_and_show(fitted_partition, capsys):
    name, effect, part = fitted_partition
    assert isinstance(part, Partition)
    assert part[0].idx == 0
    assert part[0].weight == 1.0
    part.show()
    assert "Feature 0" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# RC2 — eval works for every region index and returns the C1 shape
# ---------------------------------------------------------------------------


def test_rc2_eval_every_region(fitted_partition):
    name, effect, part = fitted_partition
    for idx in range(len(part)):
        y = part.eval(idx, XS, centering=False)
        assert isinstance(y, np.ndarray)
        assert y.shape == XS.shape
        assert np.all(np.isfinite(y))


def test_rc2_invalid_region_idx_raises(fitted_partition):
    name, effect, part = fitted_partition
    with pytest.raises((IndexError, ValueError)):
        part.eval(len(part), XS)


def test_rc2_eval_heter_root(fitted_partition):
    name, effect, part = fitted_partition
    h = part.eval_heter(0, XS)
    assert isinstance(h, np.ndarray)
    assert h.shape == XS.shape
    assert np.all(h >= 0)


# ---------------------------------------------------------------------------
# RC3 — plot contract (R7)
# ---------------------------------------------------------------------------


def test_rc3_plot_returns_fig_ax(fitted_partition):
    name, effect, part = fitted_partition
    ret = part.plot(0, show_plot=False)
    assert isinstance(ret, tuple) and len(ret) == 2
    assert isinstance(ret[0], plt.Figure)


def test_rc3_plot_smoke(fitted_partition):
    name, effect, part = fitted_partition
    part.plot(0)


# ---------------------------------------------------------------------------
# RC4 — fit-kwargs propagation is now just replay: each region's heterogeneity
# equals the masked heter_score with the feature's fitted binning.
# ---------------------------------------------------------------------------


def _quad_model(x):
    return x[:, 0] ** 2


def _quad_jac(x):
    jac = np.zeros_like(x)
    jac[:, 0] = 2 * x[:, 0]
    return jac


def test_rc4_fit_kwargs_replayed_in_regions():
    rng = np.random.default_rng(21)
    data = rng.uniform(-1, 1, size=(500, 2))
    rhale = effector.RHALE(data, _quad_model, model_jac=_quad_jac, nof_instances="all")
    rhale.fit(0, binning_method=ap.Fixed(nof_bins=7), centering=False)
    part = rhale.find_regions(0, finder=effector.space_partitioning.Best(max_depth=1))
    for r in part:
        np.testing.assert_allclose(
            r.heterogeneity, rhale.heter_score(0, mask=r.mask), atol=1e-10
        )


# ---------------------------------------------------------------------------
# RC5 — finder argument forms + non-mutation of the passed finder
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rc5_effect():
    data = make_regional_data(n=300)
    fx = effector.PDP(data, gated_model, nof_instances="all")
    fx.fit(0, centering=False)
    return fx


def test_rc5_best_works(rc5_effect):
    assert isinstance(rc5_effect.find_regions(0, finder="best"), Partition)


def test_rc5_best_level_wise_works(rc5_effect):
    assert isinstance(rc5_effect.find_regions(0, finder="best_level_wise"), Partition)


@pytest.mark.parametrize("junk", ["cart", "worst", ""])
def test_rc5_junk_raises(rc5_effect, junk):
    with pytest.raises((ValueError, AssertionError)):
        rc5_effect.find_regions(0, finder=junk)


def test_rc5_finder_instance_not_mutated(rc5_effect):
    finder = effector.space_partitioning.Best(max_depth=2)
    rc5_effect.find_regions(0, finder=finder)
    # compile() ran on a deepcopy — the caller's instance stays pristine
    assert finder.data is None
    assert finder.feature is None


# ---------------------------------------------------------------------------
# RC6 — the split search is model-free: model/jacobian calls do not grow with
# the candidate-split count (grid size).
# ---------------------------------------------------------------------------


def _find_counted(name, data, grid):
    model = CountingModel(gated_model)
    jac = CountingModel(gated_model_jac)
    finder = effector.space_partitioning.Best(
        max_depth=2, numerical_features_grid_size=grid
    )
    if name == "pdp":
        fx = effector.PDP(data, model, nof_instances="all")
    elif name == "derpdp":
        fx = effector.DerPDP(data, model, model_jac=jac, nof_instances="all")
    elif name == "ale":
        fx = effector.ALE(data, model, nof_instances="all")
    elif name == "rhale":
        fx = effector.RHALE(data, model, model_jac=jac, nof_instances="all")
    elif name == "shapdp":
        shap_values = np.random.RandomState(0).normal(size=data.shape)
        fx = effector.ShapDP(
            data, model, shap_values=shap_values, nof_instances="all"
        )
    else:
        raise ValueError(name)
    fx.find_regions(0, finder=finder)  # triggers the one lazy fit + model-free search
    return model.n_calls + jac.n_calls


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_rc6_split_search_is_model_free(name):
    data = make_regional_data()
    n_small = _find_counted(name, data, grid=5)
    n_large = _find_counted(name, data, grid=40)
    assert n_small == n_large, (
        f"{name}: model calls grew with the candidate count "
        f"({n_small} at grid=5 vs {n_large} at grid=40)"
    )
    assert n_large < 40, f"{name}: {n_large} model calls looks like per-candidate work"


# ---------------------------------------------------------------------------
# RC7 — one truth: each region's heterogeneity == masked heter_score.
# ---------------------------------------------------------------------------


def test_rc7_region_heterogeneity_equals_masked_heter_score(fitted_partition):
    name, effect, part = fitted_partition
    for r in part:
        np.testing.assert_allclose(
            r.heterogeneity,
            effect.heter_score(0, mask=r.mask),
            atol=1e-10,
            err_msg=f"{name}, region {r.idx}",
        )


# ---------------------------------------------------------------------------
# RC8 — model-free surfaces: after find_regions, looping eval/eval_heter/plot
# over regions adds ZERO model calls (xs on the cache grid; PDP off-grid is the
# pinned exception in test_contract_masked.py).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_rc8_region_surfaces_are_model_free(name):
    from effector import helpers

    data = make_regional_data()
    model = CountingModel(gated_model)
    jac = CountingModel(gated_model_jac)
    if name == "pdp":
        fx = effector.PDP(data, model, nof_instances="all")
    elif name == "derpdp":
        fx = effector.DerPDP(data, model, model_jac=jac, nof_instances="all")
    elif name == "ale":
        fx = effector.ALE(data, model, nof_instances="all")
    elif name == "rhale":
        fx = effector.RHALE(data, model, model_jac=jac, nof_instances="all")
    elif name == "shapdp":
        rng = np.random.default_rng(0)
        shap_values = rng.normal(0, 0.05, data.shape)
        ind = np.logical_and(data[:, 1] > 0, data[:, 2] == 0)
        shap_values[ind, 0] += 5 * data[ind, 0]
        fx = effector.ShapDP(data, model, shap_values=shap_values, nof_instances="all")
    else:
        raise ValueError(name)

    part = fx.find_regions(0, finder=effector.space_partitioning.Best(max_depth=2))
    n0 = model.n_calls + jac.n_calls
    grid = np.linspace(
        fx.axis_limits[0, 0], fx.axis_limits[1, 0], helpers.NOF_INTERNAL_POINTS
    )
    for idx in range(len(part)):
        part.eval(idx, grid, centering=True)
        part.eval_heter(idx, grid)
        part.plot(idx, show_plot=False)
    plt.close("all")
    assert model.n_calls + jac.n_calls == n0, (
        f"{name}: region surfaces re-queried the model "
        f"({model.n_calls + jac.n_calls - n0} extra calls)"
    )


# ---------------------------------------------------------------------------
# RC9 — values-not-state: find_regions is a pure query. Two calls return equal
# but distinct partitions, add no public attribute to the effect, and the second
# call makes zero model calls.
# ---------------------------------------------------------------------------


def test_rc9_values_not_state():
    data = make_regional_data()
    model = CountingModel(gated_model)
    fx = effector.PDP(data, model, nof_instances="all")
    fx.fit(0, centering=False)

    before = {k for k in vars(fx) if not k.startswith("_")}
    p1 = fx.find_regions(0)
    n_after_first = model.n_calls
    p2 = fx.find_regions(0)
    after = {k for k in vars(fx) if not k.startswith("_")}

    assert p1 is not p2
    assert len(p1) == len(p2)
    for a, b in zip(p1, p2):
        assert np.array_equal(a.mask, b.mask)
        assert a.heterogeneity == b.heterogeneity
    assert after == before, "find_regions added public state to the effect"
    assert model.n_calls == n_after_first, "second find_regions re-queried the model"

"""Contract layer, reproducibility (PLAN III §6.1).

Every public effect-class constructor takes ``random_state`` (default ``21``):
two identical constructions give identical ``eval``/``fit``/``plot`` output,
``None`` opts into fresh randomness, and no effect-class code touches the
global ``np.random`` state.  Parametrized over the 5 global classes on the
shared tiny models (see conftest); ``nof_instances`` is always set below the
dataset size so the subsampling draw is actually exercised.  D5 extends the
same reproducibility contract to ``find_regions``: two independent runs of the
same global class on the same seed must yield an identical ``Partition``.
"""

import numpy as np
import pytest

import effector
from tests.conftest import (
    GLOBAL_NAMES,
    analytic_shap_values,
    eval_mean,
    make_gated_global,
    make_global,
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
# D5 — find_regions: same seed -> identical partition (masks, heterogeneities,
# split metadata).  Regional questions are now asked via the GLOBAL class's
# ``find_regions`` -> ``Partition``; the reproducibility contract carries the
# constructor's default random_state end-to-end into the split search (for
# shapdp, all the way into the shap backend).
# ---------------------------------------------------------------------------


def fit_find_subsampled(name, data):
    """Construct the GLOBAL class with forced subsampling and *no* explicit seed
    kwargs, fit feature 0, then find_regions — twice-callable so two independent
    runs can be compared for determinism.  For shapdp keep the old
    N=50/budget=128/seed convention that keeps the gate fast and stable."""
    finder = effector.space_partitioning.Best(max_depth=2)
    if name == "shapdp":
        fx = make_gated_global(
            "shapdp",
            data[:50],
            budget=128,
            shap_explainer_kwargs={"seed": 0},
            nof_instances="all",
        )
        fx.fit(0, centering=False)
        return fx, fx.find_regions(0, finder=finder)
    fx = make_gated_global(name, data, nof_instances=300)
    fx.fit(0, centering=False)
    return fx, fx.find_regions(0, finder=finder)


@pytest.mark.parametrize("name", params(GLOBAL_NAMES))
def test_d5_find_regions_deterministic(name, regional_data):
    fx1, part1 = fit_find_subsampled(name, regional_data)
    fx2, part2 = fit_find_subsampled(name, regional_data)
    # same subsample end-to-end, then an identical partition
    assert np.array_equal(fx1.data, fx2.data)
    assert len(part1) == len(part2)
    for r1, r2 in zip(part1, part2, strict=True):
        assert np.array_equal(r1.mask, r2.mask)
        np.testing.assert_allclose(r1.heterogeneity, r2.heterogeneity, atol=1e-12)
        # split metadata (None on the root) must match exactly
        assert r1.foc_index == r2.foc_index
        assert r1.foc_split_position == r2.foc_split_position
        assert r1.comparison == r2.comparison
        assert r1.level == r2.level
        assert r1.parent_idx == r2.parent_idx


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

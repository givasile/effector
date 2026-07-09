"""R14 — the two-block lifecycle as arithmetic.

Every test here replays one of the agreed lifecycle traces and pins the EXACT
model-call count of every step (the `budget` context manager), plus the state
transitions the constitution promises (`_local[f]["frame"]`, `_epoch[f]`,
position-store growth). Any future edit that sneaks in an extra model touch,
loses a cache hit, or forgets a frame comparison flips one of these integers
and fails loudly.

The traces:
- cache (a) fills once per frame; same frame re-fitted = 0 calls
- a frame change (ALE grid, categorical order) recomputes and bumps the epoch
- the (d-)PDP position store grows missing-only and never bumps the epoch
- repeated evals/plots, centering switches, scores, masked surfaces: 0 calls
- `ToyEffect` (tests/toy_method.py): a brand-new method written with zero
  cache logic gets the whole constitution for free
- lazy ≡ eager: never-fit queries equal fit-then-query, byte for byte
"""

import numpy as np
import pytest

import effector
import effector.axis_partitioning as ap

from .conftest import (
    GLOBAL_NAMES,
    CountingModel,
    analytic_shap_values,
    budget,
    linear_model,
    linear_model_jac,
    make_global,
    make_global_data,
)
from .toy_method import ToyEffect

XS = np.linspace(-0.8, 0.8, 7)


def make_counting(name, data):
    """A fresh effect on counting model + counting jacobian. Returns
    (effect, [counters]) — sum the counters for the budget."""
    model = CountingModel(linear_model)
    jac = CountingModel(linear_model_jac)
    if name == "pdp":
        return effector.PDP(data, model), [model]
    if name == "derpdp":
        return effector.DerPDP(data, model, model_jac=jac), [model, jac]
    if name == "ale":
        return effector.ALE(data, model), [model]
    if name == "rhale":
        return effector.RHALE(data, model, model_jac=jac), [model, jac]
    if name == "shapdp":
        return effector.ShapDP(data, model, shap_values=analytic_shap_values(data)), [
            model
        ]
    raise ValueError(name)


# ---------------------------------------------------------------------------
# L1 — the five lifecycle traces, exact budgets
# ---------------------------------------------------------------------------


def test_l1_pdp_trace(global_data):
    m, counters = make_counting("pdp", global_data)

    with budget(*counters, expected=1):  # ICE on the 30-pt canonical grid
        m.fit(features=0, centering=True)  # payload + const DERIVED: 0 extra
    with budget(*counters, expected=1):  # only the missing display positions
        m.plot(0, show_plot=False)
    with budget(*counters, expected=0):  # every position cached
        m.plot(0, show_plot=False)
    grid = m._local[0]["pos"]
    with budget(*counters, expected=0):  # on-cached-positions eval
        m.eval(0, grid[3:8])
    with budget(*counters, expected=0):  # centering switch = derive a constant
        m.eval(0, grid[3:8], centering="zero_start")
    with budget(*counters, expected=0):
        m.heter_score(0)
        m.importance(0)


def test_l1_derpdp_trace(global_data):
    m, counters = make_counting("derpdp", global_data)

    with budget(*counters, expected=1):  # d-ICE on the canonical grid (jac)
        m.fit(features=0)
    with budget(*counters, expected=0):  # on-grid positions -> cache read
        m.eval(0, m._local[0]["pos"][:5])
    with budget(*counters, expected=1):  # missing display positions
        m.plot(0, show_plot=False)
    with budget(*counters, expected=0):
        m.plot(0, show_plot=False)


def test_l1_ale_trace(global_data):
    m, counters = make_counting("ale", global_data)

    with budget(*counters, expected=2):  # secants: model(right), model(left)
        m.fit(features=0)  # centering const derived: 0 extra
    with budget(*counters, expected=0):  # payload read
        m.plot(0, show_plot=False)
    with budget(*counters, expected=0):  # different centering = new constant
        m.eval(0, XS, centering="zero_start")
    with budget(*counters, expected=2):  # frame change -> recompute secants
        m.fit(features=0, binning_method=ap.Fixed(nof_bins=5, min_points_per_bin=0))
    assert len(m.payload(0)["limits"]) == 6  # the stale-grid bug fix, pinned
    with budget(*counters, expected=0):  # same frame re-fitted = cache hit
        m.fit(features=0, binning_method=ap.Fixed(nof_bins=5, min_points_per_bin=0))


def test_l1_rhale_trace(global_data):
    m, counters = make_counting("rhale", global_data)

    with budget(*counters, expected=1):  # the whole (N, D) jacobian, shared
        m.fit(features=0, binning_method="dp")
    with budget(*counters, expected=0):
        m.plot(0, show_plot=False)
    with budget(*counters, expected=0):  # binning is a summary-stage param
        m.fit(features=0, binning_method="greedy")
    with budget(*counters, expected=0):  # the shared table covers feature 1
        m.fit(features=1)
        m.eval(1, XS)


def test_l1_shapdp_trace(global_data, monkeypatch):
    calls = {"n": 0}

    def fake_shap(model, data, *args, **kwargs):
        calls["n"] += 1
        return analytic_shap_values(data)

    monkeypatch.setattr(effector.global_effect_shap, "_compute_shap_values", fake_shap)
    model = CountingModel(linear_model)
    m = effector.ShapDP(global_data, model)

    m.fit(features=0)
    assert calls["n"] == 1  # the explainer ran exactly once
    with budget(model, expected=0):
        m.plot(0, heterogeneity="shap_values", show_plot=False)
        m.eval(0, XS, centering=False)
        m.fit(features=1)  # the (N, D) phi table is shared
        m.eval(1, XS)
    assert calls["n"] == 1  # ... and never again


# ---------------------------------------------------------------------------
# L2 — the categorical order frame (ALE): 2(L-1) per frame, 0 on re-fit
# ---------------------------------------------------------------------------

CAT_SCHEMA = {"feature_types": ["continuous", "continuous", "nominal"]}


def make_cat_data(n=90, seed=3):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-1, 1, n),
            rng.uniform(-1, 1, n),
            rng.integers(0, 3, n).astype(float),
        ],
        axis=1,
    )


def test_l2_categorical_order_frame():
    data = make_cat_data()
    model = CountingModel(linear_model)
    m = effector.ALE(data, model, schema=CAT_SCHEMA)
    L = 3
    n_pairs = L * (L - 1) // 2

    # nominal fit warms cache (a) — 2 calls per chain transition — AND
    # cache (a′) — 2 calls per level pair (the order-free scalar material)
    with budget(model, expected=2 * (L - 1) + 2 * n_pairs):
        m.fit(features=2)
    with budget(model, expected=0):  # same order = same frame (was: recompute)
        m.fit(features=2)
    # new order = new chain frame; the pairwise cache is order-free and stays
    with budget(model, expected=2 * (L - 1)):
        m.fit(features=2, order=[2.0, 0.0, 1.0])
    np.testing.assert_array_equal(m.payload(2)["levels"], [2.0, 0.0, 1.0])
    with budget(model, expected=0):  # scalars re-summarize (a′), no model
        m.heter_score(2)
        m.importance(2)


# ---------------------------------------------------------------------------
# L3 — state pokes: the decisions themselves, not just their cost
# ---------------------------------------------------------------------------


def test_l3_ale_frame_and_epoch(global_data):
    m = make_global("ale", global_data)
    m.fit(features=0)
    assert m._local[0]["frame"][0] == "fixed"
    e0 = m._epoch[0]
    m.eval(0, XS)  # queries never bump the epoch
    assert m._epoch[0] == e0
    m.fit(features=0, binning_method=ap.Fixed(nof_bins=5, min_points_per_bin=0))
    assert m._epoch[0] > e0  # frame replace bumped it
    assert m._local[0]["frame"] == ("fixed", 5, 0)


def test_l3_rhale_config_change_bumps_epoch_without_recompute(global_data):
    m = make_global("rhale", global_data)
    m.fit(features=0, binning_method="dp")
    entry0 = m._local[0]
    e0 = m._epoch[0]
    m.fit(features=0, binning_method="greedy")
    assert m._local[0] is entry0  # cache (a) untouched (no frame)
    assert m._epoch[0] > e0  # but dp-binned summaries are unreachable


def test_l3_pdp_store_grows_without_epoch_bump(global_data):
    m = make_global("pdp", global_data)
    m.fit(features=0)
    n0 = len(m._local[0]["pos"])
    e0 = m._epoch[0]
    h0 = m.eval_heter(0, XS)
    m.eval(0, np.array([0.123456, 0.654321]), centering=False)
    assert len(m._local[0]["pos"]) == n0 + 2  # grew by the missing positions
    assert m._epoch[0] == e0  # growth is additive: no invalidation
    np.testing.assert_array_equal(m.eval_heter(0, XS), h0)  # summaries stable


def test_l3_summaries_memoized_per_mask(global_data):
    m = make_global("ale", global_data)
    m.fit(features=0)
    mask = global_data[:, 0] > 0
    p1 = m._summary(0, mask)
    p2 = m._summary(0, mask)
    assert p1 is p2  # same key -> same object (memo hit)
    assert m._summary(0, None) is m._summary(0, np.ones(len(global_data), bool))


# ---------------------------------------------------------------------------
# L4 — avg-output: one y_pred per object, ever
# ---------------------------------------------------------------------------


def test_l4_avg_output_costs_one_call_total(global_data):
    m, counters = make_counting("ale", global_data)
    m.fit(features=0)
    with budget(*counters, expected=1):  # y_pred, once
        m.plot(0, show_avg_output=True, show_plot=False)
    with budget(*counters, expected=0):  # cached — masked variant included
        m.plot(0, show_avg_output=True, show_plot=False)
        m.plot(0, show_avg_output=True, show_plot=False, mask=global_data[:, 0] > 0)


# ---------------------------------------------------------------------------
# L5 — lazy ≡ eager, byte for byte, every method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_l5_lazy_equals_eager(name, global_data):
    eager = make_global(name, global_data)
    eager.fit(features=0)
    lazy = make_global(name, global_data)

    np.testing.assert_array_equal(eager.eval(0, XS), lazy.eval(0, XS))
    np.testing.assert_array_equal(eager.eval_heter(0, XS), lazy.eval_heter(0, XS))
    assert eager.heter_score(0) == lazy.heter_score(0)
    assert eager.importance(0) == lazy.importance(0)


# ---------------------------------------------------------------------------
# L6 — ToyEffect: a zero-cache-logic method gets the constitution for free
# ---------------------------------------------------------------------------


def test_l6_toy_method_full_constitution(global_data):
    model = CountingModel(linear_model)
    m = ToyEffect(global_data, model)

    with budget(model, expected=1):  # one prediction pass, ever
        m.fit(features=0, nof_bins=8)
    with budget(model, expected=0):  # plots, scores, masks, regions: free
        m.plot(0, heterogeneity=True, show_plot=False)
        m.eval(0, XS, centering="zero_integral")
        m.heter_score(0)
        m.importances()
        part = m.find_regions(0)
        for r in part.leaves:
            part.plot(r.idx, show_plot=False)
    with budget(model, expected=0):  # nof_bins is a summary knob, not a frame
        m.fit(features=0, nof_bins=4)
    assert len(m.payload(0)["limits"]) == 5

    # lazy ≡ eager holds for the toy too
    lazy = ToyEffect(make_global_data(), linear_model)
    eager = ToyEffect(make_global_data(), linear_model)
    eager.fit(features=0)
    np.testing.assert_array_equal(eager.eval(0, XS), lazy.eval(0, XS))


# ---------------------------------------------------------------------------
# L7 — repeated plots cost nothing, every method
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_l7_repeat_plot_costs_nothing(name, global_data):
    m, counters = make_counting(name, global_data)
    m.plot(0, show_plot=False)  # warm (lazy — no fit needed)
    with budget(*counters, expected=0):
        m.plot(0, show_plot=False)
        m.plot(0, show_plot=False)

"""CALM snapshots and select_regions (the reified decision sequence).

Pins the chain invariants — R² non-decreasing, sequential marginals summing
exactly to `regional_r2 − gam_r2` — plus the value-object contract: stamped
scalars equal the live masked verbs (the weighted-mean bridge), to_dict /
from_dict round-trips unbound and rebinds with mask verification, and
select_regions costs exactly one prediction pass beyond fit.
"""

import numpy as np
import pytest

import effector
from effector import explained_variance as ev
from tests.conftest import (
    CountingModel,
    budget,
    gated_model,
    make_regional_data,
)


def switch_model(x):
    # all of x0's interaction variance is convertible by an x2-split
    return x[:, 0] * np.where(x[:, 2] > 0, 3.0, 1.0) + x[:, 1]


def make_uniform(n=4000, d=3, seed=0):
    return np.random.default_rng(seed).uniform(-1, 1, (n, d))


def fitted_pdp(data, model):
    m = effector.PDP(data, model, nof_instances="all")
    m.fit(features="all")
    return m


@pytest.fixture(scope="module")
def switch_chain():
    data = make_uniform()
    m = fitted_pdp(data, switch_model)
    return m, m.select_regions()


# ---------------------------------------------------------------------------
# chain invariants
# ---------------------------------------------------------------------------


def test_chain_r2_is_non_decreasing_and_marginals_sum_exactly(switch_chain):
    _, chain = switch_chain
    r2s = [c.r2 for c in chain]
    assert all(b >= a for a, b in zip(r2s, r2s[1:]))
    gains = sum(st["delta_r2"] for st in chain.stages)
    np.testing.assert_allclose(gains, chain.regional_r2 - chain.gam_r2, atol=1e-12)
    # snapshots are cumulative: each adds exactly one split feature
    for prev, nxt in zip(chain, list(chain)[1:]):
        added = set(nxt.features) - set(prev.features)
        assert len(added) == 1 and next(iter(added)) == nxt.stage["feature"]


def test_gam_snapshot_equals_the_engine_globals(switch_chain):
    m, chain = switch_chain
    gam = chain.gam
    assert gam.is_gam and gam.index == 0 and gam.stage is None
    for f in range(m.dim):
        assert gam.importance(f) == pytest.approx(m.importance(f))
        assert gam.heter_score(f) == pytest.approx(m.heter_score(f))


def test_split_feature_scalars_are_the_weighted_mean_bridge(switch_chain):
    m, chain = switch_chain
    final = chain.final
    assert final.features, "the switch model must accept at least one split"
    for f in final.features:
        part = final.partitions[f]
        # the stamped heterogeneity reproduces _heter_pair's "after" exactly
        np.testing.assert_allclose(
            final.heter_score(f), ev._heter_pair(part)[1], atol=1e-12
        )
        # the stamped importance is the instance-weighted mean over the leaves
        w = np.array([leaf.nof_instances for leaf in part.leaves], dtype=float)
        vals = [m.importance(f, mask=part.mask(leaf.idx)) for leaf in part.leaves]
        np.testing.assert_allclose(
            final.importance(f), np.average(vals, weights=w), atol=1e-12
        )
        # per-region views agree with the masked verbs
        assert final.importance(f, per_region=True) == pytest.approx(vals)


def test_select_regions_accepts_the_known_split_on_the_gated_model():
    data = make_regional_data(n=800)
    m = fitted_pdp(data, gated_model)
    chain = m.select_regions()
    assert chain.regional_r2 > chain.gam_r2
    assert 0 in chain.final.features  # x0 is the gated feature
    for sk in chain.skipped:
        assert sk["reason"] in ("redundant", "below_threshold")


def test_min_r2_gain_prunes_the_chain():
    data = make_uniform()
    m = fitted_pdp(data, switch_model)
    # an impossible bar: nothing can add 90 pts, so the chain is the GAM alone
    chain = m.select_regions(min_r2_gain=0.9)
    assert len(chain) == 1 and chain.final.is_gam
    assert chain.skipped, "the pruned splits must be reported, not dropped"


# ---------------------------------------------------------------------------
# value-object contract
# ---------------------------------------------------------------------------


def test_chain_roundtrips_unbound_and_rebinds(switch_chain):
    m, chain = switch_chain
    import json

    d = json.loads(json.dumps(chain.to_dict()))  # a real serialization boundary
    chain2 = effector.CalmSequence.from_dict(d)
    # unbound: scalars, show, and the triage figure work from stamped values
    assert chain2.final.importances() == pytest.approx(chain.final.importances())
    assert chain2.final.heter_scores() == pytest.approx(chain.final.heter_scores())
    chain2.show()
    fig, _ = chain2.final.plot_triage(show_plot=False)
    assert fig is not None
    # unbound partitions refuse the live verbs...
    f = chain.final.features[0]
    with pytest.raises(RuntimeError):
        chain2.final.importance(f, per_region=True)
    # ...until bind reattaches (and verifies the masks)
    chain2.bind(m)
    assert chain2.final.importance(f, per_region=True) == pytest.approx(
        chain.final.importance(f, per_region=True)
    )


def test_bind_rejects_an_effect_with_different_data(switch_chain):
    _, chain = switch_chain
    other = fitted_pdp(make_uniform(seed=99), switch_model)
    rebuilt = effector.CalmSequence.from_dict(chain.to_dict())
    with pytest.raises(ValueError):
        rebuilt.bind(other)


def test_select_regions_raises_for_derivative_scale_methods():
    from tests.conftest import gated_model_jac

    data = make_regional_data(n=500)
    m = effector.DerPDP(data, gated_model, model_jac=gated_model_jac,
                        nof_instances="all")
    m.fit(features="all")
    with pytest.raises(ValueError, match="derivative-scale"):
        m.select_regions()


def test_summarize_is_a_thin_serializer_over_select_regions():
    data = make_regional_data(n=800)
    m = fitted_pdp(data, gated_model)
    parts = {
        m._resolve_feature(k): p
        for k, p in m.find_regions(features="heterogeneous").items()
    }
    supported = list(range(m.dim))
    flat = ev.summarize(m, parts, supported)
    d = m.select_regions(partitions=parts).to_dict()
    assert flat == {k: d[k] for k in flat}


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------


def test_select_regions_costs_one_prediction_pass_after_fit():
    data = make_regional_data(n=500)
    counting = CountingModel(gated_model)
    m = effector.PDP(data, counting, nof_instances="all")
    m.fit(features="all")
    fit_calls = counting.n_calls
    with budget(counting, expected=1):
        m.select_regions()
    # a second chain reuses the cached prediction — zero further calls
    with budget(counting, expected=0):
        m.select_regions()
    assert counting.n_calls == fit_calls + 1

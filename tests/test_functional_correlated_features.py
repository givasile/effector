"""F5 (LOGBOOK #8): CorrelatedInteraction (Gkolemis et al. 2023) ground truths.

f(x) = sin(2 pi x1) (1{x1<0} - 2 * 1{x3<0}) + x1 x2 + x2, with x3 strongly
correlated with x1 (x3 = x1 + N(0, 0.01)) and x1 mixture-uniform (5/6 of the
mass below 0).

The only pair where the methods provably *differ* — each assert locks that
method's own handling of correlation:
- PDP averages over the marginal of x3 (the -2 sin term acts with prob. 5/6
  everywhere),
- ALE/RHALE condition on x3 ~ x1 (the term acts only where x1 < 0, flipping
  the sine there),
- d-PDP locks the derivative-space PDP.

Tolerances are per-half: only ~1/6 of the mass lies at x1 > 0, so estimates
there are intrinsically noisier (calibrated at seed 21, N=170 — the pair's
canonical sample size from the paper).

SHAP: notebook 02's closed form (-5/6 sin(2 pi x)) does NOT match interventional
Shapley — an exact brute-force computation and the shap package agree with each
other (max residual ~0.29, not shrinking with N) and both disagree with it.
The closed form needs re-derivation; until then the SHAP assert is skipped
(see benchmarks.CorrelatedInteraction.shap_gt docstring).
"""

import numpy as np
import pytest

import effector

N = 170
XS = np.linspace(-0.5, 0.5, 100)
DENSE = XS <= 0  # 5/6 of the x1 mass


@pytest.fixture(scope="module")
def bench():
    return effector.benchmarks.CorrelatedInteraction()


@pytest.fixture(scope="module")
def data(bench):
    return bench.generate_data(N, seed=21)


def assert_per_half(y, gt, atol_dense, atol_sparse):
    np.testing.assert_allclose(y[DENSE], gt[DENSE], atol=atol_dense)
    np.testing.assert_allclose(y[~DENSE], gt[~DENSE], atol=atol_sparse)


def test_pdp(bench, data):
    pdp = effector.PDP(data, bench.predict, axis_limits=bench.axis_limits)
    y = pdp.eval(feature=0, xs=XS, centering=True)
    np.testing.assert_allclose(y, bench.pdp_gt(XS), atol=1e-1)


def test_d_pdp(bench, data):
    dpdp = effector.DerPDP(
        data, bench.predict, model_jac=bench.jacobian, axis_limits=bench.axis_limits
    )
    y = dpdp.eval(feature=0, xs=XS, centering=False)
    # amplitude is ~10.5 (derivative space); 0.5 is ~5% relative
    np.testing.assert_allclose(y, bench.d_pdp_gt(XS), atol=5e-1)


def test_ale(bench, data):
    ale = effector.ALE(data, bench.predict, axis_limits=bench.axis_limits)
    ale.fit(
        features=0, binning_method=effector.axis_partitioning.Fixed(nof_bins=31)
    )
    y = ale.eval(feature=0, xs=XS, centering=True)
    assert_per_half(y, bench.ale_gt(XS), atol_dense=1.5e-1, atol_sparse=3e-1)


def test_rhale(bench, data):
    rhale = effector.RHALE(
        data, bench.predict, bench.jacobian, axis_limits=bench.axis_limits
    )
    rhale.fit(features=0)
    y = rhale.eval(feature=0, xs=XS, centering=True, heterogeneity=False)
    assert_per_half(y, bench.rhale_gt(XS), atol_dense=2e-1, atol_sparse=4e-1)


def test_methods_provably_differ(bench):
    """The point of this pair: PDP and ALE ground truths disagree by design.

    On the positive half PDP follows -5/3 sin(2 pi x) while ALE is flat 0 —
    if a refactor ever collapses conditional into marginal averaging (or vice
    versa), the per-method asserts above fail; this assert documents that the
    two targets themselves are far apart, so those failures cannot cancel."""
    gap = np.abs(bench.pdp_gt(XS) - bench.ale_gt(XS))
    assert gap.max() > 1.0


@pytest.mark.skip(
    reason="notebook 02's SHAP closed form (-5/6 sin) does not match "
    "interventional Shapley: exact brute-force and the shap package agree "
    "with each other and both disagree with it (max residual ~0.29, "
    "N-independent). Re-derive the closed form, then wire this in."
)
def test_shap(bench, data):
    sh = effector.ShapDP(
        data, bench.predict, axis_limits=bench.axis_limits, nof_instances="all"
    )
    sh.fit(features=0)
    y = sh.eval(feature=0, xs=XS, centering=True, heterogeneity=False)
    np.testing.assert_allclose(y, bench.shap_gt(XS), atol=2e-1)

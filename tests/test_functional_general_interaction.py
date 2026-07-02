"""F3 (LOGBOOK #8): GeneralInteractionUniform ground truths.

f(x) = x1 * x2^2 + e^{x3},  x ~ U[-1, 1]^3 iid.

Unlike the conditional-interaction pair, all methods agree here — what this
locks is *averaging correctness*: x1's mean effect has slope exactly
E[x2^2] = 1/3 (it exists only because the method integrates x2^2 over the
marginal correctly), and the interaction is invisible in x2's mean effect
(E[x1] = 0) while fully visible in its heterogeneity — h(x2) closed form is
new here (LOGBOOK #8 F3 addendum), it is not derived in notebook 06.
"""

import numpy as np
import pytest

import effector

N = 1_000
N_HETER = 10_000
NOF_BINS = 31
ATOL = 1e-1
XS = np.linspace(-1, 1, 100)


@pytest.fixture(scope="module")
def bench():
    return effector.benchmarks.GeneralInteractionUniform()


@pytest.fixture(scope="module")
def data(bench):
    return bench.generate_data(N, seed=21)


@pytest.fixture(scope="module")
def pdp(bench, data):
    method = effector.PDP(data, bench.model.predict, axis_limits=bench.axis_limits)
    method.fit(features="all", centering=True)
    return method


@pytest.fixture(scope="module")
def ale(bench, data):
    method = effector.ALE(data, bench.model.predict, axis_limits=bench.axis_limits)
    method.fit(
        features="all",
        centering=True,
        binning_method=effector.axis_partitioning.Fixed(nof_bins=NOF_BINS),
    )
    return method


@pytest.fixture(scope="module")
def rhale(bench, data):
    method = effector.RHALE(
        data, bench.model.predict, bench.model.jacobian, axis_limits=bench.axis_limits
    )
    method.fit(
        features="all",
        centering=True,
        binning_method=effector.axis_partitioning.Fixed(nof_bins=NOF_BINS),
    )
    return method


class TestGlobalEffects:
    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_pdp(self, pdp, bench, feature):
        y = pdp.eval(feature=feature, xs=XS, centering=True)
        np.testing.assert_allclose(y, bench.pdp_gt(feature, XS), atol=ATOL)

    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_ale(self, ale, bench, feature):
        y = ale.eval(feature=feature, xs=XS, centering=True)
        np.testing.assert_allclose(y, bench.ale_gt(feature, XS), atol=ATOL)

    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_rhale(self, rhale, bench, feature):
        y = rhale.eval(feature=feature, xs=XS, centering=True)
        np.testing.assert_allclose(y, bench.rhale_gt(feature, XS), atol=ATOL)


class TestHeterogeneity:
    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_pdp_heterogeneity(self, bench, feature):
        data = bench.generate_data(N_HETER, seed=21)
        pdp = effector.PDP(data, bench.model.predict, axis_limits=bench.axis_limits)
        _, heter = pdp.eval(feature=feature, xs=XS, centering=True, heterogeneity=True)
        np.testing.assert_allclose(heter, bench.pdp_heter_gt(feature, XS), atol=ATOL)

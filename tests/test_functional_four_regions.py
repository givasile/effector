"""F4 (LOGBOOK #8): ConditionalInteraction4RegionsUniform ground truths.

f(x) = {-x1^2, +x1^2, -x1^4, +x1^4} gated by the signs of (x2, x3), plus e^{x4},
x ~ U[-1, 1]^4 iid.

What only this pair covers: D=4 (where feature-indexing bugs hide — B10, the
model's exp term reading x3 instead of x4, was found by exactly these asserts),
the x1^2-vs-x1^4 distinction between regions, and the only *two-level* regional
ground truth: level 1 must split on x3 (the sign gate), level 2 on x2 (the
power gate), yielding 4 leaves with deterministic effects.

RHALE tolerances are per-feature: x1's RHALE is 0 only in expectation — the
per-bin means of +-2x / +-4x^3 carry O(1/sqrt(N)) noise that accumulates
through the integration (~0.11 at N=1000, shrinking ~1/sqrt(N)); the other
features' ground truths are deterministic and stay tight.
"""

import numpy as np
import pytest

import effector

N = 1_000
NOF_BINS = 31
ATOL = 1e-1
# feature 0's ALE/RHALE are 0 only in expectation (see module docstring)
ALE_ATOL = {0: 1.5e-1, 1: 1e-1, 2: 1e-1, 3: 1e-1}
RHALE_ATOL = {0: 1.5e-1, 1: 1e-2, 2: 1e-2, 3: 1e-2}
# per-leaf heterogeneity: RHALE reports the within-bin std of the derivative;
# for the x1^4 leaves the edge bins carry ~0.07 of real within-bin variation
# (4x^3 varies fastest near |x|=1) plus std-estimation noise at ~8 points per
# bin per leaf — measured up to ~0.16 at seed 21. This is a resolution
# artifact, not region inhomogeneity; the <5%-of-root ratio test below carries
# the substantive "regions are homogeneous" claim.
LEAF_HETER_ATOL = 2e-1
XS = np.linspace(-1, 1, 100)


@pytest.fixture(scope="module")
def bench():
    return effector.benchmarks.ConditionalInteraction4RegionsUniform()


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


def mask_jump_bin(y, xs, nof_bins=NOF_BINS):
    """Zero out the bin containing the discontinuity at 0 (binned methods
    interpolate the jump inside it — an artifact, not an error)."""
    y = y.copy()
    y[np.logical_and(xs > -1 / nof_bins, xs < 1 / nof_bins)] = 0
    return y


class TestGlobalEffects:
    @pytest.mark.parametrize("feature", [0, 1, 2, 3])
    def test_pdp(self, pdp, bench, feature):
        y = pdp.eval(feature=feature, xs=XS, centering=True)
        np.testing.assert_allclose(y, bench.pdp_gt(feature, XS), atol=ATOL)

    @pytest.mark.parametrize("feature", [0, 1, 2, 3])
    def test_ale(self, ale, bench, feature):
        y = ale.eval(feature=feature, xs=XS, centering=True)
        gt = bench.ale_gt(feature, XS)
        if feature == 2:
            y, gt = mask_jump_bin(y, XS), mask_jump_bin(gt, XS)
        np.testing.assert_allclose(y, gt, atol=ALE_ATOL[feature])

    @pytest.mark.parametrize("feature", [0, 1, 2, 3])
    def test_rhale(self, rhale, bench, feature):
        y = rhale.eval(feature=feature, xs=XS, centering=True)
        np.testing.assert_allclose(
            y, bench.rhale_gt(feature, XS), atol=RHALE_ATOL[feature]
        )


class TestRegionalEffects:
    """The only multi-level regional ground truth in the suite.

    Built on GLOBAL effects + `find_regions(finder=Best(max_depth=2))`, which
    yields a 2-level `Partition` (root + 2 + 4 = 7 regions). The hierarchy is
    walked via `region.parent_idx` / `part[region.parent_idx]`.
    """

    @pytest.fixture(scope="class", params=["pdp", "rhale"])
    def fitted(self, request, bench, data):
        if request.param == "pdp":
            fx = effector.PDP(
                data, bench.model.predict, axis_limits=bench.axis_limits
            )
        else:
            fx = effector.RHALE(
                data,
                bench.model.predict,
                bench.model.jacobian,
                axis_limits=bench.axis_limits,
            )
        fx.fit(0, centering=True)
        return fx.find_regions(
            0, finder=effector.space_partitioning.Best(max_depth=2)
        )

    def test_two_levels_sign_gate_then_power_gate(self, fitted, bench):
        part = fitted
        # root + 2 + 4 = 7 regions in a two-level tree
        assert len(part) == 7
        level1 = [r for r in part if r.level == 1]
        level2 = [r for r in part if r.level == 2]
        assert len(level1) == 2 and len(level2) == 4
        for region in level1:
            assert region.foc_index == bench.regional_level1_split_feature
            assert abs(region.foc_split_position) <= 0.15
        for region in level2:
            assert region.foc_index == bench.regional_level2_split_feature
            assert abs(region.foc_split_position) <= 0.15

    def test_leaf_effects_match_the_four_regions(self, fitted, bench):
        part = fitted
        for region in (r for r in part if r.level == 2):
            parent = part[region.parent_idx]
            x3_side = "left" if parent.comparison == "<=" else "right"
            x2_side = "left" if region.comparison == "<=" else "right"
            y = part.eval(region.idx, XS, centering=True)
            heter = part.eval_heter(region.idx, XS)
            gt = bench.regional_effect_gt(x2_side, x3_side, XS)
            np.testing.assert_allclose(y, gt, atol=ATOL)
            np.testing.assert_allclose(heter, np.zeros_like(XS), atol=LEAF_HETER_ATOL)

    def test_partition_removes_the_heterogeneity(self, fitted):
        part = fitted
        root = part[0]
        for region in (r for r in part if r.level == 2):
            assert region.heterogeneity < 0.05 * root.heterogeneity

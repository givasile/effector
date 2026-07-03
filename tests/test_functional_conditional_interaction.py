"""F1/F2 (LOGBOOK #8): ConditionalInteractionUniform ground truths.

f(x) = -x1^2 1{x2<0} + x1^2 1{x2>=0} + e^{x3},  x ~ U[-1, 1]^3 iid.

F1 — global: centered PDP/ALE/RHALE and heterogeneity vs closed form. The x2
row locks method *semantics*: PDP/ALE see a -+1/3 step, RHALE (zero derivative
a.e.) sees 0.
F2 — regional: the x1 heterogeneity is fully explained by sign(x2), so the
optimal split is on x2 at 0 and the per-region effects are deterministic (+-x1^2).
"""

import numpy as np
import pytest

import effector

N = 1_000
N_HETER = 10_000  # variance GTs need the notebook's N: at 2k the big bins are ~7% off
NOF_BINS = 31
ATOL = 1e-1
XS = np.linspace(-1, 1, 100)


@pytest.fixture(scope="module")
def bench():
    return effector.benchmarks.ConditionalInteractionUniform()


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
    """Zero out the bin containing the discontinuity at 0: inside it the
    binned estimate interpolates the jump, which is an artifact, not an error."""
    y = y.copy()
    y[np.logical_and(xs > -1 / nof_bins, xs < 1 / nof_bins)] = 0
    return y


class TestGlobalEffects:
    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_pdp(self, pdp, bench, feature):
        y = pdp.eval(feature=feature, xs=XS, centering=True)
        np.testing.assert_allclose(y, bench.pdp_gt(feature, XS), atol=ATOL)

    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_ale(self, ale, bench, feature):
        y = ale.eval(feature=feature, xs=XS, centering=True)
        gt = bench.ale_gt(feature, XS)
        if feature == 1:
            y, gt = mask_jump_bin(y, XS), mask_jump_bin(gt, XS)
        np.testing.assert_allclose(y, gt, atol=ATOL)

    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_rhale(self, rhale, bench, feature):
        y = rhale.eval(feature=feature, xs=XS, centering=True)
        np.testing.assert_allclose(y, bench.rhale_gt(feature, XS), atol=ATOL)


class TestHeterogeneity:
    @pytest.fixture(scope="class")
    def data_heter(self, bench):
        return bench.generate_data(N_HETER, seed=21)

    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_pdp_heterogeneity(self, bench, data_heter, feature):
        pdp = effector.PDP(
            data_heter, bench.model.predict, axis_limits=bench.axis_limits
        )
        _, heter = pdp.eval(feature=feature, xs=XS, centering=True, heterogeneity=True)
        np.testing.assert_allclose(heter, bench.pdp_heter_gt(feature, XS), atol=ATOL)

    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_ale_bin_variance(self, bench, data_heter, feature):
        ale = effector.ALE(
            data_heter, bench.model.predict, axis_limits=bench.axis_limits
        )
        ale.fit(
            features=feature,
            binning_method=effector.axis_partitioning.Fixed(nof_bins=NOF_BINS),
        )
        bin_var = ale.feature_effect[f"feature_{feature}"]["bin_variance"]
        gt_var = bench.ale_bin_variance_gt(feature, nof_bins=NOF_BINS)
        mask = ~np.isnan(gt_var)  # the jump bin's variance is an artifact
        np.testing.assert_allclose(bin_var[mask], gt_var[mask], atol=ATOL)

    @pytest.mark.parametrize("feature", [0, 1, 2])
    def test_rhale_bin_variance(self, bench, data_heter, feature):
        rhale = effector.RHALE(
            data_heter,
            bench.model.predict,
            bench.model.jacobian,
            axis_limits=bench.axis_limits,
        )
        rhale.fit(
            features=feature,
            binning_method=effector.axis_partitioning.Fixed(nof_bins=NOF_BINS),
        )
        bin_var = rhale.feature_effect[f"feature_{feature}"]["bin_variance"]
        gt_var = bench.rhale_bin_variance_gt(feature, nof_bins=NOF_BINS)
        np.testing.assert_allclose(bin_var, gt_var, atol=ATOL)


class TestRegionalEffects:
    """F2: the strongest guard for the regional refactor (PLAN III steps 2.5-2.7)."""

    @pytest.fixture(scope="class", params=["pdp", "ale", "rhale"])
    def fitted(self, request, bench, data):
        if request.param == "pdp":
            reg = effector.RegionalPDP(
                data, bench.model.predict, axis_limits=bench.axis_limits
            )
        elif request.param == "ale":
            reg = effector.RegionalALE(
                data, bench.model.predict, axis_limits=bench.axis_limits
            )
        else:
            reg = effector.RegionalRHALE(
                data,
                bench.model.predict,
                bench.model.jacobian,
                axis_limits=bench.axis_limits,
            )
        reg.fit(0)
        return reg

    def test_split_is_on_x2_at_zero(self, fitted, bench):
        tree = fitted.tree["feature_0"]
        children = [n for n in tree.nodes if n.info["level"] == 1]
        assert len(children) == 2
        for node in children:
            assert node.info["foc_index"] == bench.regional_split_feature
            assert (
                abs(node.info["foc_split_position"] - bench.regional_split_position)
                <= 0.15
            )

    def test_region_effects_are_plus_minus_x_squared(self, fitted, bench):
        tree = fitted.tree["feature_0"]
        children = [n for n in tree.nodes if n.info["level"] == 1]
        for node in children:
            side = "left" if node.info["comparison"] == "<=" else "right"
            y, heter = fitted.eval(0, node.idx, XS, heterogeneity=True, centering=True)
            gt = bench.regional_effect_gt(side, XS)
            np.testing.assert_allclose(y, gt, atol=ATOL)
            np.testing.assert_allclose(heter, np.zeros_like(XS), atol=ATOL)

    def test_split_removes_the_heterogeneity(self, fitted):
        tree = fitted.tree["feature_0"]
        root = next(n for n in tree.nodes if n.info["level"] == 0)
        children = [n for n in tree.nodes if n.info["level"] == 1]
        for node in children:
            assert node.info["heterogeneity"] < 0.1 * root.info["heterogeneity"]

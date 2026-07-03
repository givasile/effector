"""Contract layer, string-argument registries (PLAN II §3.1 / R5, R6, R9).

One menu per concept (LOGBOOK #5): the accepted strings, the resolver, and the
per-method restrictions must all agree.  Green = holds today;
``xfail(strict=True)`` = the refactor must make it true (B2 and the R9
error-type conversion are the known offenders).
"""

import numpy as np
import pytest

import effector
import effector.axis_partitioning as ap
import effector.space_partitioning as sp
from effector import helpers
from tests.conftest import linear_model, linear_model_jac, make_global, make_global_data


@pytest.fixture(scope="module")
def data():
    return make_global_data()


# ---------------------------------------------------------------------------
# binning strings per method (R6)
# ---------------------------------------------------------------------------


def test_ale_accepts_fixed_only(data):
    m = effector.ALE(data, linear_model)
    m.fit(features=0, binning_method="fixed")

    m2 = effector.ALE(data, linear_model)
    m2.fit(features=0, binning_method=ap.Fixed(nof_bins=11))

    for bad in ["greedy", "dp", "junk"]:
        with pytest.raises((AssertionError, ValueError)):
            effector.ALE(data, linear_model).fit(features=0, binning_method=bad)


@pytest.mark.parametrize("binning", ["fixed", "greedy"])
def test_rhale_accepts_fixed_and_greedy(data, binning):
    m = effector.RHALE(data, linear_model, model_jac=linear_model_jac)
    m.fit(features=0, binning_method=binning)


@pytest.mark.xfail(
    strict=True,
    reason="B2: 'dp' fails RHALE's assert although return_default resolves it "
    "-> DP binning unreachable by string",
)
def test_rhale_accepts_dp_string(data):
    m = effector.RHALE(data, linear_model, model_jac=linear_model_jac)
    m.fit(features=0, binning_method="dp")


@pytest.mark.parametrize(
    "binning",
    [ap.Fixed(nof_bins=11), ap.Greedy(), ap.DynamicProgramming()],
    ids=["fixed-inst", "greedy-inst", "dp-inst"],
)
def test_rhale_accepts_instances(data, binning):
    m = effector.RHALE(data, linear_model, model_jac=linear_model_jac)
    m.fit(features=0, binning_method=binning)


def test_rhale_junk_binning_raises(data):
    m = effector.RHALE(data, linear_model, model_jac=linear_model_jac)
    with pytest.raises((AssertionError, ValueError)):
        m.fit(features=0, binning_method="junk")


@pytest.mark.parametrize("binning", ["fixed", "greedy"])
def test_shapdp_accepts_fixed_and_greedy(data, binning):
    m = make_global("shapdp", data)
    m.fit(features=0, binning_method=binning)


def test_shapdp_junk_binning_raises(data):
    m = make_global("shapdp", data)
    with pytest.raises((AssertionError, ValueError)):
        m.fit(features=0, binning_method="junk")


# ---------------------------------------------------------------------------
# resolver round-trips (R6)
# ---------------------------------------------------------------------------


def test_axis_partitioning_return_default_roundtrip():
    assert isinstance(ap.return_default("fixed"), ap.Fixed)
    assert isinstance(ap.return_default("greedy"), ap.Greedy)
    assert isinstance(ap.return_default("dp"), ap.DynamicProgramming)
    inst = ap.Fixed(nof_bins=7)
    assert ap.return_default(inst) is inst
    with pytest.raises((AssertionError, ValueError)):
        ap.return_default("junk")


def test_space_partitioning_return_default_roundtrip():
    assert isinstance(sp.return_default("best"), sp.Best)
    assert isinstance(sp.return_default("best_level_wise"), sp.BestLevelWise)
    with pytest.raises(ValueError):
        sp.return_default("junk")


# ---------------------------------------------------------------------------
# centering strings (R3)
# ---------------------------------------------------------------------------


def test_prep_centering_vocabulary():
    assert helpers.prep_centering(False) is False
    assert helpers.prep_centering(True) == "zero_integral"
    assert helpers.prep_centering("zero_integral") == "zero_integral"
    assert helpers.prep_centering("zero_start") == "zero_start"
    with pytest.raises((AssertionError, ValueError)):
        helpers.prep_centering("zero_mean")


# ---------------------------------------------------------------------------
# prep_features (spec: out-of-range must be rejected)
# ---------------------------------------------------------------------------


def test_prep_features_variants():
    assert helpers.prep_features("all", 3) == [0, 1, 2]
    assert helpers.prep_features(1, 3) == [1]
    assert helpers.prep_features([0, 2], 3) == [0, 2]


@pytest.mark.xfail(
    strict=True,
    reason="spec (PLAN II §3.1): prep_features must reject out-of-range indices",
)
def test_prep_features_rejects_out_of_range():
    with pytest.raises(ValueError):
        helpers.prep_features(5, 3)


# ---------------------------------------------------------------------------
# R9 — user-input errors must be ValueError/TypeError, not bare assert
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="R9: user-input rejection must raise ValueError, not AssertionError",
)
@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: helpers.prep_centering("zero_mean"), id="centering"),
        pytest.param(lambda: ap.return_default("junk"), id="binning"),
        pytest.param(lambda: helpers.prep_nof_instances("some", 100), id="nof-inst"),
    ],
)
def test_r9_valueerror_for_user_input(call):
    with pytest.raises(ValueError):
        call()


def _r9_probe():
    """Representative eval-path probe for R9: junk centering string."""
    data = make_global_data(n=50)
    m = effector.PDP(data, linear_model)
    m.eval(0, np.linspace(-0.5, 0.5, 5), centering="zero_mean")


@pytest.mark.xfail(
    strict=True,
    reason="R9: junk centering through eval must raise ValueError",
)
def test_r9_eval_junk_centering_is_valueerror():
    with pytest.raises(ValueError):
        _r9_probe()

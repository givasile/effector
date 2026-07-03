"""F7 (LOGBOOK #8): every global method x backend x jac-path on one GAM.

f(x) = x1^3/5 + x2^2/5, x ~ U(0, 1)^2 iid — additive, no interactions, zero
heterogeneity. For an additive model with independent features every method's
mean effect equals the (centered) component function, and the SHAP values do
too, so one pair covers all five classes and both SHAP backends.

Replaces test_functional_linear.py (its DerPDP case compared the derivative
effect against the *effect* — wrong, and unnoticed because the old asserts
were no-op `np.allclose` without `assert`) and the no-op test_functional_gam.py.

Test-local ground truths (no benchmark object): the closed forms are the
component functions themselves and there is no notebook twin to share with.
"""

import numpy as np
import pytest

import effector

N = 1_000
NOF_INSTANCES = 100
XS = np.linspace(0, 1, 100)
ATOL = 1e-1


def model(x):
    return x[:, 0] ** 3 / 5 + x[:, 1] ** 2 / 5


def model_jac(x):
    return np.stack([3 * x[:, 0] ** 2 / 5, 2 * x[:, 1] / 5], axis=1)


# zero_start-centered ground truths (all components are 0 at x=0 already)
GT = {
    "effect": {0: XS**3 / 5, 1: XS**2 / 5},
    "derivative": {0: 3 * XS**2 / 5, 1: 2 * XS / 5},
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(21)
    return rng.uniform(0, 1, size=(N, 2))


CASES = [
    pytest.param("pdp", {}, "effect", id="pdp"),
    pytest.param("derpdp", {}, "derivative", id="derpdp-nojac"),
    pytest.param("derpdp", {"use_jac": True}, "derivative", id="derpdp-jac"),
    pytest.param("ale", {}, "effect", id="ale"),
    pytest.param("rhale", {}, "effect", id="rhale-nojac"),
    pytest.param("rhale", {"use_jac": True}, "effect", id="rhale-jac"),
    pytest.param(
        "shapdp",
        {"backend": "shap"},
        "effect",
        id="shapdp-shap",
        marks=pytest.mark.slow,
    ),
    pytest.param(
        "shapdp",
        {"backend": "shapiq"},
        "effect",
        id="shapdp-shapiq",
        marks=pytest.mark.slow,
    ),
]


def make_method(kind, opts, data, nof_instances=NOF_INSTANCES):
    if kind == "pdp":
        return effector.PDP(data, model, nof_instances=nof_instances)
    if kind == "derpdp":
        jac = model_jac if opts.get("use_jac") else None
        return effector.DerPDP(data, model, model_jac=jac, nof_instances=nof_instances)
    if kind == "ale":
        return effector.ALE(data, model, nof_instances=nof_instances)
    if kind == "rhale":
        jac = model_jac if opts.get("use_jac") else None
        return effector.RHALE(data, model, model_jac=jac, nof_instances=nof_instances)
    if kind == "shapdp":
        # SHAP cost scales with instances; 50 keeps both backends within the
        # tier-2 budget (PLAN II 5) without loosening the tolerance
        return effector.ShapDP(data, model, nof_instances=50, backend=opts["backend"])


@pytest.mark.parametrize("kind,opts,gt_key", CASES)
@pytest.mark.parametrize("feature", [0, 1])
def test_effect_matches_component(kind, opts, gt_key, feature, data):
    method = make_method(kind, opts, data)
    y = method.eval(feature, XS, centering="zero_start")
    heter = method.eval_heter(feature, XS)
    np.testing.assert_allclose(y, GT[gt_key][feature], atol=ATOL)
    np.testing.assert_allclose(heter, np.zeros_like(XS), atol=ATOL)


@pytest.mark.parametrize("kind", ["derpdp", "rhale"])
@pytest.mark.parametrize("feature", [0, 1])
def test_jac_and_nojac_paths_agree(kind, feature, data):
    """The finite-difference fallback must estimate the same thing as the
    analytic jacobian — tighter than the GT tolerance on purpose."""
    with_jac = make_method(kind, {"use_jac": True}, data)
    without = make_method(kind, {}, data)
    y_jac = with_jac.eval(feature=feature, xs=XS, centering="zero_start")
    y_fd = without.eval(feature=feature, xs=XS, centering="zero_start")
    np.testing.assert_allclose(y_jac, y_fd, atol=1e-2)


def test_shap_tiny_in_the_gate(data):
    """One fast SHAP case so `make test` is never SHAP-blind (PLAN II 3.3)."""
    method = effector.ShapDP(data[:50], model, nof_instances=50, backend="shap")
    method.fit(features=0, budget=128)
    y = method.eval(feature=0, xs=XS, centering="zero_start")
    np.testing.assert_allclose(y, GT["effect"][0], atol=1.5e-1)

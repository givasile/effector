"""F8 (LOGBOOK #8): every regional method x backend on one gated-linear model.

f(x) = 5 x0 if (x1 > 0 and x2 == 0) else 0, with x0, x1 ~ U(-1, 1) and x2
binary in {0, 1} — the only discrete feature in the functional suite, so this
also locks the categorical split path ('==' / '!=' comparisons).

The x0 heterogeneity is fully explained by the gate, so the fitted tree must
isolate the active region in two levels (x1 > 0, then x2 == 0); inside it the
effect is exactly 5 x0 (derivative: the constant 5) with zero heterogeneity,
and every other leaf is exactly flat.

Replaces the no-op test_regional_methods.py: real asserts, the target leaf is
*derived from the fitted tree* instead of hardcoding node_idx=3 (which crashed
notebook 04 when the tree came out smaller), and a per-method parametrization
so a failure names the culprit.

Test-local ground truths (no benchmark object): trivial closed form, no
notebook twin.
"""

import numpy as np
import pytest

import effector

N = 1_000
XS = np.linspace(-1, 1, 100)


def model(x):
    y = np.zeros_like(x[:, 0])
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind] = 5 * x[ind, 0]
    return y


def model_jac(x):
    y = np.zeros_like(x)
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind, 0] = 5
    return y


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(21)
    return np.stack(
        [
            rng.uniform(-1, 1, N),
            rng.uniform(-1, 1, N),
            rng.integers(0, 2, N).astype(float),
        ],
        axis=1,
    )


CASES = [
    pytest.param("pdp", {}, id="pdp"),
    pytest.param("derpdp", {}, id="derpdp"),
    pytest.param("ale", {}, id="ale"),
    pytest.param("rhale", {}, id="rhale"),
    pytest.param(
        "shapdp", {"backend": "shap"}, id="shapdp-shap", marks=pytest.mark.slow
    ),
    pytest.param(
        "shapdp",
        {"backend": "shapiq"},
        id="shapdp-shapiq",
        marks=pytest.mark.slow,
    ),
]


def make_fitted(kind, opts, data):
    """Build the GLOBAL effect, fit feature 0, and return its find_regions Partition."""
    finder = effector.space_partitioning.Best(max_depth=2)
    if kind == "pdp":
        fx = effector.PDP(data, model, nof_instances=N)
    elif kind == "derpdp":
        fx = effector.DerPDP(data, model, model_jac=model_jac, nof_instances=N)
    elif kind == "ale":
        fx = effector.ALE(data, model, nof_instances=N)
    elif kind == "rhale":
        fx = effector.RHALE(data, model, model_jac=model_jac, nof_instances=N)
    elif kind == "shapdp":
        # 100 instances: at 50 the subsample is too thin for the partitioner
        # to find the second (categorical) split at all.
        # Unseeded explainers make even the fitted tree vary between runs, so
        # seed both backends via shap_explainer_kwargs (effector exposes no
        # first-class seed — HOMOGENIZATION candidate).
        np.random.seed(0)
        seed_kw = {"shap": {"seed": 0}, "shapiq": {"random_state": 0}}[opts["backend"]]
        fx = effector.ShapDP(
            data,
            model,
            nof_instances=100,
            backend=opts["backend"],
            shap_explainer_kwargs=seed_kw,
        )
        fx.fit(0, centering=False)
        return fx.find_regions(0, finder=finder)
    fx.fit(0, centering=False)
    return fx.find_regions(0, finder=finder)


def find_active_leaf(part):
    """The leaf region whose rule is {x1 >= split, x2 == 0} — derived, not
    hardcoded. The region's rule encodes the whole root->leaf path: the right
    side of the x1 split is a lower-bounded interval, the '==' side of the x2
    split a singleton level set."""

    def is_active(region):
        iv = region.rule.get(1)
        ls = region.rule.get(2)
        return (
            isinstance(iv, effector.rules.Interval)
            and np.isfinite(iv.lo)
            and isinstance(ls, effector.rules.LevelSet)
            and ls.levels == frozenset({0.0})
        )

    matches = [r for r in part if is_active(r)]
    assert len(matches) == 1, f"expected exactly one active leaf, got {len(matches)}"
    return matches[0]


@pytest.mark.parametrize("kind,opts", CASES)
def test_active_region_effect(kind, opts, data):
    part = make_fitted(kind, opts, data)

    # the gate must be found: a split on x1 near 0, then x2 == 0
    leaf = find_active_leaf(part)
    assert set(leaf.rule.features) == {1, 2}

    if kind == "derpdp":
        gt = np.full_like(XS, 5.0)  # derivative of 5*x0
        y = part.eval(leaf.idx, XS, centering=False)
        heter = part.eval_heter(leaf.idx, XS)
        atol = 1e-1
    elif kind == "shapdp":
        # find_regions ShapDP does NOT recompute shap values within the region:
        # it subsets the precomputed GLOBAL attributions (global_shap_values).
        # For gate instances the exact interventional Shapley slope is
        # 5*(1/3*Pg + 1/6*P2 + 1/6*P1 + 1/3) = 5*7/12 ~ 2.9
        # (P1 = P(x1>0) = P2 = P(x2=0) = 1/2) — NOT 5, which recompute-within-
        # region semantics would give. Subsample-P jitter and the spline
        # aggregation move the realized slope by a few tenths, so the assert
        # is at slope granularity: near 35/12, decisively far from 5.
        y = part.eval(leaf.idx, XS, centering=True)
        heter = part.eval_heter(leaf.idx, XS)
        A = np.vstack([XS, np.ones_like(XS)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        assert 2.4 < slope < 3.5, f"slope {slope:.2f}: not subset-global semantics?"
        assert abs(intercept) < 0.2
        residuals = y - (slope * XS + intercept)
        np.testing.assert_allclose(residuals[3:-3], 0, atol=3e-1)  # linear-ish
        np.testing.assert_allclose(heter[3:-3], 0, atol=3e-1)
        return
    else:
        gt = 5 * XS  # already zero-integral over [-1, 1]
        y = part.eval(leaf.idx, XS, centering=True)
        heter = part.eval_heter(leaf.idx, XS)
        atol = 1e-1
    np.testing.assert_allclose(y, gt, atol=atol)
    np.testing.assert_allclose(heter, np.zeros_like(heter), atol=atol)


@pytest.mark.parametrize("kind,opts", CASES[:2])
def test_inactive_leaves_are_flat(kind, opts, data):
    """ICE-based methods only: the empirical split position (~0.003, not
    exactly 0) leaks a sliver of active points into the inactive branch, which
    makes the derivative-based trees legitimately split it again — their
    'inactive' leaves are not flat by construction, only PDP/DerPDP's are."""
    part = make_fitted(kind, opts, data)
    active = find_active_leaf(part)
    leaves = [r for r in part.leaves if r.level > 0]
    for leaf in leaves:
        if leaf.idx == active.idx:
            continue
        y = part.eval(leaf.idx, XS, centering=True)
        heter = part.eval_heter(leaf.idx, XS)
        np.testing.assert_allclose(y, np.zeros_like(XS), atol=1e-1)
        if kind in ("pdp", "derpdp"):
            # bin-based methods (ale/rhale) spike in the 1-2 bins holding the
            # sliver of active points between the empirical split position
            # (~0.003) and the true gate at 0 — boundary noise, not a flatness
            # violation, so their heterogeneity is not asserted here
            np.testing.assert_allclose(heter, np.zeros_like(XS), atol=1e-1)

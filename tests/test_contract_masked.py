"""Contract layer, the masked global surface (regional ≡ masked global).

A mask is the ONLY thing that differentiates a regional question from a global
one: `eval`/`eval_heter`/`heter_score`/`plot` with `mask=` summarize the cached
local effects over the subregion, on the global frame, without model calls.
The rules pinned here:

  - M1  mask of all-ones ≡ mask=None (eval / eval_heter / heter_score / plot),
        continuous and categorical features of interest;
  - M2  masked calls after the local-effects cache is sealed make ZERO model
        calls — the single-model-touch constitution extended to eval/plot.
        The one documented exception: (d-)PDP `eval` at off-grid `xs`
        recomputes ICE on `data[mask]` (the exact-evaluation retouch);
  - M3  invalid masks fail loudly: wrong dtype/shape, empty, or a degenerate
        masked interval;
  - M4  masked centering is computed over the subregion's own effective
        interval (zero_integral / zero_start semantics within the region);
  - M5  `binning_scope` ("global"/"effective") exists on RHALE/ShapDP only,
        is validated, and controls the x-range of the masked re-binning;
  - M6  masked plots return (fig, ax), window the x-axis to the effective
        interval, and honor `feature_label`.
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

import effector
from effector import helpers
from tests.conftest import (
    GLOBAL_NAMES,
    CountingModel,
    analytic_shap_values,
    linear_model,
    linear_model_jac,
    make_global,
    make_global_data,
)

XS = np.linspace(-0.8, 0.8, 40)


def params(names=GLOBAL_NAMES):
    return [pytest.param(name, id=name) for name in names]


def half_mask(data):
    """A mask correlated with feature 0 — a genuinely restrictive subregion."""
    return data[:, 0] < 0.3


# ---------------------------------------------------------------------------
# categorical mirror: ordinal FOI (feature 1), so RHALE participates too;
# DerPDP is continuous-only and sits the categorical tests out
# ---------------------------------------------------------------------------

CAT_NAMES = ["pdp", "ale", "rhale", "shapdp"]
CAT_SCHEMA = {"feature_types": ["continuous", "ordinal", "continuous"]}


def cat_model(x):
    return (
        2.0 * x[:, 0]
        + 1.5 * (x[:, 1] == 1)
        - 0.5 * (x[:, 1] == 2)
        + (x[:, 1] == 1) * (x[:, 2] > 0)
    )


def cat_model_jac(x):
    j = np.zeros_like(x)
    j[:, 0] = 2.0
    return j


def make_cat_data(n=200, seed=3):
    rng = np.random.default_rng(seed)
    return np.stack(
        [
            rng.uniform(-1, 1, n),
            rng.integers(0, 3, n).astype(float),
            rng.uniform(-1, 1, n),
        ],
        axis=1,
    )


def make_cat_global(name, data):
    """The 4 cat-capable methods on the categorical mirror. ShapDP gets
    deterministic (seeded, not meaningful) attributions — the masked contract
    tests determinism and model-freedom, not SHAP correctness."""
    if name == "pdp":
        return effector.PDP(data, cat_model, schema=CAT_SCHEMA)
    if name == "ale":
        return effector.ALE(data, cat_model, schema=CAT_SCHEMA)
    if name == "rhale":
        return effector.RHALE(data, cat_model, cat_model_jac, schema=CAT_SCHEMA)
    if name == "shapdp":
        rng = np.random.default_rng(7)
        return effector.ShapDP(
            data,
            cat_model,
            schema=CAT_SCHEMA,
            shap_values=rng.normal(size=data.shape),
        )
    raise ValueError(name)


# ---------------------------------------------------------------------------
# M1 — all-ones mask ≡ no mask
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_m1_ones_mask_equals_none_continuous(name, global_data):
    m = make_global(name, global_data)
    m.fit(features=0)
    ones = np.ones(m.data.shape[0], dtype=bool)
    for centering in [False, "zero_integral", "zero_start"]:
        np.testing.assert_allclose(
            m.eval(0, XS, centering=centering, mask=ones),
            m.eval(0, XS, centering=centering),
            atol=1e-8,
        )
    np.testing.assert_allclose(
        m.eval_heter(0, XS, mask=ones), m.eval_heter(0, XS), atol=1e-8
    )
    np.testing.assert_allclose(m.heter_score(0, mask=ones), m.heter_score(0), atol=1e-8)


@pytest.mark.parametrize("name", params(CAT_NAMES))
def test_m1_ones_mask_equals_none_categorical(name):
    data = make_cat_data()
    m = make_cat_global(name, data)
    m.fit(features=1)
    ones = np.ones(m.data.shape[0], dtype=bool)
    levels = np.unique(data[:, 1])
    for centering in [False, "zero_integral"]:
        np.testing.assert_allclose(
            m.eval(1, levels, centering=centering, mask=ones),
            m.eval(1, levels, centering=centering),
            atol=1e-8,
        )
    np.testing.assert_allclose(
        m.eval_heter(1, levels, mask=ones), m.eval_heter(1, levels), atol=1e-8
    )
    np.testing.assert_allclose(m.heter_score(1, mask=ones), m.heter_score(1), atol=1e-8)


@pytest.mark.parametrize("name", params())
def test_m1_masked_eval_deterministic(name, global_data):
    """Two identical masked calls agree — the transient payload is
    reproducible and never contaminates stored state."""
    m = make_global(name, global_data)
    m.fit(features=0)
    mask = half_mask(m.data)
    y_stored_before = m.eval(0, XS, centering="zero_integral")
    y1 = m.eval(0, XS, centering="zero_integral", mask=mask)
    y2 = m.eval(0, XS, centering="zero_integral", mask=mask)
    np.testing.assert_allclose(y1, y2, atol=1e-12)
    # stored state untouched by the masked calls
    np.testing.assert_allclose(
        m.eval(0, XS, centering="zero_integral"), y_stored_before, atol=1e-12
    )


# ---------------------------------------------------------------------------
# M2 — the constitution on the masked surface: zero model calls
# ---------------------------------------------------------------------------


def make_counting(name, data):
    """A global object whose model/jacobian invocations are counted."""
    model = CountingModel(linear_model)
    jac = CountingModel(linear_model_jac)
    if name == "pdp":
        obj = effector.PDP(data, model)
    elif name == "derpdp":
        obj = effector.DerPDP(data, model, jac)
    elif name == "ale":
        obj = effector.ALE(data, model)
    elif name == "rhale":
        obj = effector.RHALE(data, model, jac)
    elif name == "shapdp":
        obj = effector.ShapDP(data, model, shap_values=analytic_shap_values(data))
    else:
        raise ValueError(name)
    return obj, model, jac


@pytest.mark.parametrize("name", params())
def test_m2_masked_surface_is_model_free(name):
    data = make_global_data()
    obj, model, jac = make_counting(name, data)
    obj.fit(features=0)
    mask = half_mask(obj.data)
    grid = np.linspace(
        obj.axis_limits[0, 0], obj.axis_limits[1, 0], helpers.NOF_INTERNAL_POINTS
    )
    # warmup: the first masked call may compute+seal the local effects (the
    # one model touch of the lifecycle, e.g. PDP's lazily cached ICE table)
    obj.eval_heter(0, grid, mask=mask)
    n0 = model.n_calls + jac.n_calls
    obj.eval(0, grid, centering="zero_integral", mask=mask)
    obj.eval(0, grid, centering="zero_start", mask=mask)
    obj.eval_heter(0, grid, mask=mask)
    obj.heter_score(0, mask=mask)
    obj.plot(0, show_plot=False, mask=mask)
    assert model.n_calls + jac.n_calls == n0
    plt.close("all")


@pytest.mark.parametrize("name", params(["pdp", "derpdp"]))
def test_m2_pdp_offgrid_masked_eval_recomputes(name):
    """The documented exception: (d-)PDP masked `eval` at off-grid `xs` is the
    exact-evaluation retouch — it DOES query the model, on `data[mask]`."""
    data = make_global_data()
    obj, model, jac = make_counting(name, data)
    obj.fit(features=0)
    mask = half_mask(obj.data)
    obj.eval_heter(0, XS, mask=mask)  # seal
    n0 = model.n_calls + jac.n_calls
    obj.eval(0, np.array([0.1234, 0.4321]), centering=False, mask=mask)
    assert model.n_calls + jac.n_calls > n0


# ---------------------------------------------------------------------------
# M3 — invalid masks fail loudly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_m3_invalid_masks_raise(name, global_data):
    m = make_global(name, global_data)
    m.fit(features=0)
    n = m.data.shape[0]
    with pytest.raises(ValueError):
        m.eval(0, XS, mask=np.zeros(n, dtype=bool))  # empty
    with pytest.raises(ValueError):
        m.eval(0, XS, mask=np.ones(n, dtype=float))  # not boolean
    with pytest.raises(ValueError):
        m.eval(0, XS, mask=np.ones(n + 1, dtype=bool))  # wrong shape
    single = np.zeros(n, dtype=bool)
    single[0] = True
    with pytest.raises(ValueError):
        m.eval(0, XS, mask=single)  # degenerate masked interval


# ---------------------------------------------------------------------------
# M4 — masked centering lives on the subregion's effective interval
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_m4_masked_zero_integral_on_effective_interval(name, global_data):
    m = make_global(name, global_data)
    m.fit(features=0)
    mask = half_mask(m.data)
    lo, hi = m.data[mask, 0].min(), m.data[mask, 0].max()
    xs = np.linspace(lo, hi, 1000)
    y = m.eval(0, xs, centering="zero_integral", mask=mask)
    np.testing.assert_allclose(np.mean(y), 0.0, atol=2e-2)


@pytest.mark.parametrize("name", params())
def test_m4_masked_zero_start_at_effective_start(name, global_data):
    m = make_global(name, global_data)
    m.fit(features=0)
    mask = half_mask(m.data)
    lo = m.data[mask, 0].min()
    y = m.eval(0, np.array([lo]), centering="zero_start", mask=mask)
    np.testing.assert_allclose(y, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# M5 — binning_scope: RHALE/ShapDP only, validated, honored
# ---------------------------------------------------------------------------


def test_m5_binning_scope_signature():
    for cls in [effector.RHALE, effector.ShapDP]:
        assert "binning_scope" in inspect.signature(cls.fit).parameters
    for cls in [effector.ALE, effector.PDP, effector.DerPDP]:
        assert "binning_scope" not in inspect.signature(cls.fit).parameters


@pytest.mark.parametrize("name", params(["rhale", "shapdp"]))
def test_m5_binning_scope_validated(name, global_data):
    m = make_global(name, global_data)
    with pytest.raises(ValueError):
        m.fit(features=0, binning_scope="bogus")


@pytest.mark.parametrize("name", params(["rhale", "shapdp"]))
def test_m5_binning_scope_controls_masked_bins(name, global_data):
    mask = half_mask(global_data)
    lo, hi = global_data[mask, 0].min(), global_data[mask, 0].max()

    m_eff = make_global(name, global_data)
    m_eff.fit(features=0, binning_scope="effective")
    p_eff = m_eff._summary(0, mask)

    m_glob = make_global(name, global_data)
    m_glob.fit(features=0)  # default: "global"
    p_glob = m_glob._summary(0, mask)

    def limits_of(p):
        if "limits" in p:
            return np.asarray(p["limits"], dtype=float)
        # ShapDP keeps the spline, not the limits: read the spline knots' span
        return np.asarray(p["spline_mean"].x, dtype=float)

    # the mask (x0 < 0.3) shares the global LEFT edge, so the scopes separate
    # on the right: effective bins stop at the masked max, global bins keep
    # covering the full axis (ShapDP knots are bin centers — compare vs hi)
    eff = limits_of(p_eff)
    glob = limits_of(p_glob)
    assert eff[0] >= lo - 1e-12 and eff[-1] <= hi + 1e-12
    assert glob[-1] > hi


# ---------------------------------------------------------------------------
# M6 — masked plot: (fig, ax), effective x-window, feature_label
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_m6_masked_plot_window_and_label(name, global_data):
    m = make_global(name, global_data)
    m.fit(features=0)
    mask = half_mask(m.data)
    lo, hi = m.data[mask, 0].min(), m.data[mask, 0].max()
    ret = m.plot(0, show_plot=False, mask=mask, feature_label="x_0 | region")
    assert isinstance(ret, tuple) and len(ret) == 2
    fig, ax = ret
    assert isinstance(fig, plt.Figure)
    ax_main = ax[0] if isinstance(ax, tuple) else ax
    xlo, xhi = ax_main.get_xlim()
    pad = 0.1 * (hi - lo)
    assert xlo >= lo - pad and xhi <= hi + pad
    labeled = ax[-1] if isinstance(ax, tuple) else ax
    assert labeled.get_xlabel() == "x_0 | region"


@pytest.mark.parametrize("name", params(CAT_NAMES))
def test_m6_masked_plot_categorical_smoke(name):
    data = make_cat_data()
    m = make_cat_global(name, data)
    m.fit(features=1)
    mask = data[:, 2] > 0
    ret = m.plot(1, show_plot=False, mask=mask)
    assert isinstance(ret, tuple) and len(ret) == 2


# ---------------------------------------------------------------------------
# M7 — the masked-summary memo is semantically INVISIBLE: a cache hit returns
# exactly what a cold twin computes; a refit with different fit kwargs bumps the
# fit-epoch so no stale masked answer is served; repeated calls are stable.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_m7_memo_matches_cold_twin(name):
    data = make_global_data()
    mask = half_mask(data)

    warm = make_global(name, data, nof_instances="all")
    warm.fit(features=0, centering=False)
    _ = warm.eval(0, XS, centering=False, mask=mask)  # warm the memo
    y_warm = warm.eval(0, XS, centering=False, mask=mask)  # cache hit
    h_warm = warm.eval_heter(0, XS, mask=mask)

    cold = make_global(name, data, nof_instances="all")
    cold.fit(features=0, centering=False)
    y_cold = cold.eval(0, XS, centering=False, mask=mask)
    h_cold = cold.eval_heter(0, XS, mask=mask)

    np.testing.assert_allclose(y_warm, y_cold, atol=1e-12)
    np.testing.assert_allclose(h_warm, h_cold, atol=1e-12)


def test_m7_refit_invalidates_memo():
    import effector
    import effector.axis_partitioning as ap

    # a quadratic model has non-zero, binning-dependent RHALE heterogeneity, so
    # a coarse vs fine binning genuinely changes the masked score
    def quad(x):
        return x[:, 0] ** 2

    def quad_jac(x):
        jac = np.zeros_like(x)
        jac[:, 0] = 2 * x[:, 0]
        return jac

    data = make_global_data()
    mask = half_mask(data)

    rhale = effector.RHALE(data, quad, model_jac=quad_jac, nof_instances="all")
    rhale.fit(0, binning_method=ap.Fixed(nof_bins=3), centering=False)
    v_coarse = rhale.heter_score(0, mask=mask)  # warms the memo at epoch e

    rhale.fit(0, binning_method=ap.Fixed(nof_bins=40), centering=False)
    v_fine_warm = rhale.heter_score(0, mask=mask)  # epoch bumped -> recomputed

    cold = effector.RHALE(data, quad, model_jac=quad_jac, nof_instances="all")
    cold.fit(0, binning_method=ap.Fixed(nof_bins=40), centering=False)
    v_fine_cold = cold.heter_score(0, mask=mask)

    # the two binnings genuinely differ (test is meaningful) ...
    assert v_coarse != v_fine_cold
    # ... and the refit served the NEW value, not the stale cached one
    assert v_fine_warm == v_fine_cold


@pytest.mark.parametrize("name", params())
def test_m7_heter_score_masked_stable(name):
    data = make_global_data()
    mask = half_mask(data)
    m = make_global(name, data, nof_instances="all")
    m.fit(features=0, centering=False)
    assert m.heter_score(0, mask=mask) == m.heter_score(0, mask=mask)

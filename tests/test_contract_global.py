"""Contract layer, global classes (PLAN II §3.1, rewritten per LOGBOOK #4).

Parametrized over the 5 global effect classes on the shared tiny linear model
(see conftest).  Each test states one rule of the constitution (Part III §1):
green = the rule already holds today; ``xfail(strict=True)`` = the
homogenization refactor must make it true (removing the marker the moment it
does — PLAN II §4).

Provisional decisions (LOGBOOK #4 left them to constitution-writing time;
flagged for Vasilis's sign-off, cheap to rename):
  - payload accessor: ``method.payload(feature) -> dict`` containing at least
    ``"h"`` (the method-specific heterogeneity object, an ``np.ndarray``);
  - agnostic score:   ``method.heterogeneity(feature) -> float`` (>= 0,
    centering-invariant);
  - ``eval`` loses the ``heterogeneity`` (and PDP's ``return_all``) kwargs:
    one return type, always (R1/R2).
"""

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tests.conftest import GLOBAL_NAMES, eval_mean, make_global

XS = np.linspace(-0.8, 0.8, 40)

xfail = pytest.mark.xfail


def params(**marks):
    """The 5 global methods, with per-method xfail marks where given."""
    return [
        pytest.param(name, marks=marks[name], id=name)
        if name in marks
        else pytest.param(name, id=name)
        for name in GLOBAL_NAMES
    ]


# ---------------------------------------------------------------------------
# C1 — return shape/type: eval(feature, xs) -> (T,) ndarray, always (R1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    params(
        shapdp=xfail(
            strict=True,
            reason="B11: ShapDP.eval defaults heterogeneity=True -> returns a tuple",
        )
    ),
)
def test_c1_eval_returns_mean_array(name, global_data):
    m = make_global(name, global_data)
    y = m.eval(0, XS)
    assert isinstance(y, np.ndarray)
    assert y.shape == XS.shape


@pytest.mark.parametrize("name", params())
def test_c1_eval_all_features(name, global_data):
    m = make_global(name, global_data)
    for feature in range(global_data.shape[1]):
        y = eval_mean(m, feature, XS)
        assert y.shape == XS.shape
        assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# C2 — centering semantics (R3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_c2_zero_integral(name, global_data):
    m = make_global(name, global_data)
    # the same 30-point linspace the fit uses for the normalization constant
    xs = np.linspace(m.axis_limits[0, 0], m.axis_limits[1, 0], 30)
    y = eval_mean(m, 0, xs, centering="zero_integral")
    np.testing.assert_allclose(np.mean(y), 0.0, atol=1e-6)


@pytest.mark.parametrize("name", params())
def test_c2_zero_start(name, global_data):
    m = make_global(name, global_data)
    xs = np.array([m.axis_limits[0, 0]])
    y = eval_mean(m, 0, xs, centering="zero_start")
    np.testing.assert_allclose(y, 0.0, atol=1e-6)


@pytest.mark.parametrize("name", params())
def test_c2_true_equals_zero_integral(name, global_data):
    y_true = eval_mean(make_global(name, global_data), 0, XS, centering=True)
    y_str = eval_mean(make_global(name, global_data), 0, XS, centering="zero_integral")
    np.testing.assert_allclose(y_true, y_str, atol=1e-12)


@pytest.mark.parametrize("name", params())
def test_c2_uncentered_is_constant_shift(name, global_data):
    y_c = eval_mean(make_global(name, global_data), 0, XS, centering="zero_integral")
    y_u = eval_mean(make_global(name, global_data), 0, XS)
    diff = y_u - y_c
    np.testing.assert_allclose(diff, diff[0], atol=1e-8)


# ---------------------------------------------------------------------------
# C3 — lazy fit + refit (exercises requires_refit; B7 territory)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_c3_eval_works_without_fit(name, global_data):
    m = make_global(name, global_data)
    assert eval_mean(m, 0, XS).shape == XS.shape


@pytest.mark.parametrize("name", params())
def test_c3_eval_idempotent(name, global_data):
    m = make_global(name, global_data)
    y1 = eval_mean(m, 0, XS, centering="zero_integral")
    y2 = eval_mean(m, 0, XS, centering="zero_integral")
    np.testing.assert_allclose(y1, y2, atol=1e-12)


@pytest.mark.parametrize("name", params())
@pytest.mark.parametrize("fit_centering", [False, "zero_start", "zero_integral"])
def test_c3_refit_on_centering_change(name, fit_centering, global_data):
    """eval(centering=X) after fit(centering=Y) must equal a fresh
    eval(centering=X): whatever was pre-fitted may never leak into the answer."""
    m = make_global(name, global_data)
    m.fit(features=0, centering=fit_centering)
    y = eval_mean(m, 0, XS, centering="zero_integral")
    y_fresh = eval_mean(
        make_global(name, global_data), 0, XS, centering="zero_integral"
    )
    np.testing.assert_allclose(y, y_fresh, atol=1e-8)


# ---------------------------------------------------------------------------
# C4 — fit(features=...) variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
@pytest.mark.parametrize("features", [0, [0], [0, 1], "all"])
def test_c4_fit_features_variants(name, features, global_data):
    m = make_global(name, global_data)
    m.fit(features=features, centering="zero_integral")
    y = eval_mean(m, 0, XS, centering="zero_integral")
    y_ref = eval_mean(make_global(name, global_data), 0, XS, centering="zero_integral")
    np.testing.assert_allclose(y, y_ref, atol=1e-8)


# ---------------------------------------------------------------------------
# C5 — plot contract (R7): show_plot=False -> (fig, ax); True -> None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_c5_plot_returns_fig_ax(name, global_data):
    m = make_global(name, global_data)
    ret = m.plot(0, show_plot=False)
    assert isinstance(ret, tuple) and len(ret) == 2
    assert isinstance(ret[0], plt.Figure)


@pytest.mark.parametrize("name", params())
def test_c5_plot_show_returns_none(name, global_data):
    m = make_global(name, global_data)
    assert m.plot(0, show_plot=True) is None


# ---------------------------------------------------------------------------
# C6 — constructor contract (R8)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", params())
def test_c6_positional_data_model_equals_kwargs(name, global_data):
    """data/model (and model_jac) positionally vs everything by keyword must
    agree — the stable positional prefix of the R8 migration."""
    import effector
    from tests.conftest import analytic_shap_values, linear_model, linear_model_jac

    if name == "pdp":
        m_pos = effector.PDP(global_data, linear_model)
    elif name == "derpdp":
        m_pos = effector.DerPDP(global_data, linear_model, linear_model_jac)
    elif name == "ale":
        m_pos = effector.ALE(global_data, linear_model)
    elif name == "rhale":
        m_pos = effector.RHALE(global_data, linear_model, linear_model_jac)
    else:
        m_pos = effector.ShapDP(
            global_data, linear_model, shap_values=analytic_shap_values(global_data)
        )
    m_kw = make_global(name, global_data)
    np.testing.assert_allclose(
        eval_mean(m_pos, 0, XS),
        eval_mean(m_kw, 0, XS),
        atol=1e-12,
    )


@pytest.mark.parametrize("name", params())
def test_c6_nof_instances(name, global_data):
    np.random.seed(21)
    m = make_global(name, global_data, nof_instances=50)
    assert m.data.shape[0] == 50
    m_all = make_global(name, global_data, nof_instances="all")
    assert m_all.data.shape[0] == global_data.shape[0]


@pytest.mark.parametrize("name", params())
def test_c6_axis_limits_filter(name, global_data):
    limits = np.array([[-0.5] * 3, [0.5] * 3])
    m = make_global(name, global_data, axis_limits=limits)
    assert m.data.shape[0] < global_data.shape[0]
    assert np.all(m.data >= -0.5) and np.all(m.data <= 0.5)


# ---------------------------------------------------------------------------
# New surface (R1/R2, LOGBOOK #4) — all xfail until the refactor lands
# ---------------------------------------------------------------------------

NEW_SURFACE = xfail(
    strict=True,
    reason="R2/LOGBOOK #4: new heterogeneity surface not implemented yet",
)


@pytest.mark.parametrize("name", params())
@pytest.mark.xfail(
    strict=True,
    reason="R1/R2: eval must have one return type — no heterogeneity/return_all kwargs",
)
def test_new_eval_signature_mean_only(name, global_data):
    m = make_global(name, global_data)
    sig = inspect.signature(type(m).eval)
    assert "heterogeneity" not in sig.parameters
    assert "return_all" not in sig.parameters


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=NEW_SURFACE) for n in GLOBAL_NAMES]
)
def test_new_payload_accessor(name, global_data):
    """Provisional: payload(feature) -> dict with the method's honest object,
    at least the key 'h' (np.ndarray)."""
    m = make_global(name, global_data)
    m.fit(features=0)
    p = m.payload(0)
    assert isinstance(p, dict)
    assert isinstance(p["h"], np.ndarray)


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=NEW_SURFACE) for n in GLOBAL_NAMES]
)
def test_new_H_scalar(name, global_data):
    """Provisional: heterogeneity(feature) -> non-negative scalar (the single
    quantity regional splitting and F2 consume)."""
    m = make_global(name, global_data)
    H = m.heterogeneity(0)
    assert np.isscalar(H)
    assert H >= 0


@pytest.mark.parametrize(
    "name", [pytest.param(n, marks=NEW_SURFACE) for n in GLOBAL_NAMES]
)
def test_new_h_centering_invariant(name, global_data):
    """h (inside the payload) is a variance-like object: centering the mean
    effect must not change it."""
    m_c = make_global(name, global_data)
    m_c.fit(features=0, centering="zero_integral")
    m_u = make_global(name, global_data)
    m_u.fit(features=0, centering=False)
    np.testing.assert_allclose(m_c.payload(0)["h"], m_u.payload(0)["h"], atol=1e-8)

"""Contract layer, global classes (PLAN II §3.1, rewritten per LOGBOOK #4).

Parametrized over the 5 global effect classes on the shared tiny linear model
(see conftest).  Each test states one rule of the constitution (Part III §1):
green = the rule already holds today; ``xfail(strict=True)`` = the
homogenization refactor must make it true (removing the marker the moment it
does — PLAN II §4).

The heterogeneity surface (agreed with Vasilis 2026-07-03, LOGBOOK #13):
  - ``eval(feature, xs, centering=...)`` -> mean effect only, one return type
    (loses the ``heterogeneity`` and PDP's ``return_all`` kwargs — R1/R2);
  - ``eval_heter(feature, xs)`` -> the heterogeneity curve h(xs), ``(T,)``,
    no centering kwarg (invariant by signature), method-specific units;
  - ``payload(feature)`` -> dict, the method's raw honest object;
  - ``heter_score(feature)`` -> float >= 0, the method-agnostic scalar.
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
    """The semantic contract: with zero_integral centering the mean effect over
    the feature's interval is ~0.  Asserted on a dense grid with a tolerance
    that covers the 30-point discretization of the normalization constant
    (since step 2 all methods share the same midpoint scheme — the tight
    per-scheme pin at 1e-6 was an accident of the old per-method grids)."""
    m = make_global(name, global_data)
    xs = np.linspace(m.axis_limits[0, 0], m.axis_limits[1, 0], 1000)
    y = eval_mean(m, 0, xs, centering="zero_integral")
    np.testing.assert_allclose(np.mean(y), 0.0, atol=1e-2)


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
# New surface (R1/R2, LOGBOOK #4 + #13 — names agreed with Vasilis 2026-07-03:
# eval / eval_heter / payload / heter_score; implemented in refactor step 2)
# ---------------------------------------------------------------------------


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


@pytest.mark.parametrize("name", params())
def test_new_payload_accessor(name, global_data):
    """payload(feature) -> non-empty dict with the method's raw honest object
    (ICE matrix, per-bin variances, shap cloud); exact schema is decided at
    refactor step 2-3."""
    m = make_global(name, global_data)
    m.fit(features=0)
    p = m.payload(0)
    assert isinstance(p, dict)
    assert len(p) > 0


@pytest.mark.parametrize("name", params())
def test_new_heter_score_scalar(name, global_data):
    """heter_score(feature) -> non-negative scalar: the single method-agnostic
    heterogeneity quantity regional splitting and F2 consume."""
    m = make_global(name, global_data)
    score = m.heter_score(0)
    assert np.isscalar(score)
    assert score >= 0


@pytest.mark.parametrize("name", params())
def test_new_eval_heter_returns_curve(name, global_data):
    """eval_heter(feature, xs) -> (T,) ndarray >= 0: the heterogeneity curve
    h(xs), method-specific units (the plot layer's bands must equal it — R1)."""
    m = make_global(name, global_data)
    h = m.eval_heter(0, XS)
    assert isinstance(h, np.ndarray)
    assert h.shape == XS.shape
    assert np.all(h >= 0)


@pytest.mark.parametrize("name", params())
def test_new_eval_heter_signature_has_no_centering(name, global_data):
    """Heterogeneity is centering-invariant by definition (R2): the signature
    itself must make it impossible to ask otherwise."""
    m = make_global(name, global_data)
    sig = inspect.signature(type(m).eval_heter)
    assert "centering" not in sig.parameters


@pytest.mark.parametrize("name", params())
def test_new_eval_heter_centering_invariant(name, global_data):
    """eval_heter returns identical values whether the object was fitted
    centered or uncentered."""
    m_c = make_global(name, global_data)
    m_c.fit(features=0, centering="zero_integral")
    m_u = make_global(name, global_data)
    m_u.fit(features=0, centering=False)
    np.testing.assert_allclose(m_c.eval_heter(0, XS), m_u.eval_heter(0, XS), atol=1e-8)

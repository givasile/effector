"""Contract layer, importance (R13).

`importance(feature, mask=None) -> float >= 0` is the μ-twin of `heter_score`:
the dispersion of the mean effect over the (masked) data. Model-free,
centering-invariant, data-weighted. `importances()` is the per-feature vector,
warning once on unsupported feature types.

Closed-form oracle: for a linear model with independent features the mean effect
is `a_j*(x_j - E[x_j])`, evaluated over the uniform grid, so the std of
PDP/ALE/RHALE is exactly `|a_j| * std(grid_j)` (the μ-twin uses heter_score's
grid); ShapDP's `mean(|phi|)` is `|a_j| * mean(|x_j - mean|)`; d-PDP's mean
|derivative| is `|a_j|`.
"""

import warnings

import numpy as np
import pytest

import effector
from effector import helpers
from tests.conftest import (
    COEF,
    GLOBAL_NAMES,
    CountingModel,
    analytic_shap_values,
    linear_model,
    linear_model_jac,
    make_global,
    make_global_data,
)


def _grid_std(m, feature):
    """std of the uniform grid heter_score / importance evaluate over."""
    grid = np.linspace(
        m.axis_limits[0, feature],
        m.axis_limits[1, feature],
        helpers.NOF_INTERNAL_POINTS,
    )
    return np.std(grid)


CONT_NAMES = ["pdp", "ale", "rhale"]  # recover the exact linear effect


def _fit(name, data, **fit_kwargs):
    m = make_global(name, data, nof_instances="all")
    m.fit(features="all", centering=False, **fit_kwargs)
    return m


# ---------------------------------------------------------------------------
# I1 — non-negative, finite, (D,) vector
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_i1_nonnegative_and_shape(name):
    m = _fit(name, make_global_data())
    imps = m.importances()
    assert imps.shape == (3,)
    for f in range(3):
        v = m.importance(f)
        assert np.isfinite(v) and v >= 0


# ---------------------------------------------------------------------------
# I2 — centering-invariant: importance does not depend on how the feature was
# fitted / centered (no centering kwarg exists on the signature).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_i2_centering_invariant(name):
    data = make_global_data()
    m_false = make_global(name, data, nof_instances="all")
    m_false.fit(features="all", centering=False)
    m_zi = make_global(name, data, nof_instances="all")
    m_zi.fit(features="all", centering="zero_integral")
    for f in range(3):
        np.testing.assert_allclose(
            m_false.importance(f), m_zi.importance(f), atol=1e-10
        )


def test_i2_no_centering_kwarg():
    import inspect

    sig = inspect.signature(effector.PDP.importance)
    assert "centering" not in sig.parameters


# ---------------------------------------------------------------------------
# I3 — model-free: after fit, importance (masked or not) makes ZERO model calls.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_i3_model_free(name):
    data = make_global_data()
    model = CountingModel(linear_model)
    jac = CountingModel(linear_model_jac)
    if name == "pdp":
        m = effector.PDP(data, model, nof_instances="all")
    elif name == "derpdp":
        m = effector.DerPDP(data, model, model_jac=jac, nof_instances="all")
    elif name == "ale":
        m = effector.ALE(data, model, nof_instances="all")
    elif name == "rhale":
        m = effector.RHALE(data, model, model_jac=jac, nof_instances="all")
    elif name == "shapdp":
        m = effector.ShapDP(
            data, model, shap_values=analytic_shap_values(data), nof_instances="all"
        )
    m.fit(features="all", centering=False)
    # warm the local effects (PDP/DerPDP compute their ICE table lazily on first
    # summarize — that is the single model touch); measure AFTER it, like RC8
    m.importances()
    n0 = model.n_calls + jac.n_calls
    mask = data[:, 0] < 0.3
    for f in range(3):
        m.importance(f)
        m.importance(f, mask=mask)
    assert model.n_calls + jac.n_calls == n0


# ---------------------------------------------------------------------------
# I4 — importances warns once (R9) on an unsupported feature type -> NaN there.
# ---------------------------------------------------------------------------


def test_i4_unsupported_type_warns_once():
    rng = np.random.default_rng(0)
    data = np.column_stack([rng.uniform(-1, 1, 500), rng.integers(0, 3, 500)])
    rhale = effector.RHALE(
        data,
        lambda x: 2 * x[:, 0],
        model_jac=lambda x: np.column_stack([2 * np.ones(len(x)), np.zeros(len(x))]),
        schema={"feature_types": ["continuous", "nominal"]},
        nof_instances="all",
    )
    rhale.fit(features=0, centering=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        imps = rhale.importances()
    user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
    assert len(user_warnings) == 1
    assert np.isfinite(imps[0])
    assert np.isnan(imps[1])  # nominal feature unsupported by RHALE


# ---------------------------------------------------------------------------
# I5 — closed form on the linear conftest model (COEF = [2, -3, 0.5]).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", CONT_NAMES)
def test_i5_closed_form_pdp_ale_rhale(name):
    data = make_global_data(n=2000)
    m = _fit(name, data)
    for f in range(3):
        expected = abs(COEF[f]) * _grid_std(m, f)  # |a_j| * std(grid_j)
        np.testing.assert_allclose(m.importance(f), expected, rtol=1e-6)


def test_i5_closed_form_shapdp():
    data = make_global_data(n=2000)
    m = make_global("shapdp", data, nof_instances="all")
    m.fit(features="all", centering=False)
    for f in range(3):
        col = data[:, f]
        expected = abs(COEF[f]) * np.mean(np.abs(col - col.mean()))
        np.testing.assert_allclose(m.importance(f), expected, rtol=1e-6)


def test_i5_closed_form_derpdp():
    data = make_global_data(n=2000)
    m = _fit("derpdp", data)
    for f in range(3):
        np.testing.assert_allclose(m.importance(f), abs(COEF[f]), rtol=1e-6)


def test_i5_importance_ratio_matches_coefficients():
    data = make_global_data(n=2000)
    for name in GLOBAL_NAMES:
        m = _fit(name, data)
        ratio = m.importance(0) / m.importance(1)
        np.testing.assert_allclose(ratio, abs(COEF[0] / COEF[1]), rtol=0.1)

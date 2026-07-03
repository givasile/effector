"""Unit layer for the PDP/ICE kernels (PLAN II §3.2): absorbs test_unit.py and
settles B9 — the vectorized and non-vectorized kernels must agree on every
path (values, d-ICE via jacobian, d-ICE via finite differences)."""

import numpy as np
import pytest

from effector.global_effect_pdp import ice_non_vectorized, ice_vectorized

T = 20
N = 30


def model(x):
    return x[:, 0] ** 2 + x[:, 0] * x[:, 1] + 2 * x[:, 1]


def model_jac(x):
    return np.stack([2 * x[:, 0] + x[:, 1], x[:, 0] + 2], axis=1)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(21)
    return rng.uniform(-1, 1, size=(N, 2))


@pytest.fixture(scope="module")
def xs():
    return np.linspace(-1, 1, T)


def test_ice_values_ground_truth(data, xs):
    """ICE of the square model along feature 0: x^2 + x*x1_i + 2*x1_i."""
    yy = ice_vectorized(model, None, data, xs, feature=0)
    gt = xs[:, None] ** 2 + xs[:, None] * data[:, 1] + 2 * data[:, 1]
    np.testing.assert_allclose(yy, gt, atol=1e-10)
    assert yy.shape == (T, N)


def test_ice_values_vectorized_equals_non_vectorized(data, xs):
    y_vec = ice_vectorized(model, None, data, xs, feature=0)
    y_non = ice_non_vectorized(model, None, data, xs, feature=0)
    np.testing.assert_allclose(y_vec, y_non, atol=1e-12)


def test_d_ice_jac_ground_truth(data, xs):
    yy = ice_vectorized(model, model_jac, data, xs, feature=0, return_d_ice=True)
    gt = 2 * xs[:, None] + data[:, 1]
    np.testing.assert_allclose(yy, gt, atol=1e-10)


def test_d_ice_jac_vectorized_equals_non_vectorized(data, xs):
    y_vec = ice_vectorized(model, model_jac, data, xs, feature=0, return_d_ice=True)
    y_non = ice_non_vectorized(model, model_jac, data, xs, feature=0, return_d_ice=True)
    np.testing.assert_allclose(y_vec, y_non, atol=1e-12)


def test_d_ice_findiff_vectorized_equals_non_vectorized(data, xs):
    """B9: the vectorized finite-difference d-ICE carries a
    'TODO: needs test, something is wrong' — settle it against the
    non-vectorized path and the analytic jacobian."""
    y_vec = ice_vectorized(model, None, data, xs, feature=0, return_d_ice=True)
    y_non = ice_non_vectorized(model, None, data, xs, feature=0, return_d_ice=True)
    np.testing.assert_allclose(y_vec, y_non, atol=1e-4)


def test_d_ice_findiff_matches_analytic(data, xs):
    gt = 2 * xs[:, None] + data[:, 1]
    y_vec = ice_vectorized(model, None, data, xs, feature=0, return_d_ice=True)
    np.testing.assert_allclose(y_vec, gt, atol=1e-4)
    y_non = ice_non_vectorized(model, None, data, xs, feature=0, return_d_ice=True)
    np.testing.assert_allclose(y_non, gt, atol=1e-4)


def test_second_feature_paths(data, xs):
    """Same checks along feature 1 (d/dx1 = x0 + 2, constant in x1)."""
    gt = np.tile(data[:, 0] + 2, (T, 1))
    y_jac = ice_vectorized(model, model_jac, data, xs, feature=1, return_d_ice=True)
    np.testing.assert_allclose(y_jac, gt, atol=1e-10)
    y_fd = ice_non_vectorized(model, None, data, xs, feature=1, return_d_ice=True)
    np.testing.assert_allclose(y_fd, gt, atol=1e-4)

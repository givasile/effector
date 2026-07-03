"""Unit layer for effector.helpers (PLAN II §3.2): the prep_* input
normalizers and the small data helpers."""

import numpy as np
import pytest

from effector import helpers


def test_prep_features():
    assert helpers.prep_features("all", 3) == [0, 1, 2]
    assert helpers.prep_features(1, 3) == [1]
    assert helpers.prep_features([0, 2], 3) == [0, 2]


def test_prep_centering():
    assert helpers.prep_centering(False) is False
    assert helpers.prep_centering(True) == "zero_integral"
    assert helpers.prep_centering("zero_integral") == "zero_integral"
    assert helpers.prep_centering("zero_start") == "zero_start"
    with pytest.raises((AssertionError, ValueError)):
        helpers.prep_centering("junk")
    with pytest.raises((AssertionError, ValueError, TypeError)):
        helpers.prep_centering(3)


def test_prep_confidence_interval():
    assert helpers.prep_confidence_interval(False) is False
    assert helpers.prep_confidence_interval(True) == "std"
    for valid in ["std", "std_err", "ice", "shap_values"]:
        assert helpers.prep_confidence_interval(valid) == valid
    with pytest.raises((AssertionError, ValueError)):
        helpers.prep_confidence_interval("junk")


def test_prep_nof_instances_subsample():
    np.random.seed(21)
    _, indices = helpers.prep_nof_instances(50, 100)
    assert indices.shape == (50,)
    assert len(np.unique(indices)) == 50  # without replacement
    assert np.all((indices >= 0) & (indices < 100))


def test_prep_nof_instances_more_than_available():
    _, indices = helpers.prep_nof_instances(200, 100)
    np.testing.assert_array_equal(indices, np.arange(100))


def test_prep_nof_instances_all():
    nof, indices = helpers.prep_nof_instances("all", 100)
    assert nof == 100
    np.testing.assert_array_equal(indices, np.arange(100))
    with pytest.raises((AssertionError, ValueError)):
        helpers.prep_nof_instances("some", 100)


def test_axis_limits_from_data():
    data = np.array([[0.0, -5.0], [1.0, 3.0], [0.5, 0.0]])
    limits = helpers.axis_limits_from_data(data)
    np.testing.assert_allclose(limits, [[0.0, -5.0], [1.0, 3.0]])


def test_indices_within_limits():
    data = np.array([[0.0, 0.0], [0.5, 0.5], [2.0, 0.5], [0.5, -2.0]])
    limits = np.array([[0.0, -1.0], [1.0, 1.0]])
    accept = helpers.indices_within_limits(data, limits)
    np.testing.assert_array_equal(accept, [True, True, False, False])


def test_indices_within_limits_none_inside_raises():
    data = np.array([[5.0, 5.0]])
    limits = np.array([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(AssertionError):
        helpers.indices_within_limits(data, limits)


def test_camel_to_snake():
    assert helpers.camel_to_snake("CamelCase") == "camel_case"
    assert helpers.camel_to_snake("Fixed") == "fixed"
    # behavioral pin: consecutive capitals split letter-by-letter
    assert helpers.camel_to_snake("PDP") == "p_d_p"


def test_get_feature_names():
    assert helpers.get_feature_names(3) == ["x_0", "x_1", "x_2"]


def test_prep_data_infers_limits_and_subsamples():
    rng = np.random.default_rng(21)
    raw = rng.uniform(-1, 1, size=(100, 3))
    np.random.seed(21)
    data, effect, limits, nof, indices = helpers.prep_data(raw, nof_instances=40)
    assert data.shape == (40, 3)
    assert effect is None
    np.testing.assert_allclose(limits, helpers.axis_limits_from_data(raw))
    assert nof == 40 and indices.shape == (40,)


def test_prep_data_filters_and_keeps_effect_aligned():
    rng = np.random.default_rng(21)
    raw = rng.uniform(-1, 1, size=(100, 2))
    raw_effect = 10 * raw  # recognizable per-row pairing
    limits = np.array([[-0.5, -0.5], [0.5, 0.5]])
    data, effect, _, _, _ = helpers.prep_data(
        raw, axis_limits=limits, nof_instances="all", data_effect=raw_effect
    )
    assert 0 < data.shape[0] < 100
    assert np.all((data >= -0.5) & (data <= 0.5))
    np.testing.assert_allclose(effect, 10 * data)  # rows stayed aligned


def test_prep_data_rejects_bad_input():
    rng = np.random.default_rng(21)
    raw = rng.uniform(-1, 1, size=(10, 2))
    with pytest.raises(ValueError):
        helpers.prep_data(raw[:, 0])  # not 2D
    with pytest.raises(ValueError):
        helpers.prep_data(raw, axis_limits=np.zeros((2, 5)))  # wrong shape
    with pytest.raises(ValueError):
        helpers.prep_data(raw, axis_limits=np.array([[1.0, 1.0], [0.0, 0.0]]))

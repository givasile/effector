import pytest

import effector


def test_independent_uniform():
    dim = 3
    X = effector.datasets.IndependentUniform(dim=dim, low=-1, high=1).generate_data(
        1000, seed=21
    )
    assert X.shape == (1000, dim)


@pytest.mark.slow
def test_bike_sharing():
    """Downloads from UCI — network, ~7 s; tier 2 only (P3)."""
    data = effector.datasets.BikeSharing()
    data.fetch_and_preprocess()
    assert data.dataset is not None


@pytest.mark.slow
def test_medical_costs():
    """Downloads a CSV — network; tier 2 only."""
    data = effector.datasets.MedicalCosts()
    assert data.dataset.shape == (1338, 7)
    assert data.x_train.shape == (1070, 6)
    assert data.feature_names[4] == "smoker"
    assert data.category_names[4] == ["no", "yes"]
    assert len(data.feature_types) == 6


@pytest.mark.slow
def test_airfoil_self_noise():
    """Downloads from UCI — network; tier 2 only."""
    data = effector.datasets.AirfoilSelfNoise()
    assert data.dataset.shape == (1503, 6)
    assert data.x_train.shape == (1202, 5)
    assert len(data.feature_names) == 5


@pytest.mark.slow
def test_adult_income():
    """Downloads from UCI — network; tier 2 only."""
    data = effector.datasets.AdultIncome()
    assert data.dataset.shape == (45222, 13)
    assert data.x_train.shape == (36177, 12)
    assert set(data.y_train.tolist()) == {0.0, 1.0}
    assert data.feature_names[8] == "capital-gain"
    sex = data.feature_names.index("sex")
    assert data.category_names[sex] == ["Female", "Male"]


def test_independent_uniform_seeded_reproducibility():
    ds = effector.datasets.IndependentUniform(dim=3, low=-1, high=1)
    a = ds.generate_data(500, seed=21)
    b = ds.generate_data(500, seed=21)
    c = ds.generate_data(500, seed=22)
    import numpy as np

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)

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


def test_independent_uniform_seeded_reproducibility():
    ds = effector.datasets.IndependentUniform(dim=3, low=-1, high=1)
    a = ds.generate_data(500, seed=21)
    b = ds.generate_data(500, seed=21)
    c = ds.generate_data(500, seed=22)
    import numpy as np

    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)

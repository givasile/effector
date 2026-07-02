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

import effector


def test_datasets():
    dim = 3

    X = effector.datasets.IndependentUniform(dim=dim, low=-1, high=1).generate_data(
        1000, seed=21
    )
    assert X.shape == (1000, dim)

    data = effector.datasets.BikeSharing()
    data.fetch_and_preprocess()
    data.postprocess
    assert data.dataset is not None

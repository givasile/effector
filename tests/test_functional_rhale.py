import numpy as np
import effector

np.random.seed(21)


def test_rhale_square():
    """
    Test the vectorized version of the PDP function for a square model
    """
    N = 1000
    T = 1000

    data = np.stack([
        np.random.rand(N + 1),
        np.random.rand(N + 1),
    ], axis=1)

    model = lambda x: x[:, 1] ** 2
    model_jac = lambda x: (
        np.stack(
            [
            np.zeros_like(x[:, 1]),
            2 * x[:, 1]
            ],
        axis=1
    ))

    x = np.linspace(0, 1, T)

    # test the finite difference version
    rhale = effector.RHALE(data, model)
    binning_method = effector.binning_methods.Fixed(nof_bins=10, min_points_per_bin=0)
    rhale.fit(features="all", binning_method=binning_method)
    rhale.plot(feature=1)

    rhale = effector.RHALE(data, model, model_jac)
    binning_method = effector.binning_methods.Fixed(nof_bins=10, min_points_per_bin=0)
    rhale.fit(features="all", binning_method=binning_method)
    rhale.plot(feature=1)

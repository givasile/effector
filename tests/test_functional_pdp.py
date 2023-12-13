import numpy as np
import effector

np.random.seed(21)


def test_pdp_linear_square():
    """
    Test the vectorized version of the PDP function for a square model
    """
    N = 10
    T = 100

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
    effector.DerivativePDP(data, model).plot(feature=1)
    effector.DerivativePDP(data, model, model_jac).plot(feature=1)

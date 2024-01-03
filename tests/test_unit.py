import numpy as np
import effector


def test_pdp_1d_vectorized():
    """
    Test the vectorized version of the PDP function for a square model
    """

    np.random.seed(21)

    N = 10
    T = 100

    data = np.stack([
        np.random.rand(N + 1),
        np.linspace(0, 1, N+1)
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
    data_effect = effector.global_effect_pdp.pdp_1d_vectorized(
        model,
        data,
        x,
        feature=1,
        heterogeneity=False,
        model_returns_jac=False,
        return_all=False,
        ask_for_derivatives=True)

    diff = np.abs(data_effect - 2 * x)
    assert np.all(diff < 1e-6)

    # test the jacobian version
    data_effect = effector.global_effect_pdp.pdp_1d_vectorized(
        model_jac,
        data,
        x,
        feature=1,
        heterogeneity=False,
        model_returns_jac=True,
        return_all=False,
        ask_for_derivatives=True)

    diff = np.abs(data_effect - 2 * x)
    assert np.all(diff < 1e-6)

    # test the pdp version
    data_effect = effector.global_effect_pdp.pdp_1d_vectorized(
        model,
        data,
        x,
        feature=1,
        heterogeneity=False,
        model_returns_jac=False,
        return_all=False,
        ask_for_derivatives=False)

    diff = np.abs(data_effect - x**2)
    assert np.all(diff < 1e-6)



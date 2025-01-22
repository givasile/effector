import numpy as np
from pandas.core.ops import make_flex_doc

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

    # test the pdp version
    yy = effector.global_effect_pdp.ice_vectorized(
        model=model,
        model_jac=None,
        data=data,
        x=x,
        feature=1
    )

    # ground truth
    yy_gt = np.stack([x**2 for _ in range(N + 1)], axis=1)
    diff = np.abs(yy - yy_gt)

    assert np.all(diff < 1e-6)

    # test the jacobian version
    yy = effector.global_effect_pdp.ice_vectorized(
        model=model,
        model_jac=model_jac,
        data=data,
        x=x,
        feature=1,
        return_d_ice=True
    )

    yy_gt = np.stack([2*x for _ in range(N + 1)], axis=1)
    diff = np.abs(yy_gt - yy)
    assert np.all(diff < 1e-6)

    # test the finite difference version
    yy = effector.global_effect_pdp.ice_vectorized(
        model=model,
        model_jac=None,
        data=data,
        x=x,
        feature=1,
        return_d_ice=True,
    )
    yy_gt = np.stack([2*x for _ in range(N + 1)], axis=1)
    diff = np.abs(yy - yy_gt)
    assert np.all(diff < 1e-6)



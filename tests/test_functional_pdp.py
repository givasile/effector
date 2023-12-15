import numpy as np
import effector

np.random.seed(21)


def test_pdp_square():
    """
    Test the vectorized version of the PDP function for a square model
    """
    N = 10
    T = 100

    data = np.stack(
        [
            np.random.rand(N + 1),
            np.random.rand(N + 1),
        ],
        axis=1,
    )

    model = lambda x: x[:, 1] ** 2 + x[:, 0] + x[:, 0] * x[:, 1]
    model_jac = lambda x: np.stack(
        [
            np.ones_like(x[:, 0]) + x[:, 1],
            2 * x[:, 1] + x[:, 0],
        ],
        axis=1,
    )

    x = np.linspace(0, 1, T)

    # compute the PDP dependence
    pdp = effector.PDP(data, model)
    pdp.fit(features="all", centering="zero_start")
    pdp.plot(feature=0, heterogeneity="std", centering="zero_integral")
    pdp.plot(feature=0, heterogeneity="std_err", centering="zero_integral")
    pdp.plot(feature=0, heterogeneity="ice", centering="zero_integral")

    # test the finite difference version
    d_pdp1 = effector.DerivativePDP(data, model)
    d_pdp1.fit(features="all", centering=False)
    d_pdp1.plot(feature=0, confidence_interval="ice", centering=False)

    d_pdp2 = effector.DerivativePDP(data, model, model_jac)
    d_pdp2.fit(features="all", centering=False)
    d_pdp2.plot(feature=0, confidence_interval="ice", centering=False)


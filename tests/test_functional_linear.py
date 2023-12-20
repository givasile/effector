import numpy as np
import effector

np.random.seed(21)


def test_shap_linear():
    """
    Test the vectorized version of the SHAP function for a linear model
    """
    N = 1000
    T = 1000

    data = np.stack(
        [
            np.random.uniform(0, 1, N + 1),
            np.random.uniform(0, 1, N + 1),
        ],
        axis=1,
    )

    model = lambda x: x[:, 0] + x[:, 1]
    model_jac = lambda x: np.stack(
        [
            np.ones_like(x[:, 0]),
            np.ones_like(x[:, 1]),
        ],
        axis=1,
    )

    x = np.linspace(0, 1, T)

    # ground truth
    y_gt = x
    heterogeneity_gt = np.zeros_like(x)
    y_der_gt = np.ones_like(x)
    heterogeneity_der_gt = np.zeros_like(x)

    # test PDP
    pdp = effector.PDP(data, model, nof_instances=100)
    y, heterogeneity = pdp.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heterogeneity, heterogeneity_gt, atol=1e-1, rtol=1e-1)

    # test d-PDP withouth Jacobian
    d_pdp = effector.DerivativePDP(data, model)
    y, heter = d_pdp.eval(feature=0, xs=x, heterogeneity=True, centering=False)
    np.allclose(y, y_der_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity_der_gt, atol=1e-1, rtol=1e-1)

    # test d-PDP with Jacobian
    d_pdp = effector.DerivativePDP(data, model, model_jac)
    y, heter = d_pdp.eval(feature=0, xs=x, heterogeneity=True, centering=False)
    np.allclose(y, y_der_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity_der_gt, atol=1e-1, rtol=1e-1)

    # test SHAP
    shap = effector.SHAPDependence(data, model, nof_instances=100)
    y, heter = shap.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity_gt, atol=1e-1, rtol=1e-1)

    # test ALE
    ale = effector.ALE(data, model, nof_instances=100)
    y, heter = ale.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity_gt, atol=1e-1, rtol=1e-1)

    # test RHALE without Jacobian
    rhale = effector.RHALE(data, model, nof_instances=100)
    y, heter = rhale.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity_gt, atol=1e-1, rtol=1e-1)

    # test RHALE with Jacobian
    rhale = effector.RHALE(data, model, model_jac, nof_instances=100)
    y, heter = rhale.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity_gt, atol=1e-1, rtol=1e-1)


def test_pdp_square():
    """
    Test the vectorized version of the PDP function for a square model
    """
    N = 1000
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

    pdp = effector.PDP(data, model)
    # pdp.plot(feature=1, heterogeneity="ice", centering="zero_start")
    y, heterogeneity = pdp.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")

    # ground truth
    y_gt = x ** 2 + .5*x
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)

    # test the finite difference version
    d_pdp1 = effector.DerivativePDP(data, model)
    y, heter = d_pdp1.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")

    # ground truth
    y_gt = 2 * x
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)

    d_pdp2 = effector.DerivativePDP(data, model, model_jac)
    y, heter = d_pdp2.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")

    # ground truth
    y_gt = 2 * x
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)


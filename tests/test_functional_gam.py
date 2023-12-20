import numpy as np
import effector

np.random.seed(21)


def test_gam():
    """
    Test the vectorized version of the SHAP function for a square model
    """
    N = 1000
    T = 10000

    data = np.stack([
            np.random.rand(N + 1),
            np.random.rand(N + 1),
        ], axis=1)

    model = lambda x: x[:, 0]**3/5 + x[:, 1] ** 2/5
    model_jac = lambda x: np.stack([3 * x[:, 0]**2/5, 2 * x[:, 1]/5], axis=1)

    x = np.linspace(0, 1, T)
    y_1_gt = x**3/5
    y_2_gt = x**2/5
    heter_gt = np.zeros_like(x)
    y_1_der_gt = 3 * x**2/5
    y_2_der_gt = 2 * x/5

    # test PDP
    pdp = effector.PDP(data, model, nof_instances=100)
    y, heterogeneity = pdp.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_1_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heterogeneity, heter_gt, atol=1e-1, rtol=1e-1)
    y, heterogeneity = pdp.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_2_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heterogeneity, heter_gt, atol=1e-1, rtol=1e-1)

    # test d-PDP without Jacobian
    d_pdp = effector.DerivativePDP(data, model)
    y, heter = d_pdp.eval(feature=0, xs=x, heterogeneity=True, centering=False)
    np.allclose(y, y_2_der_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity, atol=1e-1, rtol=1e-1)
    y, heter = d_pdp.eval(feature=1, xs=x, heterogeneity=True, centering=False)
    np.allclose(y, y_1_der_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity, atol=1e-1, rtol=1e-1)

    # test d-PDP with Jacobian
    d_pdp = effector.DerivativePDP(data, model, model_jac)
    y, heter = d_pdp.eval(feature=0, xs=x, heterogeneity=True, centering=False)
    np.allclose(y, y_2_der_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heterogeneity, atol=1e-1, rtol=1e-1)
    y, heter = d_pdp.eval(feature=1, xs=x, heterogeneity=True, centering=False)
    np.allclose(y, y_1_der_gt, atol=1e-1, rtol=1e-1)

    # test SHAP
    shap_dep = effector.SHAPDependence(data, model, nof_instances=100)
    y, heter = shap_dep.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_1_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heter_gt, atol=1e-1, rtol=1e-1)
    y, heter = shap_dep.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_2_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heter_gt, atol=1e-1, rtol=1e-1)

    # test ALE
    ale = effector.ALE(data, model, nof_instances=100)
    y, heter = ale.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_1_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heter_gt, atol=1e-1, rtol=1e-1)
    y, heter = ale.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_2_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heter_gt, atol=1e-1, rtol=1e-1)

    # test RHALE without Jacobian
    rhale = effector.RHALE(data, model, nof_instances=100)
    y, heter = rhale.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_1_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heter_gt, atol=1e-1, rtol=1e-1)
    y, heter = rhale.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_2_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heter_gt, atol=1e-1, rtol=1e-1)

    # test RHALE with Jacobian
    rhale = effector.RHALE(data, model, model_jac, nof_instances=100)
    y, heter = rhale.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_1_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heter_gt, atol=1e-1, rtol=1e-1)
    y, heter = rhale.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")
    np.allclose(y, y_2_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heter, heter_gt, atol=1e-1, rtol=1e-1)


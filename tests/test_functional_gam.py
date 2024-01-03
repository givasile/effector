import numpy as np
import effector

def test_gam():
    """
    Test the vectorized version of the SHAP function for a square model
    """

    np.random.seed(21)

    N = 1000
    T = 1000
    atol = 0.1
    rtol = 0.1

    data = np.stack([
            np.random.rand(N + 1),
            np.random.rand(N + 1),
        ], axis=1)

    model = lambda x: x[:, 0]**3/5 + x[:, 1] ** 2/5
    model_jac = lambda x: np.stack([3 * x[:, 0]**2/5, 2 * x[:, 1]/5], axis=1)

    x = np.linspace(0, 1, T)
    gt = {"x1": x**3/5, "x2": x**2/5, "heterogeneity": np.zeros_like(x), "x1_der": 3 * x**2/5, "x2_der": 2 * x/5}

    # Define test cases
    test_cases = [
        {"method": effector.PDP, "kwargs": {}},
        {"method": effector.DerivativePDP, "kwargs": {"model_jac": None}},
        {"method": effector.DerivativePDP, "kwargs": {"model_jac": model_jac}},
        {"method": effector.SHAPDependence, "kwargs": {}},
        {"method": effector.ALE, "kwargs": {}},
        {"method": effector.RHALE, "kwargs": {"model_jac": None}},
        {"method": effector.RHALE, "kwargs": {"model_jac": model_jac}}
    ]

    for test_case in test_cases:
        effector_class = test_case["method"]
        kwargs = test_case["kwargs"]

        # Instantiate the effector class
        eff = effector_class(data, model, nof_instances=100, **kwargs)

        for feature in [0, 1]:
            # Evaluate the effector and retrieve results
            y, heterogeneity = eff.eval(feature=feature, xs=x, heterogeneity=True, centering="zero_start")

            np.allclose(heterogeneity, gt["heterogeneity"], atol=atol, rtol=rtol)

            if effector_class not in [effector.DerivativePDP]:
                if feature == 0:
                    np.allclose(y, gt["x1"], atol=atol, rtol=rtol)
                elif feature == 1:
                    np.allclose(y, gt["x2"], atol=atol, rtol=rtol)
            else:
                if feature == 0:
                    np.allclose(y, gt["x1_der"], atol=atol, rtol=rtol)
                elif feature == 1:
                    np.allclose(y, gt["x2_der"], atol=atol, rtol=rtol)

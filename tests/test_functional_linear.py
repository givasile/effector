import numpy as np
import effector

def test_linear():
    """
    Test the vectorized version of the SHAP function for a linear model
    """

    np.random.seed(21)

    N = 1000
    T = 1000
    rtol = .1
    atol = .1

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

    # Create a list of test cases
    test_cases = [
        {"method": effector.PDP, "init_kwargs": {}},
        {"method": effector.DerivativePDP, "init_kwargs": {}},
        {"method": effector.DerivativePDP, "init_kwargs": {"model_jac": model_jac}},
        {"method": effector.SHAPDependence, "init_kwargs": {}},
        {"method": effector.ALE, "init_kwargs": {}},
        {"method": effector.RHALE, "init_kwargs": {}},
        {"method": effector.RHALE, "init_kwargs": {"model_jac": model_jac}}
    ]

    # Iterate through test cases
    for test_case in test_cases:
        effector_class = test_case["method"]
        kwargs = test_case["init_kwargs"]

        # Instantiate the effector class
        eff = effector_class(data, model, **kwargs)

        # Evaluate the effector and retrieve results
        y, heterogeneity = eff.eval(feature=0, xs=x, heterogeneity=True, centering="zero_start")

        # Check assertions
        np.allclose(y, x, atol=atol, rtol=rtol)
        np.allclose(heterogeneity, np.zeros_like(x), atol=atol, rtol=rtol)
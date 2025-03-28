import numpy as np
import effector


def test_regional():
    np.random.seed(21)

    N = 1000
    T = 1000

    # create data, model
    data = np.stack(
        [
            np.random.uniform(-1, 1, N),
            np.random.uniform(-1, 1, N),
            np.random.randint(0, 2, N)
        ],
        axis=1)

    def model(x):
        y = np.zeros_like(x[:, 0])
        ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
        y[ind] = 5*x[ind, 0]
        return y

    def model_jac(x):
        y = np.zeros_like(x)
        ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
        y[ind, 0] = 5
        return y

    xs = np.linspace(-1, 1, T)
    methods = ["pdp", "d-pdp", "ale", "rhale", "shap", "shapiq"]
    for method in methods:
        if method == "pdp":
            reg_eff = effector.RegionalPDP(data, model, nof_instances=1000)
            reg_eff.fit(0)
            y, std = reg_eff.eval(0, 3, xs, heterogeneity=True, centering=True)
        elif method == "d-pdp":
            reg_eff = effector.RegionalDerPDP(data, model, model_jac, nof_instances=1000)
            reg_eff.fit(0)
            y, std = reg_eff.eval(0, 3, xs, heterogeneity=True, centering=False)
        elif method == "ale":
            reg_eff = effector.RegionalALE(data, model, nof_instances=1000)
            reg_eff.fit(0)
            y, std = reg_eff.eval(0, 3, xs, heterogeneity=True)
        elif method == "rhale":
            reg_eff = effector.RegionalRHALE(data, model, model_jac, nof_instances=1000)
            reg_eff.fit(0)
            y, std = reg_eff.eval(0, 3, xs, heterogeneity=True)
        elif method == "shap":
            reg_eff = effector.RegionalShapDP(data, model, nof_instances=100, backend="shap")
            reg_eff.fit(0)
            y, std = reg_eff.eval(0, 3, xs, heterogeneity=True)
        elif method == "shapiq":
            reg_eff = effector.RegionalShapDP(data, model, nof_instances=100, backend="shapiq")
            reg_eff.fit(0)
            y, std = reg_eff.eval(0, 3, xs, heterogeneity=True)


        np.allclose(y, 5*xs, atol=0.1, rtol=0.1)
        np.allclose(std, 0, atol=0.1, rtol=0.1)

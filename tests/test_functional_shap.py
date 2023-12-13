import numpy as np
import effector
import shap
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

np.random.seed(21)


def test_shap_square():
    """
    Test the vectorized version of the SHAP function for a square model
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

    # compute the SHAP dependence
    shap_dep = effector.SHAPDependence(data, model)
    shap_dep.fit(features="all", centering="zero_integral")
    yy = shap_dep.eval(feature=1, xs=x, centering="zero_integral")

    # plot
    plt.figure()
    plt.plot(x, yy)
    plt.plot(x, shap_dep.eval(0, x))
    plt.show()

    # shap_explainer = shap.Explainer(model, data)
    # explanation = shap_explainer(data)
    # yy = explanation.values[:, 1]
    # xx = data[:, 1]
    #
    # # make xx monotonic
    # idx = np.argsort(xx)
    # xx = xx[idx]
    # yy = yy[idx]
    #
    # spline = UnivariateSpline(xx, yy)
    #
    # plt.figure()
    # plt.plot(xx, yy, "rx")
    # plt.plot(xx, spline(xx), "b-")
    # plt.show()


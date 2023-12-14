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

    model = lambda x: x[:,0] * x[:, 1] ** 2

    x = np.linspace(0, 1, T)

    # compute the SHAP dependence
    shap_dep = effector.SHAPDependence(data, model)
    shap_dep.fit(features="all", centering="zero_integral")
    yy = shap_dep.eval(feature=1, xs=x, centering="zero_integral")

    shap_dep.plot(feature=1, confidence_interval="std", centering="zero_integral")
    shap_dep.plot(feature=1, confidence_interval="shap_values", centering="zero_integral")


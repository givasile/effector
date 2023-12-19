import numpy as np
import effector

np.random.seed(21)


def test_shap_linear():
    """
    Test the vectorized version of the SHAP function for a linear model
    """
    N = 1000
    T = 10000

    data = np.stack(
        [
            np.random.uniform(0, 1, N + 1),
            np.random.uniform(0, 1, N + 1),
        ],
        axis=1,
    )

    model = lambda x: x[:, 0] + x[:, 1]

    x = np.linspace(0, 1, T)

    # compute the SHAP dependence
    shap_dep = effector.SHAPDependence(data, model, nof_instances=100)
    shap_dep.fit(features="all", centering=True)
    y, heterogeneity, _ = shap_dep.eval(feature=1, xs=x, heterogeneity=True, centering=True)

    # ground truth
    y_gt = x
    heterogeneity_gt = np.zeros_like(x)
    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)
    np.allclose(heterogeneity, heterogeneity_gt, atol=1e-1, rtol=1e-1)

    # shap_dep.plot(
    #     feature=0,
    #     heterogeneity="shap_values",
    #     centering=False,
    #     show_avg_output=False,
    #     nof_points = 100
    # )

    # shap_dep.plot(
    #     feature=1,
    #     heterogeneity="shap_values",
    #     centering=False,
    #     show_avg_output=False,
    # )


def test_shap_square():
    """
    Test the vectorized version of the SHAP function for a square model
    """
    N = 1000
    T = 10000

    data = np.stack(
        [
            np.random.uniform(0, 1, N + 1),
            np.random.uniform(0, 1, N + 1),
        ],
        axis=1,
    )

    model = lambda x: x[:, 0] * x[:, 1] ** 2

    x = np.linspace(0, 1, T)

    # compute the SHAP dependence
    shap_dep = effector.SHAPDependence(data, model, nof_instances=100)
    shap_dep.fit(features="all", centering=True)
    y, heter, _ = shap_dep.eval(feature=1, xs=x, heterogeneity=True, centering="zero_start")

    y_gt = x ** 2 / 2

    np.allclose(y, y_gt, atol=1e-1, rtol=1e-1)

    # shap_dep.plot(
    #     feature=0,
    #     heterogeneity="shap_values",
    #     centering=False,
    #     y_limits=[-0.5, 0.5],
    #     show_avg_output=False,
    # )
    #
    # shap_dep.plot(
    #     feature=1,
    #     heterogeneity="std",
    #     centering=True,
    #     y_limits=[-0.5, 0.5],
    #     show_avg_output=False,
    # )

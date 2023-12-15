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
    shap_dep = effector.SHAPDependence(data, model)
    shap_dep.fit(features="all", centering=False)
    # yy = shap_dep.eval(feature=1, xs=x, centering=False)

    shap_dep.plot(
        feature=0,
        confidence_interval="shap_values",
        centering=False,
        y_limits=[-0.5, 0.5],
        show_avg_output=False,
    )

    shap_dep.plot(
        feature=1,
        confidence_interval="shap_values",
        centering=False,
        y_limits=[-0.5, 0.5],
        show_avg_output=False,
    )


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
    shap_dep = effector.SHAPDependence(data, model)
    shap_dep.fit(features="all", centering=False)
    # yy = shap_dep.eval(feature=1, xs=x, centering=False)

    shap_dep.plot(
        feature=0,
        confidence_interval="shap_values",
        centering=False,
        y_limits=[-0.5, 0.5],
        show_avg_output=False,
    )

    shap_dep.plot(
        feature=1,
        confidence_interval="shap_values",
        centering=False,
        y_limits=[-0.5, 0.5],
        show_avg_output=False,
    )

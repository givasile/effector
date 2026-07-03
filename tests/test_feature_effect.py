import numpy as np
import pytest

import effector


def _linear_dataset(N=1000, seed=21):
    np.random.seed(seed)
    x1 = np.random.uniform(0, 1, size=N)
    x2 = np.random.normal(loc=x1, scale=0.1)
    x3 = np.random.uniform(0, 1, size=N)
    return np.stack((x1, x2, x3), axis=-1)


def predict(x):
    return 7 * x[:, 0] - 3 * x[:, 1] + 4 * x[:, 2]


def predict_grad(x):
    df_dx1 = 7 * np.ones([x.shape[0]])
    df_dx2 = -3 * np.ones([x.shape[0]])
    df_dx3 = 4 * np.ones([x.shape[0]])
    return np.stack([df_dx1, df_dx2, df_dx3], axis=-1)


def test_comparison_returns_fig_ax():
    X = _linear_dataset()
    fe = effector.FeatureEffect(X, predict, model_jac=predict_grad)
    for i in range(3):
        ret = fe.plot(feature=i, methods=["PDP", "ALE", "RHALE"], show_plot=False)
        assert ret is not None
        fig, ax = ret
        # one line per method (+ none extra, since show_avg_output is False)
        assert len(ax.get_lines()) == 3


def test_method_objects_are_cached():
    X = _linear_dataset()
    fe = effector.FeatureEffect(X, predict, model_jac=predict_grad)
    fe.plot(feature=0, methods=["PDP", "ALE"], show_plot=False)
    fe.plot(feature=1, methods=["PDP", "ALE"], show_plot=False)
    assert set(fe._methods.keys()) == {"pdp", "ale"}


def test_unknown_method_raises():
    X = _linear_dataset()
    fe = effector.FeatureEffect(X, predict)
    with pytest.raises(ValueError):
        fe.plot(feature=0, methods=["not_a_method"], show_plot=False)


def test_rhale_without_jac_warns_and_still_works():
    X = _linear_dataset()
    fe = effector.FeatureEffect(X, predict)  # no model_jac
    with pytest.warns(UserWarning):
        ret = fe.plot(feature=0, methods=["RHALE"], show_plot=False)
    assert ret is not None


def test_centering_false_is_coerced_with_warning():
    X = _linear_dataset()
    fe = effector.FeatureEffect(X, predict)
    with pytest.warns(UserWarning):
        ret = fe.plot(
            feature=0, methods=["PDP", "ALE"], centering=False, show_plot=False
        )
    assert ret is not None


def test_shap_alias():
    X = _linear_dataset()
    fe = effector.FeatureEffect(X, predict)
    assert fe._canonical("SHAP") == "shapdp"
    assert fe._canonical("shap-dp") == "shapdp"

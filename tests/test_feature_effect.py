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


def test_overlaid_curves_match_each_methods_eval():
    """§3.5: every overlaid curve equals the corresponding method's own
    centered eval on the same grid (the facade adds no computation of its own)."""
    X = _linear_dataset()
    fe = effector.FeatureEffect(X, predict, model_jac=predict_grad)
    ret = fe.plot(feature=0, methods=["PDP", "ALE", "RHALE"], show_plot=False)
    fig, ax = ret
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    assert set(lines) == {"PDP", "ALE", "RHALE"}
    for name, line in lines.items():
        method = fe._methods[fe._canonical(name)]
        y = method.eval(0, line.get_xdata(), centering="zero_integral")
        np.testing.assert_allclose(line.get_ydata(), y, atol=1e-8)


# --- categorical feature support ------------------------------------------

def _cat_dataset(N=1500, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, N).astype(float)
    g[:3] = [0.0, 1.0, 2.0]  # ensure all 3 levels present
    return np.column_stack([g, rng.uniform(-1, 1, N)])


def _cat_predict(x):
    return x[:, 0] + 0.5 * x[:, 1]


CAT_SCHEMA = {
    "feature_types": ["nominal", "continuous"],
    "category_names": [["a", "b", "c"], None],
}


def test_facade_categorical_skips_unsupported_and_labels():
    fe = effector.FeatureEffect(_cat_dataset(), _cat_predict, schema=CAT_SCHEMA)
    with pytest.warns(UserWarning, match="Skipping.*RHALE"):
        fig, ax = fe.plot(0, methods=["PDP", "ALE", "RHALE"], show_plot=False)
    assert len(ax.get_lines()) == 2  # RHALE dropped for nominal -> PDP + ALE
    assert [t.get_text() for t in ax.get_xticklabels()] == ["a", "b", "c"]


def test_facade_categorical_eval_at_levels():
    fe = effector.FeatureEffect(_cat_dataset(), _cat_predict, schema=CAT_SCHEMA)
    curves = fe.eval(0, np.array([0.0, 1.0, 2.0]), methods=["PDP", "ALE"])
    assert len(curves) == 2
    for y in curves.values():
        assert y.shape == (3,)


def test_facade_categorical_all_unsupported_raises():
    fe = effector.FeatureEffect(_cat_dataset(), _cat_predict, schema=CAT_SCHEMA)
    with pytest.raises(ValueError, match="No requested method supports"):
        fe.plot(0, methods=["RHALE"], show_plot=False)


def test_facade_ordinal_keeps_rhale():
    rng = np.random.default_rng(0)
    X = np.column_stack([rng.integers(0, 4, 1500).astype(float), rng.uniform(-1, 1, 1500)])
    jac = lambda z: np.column_stack([np.zeros(len(z)), 0.5 * np.ones(len(z))])
    fe = effector.FeatureEffect(
        X, _cat_predict, model_jac=jac, schema={"feature_types": ["ordinal", "continuous"]}
    )
    fig, ax = fe.plot(0, methods=["PDP", "ALE", "RHALE"], show_plot=False)
    assert len(ax.get_lines()) == 3  # RHALE supported for ordinal

"""Unit tests for effector/adapters.py — facilitation-only model wrappers.

Layer: unit (Tier-1, tiny N). Adapters return plain numpy->numpy callables and
never run automatically; the duck-typed fakes below cover the contract without
framework dependencies. Real-sklearn tests run too (sklearn is in the test
group); torch tests skip unless torch is installed locally.
"""

import numpy as np
import pytest

import effector
from effector.adapters import check, classifier_proba, from_sklearn, from_torch


class FakeRegressor:
    def predict(self, X):
        return np.asarray(X)[:, 0] * 2.0


class FakeColumnRegressor:
    """predict returns (N, 1) — adapters must ravel it."""

    def predict(self, X):
        return (np.asarray(X)[:, 0] * 2.0)[:, None]


class FakeBadRegressor:
    """predict returns (N, 2) — adapters must reject it loudly."""

    def predict(self, X):
        X = np.asarray(X)
        return np.stack([X[:, 0], X[:, 0]], axis=1)


class FakeClassifier:
    classes_ = np.array(["no", "maybe", "yes"])

    def predict(self, X):
        return np.repeat("no", len(X))

    def predict_proba(self, X):
        X = np.asarray(X)
        p = 1 / (1 + np.exp(-X[:, 0]))
        rest = (1 - p) / 2
        return np.stack([rest, rest, p], axis=1)


X = np.linspace(-1, 1, 10).reshape(-1, 2)


# --- from_sklearn ---


def test_from_sklearn_wraps_predict():
    model = from_sklearn(FakeRegressor())
    y = model(X)
    assert y.shape == (len(X),)
    np.testing.assert_allclose(y, X[:, 0] * 2.0)


def test_from_sklearn_ravels_column_output():
    model = from_sklearn(FakeColumnRegressor())
    assert model(X).shape == (len(X),)


def test_from_sklearn_bad_shape_message():
    model = from_sklearn(FakeBadRegressor())
    with pytest.raises(ValueError, match=r"from_sklearn.*shape.*expected"):
        model(X)


def test_from_sklearn_rejects_classifier():
    with pytest.raises(ValueError, match="classifier_proba"):
        from_sklearn(FakeClassifier())


def test_from_sklearn_rejects_non_estimator():
    with pytest.raises(TypeError, match="predict"):
        from_sklearn(lambda x: x)


# --- classifier_proba ---


def test_classifier_proba_by_label():
    model = classifier_proba(FakeClassifier(), class_="yes")
    y = model(X)
    assert y.shape == (len(X),)
    np.testing.assert_allclose(y, 1 / (1 + np.exp(-X[:, 0])))


def test_classifier_proba_by_index():
    model = classifier_proba(FakeClassifier(), class_=2)
    np.testing.assert_allclose(model(X), 1 / (1 + np.exp(-X[:, 0])))


def test_classifier_proba_label_beats_index():
    """Integer classes_: an int class_ matching a label resolves as a label."""

    class IntClassifier(FakeClassifier):
        classes_ = np.array([5, 7, 9])

    model = classifier_proba(IntClassifier(), class_=9)  # label 9 -> column 2
    np.testing.assert_allclose(model(X), 1 / (1 + np.exp(-X[:, 0])))


def test_classifier_proba_unknown_class_lists_classes():
    with pytest.raises(ValueError, match=r"maybe.*yes|classes_"):
        classifier_proba(FakeClassifier(), class_="nope")


def test_classifier_proba_rejects_regressor():
    with pytest.raises(TypeError, match="from_sklearn"):
        classifier_proba(FakeRegressor())


# --- check ---


def test_check_passes_good_model():
    assert check(lambda x: np.asarray(x)[:, 0], X) is None


def test_check_rejects_non_callable():
    with pytest.raises(TypeError, match="adapters"):
        check(FakeRegressor(), X)  # the raw estimator, a classic mistake


def test_check_rejects_bad_shape():
    with pytest.raises(ValueError, match="shape"):
        check(lambda x: np.asarray(x), X)  # returns (2, D)


def test_check_wraps_model_exception():
    def broken(x):
        raise AttributeError("no columns")

    with pytest.raises(AttributeError, match="numpy-in/numpy-out"):
        check(broken, X)


def test_check_validates_jacobian_shape():
    model = from_sklearn(FakeRegressor())
    good_jac = lambda x: np.tile([2.0, 0.0], (len(x), 1))  # noqa: E731
    assert check(model, X, model_jac=good_jac) is None
    bad_jac = lambda x: np.zeros(len(x))  # noqa: E731
    with pytest.raises(ValueError, match="model_jac"):
        check(model, X, model_jac=bad_jac)


# --- ingest error path ---


def test_ingest_non_callable_model_points_to_adapters():
    with pytest.raises(TypeError, match="effector.adapters"):
        effector.PDP(X, FakeRegressor())


def test_ingest_non_callable_jac_rejected():
    with pytest.raises(TypeError, match="model_jac"):
        effector.RHALE(X, lambda x: np.asarray(x)[:, 0], "not-a-callable")


# --- end-to-end: adapter output feeds an engine ---


def test_adapter_feeds_engine():
    model = from_sklearn(FakeRegressor())
    pdp = effector.PDP(X, model)
    pdp.fit(features=[0])
    xs = np.linspace(-1, 1, 5)
    assert pdp.eval(0, xs).shape == (5,)


# --- real frameworks ---


def test_real_sklearn_regressor():
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    from sklearn.linear_model import LinearRegression

    est = LinearRegression().fit(X, X[:, 0] * 3.0)
    model = from_sklearn(est)
    check(model, X)
    np.testing.assert_allclose(model(X), X[:, 0] * 3.0, atol=1e-10)


def test_real_sklearn_classifier_proba():
    sklearn = pytest.importorskip("sklearn")  # noqa: F841
    from sklearn.linear_model import LogisticRegression

    y = (X[:, 0] > 0).astype(int)
    est = LogisticRegression().fit(X, y)
    model = classifier_proba(est, class_=1)
    check(model, X)
    proba = model(X)
    assert proba.min() >= 0 and proba.max() <= 1


def test_real_torch_forward_and_jacobian():
    torch = pytest.importorskip("torch")

    lin = torch.nn.Linear(2, 1)
    with torch.no_grad():
        lin.weight[:] = torch.tensor([[2.0, -1.0]])
        lin.bias[:] = 0.0
    model, model_jac = from_torch(lin, jacobian=True)
    check(model, X, model_jac=model_jac)
    np.testing.assert_allclose(model(X), X @ [2.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(
        model_jac(X), np.tile([2.0, -1.0], (len(X), 1)), atol=1e-6
    )

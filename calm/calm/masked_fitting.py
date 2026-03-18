import copy
import numpy as np
import matplotlib.pyplot as plt

from pygam import LinearGAM, LogisticGAM, s, f
from pygam.terms import SplineTerm, TermList
from sklearn.tree import DecisionTreeRegressor
from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier,
)

from calm import neural_additive_models


class MaskedGAM:
    def __init__(self):
        self.constants = None
        self.offset = None

    def _prepare_X(self, X):
        """
        If X is a DataFrame, convert it to a numpy array.
        Raise an error if the resulting array is not numeric.
        """
        # Convert pandas DataFrame to numpy array
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if not np.issubdtype(X.dtype, np.number):
            raise ValueError("Input data X must be numeric.")
        return X

    def _default_mask(self, X, mask):
        """
        If mask is None, return an all-ones mask with the same shape as X.
        If mask is provided and is a DataFrame, convert it to a numpy array.
        """
        if mask is None:
            return np.ones_like(X)
        else:
            return self._prepare_X(mask)

    def fit(self, X, y, mask, axis_limits=None):
        raise NotImplementedError()

    def _find_constants(self):
        constant_vec = np.zeros(self.dim)
        for i in range(self.dim):
            min_val = self.axis_limits[0, i]
            max_val = self.axis_limits[1, i]
            xx = np.linspace(min_val, max_val, 100)
            yy = self.effect_unnorm(xx, i)
            constant_vec[i] = np.mean(yy)
        self.constants = constant_vec
        self.offset = np.sum(constant_vec)

    def predict(self, X, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[1]):
            y_pred += self.effect_unnorm(X[:, i], i) * mask[:, i]
        return y_pred

    def effect_unnorm(self, xs, i):
        raise NotImplementedError()

    def effect(self, xs, i):
        if self.constants is None:
            self._find_constants()
        y = self.effect_unnorm(xs, i)
        return y - self.constants[i]

    def plot_feature(self, i, axis_limits=None, centering=True, x_lim=None, y_lim=None):
        if axis_limits is None:
            axis_limits = self.axis_limits[:, i]
        x = np.linspace(axis_limits[0], axis_limits[1], 100)
        y = self.effect(x, i) if centering else self.effect_unnorm(x, i)
        plt.figure()
        plt.title(f"Feature {i}")
        plt.plot(x, y)
        plt.xlabel("x_%d" % i)
        plt.ylabel("y")
        if x_lim is not None:
            plt.xlim(x_lim)
        if y_lim is not None:
            plt.ylim(y_lim)
        plt.show(block=False)


class MaskedNAM(MaskedGAM):
    def __init__(self, dim, subnetwork=None, classification=False, loss="mse"):
        self.dim = dim
        self.model = neural_additive_models.simple_nam(
            dim, subnetwork, classification=classification
        )
        self.loss = loss
        super().__init__()

    def fit(self, X, y, mask=None, axis_limits=None, optimizer="adam", epochs=10):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        if axis_limits is None:
            self.axis_limits = np.array(
                [[np.min(X[:, i]), np.max(X[:, i])] for i in range(X.shape[1])]
            ).T
        else:
            self.axis_limits = axis_limits

        self.model.compile(optimizer=optimizer, loss=self.loss)
        self.model.fit([X, mask], y, epochs=epochs)

        # Compute per-feature constants
        constant_vec = np.zeros(X.shape[1])
        for i in range(X.shape[1]):
            xx = np.linspace(self.axis_limits[0, i], self.axis_limits[1, i], 100)
            yy = self.effect_unnorm(xx, i)
            constant_vec[i] = np.mean(yy)
        self.constants = constant_vec
        self.offset = np.sum(constant_vec)

    def predict(self, X, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        return self.model.predict([X, mask]).flatten()

    def effect_unnorm(self, xs, i):
        submodule = self.model.layers[3].submodels[i]  # Access the i-th submodel
        xs = np.expand_dims(xs, axis=-1)
        return np.squeeze(submodule.predict(xs))


class MaskedNAMRegressor(MaskedNAM):
    def __init__(self, dim, subnetwork=None):
        super().__init__(dim, subnetwork, classification=False, loss="mse")


class MaskedNAMClassifier(MaskedNAM):
    def __init__(self, dim, subnetwork=None):
        super().__init__(
            dim, subnetwork, classification=True, loss="binary_crossentropy"
        )

    def predict_proba(self, X, mask=None):
        """Return probabilities for class 1."""
        return super().predict(X, mask)

    def predict(self, X, mask=None):
        """Return binary predictions (0 or 1)."""
        return (self.predict_proba(X, mask) >= 0.5).astype(int)


class NoInteractionsEBM(MaskedGAM):
    def __init__(self):
        super().__init__()

    def fit(self, X, y, mask=None, axis_limits=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        if axis_limits is None:
            self.axis_limits = np.array(
                [[np.min(X[:, i]), np.max(X[:, i])] for i in range(X.shape[1])]
            ).T
        else:
            self.axis_limits = axis_limits

        X_copy = X.copy()
        X_copy[mask == 0] = np.nan  # Mask inactive features

        self.model.fit(X_copy, y)  # Train the EBM model

    def effect_unnorm(self, xs, i):
        bins = self.model.bins_[i][0]
        bin_idx = np.digitize(xs, bins) + 1
        return self.model.term_scores_[i][bin_idx]

    def predict(self, X, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        X_copy = X.copy()
        X_copy[mask == 0] = np.nan
        return self.model.predict(X_copy)


class NoInteractionsEBMRegressor(NoInteractionsEBM):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = ExplainableBoostingRegressor(
            interactions=0, random_state=42, **kwargs
        )


class NoInteractionsEBMClassifier(NoInteractionsEBM):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = ExplainableBoostingClassifier(
            interactions=0, random_state=42, **kwargs
        )

    def predict_proba(self, X, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        X_copy = X.copy()
        X_copy[mask == 0] = np.nan
        return self.model.predict_proba(X_copy)[:, 1]

    def predict(self, X, mask=None):
        return (self.predict_proba(X, mask) >= 0.5).astype(int)


class WithInteractionsEBM(MaskedGAM):
    def __init__(self):
        super().__init__()

    def fit(self, X, y, mask=None, axis_limits=None, nof_interactions=2):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        if axis_limits is None:
            self.axis_limits = np.array(
                [[np.min(X[:, i]), np.max(X[:, i])] for i in range(X.shape[1])]
            ).T
        else:
            self.axis_limits = axis_limits

        X_copy = X.copy()
        X_copy[mask == 0] = np.nan  # Mask inactive features

        self.model.fit(X_copy, y)  # Train the EBM model

    def effect_unnorm(self, xs, i):
        bins = self.model.bins_[i][0]
        bin_idx = np.digitize(xs, bins) + 1
        return self.model.term_scores_[i][bin_idx]

    def effect_interactions(self, xs, i, j):
        raise NotImplementedError("Interaction effects computation is not implemented.")

    def predict(self, X, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        X_copy = X.copy()
        X_copy[mask == 0] = np.nan
        return self.model.predict(X_copy)


class WithInteractionsEBMRegressor(WithInteractionsEBM):
    def __init__(self, nof_interactions=2):
        super().__init__()
        self.model = ExplainableBoostingRegressor(
            interactions=nof_interactions, random_state=42
        )


class WithInteractionsEBMClassifier(WithInteractionsEBM):
    def __init__(self, nof_interactions=2):
        super().__init__()
        self.model = ExplainableBoostingClassifier(
            interactions=nof_interactions, random_state=42
        )

    def predict_proba(self, X, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        X_copy = X.copy()
        X_copy[mask == 0] = np.nan
        return self.model.predict_proba(X_copy)[:, 1]

    def predict(self, X, mask=None):
        return (self.predict_proba(X, mask) >= 0.5).astype(int)


class PyGAM(MaskedGAM):
    def __init__(self, dim):
        self.dim = dim
        super().__init__()

    def fit(self, X, y, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        XX = np.column_stack([X, mask])

        constant_vec = np.zeros(self.dim)
        for i in range(self.dim):
            xx = np.linspace(-1, 1, 100)
            yy = self.effect_unnorm(xx, i)
            constant_vec[i] = np.mean(yy)
        self.constants = constant_vec

    def predict(self, X, mask=None):
        X = self._prepare_X(X)
        if mask is None:
            mask = np.ones_like(X)
        else:
            mask = self._default_mask(X, mask)
        XX = np.column_stack([X, mask])
        return self.model.predict(XX)

    def effect_unnorm(self, xx, i):
        xx_matrix = np.zeros((xx.shape[0], self.dim * 2))
        xx_matrix[:, i] = xx
        xx_matrix[:, i + self.dim] = 1
        return self.model.partial_dependence(term=i, X=xx_matrix)


class PyGAMRegressor(PyGAM):
    def __init__(self, dim, use_grid_lam_search=False):
        super().__init__(dim)
        self.use_grid_lam_search = use_grid_lam_search

        terms = TermList(*[SplineTerm(i) for i in range(self.dim)])
        self.model = LinearGAM(terms)

    def fit(self, X, y, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        XX = np.column_stack([X, mask])

        if self.use_grid_lam_search:
            self.model.gridsearch(XX, y, lam=[1, 10, 100], keep_best=True)
        else:
            self.model.fit(XX, y)

        super().fit(X, y, mask=mask)


class PyGAMClassifier(PyGAM):
    def __init__(self, dim, use_grid_lam_search=False):
        super().__init__(dim)
        self.use_grid_lam_search = use_grid_lam_search

        terms = TermList(*[SplineTerm(i) for i in range(self.dim)])
        self.model = LogisticGAM(terms)

    def fit(self, X, y, mask=None):
        X = self._prepare_X(X)
        mask = self._default_mask(X, mask)
        XX = np.column_stack([X, mask])

        if self.use_grid_lam_search:
            self.model.gridsearch(XX, y, lam=[1, 10, 100], keep_best=True)
        else:
            self.model.fit(XX, y)

        super().fit(X, y, mask=mask)


# Dictionaries registering all masked GAM models by task
MASKED_GAM_CLASSIFICATION_MODELS = {
    "MaskedNAMClassifier": MaskedNAMClassifier,
    "NoInteractionsEBMClassifier": NoInteractionsEBMClassifier,
    "WithInteractionsEBMClassifier": WithInteractionsEBMClassifier,
    "PyGAMClassifier": PyGAMClassifier,
}

MASKED_GAM_REGRESSION_MODELS = {
    "MaskedNAMRegressor": MaskedNAMRegressor,
    "NoInteractionsEBMRegressor": NoInteractionsEBMRegressor,
    "WithInteractionsEBMRegressor": WithInteractionsEBMRegressor,
    "PyGAMRegressor": PyGAMRegressor,
}

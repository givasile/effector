import numpy as np
from pygam import LinearGAM, LogisticGAM, s, f
from pygam.terms import SplineTerm, TermList
from interpret.glassbox import (
    ExplainableBoostingRegressor,
    ExplainableBoostingClassifier,
)
from nodegam.sklearn import (
    NodeGAMClassifier as NodeGAMClassifierOriginal,
    NodeGAMRegressor as NodeGAMRegressorOriginal,
)
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from abc import ABC, abstractmethod
from uuid import uuid4
from gaminet import GAMINet


# -----------------------
# GAM Models using interpret.glassbox
# -----------------------


class BaseGAMModel(ABC):
    def __init__(self, nof_interactions, constr_func):
        self.nof_interactions = nof_interactions
        self.constr_func = constr_func
        self.model = None

    def fit(self, X, y):
        self.model = self.constr_func(
            interactions=self.nof_interactions, random_state=42
        )
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        return self.model.predict(X)


class EBMRegressor(BaseGAMModel):
    def __init__(self):
        super().__init__(nof_interactions=0, constr_func=ExplainableBoostingRegressor)


class EBMClassifier(BaseGAMModel):
    def __init__(self):
        super().__init__(nof_interactions=0, constr_func=ExplainableBoostingClassifier)


class EBM2Regressor(BaseGAMModel):
    def __init__(self, nof_interactions=0.9):
        super().__init__(
            nof_interactions=nof_interactions, constr_func=ExplainableBoostingRegressor
        )


class EBM2Classifier(BaseGAMModel):
    def __init__(self, nof_interactions=0.9):
        super().__init__(
            nof_interactions=nof_interactions, constr_func=ExplainableBoostingClassifier
        )


# -----------------------
# PyGAM Models
# -----------------------


class BasePyGAMModel(ABC):
    def __init__(self, constr_func):
        self.model = None
        self.constr_func = constr_func

    def fit(self, X, y):
        terms = TermList(*[SplineTerm(i) for i in range(X.shape[1])])
        self.model = self.constr_func(terms=terms)
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        # Ensure X is properly formatted (using column_stack to mimic expected behavior)
        XX = np.column_stack([X])
        return self.model.predict(XX)


class PyGAMRegressor(BasePyGAMModel):
    def __init__(self):
        super().__init__(LinearGAM)


class PyGAMClassifier(BasePyGAMModel):
    def __init__(self):
        super().__init__(LogisticGAM)


# -----------------------
# NodeGAM Models
# -----------------------


class BaseNodeGAMModel(ABC):
    def __init__(self, constr_func, device="cuda", **kwargs):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available; falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        print(f"Selected device: {self.device}")
        self.constr_func = constr_func
        self.model = None
        self.kwargs = kwargs

    def fit(self, X, y):
        # BUGFIX: name should be unique for each experiment, else it tries to
        # load old checkpoints and (sometimes) fails
        if "name" not in self.kwargs:
            self.kwargs["name"] = "nodegam_model" + str(uuid4())

        # NodeGAM requires pandas DataFrame input
        self.model = self.constr_func(
            in_features=X.shape[1], seed=42, device=self.device, **self.kwargs
        )
        if not isinstance(y, np.ndarray):
            if isinstance(y, pd.Series):
                y = y.to_numpy()
            else:
                y = np.array(y)
        self.model.fit(pd.DataFrame(X), y)
        return self.model

    def predict(self, X):
        X_df = pd.DataFrame(X)
        preds = self.model.predict(X_df)
        return self._post_process_prediction(preds)

    @abstractmethod
    def _post_process_prediction(self, predictions):
        pass


class NodeGAMClassifier(BaseNodeGAMModel):
    def __init__(self, device="cuda", **kwargs):
        kwargs["ga2m"] = 0
        super().__init__(constr_func=NodeGAMClassifierOriginal, device=device, **kwargs)

    def _post_process_prediction(self, predictions):
        return (predictions > 0.5).astype(int)


class NodeGAMRegressor(BaseNodeGAMModel):
    def __init__(self, device="cuda", **kwargs):
        kwargs["ga2m"] = 0
        super().__init__(constr_func=NodeGAMRegressorOriginal, device=device, **kwargs)

    def _post_process_prediction(self, predictions):
        return predictions


class NodeGAM2Classifier(BaseNodeGAMModel):
    def __init__(self, device="cuda", **kwargs):
        kwargs["ga2m"] = 1
        super().__init__(constr_func=NodeGAMClassifierOriginal, device=device, **kwargs)

    def _post_process_prediction(self, predictions):
        return (predictions > 0.5).astype(int)


class NodeGAM2Regressor(BaseNodeGAMModel):
    def __init__(self, device="cuda", **kwargs):
        kwargs["ga2m"] = 1
        super().__init__(constr_func=NodeGAMRegressorOriginal, device=device, **kwargs)

    def _post_process_prediction(self, predictions):
        return predictions


class BaseGAMINetModel(ABC):
    def __init__(self, meta_info, task_type):
        self.model = GAMINet(meta_info=meta_info, task_type=task_type)

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.values
        self.model.fit(X, y)
        return self.model

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        preds = self.model.predict(X)
        return self._post_process_prediction(preds)

    @abstractmethod
    def _post_process_prediction(self, predictions):
        pass


class GAMINetRegressor(BaseGAMINetModel):
    def __init__(self, meta_info, task_type):
        super().__init__(meta_info, task_type)

    def _post_process_prediction(self, predictions):
        return predictions


class GAMINetClassifier(BaseGAMINetModel):
    def __init__(self, meta_info, task_type):
        super().__init__(meta_info, task_type)

    def _post_process_prediction(self, predictions):
        return (predictions > 0.5).astype(int)


# -----------------------
# Dictionaries registering competitor models
# -----------------------

COMPETITOR_CLASSIFICATION_MODELS = {
    "EBMClassifier": EBMClassifier,
    "EBM2Classifier": EBM2Classifier,
    "PyGAMClassifier": PyGAMClassifier,
    "NodeGAMClassifier": NodeGAMClassifier,
    "NodeGAM2Classifier": NodeGAM2Classifier,
    "GAMINetClassifier": GAMINetClassifier,
}

COMPETITOR_REGRESSION_MODELS = {
    "EBMRegressor": EBMRegressor,
    "EBM2Regressor": EBM2Regressor,
    "PyGAMRegressor": PyGAMRegressor,
    "NodeGAMRegressor": NodeGAMRegressor,
    "NodeGAM2Regressor": NodeGAM2Regressor,
    "GAMINetRegressor": GAMINetRegressor,
}

from typing import Tuple, List
import pandas as pd
import numpy as np

from aif360.sklearn.datasets import fetch_compas

from .base import Dataset

class COMPAS(Dataset):
    def fetch(self):
        self.X, self.y = fetch_compas() # type: ignore
        self.all_features = self.X.columns.tolist()
        self.numerical_features = self.X.select_dtypes("number").columns.tolist()
        self.categorical_features = self.X.columns.difference(self.numerical_features)
        return self
    
    def preprocess(self):
        X = self.X
        y = self.y

        X, y = preprocess_compas(X, y)
        self.all_features = X.columns.tolist()
        self.numerical_features = X.select_dtypes("number").columns.tolist()
        self.categorical_features = X.columns.difference(self.numerical_features).tolist()

        self.X = X
        self.y = y
        return self
    
    def get_Xy(self):
        return self.X, self.y
    
    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features

def preprocess_compas(X: pd.DataFrame, y: pd.Series):
    X['target'] = y
    X = X.reset_index(drop=True)
    X = X.drop(columns=["c_charge_desc"])
    X["target"] = X["target"].map({"Recidivated": 0, "Survived": 1}).cat.codes

    y = X["target"]
    X = X.drop(columns=["target"])

    # random shuffle
    shuffled_index = np.random.permutation(X.shape[0])
    X = X.iloc[shuffled_index]
    y = y.iloc[shuffled_index]
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    
    return X, y

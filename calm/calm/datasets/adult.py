from typing import Tuple, List, Optional
import numpy as np
import pandas as pd

from ucimlrepo import fetch_ucirepo

from .base import Dataset

class Adult(Dataset):
    def fetch(self):
        adult = fetch_ucirepo(id=2)
        
        self.X = adult.data.features
        self.y = adult.data.targets["income"]
        
        varbls: pd.DataFrame = adult.variables
        self.all_features = varbls["name"][varbls["role"] == "Feature"].tolist()
        num_indicator = (varbls["type"].isin(["Integer", "Continuous"])) & (varbls["role"] == "Feature")
        self.numerical_features = varbls["name"][num_indicator].tolist()
        cate_indicator = (varbls["type"].isin(["Categorical", "Binary"])) & (varbls["role"] == "Feature")
        self.categorical_features = varbls["name"][cate_indicator].tolist()

        return self
    
    def preprocess(self):
        self.X, self.y, self.all_features, self.numerical_features, self.categorical_features = preprocess_adult(self.X, self.y, self.all_features, self.numerical_features, self.categorical_features)
        return self
    
    def get_Xy(self):
        return self.X, self.y
    
    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features

def preprocess_adult(
    X: pd.DataFrame,
    y: pd.Series,
    all_features: Optional[List[str]] = None,
    numerical_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str], List[str]]:
    X = X.drop(columns=["education"])
    if categorical_features is not None:
        categorical_features.remove("education")
    if all_features is not None:
        all_features.remove("education")
    
    # random shuffle
    shuffled_index = np.random.permutation(X.shape[0])
    X = X.iloc[shuffled_index]
    y = y.iloc[shuffled_index]

    X.replace('?', np.nan, inplace=True)
    X = X.dropna()
    y = y.loc[X.index]
    X.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    X["relationship"] = X["relationship"].replace(["Husband", "Wife"], "Married")
    y = y.map({"<=50K": 0, "<=50K.": 0, ">50K": 1, ">50K.": 1})
    assert np.issubdtype(y.dtype, np.integer)

    if all_features is None or numerical_features is None or categorical_features is None:
        return X, y
    
    return X, y, all_features, numerical_features, categorical_features


from typing import Tuple, List
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from .base import Dataset


class Wine(Dataset):
    def fetch(self):
        wine = fetch_ucirepo(id=186)

        self.X = wine.data.features
        self.y = wine.data.targets["quality"]
        
        varbls: pd.DataFrame = wine.variables
        self.all_features = varbls["name"][varbls["role"] == "Feature"].tolist()
        num_indicator = (varbls["type"].isin(["Integer", "Continuous"])) & (varbls["role"] == "Feature")
        self.numerical_features = varbls["name"][num_indicator].tolist()
        cate_indicator = (varbls["type"].isin(["Categorical", "Binary"])) & (varbls["role"] == "Feature")
        self.categorical_features = varbls["name"][cate_indicator].tolist()

        return self

    def preprocess(self):
        # random shuffle
        shuffled_index = np.random.permutation(self.X.shape[0])
        self.X = self.X.iloc[shuffled_index]
        self.y = self.y.iloc[shuffled_index]
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

        return self
    
    def get_Xy(self):
        return self.X, self.y
    
    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features

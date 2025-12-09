import pandas as pd
from typing import Tuple, List
from ucimlrepo import fetch_ucirepo
import numpy as np
from .base import Dataset


class ParkinsonsTotal(Dataset):
    def fetch(self):
        parkinsons = fetch_ucirepo(id=189)

        self.X = parkinsons.data.features
        self.y = parkinsons.data.targets["total_UPDRS"]
        
        varbls: pd.DataFrame = parkinsons.variables
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

class ParkinsonsMotor(Dataset):
    def fetch(self):
        parkinsons = fetch_ucirepo(id=189)

        self.X = parkinsons.data.features
        self.y = parkinsons.data.targets["motor_UPDRS"]
        
        varbls: pd.DataFrame = parkinsons.variables
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
from typing import Tuple, List
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo

from .base import Dataset

def prepare_bike_sharing():
    bike_sharing_dataset = fetch_ucirepo(id=275)
    X_df, y_ser = preprocess_bike_sharing(df=bike_sharing_dataset["data"]["original"])

    return X_df, y_ser

def preprocess_bike_sharing(df: pd.DataFrame):
    df = df.drop(["instant", "dteday", "casual", "registered", "atemp"], axis=1)

    # Standarize X
    X_df = df.drop(["cnt"], axis=1)
    X_df = (X_df - X_df.mean()) / X_df.std()

    # Standarize Y
    y_df = df["cnt"]
    y_df = (y_df - y_df.mean()) / y_df.std()

    return X_df, y_df

class BikeSharing(Dataset):
    def fetch(self):
        bike = fetch_ucirepo(id=275)
        
        self.X = bike.data.features
        self.y = bike.data.targets["cnt"]
        
        varbls: pd.DataFrame = bike.variables
        self.all_features = varbls["name"][varbls["role"] == "Feature"].tolist()
        num_indicator = (varbls["type"].isin(["Integer", "Continuous"])) & (varbls["role"] == "Feature")
        self.numerical_features = varbls["name"][num_indicator].tolist()
        cate_indicator = (varbls["type"].isin(["Categorical", "Binary", "Date"])) & (varbls["role"] == "Feature")
        self.categorical_features = varbls["name"][cate_indicator].tolist()

        return self
    
    def preprocess(self):
        feats_to_drop = ["dteday", "atemp"]
        self.X = self.X.drop(feats_to_drop, axis=1)
        for f in feats_to_drop:
            if f in self.all_features:
                self.all_features.remove(f)
            if f in self.numerical_features:
                self.numerical_features.remove(f)
            if f in self.categorical_features:
                self.categorical_features.remove(f)
        
        # random shuffle
        shuffled_index = np.random.permutation(self.X.shape[0])
        self.X = self.X.iloc[shuffled_index]
        self.y = self.y.iloc[shuffled_index]
        self.X.reset_index(drop=True, inplace=True)
        self.y.reset_index(drop=True, inplace=True)

        return self
    
    def get_Xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y
    
    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features

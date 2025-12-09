from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from .base import Dataset

def prepare_california():
    california = fetch_california_housing(as_frame=True)
    df = california["frame"]

    # shuffle
    df.sample(frac=1).reset_index(drop=True)
    X_df, Y_df = preprocess_california(df)
    return X_df, Y_df

def preprocess_california(df: pd.DataFrame):
    df = df.dropna()
    X_df = df.iloc[:, :-1]
    Y_df = df.iloc[:, -1]

    # normalize
    X_df = (X_df - X_df.mean()) / X_df.std()

    Y_df = (Y_df - Y_df.mean()) / Y_df.std()
    
    X_df = X_df.reset_index(drop = True)
    Y_df = Y_df.reset_index(drop = True)

    return X_df, Y_df


class CaliforniaHousing(Dataset):
    def fetch(self):
        california = fetch_california_housing(as_frame=True)
        df: pd.DataFrame = california["frame"]
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]

        self.all_features = self.numerical_features = self.X.columns.tolist()
        self.categorical_features = []

        return self
    
    def preprocess(self):
        df = pd.concat([self.X, self.y], axis="columns")
        df = df.dropna().reset_index(drop=True)
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]
        
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

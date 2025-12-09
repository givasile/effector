from typing import List, Tuple
import pandas as pd
import numpy as np
from nodegam.data import fetch_MIMIC2

from .base import Dataset


class MIMIC2(Dataset):
    def fetch(self):
        mimic = fetch_MIMIC2(path="../../../data/")

        self.X = pd.concat([mimic["X_train"], mimic["X_test"]], ignore_index=True)
        self.y = pd.Series(
            np.concatenate([mimic["y_train"], mimic["y_test"]]), name="target"
        )

        self.all_features = self.X.columns.tolist()
        self.categorical_features = mimic["cat_features"]
        self.numerical_features = self.X.columns.difference(
            self.categorical_features
        ).tolist()

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

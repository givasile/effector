import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from .base import Dataset


class HELOC(Dataset):
    def __init__(self, dataset_directory: os.PathLike = "../../../data/"):
        self.dataset_directory = dataset_directory

    def fetch(self):
        self._df = pd.read_csv(
            Path(self.dataset_directory) / "heloc_dataset_v1 (1).csv"
        )
        self.all_features = self.numerical_features = self.categorical_features = []

        return self

    def preprocess(self):
        self._df["RiskPerformance"] = self._df["RiskPerformance"].map(
            {"Good": 1, "Bad": 0}
        )
        self.y = self._df["RiskPerformance"]
        self.X = self._df.drop(columns=["RiskPerformance"])

        self.categorical_features = []
        self.all_features = self.numerical_features = self.X.columns.difference(
            self.categorical_features
        ).tolist()

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

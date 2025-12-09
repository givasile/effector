import openml
from typing import Tuple, List
import pandas as pd
from .base import Dataset


class Electrical(Dataset):
    def fetch(self):
        dataset = openml.datasets.get_dataset(43007)

        X, y, categorical_indicator, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )

        y = X["stabf"]
        X = X.drop(columns=["stabf"])
        y = y.map(
            {
                "unstable": 0,
                "stable": 1,
            }
        )

        self.X = X
        self.y = y
        self.all_features = self.X.columns.tolist()
        self.categorical_features = []
        self.numerical_features = self.X.columns.tolist()

        return self

    def preprocess(self):
        return self

    def get_Xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features

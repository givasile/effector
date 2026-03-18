from typing import Tuple, List
import pandas as pd
from .base import Dataset
from ucimlrepo import fetch_ucirepo


class Bank(Dataset):
    def fetch(self):
        bank_marketing = fetch_ucirepo(id=222)
        self.categorical_features = bank_marketing.variables[
            bank_marketing.variables["type"] == "Categorical"
        ]["name"].tolist()

        X = bank_marketing.data.features
        y = bank_marketing.data.targets

        X["month"] = (
            X["month"]
            .replace("jan", 1)
            .replace("feb", 2)
            .replace("mar", 3)
            .replace("apr", 4)
            .replace("may", 5)
            .replace("jun", 6)
            .replace("jul", 7)
            .replace("aug", 8)
            .replace("sep", 9)
            .replace("oct", 10)
            .replace("nov", 11)
            .replace("dec", 12)
        )

        X[self.categorical_features] = X[self.categorical_features].fillna("unknown")

        for col in ["default", "housing", "loan"]:
            X[col] = X[col].map({"no": 0, "yes": 1})

        y["y"] = y["y"].map({"no": 0, "yes": 1})
        y = y["y"].values

        self.X = X
        self.y = y
        self.all_features = list(self.X.columns)
        self.numerical_features = [
            col for col in self.X.columns if col not in self.categorical_features
        ]

        return self

    def preprocess(self):
        return self

    def get_Xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features

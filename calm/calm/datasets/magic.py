from typing import Tuple, List
import pandas as pd
from .base import Dataset
from ucimlrepo import fetch_ucirepo


class Magic(Dataset):
    def fetch(self):
        combined_cycle_power_plant = fetch_ucirepo(id=159)
        X = combined_cycle_power_plant.data.features
        y = combined_cycle_power_plant.data.targets
        y = y["class"].map(
            {
                "g": 1,
                "h": 0,
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

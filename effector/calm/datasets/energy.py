from typing import Tuple, List
import pandas as pd
from .base import Dataset


class Energy(Dataset):
    def fetch(self, path="../../../data/"):
        df = pd.read_csv(f"{path}energydata_complete.csv", parse_dates=["date"])

        df["NSM"] = (
            df["date"].dt.hour * 3600 + df["date"].dt.minute * 60 + df["date"].dt.second
        )
        df["WeekStatus"] = (
            df["date"]
            .dt.day_name()
            .apply(lambda x: "Weekend" if x in ["Saturday", "Sunday"] else "Weekday")
        )
        df["Day_of_week"] = df["date"].dt.day_name()

        df = df.drop(columns=["date", "rv1", "rv2"], errors="ignore")

        self.X = df.drop(columns="Appliances")
        self.y = df["Appliances"]
        self.all_features = list(self.X.columns)
        self.categorical_features = ["WeekStatus", "Day_of_week"]
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

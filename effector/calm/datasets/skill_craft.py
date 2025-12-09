from typing import Tuple, List
import pandas as pd
from .base import Dataset


class SkillCraft(Dataset):
    def fetch(self, path: str = "../../../data/"):

        df = pd.read_csv(f"{path}SkillCraft1_Dataset.csv")
        df = df[
            (df["Age"] != "?") & (df["HoursPerWeek"] != "?") & (df["TotalHours"] != "?")
        ]
        X = df.drop(columns=["LeagueIndex"])
        y = df["LeagueIndex"]

        self.X = X
        self.y = y
        self.all_features = list(self.X.columns)
        self.numerical_features = list(self.X.columns)
        self.categorical_features = []

        return self

    def preprocess(self):
        return self

    def get_Xy(self) -> Tuple[pd.DataFrame, pd.Series]:
        return self.X, self.y

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        return self.all_features, self.numerical_features, self.categorical_features

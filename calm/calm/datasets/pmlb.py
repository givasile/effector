from typing import List, Tuple
import pmlb
import numpy as np
from .base import Dataset

"pmlb_appendicitis, pmlb_phoneme, pmlb_spectf"


class PMLB_APPENDICITIS(Dataset):
    def fetch(self):
        df = pmlb.fetch_data("appendicitis")
        self.X = df.drop(columns="target")
        self.y = df["target"]
        self.all_features = self.numerical_features = self.X.columns.tolist()
        self.categorical_features = []
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


class PMLB_PHONEME(Dataset):
    def fetch(self):
        df = pmlb.fetch_data("phoneme")
        self.X = df.drop(columns="target")
        self.y = df["target"]
        self.all_features = self.numerical_features = self.X.columns.tolist()
        self.categorical_features = []
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


class PMLB_SPECTF(Dataset):
    def fetch(self):
        df = pmlb.fetch_data("spectf")
        self.X = df.drop(columns="target")
        self.y = df["target"]
        self.all_features = self.numerical_features = self.X.columns.tolist()
        self.categorical_features = []
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


class PMLB_CHURN(Dataset):
    def fetch(self):
        df = pmlb.fetch_data("churn")
        df = df.drop(["phone number"], axis=1)
        df["target"] = df["target"] - np.min(df["target"])

        self.X = df.drop(columns="target")
        self.y = df["target"]
        self.all_features = self.X.columns.tolist()
        self.categorical_features = [
            "state",
            "voice mail plan",
            "area code",
            "international plan",
        ]
        self.numerical_features = [
            col for col in self.X.columns if col not in self.categorical_features
        ]
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

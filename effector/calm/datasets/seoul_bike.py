from typing import Tuple, List
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from .base import Dataset


class SeoulBike(Dataset):
    def fetch(self):
        seoul_bike = fetch_ucirepo(id=560)

        self.X = seoul_bike.data.features.copy()
        self.y = self.X['Rented Bike Count']
        
        varbls: pd.DataFrame = seoul_bike.variables
        self.all_features = varbls["name"][varbls["role"] == "Feature"].tolist()
        num_indicator = (varbls["type"].isin(["Integer", "Continuous"])) & (varbls["role"] == "Feature")
        self.numerical_features = varbls["name"][num_indicator].tolist()
        cate_indicator = (varbls["type"].isin(["Categorical", "Binary", "Date"])) & (varbls["role"] == "Feature")
        self.categorical_features = varbls["name"][cate_indicator].tolist()

        return self

    def preprocess(self):
        self.X['Date'] = pd.to_datetime(self.X['Date'], dayfirst=True) 
        self.X['Day'] = self.X['Date'].dt.day
        self.X['Month'] = self.X['Date'].dt.month
        self.X['Year'] = self.X['Date'].dt.year

        self.X = self.X.drop(columns = ['Date', 'Rented Bike Count'])
        self.numerical_features += ["Day", "Month", "Year"]
        self.all_features += ["Day", "Month", "Year"]
        self.numerical_features.remove('Rented Bike Count')
        self.all_features.remove('Rented Bike Count')
        self.categorical_features.remove("Date")
        self.all_features.remove("Date")

        self.X = self.X[self.y != 0]
        self.y = self.y[self.y != 0]

        self.X = self.X.reset_index(drop = True)
        self.y = self.y.reset_index(drop = True)

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

from typing import Protocol, Tuple, TypeVar, List
import pandas as pd

Self = TypeVar("Self", bound="Dataset")


class Dataset(Protocol):
    def fetch(self: Self) -> Self: ...

    def preprocess(self: Self) -> Self: ...

    def get_Xy(self) -> Tuple[pd.DataFrame, pd.Series]: ...

    def get_feature_names(self) -> Tuple[List[str], List[str], List[str]]:
        """Should return the names of all, numerical and categorical features

        Returns:
            Tuple[List[str], List[str], List[str]]: should follow the convention (all features, numerical features, categorical features)
        """
        ...

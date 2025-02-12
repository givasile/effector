import numpy as np
from typing import Callable, List, Optional, Union, Tuple
from effector import helpers
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt


class GlobalEffectBase(ABC):
    empty_symbol = helpers.EMPTY_SYMBOL

    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        data_effect: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        axis_limits: Optional[np.ndarray] = None,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ) -> None:
        """
        Constructor for the FeatureEffectBase class.
        """
        assert data.ndim == 2

        self.method_name = method_name.lower()
        self.model = model
        self.model_jac = model_jac

        self.dim = data.shape[1]

        # data preprocessing (i): if axis_limits passed manually,
        # keep only the points within,
        # otherwise, compute the axis limits from the data
        if axis_limits is not None:
            assert axis_limits.shape == (2, self.dim)
            assert np.all(axis_limits[0, :] <= axis_limits[1, :])

            # drop points outside of limits
            accept_indices = helpers.indices_within_limits(data, axis_limits)
            data = data[accept_indices, :]
            data_effect = data_effect[accept_indices, :] if data_effect is not None else None
        else:
            axis_limits = helpers.axis_limits_from_data(data)
        self.axis_limits: np.ndarray = axis_limits

        # data preprocessing (ii): select nof_instances from the remaining data
        self.nof_instances, self.indices = helpers.prep_nof_instances(nof_instances, data.shape[0])
        data = data[self.indices, :]
        data_effect = data_effect[self.indices, :] if data_effect is not None else None

        # store the data
        self.data: np.ndarray = data
        self.data_effect: Optional[np.ndarray] = data_effect

        self.avg_output = None

        # set feature names
        feature_names: list[str] = (
            helpers.get_feature_names(axis_limits.shape[1])
            if feature_names is None
            else feature_names
        )
        self.feature_names: list = feature_names
        self.target_name = "y" if target_name is None else target_name

        # state variable
        self.is_fitted: np.ndarray = np.ones([self.dim]) < 0

        # parameters used when fitting the feature effect
        self.fit_args: dict = {}

        # dict, like {"feature_i": {"quantity_1": value_1, "quantity_2": value_2, ...}} for the i-th
        self.feature_effect: dict = {}

    @abstractmethod
    def fit(
            self,
            features: Union[int, str, list] = "all",
            centering: Union[bool, str] = False,
            **kwargs
    ) -> None:
        """Fit, i.e., compute the quantities that are necessary for evaluating and plotting the feature effect, for the given features.

        Args:
            features: the features to fit. If set to "all", all the features will be fitted.
            centering: whether to center the feature effect plot

                    - If `centering` is `False`, the plot is not centered
                    - If `centering` is `True` or `zero_integral`, the plot is centered around the `y` axis.
                    - If `centering` is `zero_start`, the plot starts from zero.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(
            self,
            feature: int,
            heterogeneity: Union[bool, str] = False,
            centering: Union[bool, str] = False,
            **kwargs
    ) -> None:
        """

        Parameters
        ----------
        feature: index of the feature to plot
        heterogeneity: whether to plot the heterogeneity measures

            - If `heterogeneity=False`, the plot shows only the mean effect
            - If `heterogeneity=True`, the plot additionally shows the heterogeneity with the default visualization, e.g., ICE plots for PDPs
            - If `heterogeneity=<str>`, the plot shows the heterogeneity using the specified method

        centering: whether to center the PDP

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.
        **kwargs: all other plot-specific arguments
        """
        raise NotImplementedError

    def requires_refit(self, feature, centering):
        """Check if refitting is needed."""
        feature_key = f"feature_{feature}"

        # if the state variable is not set, refit
        if not self.is_fitted[feature]:
            return True

        # if the feature info does not exist, refit
        if self.feature_effect.get(feature_key) is None:
            return True

        # if the above are ok and centering is False, no need to refit
        if not centering:
            return False

        # if centering is not None and the norm_const is not set, refit
        norm_const = self.feature_effect.get(feature_key, {}).get("norm_const")
        if norm_const is None:
            return True

        # if centering is not None and is different from the centering when fitting, refit
        if self.fit_args.get(feature_key, {}).get("centering") != centering:
            return True

        return False

    @abstractmethod
    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        heterogeneity: bool = False,
        centering: Union[bool, str] = False,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    @abstractmethod
    def importance(self, feature):
        raise NotImplementedError

    def plot_importances(self, features="all"):
        features = helpers.prep_features(features, self.dim)
        absolute_importances = np.array([self.importance(feature) for feature in features])
        total_importance = sum(absolute_importances)
        normalized_importances = absolute_importances / total_importance
        feature_names = [self.feature_names[feature] for feature in features]

        # Sort features by importance
        sorted_indices = np.argsort(normalized_importances)[::-1]
        sorted_importances = normalized_importances[sorted_indices]
        sorted_feature_names = np.array(feature_names)[sorted_indices]

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(sorted_feature_names, sorted_importances, color="royalblue")

        # Labels & Title
        ax.set_xlabel("Relative Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Effect-based Importance")

        # Display values on bars
        for index, value in enumerate(sorted_importances):
            ax.text(value + 0.01, index, f"{value:.2f}", va='center')

        # Invert y-axis to have the highest importance at the top
        ax.invert_yaxis()

        plt.show()
        # plot the importance


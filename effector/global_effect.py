import numpy as np
from typing import Callable, List, Optional, Union, Tuple
from effector import helpers
from abc import ABC, abstractmethod


class GlobalEffectBase(ABC):
    empty_symbol = helpers.EMPTY_SYMBOL

    def __init__(
            self,
            method_name: str,
            data: np.ndarray,
            model: Callable,
            nof_instances: Union[int, str] = 1000,
            axis_limits: Optional[np.ndarray] = None,
            avg_output: Optional[float] = None,
            feature_names: Optional[List] = None,
            target_name: Optional[str] = None,
    ) -> None:
        """
        Constructor for the FeatureEffectBase class.

        Args:
            method_name: the name of the method used to compute the feature effect
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N, )`

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            avg_output: the average output of the model on the data

                - use a `float`, to specify it manually
                - use `None`, to be inferred as `np.mean(model(data))`

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`

        """
        assert data.ndim == 2

        self.method_name: str = method_name.lower()

        # select nof_instances from the data
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        data = data[self.indices, :]

        self.data: np.ndarray = data
        self.dim = self.data.shape[1]

        self.model: Callable = model

        self.avg_output = (
            avg_output if avg_output is not None else np.mean(self.model(self.data))
        )

        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        self.axis_limits: np.ndarray = axis_limits

        self.feature_names: Union[None, list] = feature_names
        self.target_name: Union[None, str] = target_name

        # state variable
        self.is_fitted: np.ndarray = np.ones([self.dim]) < 1

        # parameters used when fitting the feature effect
        self.method_args: dict = {}

        # dictionary with all the information required for plotting or evaluating the feature effect
        self.feature_effect: dict = {}

    @abstractmethod
    def fit(self, features: Union[int, str, list] = "all", **kwargs) -> None:
        """Fit the feature effect for the given features.

        Args:
            features: the features to fit. If set to "all", all the features will be fitted.
        """
        raise NotImplementedError

    @abstractmethod
    def plot(self, feature: int, *args) -> None:
        """

        Parameters
        ----------
        feature: index of the feature to plot
        *args: all other plot-specific arguments
        """
        raise NotImplementedError

    def refit(self, feature, centering):
        """Checks if refitting is needed.
        """
        if not self.is_fitted[feature]:
            return True
        else:
            if centering is not False:
                if self.method_args["feature_" + str(feature)]["centering"] != centering:
                    return True
        return False

    @abstractmethod
    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        heterogeneity: bool = False,
        centering: Union[bool, str] = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Notes:
            This is a common method among all the FE classes.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot

              - `np.ndarray` of shape `(T, )`

            heterogeneity: whether to return the heterogeneity measures.

                  - if `heterogeneity=False`, the function returns the mean effect at the given `xs`
                  - If `heterogeneity=True`, the function returns `(y, std)` where `y` is the mean effect and `std` is the standard deviation of the mean effect

            centering: whether to center the PDP

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, std, estimator_var)` otherwise

        Notes:
            * If `centering` is `False`, the plot is not centered
            * If `centering` is `True` or `"zero_integral"`, the plot is centered by subtracting its mean.
            * If `centering` is `"zero_start"`, the plot starts from zero.

        Notes:
            * If `heterogeneity` is `False`, the plot returns only the mean effect `y` at the given `xs`.
            * If `heterogeneity` is `True`, the plot returns `(y, std)` where:
                * `y` is the mean effect
                * `std` is the standard deviation of the mean effect
        """
        raise NotImplementedError

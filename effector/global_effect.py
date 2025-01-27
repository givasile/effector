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
        model_jac: Optional[Callable] = None,
        data_effect: Optional[np.ndarray] = None,
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
        self.model_jac: Optional[Callable] = model_jac

        self.avg_output = (
            avg_output if avg_output is not None else np.mean(self.model(self.data))
        )

        if axis_limits is not None:
            assert axis_limits.shape == (2, self.dim)
            assert np.all(axis_limits[0, :] <= axis_limits[1, :])
        else:
           axis_limits = helpers.axis_limits_from_data(data)

        # drop points outside of limits
        # drop points outside of limits
        accept_indices = np.ones([self.data.shape[0]]) > 0
        for feature in range(self.dim):
            accept_left = self.data[:, feature] >= axis_limits[0, feature]
            accept_right = self.data[:, feature] <= axis_limits[1, feature]
            accept_indices = np.logical_and(accept_indices, accept_left)
            accept_indices = np.logical_and(accept_indices, accept_right)
        self.data = self.data[accept_indices, :]
        if data_effect is not None:
            data_effect = data_effect[accept_indices, :]
        self.data_effect = data_effect

        self.axis_limits: np.ndarray = axis_limits

        self.feature_names: Union[None, list] = feature_names
        self.target_name: Union[None, str] = target_name

        # state variable
        self.is_fitted: np.ndarray = np.ones([self.dim]) < 1

        # parameters used when fitting the feature effect
        self.fit_args: dict = {}

        # dict, like {"feature_i": {"quantity_1": value_1, "quantity_2": value_2, ...}} for the i-th
        self.feature_effect: dict = {}

        # .eval() method cache, each element is a dict like {"feature_i": {"xs": xs, "data": {"y": y, "std": std}}}
        self.eval_cache: dict[str, list[dict]] = {}

    def update_cache(self, feature, xs, data, parameters):
        """Cache stores up to 3 items per feature"""
        if "feature_" + str(feature) not in self.eval_cache.keys():
            self.eval_cache["feature_" + str(feature)] = []

        # choose the feature-specific cache
        cache = self.eval_cache["feature_" + str(feature)]
        if len(cache) < 3:
            cache.append({"xs": xs, "data": data, "parameters": parameters})
        else:
            # drop the first element
            cache = cache[1:]
            cache.append({"feature_" + str(feature): {"xs": xs, "data": data, "parameters": parameters}})

    def retrieve_from_cache(self, feature, xs, parameters):
        """Retrieve data from the cache"""
        if "feature_" + str(feature) not in self.eval_cache.keys():
            return None
        else:
            cache = self.eval_cache["feature_" + str(feature)]
            for i in range(len(cache)):
                if cache[i]["xs"].shape == xs.shape:
                    if np.allclose(cache[i]["xs"], xs) and self._parameters_equal(parameters, cache[i]["parameters"]):
                        return cache[i]["data"]
        return None

    def _parameters_equal(self, param_1, param_2):
        return True

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
        """Check if refitting is needed
        """
        if not self.is_fitted[feature]:
            return True
        else:
            if centering is not False and centering != self.fit_args["feature_" + str(feature)]["centering"]:
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
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, heterogeneity)` otherwise

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

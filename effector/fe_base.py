import numpy as np
import typing
from effector import helpers
from effector import utils_integrate
from abc import ABC, abstractmethod


class FeatureEffectBase(ABC):
    empty_symbol = 1e8

    def __init__(self, axis_limits: np.ndarray) -> None:
        """
        Constructor for the FeatureEffectBase class.

        Args:
            axis_limits: The axis limits defined as start and stop values for each axis, (2, D)

        """
        self.axis_limits: np.ndarray = axis_limits
        self.dim = self.axis_limits.shape[1]

        # state variable
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

        # parameters used when fitting the feature effect
        self.method_args: typing.Dict = {}

        # normalization constant per feature for centering the FE plot
        self.norm_const: np.ndarray = np.ones([self.dim]) * self.empty_symbol

        # dictionary with all the information required for plotting or evaluating the feautere effect
        self.feature_effect: typing.Dict = {}

    def _eval_unnorm(self,
                     feature: int,
                     x: np.ndarray,
                     uncertainty: bool = False
                     ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Compute the effect of the s-th feature at positions `x`.

        If uncertainty is False, returns a [N,] np.ndarray with the evaluation of the plot

        If uncertainty is True, returns a tuple (y, sigma, stderr) where:
         - y: is a [N,] np.ndarray with the expected effect
         - sigma: is a [N,] np.ndarray with the std of the expected effect
         - stderr: is a [N,] np.ndarray with the standard error of the expeceted effect

        Parameters
        ----------
        feature: int, index of the feature
        x: [N,] np.array, the points of the s-th axis to evaluate the FE plot
        uncertainty: bool, whether to provide uncertainty measures

        Returns
        -------
        - np.ndarray [N,], if uncertainty=False
        - tuple (np.ndarray [N,], np.ndarray [N,], np.ndarray [N,]), if uncertainty=True
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, features: typing.Union[int, str, list] = "all", **kwargs) -> None:
        """Iterates over _fit_feature for all features,
        computes the normalization constant if asked
        and updates self.is_fitted.

        Parameters
        ----------
        features
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

    def _compute_norm_const(self, feature: int, method: str = "zero_integral") -> float:
        """Compute the normalization constant.
        """
        assert method in ["zero_integral", "zero_start"]

        def create_partial_eval(feature):
            return lambda x: self._eval_unnorm(feature, x, uncertainty=False)
        partial_eval = create_partial_eval(feature)
        start = self.axis_limits[0, feature]
        stop = self.axis_limits[1, feature]

        if method == "zero_integral":
            z = utils_integrate.mean_1d_linspace(partial_eval, start, stop)
        else:
            z = partial_eval(np.array([start])).item()
        return z

    def eval(self,
             feature: int,
             xs: np.ndarray,
             uncertainty: bool = False,
             centering: typing.Union[bool, str] = False
             ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Notes:
            This is a common method among all the FE classes.

        Parameters:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot, (T)
            uncertainty: whether to return the uncertainty measures
            centering: whether to center the plot

        Returns:
            the mean effect `y`, if `uncertainty=False` (default) or a tuple `(y, std, estimator_var)` otherwise

        Notes:
            * If `centering` is `False`, the PDP is not centered
            * If `centering` is `True` or `"zero-integral"`, the PDP is centered by subtracting the mean of the PDP.
            * If `centering` is `"zero-start"`, the PDP is centered by subtracting the value of the PDP at the first point.

        Notes:
            * If `uncertainty` is `False`, the PDP returns only the mean effect `y` at the given `xs`.
            * If `uncertainty` is `True`, the PDP returns `(y, std, estimator_var)` where:
                * `y` is the mean effect
                * `std` is the standard deviation of the mean effect
                * `estimator_var` is the variance of the mean effect estimator
        """
        centering = helpers.prep_centering(centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        # Fit the feature if not already fitted
        if not self.is_fitted[feature]:
            arg_list = self.fit.__code__.co_varnames
            self.fit(features=feature, centering=centering) if "centering" in arg_list else self.fit(features=feature)

        # Evaluate the feature
        yy = self._eval_unnorm(feature, xs, uncertainty=uncertainty)
        y, std, estimator_var = yy if uncertainty else (yy, None, None)

        # Center if asked
        y = y - self.norm_const[feature] if self.norm_const[feature] != self.empty_symbol and centering else y

        return (y, std, estimator_var) if uncertainty is not False else y

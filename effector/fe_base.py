import numpy as np
import typing
from effector import helpers
from effector import utils_integrate
from abc import ABC, abstractmethod


class FeatureEffectBase(ABC):
    empty_symbol = 1e8

    def __init__(self, axis_limits: np.ndarray) -> None:
        """
        Initializes FeatureEffectBase.

        Parameters
        ----------
        axis_limits: [2, D] np.ndarray, axis limits for the FE plot
        """
        # setters
        self.axis_limits: np.ndarray = axis_limits
        self.dim = self.axis_limits.shape[1]

        # state variable
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

        # input variables, arguments of fit
        self.method_args: typing.Dict = {}

        # output variables, after fit
        # normalization constant per feature for centering the FE plot
        self.norm_const: np.ndarray = np.ones([self.dim]) * self.empty_symbol
        # dictionary with fe plot details. keys are "feature_s", where s is the index of the feature
        self.feature_effect: typing.Dict = {}
        # boolean variable for whether a FE plot has been computed

    def _eval_unnorm(self,
                     feature: int,
                     x: np.ndarray,
                     uncertainty: bool = False
                     ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute the effect of the s-th feature at x.
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
    def fit(self, features: typing.Union[int, str, list] = "all"):
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
             x: np.ndarray,
             uncertainty: bool = False,
             centering: typing.Union[bool, str] = False
             ) -> typing.Union[np.ndarray, typing.Tuple]:
        """Evaluate the feature effect method at x positions.

        Parameters
        ----------
        x: np.array (N,)
        feature: index of feature of interest
        uncertainty: whether to return the std and the estimator variance
        centering: whether to centering the plot

        Returns
        -------
        - np.array (N,), if uncertainty=False
        - tuple (y, std, estimator_var), if uncertainty=True
        """
        centering = helpers.prep_centering(centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        # Fit the feature if not already fitted
        if not self.is_fitted[feature]:
            arg_list = self.fit.__code__.co_varnames
            self.fit(features=feature, centering=centering) if "centering" in arg_list else self.fit(features=feature)

        # Evaluate the feature with or without uncertainty
        yy = self._eval_unnorm(feature, x, uncertainty=uncertainty)
        y, std, estimator_var = yy if uncertainty else (yy, None, None)

        # Center the plot if asked
        y = y - self.norm_const[feature] if self.norm_const[feature] != self.empty_symbol and centering else y

        return (y, std, estimator_var) if uncertainty is not False else y

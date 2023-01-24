import numpy as np
import typing
from pythia import utils_integrate


class FeatureEffectBase:
    empty_symbol = 1e8

    def __init__(self, axis_limits: np.ndarray) -> None:
        """
        :param axis_limits: np.ndarray (2, D), limits for computing the effects
        """
        # setters
        self.axis_limits: np.ndarray = axis_limits
        self.dim = self.axis_limits.shape[1]

        # init now -> will be filled later
        # normalization constant per feature for centering the FE plot
        self.norm_const: np.ndarray = np.ones([self.dim]) * self.empty_symbol
        # dictionary with fe plot details. keys are "feature_s", where s is the index of the feature
        self.feature_effect: typing.Dict = {}
        # boolean variable for whether a FE plot has been computed
        self.is_fitted: np.ndarray = np.ones([self.dim]) * False

    def _eval_unnorm(
        self, feature: int, x: np.ndarray, uncertainty: int = False
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute the effect of the s-th feature at x.
        If uncertainty is False, returns a [N,] np.ndarray with the evaluation of the plot
        If uncertainty is True, returns a tuple (y, sigma, stderr) where:
         - y: is a [N,] np.ndarray with the expected effect
         - sigma: is a [N,] np.ndarray with the expected uncertainty
         - stderr: is a [N,] np.ndarray with the standard error of the expeceted effect

        Parameters
        ----------
        feature: int, index of the feature
        x: [N,] np.array, the points of the s-th axis to evaluate the FE plot
        uncertainty: bool, whether to provide uncertainty measures

        Returns
        -------
        - np.ndarray [N,], if uncertainty=False
        - tuple, if uncertainty=True
        """
        raise NotImplementedError

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        """Fit a feature effect plot.

        Parameters
        ----------
        feat: int, the index of the feature
        params: method-specific parameters for fitting the feature

        Returns
        -------
        Dictionary, with all the important quantities for the feature effect plot
        """
        raise NotImplementedError

    def fit(self, features, *args):
        """Iterates over _fit_feature for all features,
        computes the normalization constant if asked
        and updates self.is_fitted.

        Parameters
        ----------
        features
        args

        Returns
        -------

        """
        raise NotImplementedError

    def plot(self, feature: int, *args) -> None:
        """

        Parameters
        ----------
        feature: index of the feature to plot
        args: all other plot-specific arguments
        """
        raise NotImplementedError

    def _compute_norm_const(self, feature: int) -> float:
        """Compute the normalization constant.
        Uses integration with linspace..
        """
        def create_func(feature):
            def func(x):
                return self._eval_unnorm(feature, x, uncertainty=False)
            return func
        func = create_func(feature)
        start = self.axis_limits[0, feature]
        stop = self.axis_limits[1, feature]
        z = utils_integrate.mean_1d_linspace(func, start, stop)
        return z

    def eval(
        self, feature: int, x: np.ndarray, uncertainty: bool = False
    ) -> typing.Union[np.ndarray, typing.Tuple]:
        """Evaluate the feature effect method at x

        Parameters
        ----------

        x: np.array (N,)
        feature: index of feature of interest
        uncertainty: whether to return the std and the estimator variance

        """
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        if not self.is_fitted[feature]:
            self.fit(features=feature)

        if self.norm_const[feature] == self.empty_symbol:
            self.norm_const[feature] = self._compute_norm_const(feature)

        if not uncertainty:
            y = self._eval_unnorm(feature, x, uncertainty=False) - self.norm_const[feature]
            return y
        else:
            y, std, estimator_var = self._eval_unnorm(feature, x, uncertainty=True)
            y = y - self.norm_const[feature]
            return y, std, estimator_var

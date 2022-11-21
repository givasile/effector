import numpy as np
from functools import partial
import typing
from pythia import utils_integrate
from pythia import helpers


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
        self.z: np.ndarray = np.ones([self.dim])*self.empty_symbol
        # dictionary with fe plot details. keys are "feature_s", where s is the index of the feature
        self.feature_effect: typing.Dict = {}
        # boolean variable for whether a FE plot has been computed
        self.fitted: np.ndarray = np.ones([self.dim]) * False

    def _eval_unnorm(self,
                     x: np.ndarray,
                     s: int,
                     uncertainty: int = False
                     ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Compute the effect of the s-th feature at x.
        If uncertainty is False, returns a [N,] np.ndarray with the evaluation of the plot
        If uncertainty is True, returna a tuple (y, sigma, stderr) where:
         - y: is a [N,] np.ndarray with the expected effect
         - sigma: is a [N,] np.ndarray with the expected uncertainty
         - stderr: is a [N,] np.ndarray with the standard error of the expeceted effect

        Parameters
        ----------
        x: [N,] np.array, the points of the s-th axis to evaluate the FE plot
        s: int, index of the feature
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

    def plot(self, s: int, *args) -> None:
        """

        Parameters
        ----------
        s: index of the feature to plot
        args: all other plot-specific arguments
        """
        raise NotImplementedError

    def _compute_z(self, s: int) -> float:
        """Compute the normalization constant.
        Uses integration with linspace..
        """
        func = partial(self._eval_unnorm, s=s, uncertainty=False)
        start = self.axis_limits[0, s]
        stop = self.axis_limits[1, s]
        z = utils_integrate.mean_1d_linspace(func, start, stop)
        return z

    def fit(self,
            features: typing.Union[int, str, list] = "all",
            params: typing.Union[None, dict] = {},
            compute_z: bool = True) -> None:
        """Fit feature effect plot for the asked features

        Parameters
        ----------
        features: features to compute the normalization constant
            - "all", all features
            - int, the index of the feature
            - list, list of indexes of the features
        params: dictionary with method-specific parameters for fitting the FE plots
        compute_z: bool, whether to compute the normalization constants
        """
        features = helpers.prep_features(features, self.dim)
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(s, params)
            if compute_z:
                self.z[s] = self._compute_z(s)
            self.fitted[s] = True

    def eval(self, x: np.ndarray, s: int, uncertainty: bool = False) -> typing.Union[np.ndarray, typing.Tuple]:
        """Evaluate the feature effect method at x

        Parameters
        ----------

        x: np.array (N,)
        s: index of feature of interest
        uncertainty: whether to return the std and the estimator variance

        """
        assert self.axis_limits[0, s] < self.axis_limits[1, s]

        if not self.fitted[s]:
            self.fit(features=s)

        if self.z[s] == self.empty_symbol:
            self.z[s] = self._compute_z(s)

        if not uncertainty:
            y = self._eval_unnorm(x, s, uncertainty=False) - self.z[s]
            return y
        else:
            y, std, estimator_var = self._eval_unnorm(x, s, uncertainty=True)
            y = y - self.z[s]
            return y, std, estimator_var

import typing
import copy
import effector.utils as utils
import effector.visualization as vis
import effector.binning_methods as bm
import effector.helpers as helpers
from effector.global_effect import GlobalEffect
import numpy as np
import shap
from scipy.interpolate import UnivariateSpline


class SHAPDependence(GlobalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        avg_output: typing.Union[None, float] = None,
        feature_names: typing.Union[None, list] = None,
        target_name: typing.Union[None, str] = None,
    ):
        """
        RHALE constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            model_jac: the black-box model Jacobian (N,D) -> (N,D)
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            data_effect:
                - if np.ndarray, the model Jacobian computed on the `data`
                - if None, the Jacobian will be computed using model_jac

        """
        super(SHAPDependence, self).__init__(
            data, model, axis_limits, avg_output, feature_names, target_name
        )

    # TODO fix so that centering and points_used_for_centering are used
    def _fit_feature(self, feature: int, centering, points_used_for_centering) -> typing.Dict:

        # drop points outside of limits
        ind = np.logical_and(
            self.data[:, feature] >= self.axis_limits[0, feature],
            self.data[:, feature] <= self.axis_limits[1, feature],
        )
        data = self.data[ind, :]

        shap_explainer = shap.Explainer(self.model, data)
        explanation = shap_explainer(data)

        # extract x and y pais
        yy = explanation.values[:, feature]
        xx = data[:, feature]

        # make xx monotonic
        idx = np.argsort(xx)
        xx = xx[idx]
        yy = yy[idx]
        spline = UnivariateSpline(xx, yy)

        # fit spline to the std
        yy_std = np.abs(yy - spline(xx))
        spline_std = UnivariateSpline(xx, yy_std)

        # compute norm constant
        x_norm = np.linspace(xx[0], xx[-1], 1000)
        y_norm = spline(x_norm)
        norm_const = np.trapz(y_norm, x_norm) / (xx[-1] - xx[0])

        return {"spline": spline, "spline_std": spline_std,
                "xx": xx, "yy": yy, "norm_const": norm_const}

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        centering: typing.Union[bool, str] = "zero_integral",
        points_used_for_centering: int = 100,
    ) -> None:
        """Fit the model.

        Args:
            features (int, str, list): the features to fit.
                - If set to "all", all the features will be fitted.
            binning_method (str):
                - If set to "greedy" or bm.Greedy, the greedy binning method will be used.
                - If set to "dynamic" or bm.DynamicProgramming, the dynamic programming binning method will be used.
                - If set to "fixed" or bm.Fixed, the fixed binning method will be used.
            centering (bool, str):
                - If set to False, no centering will be applied.
                - If set to "zero_integral" or True, the integral of the feature effect will be set to zero.
                - If set to "zero_mean", the mean of the feature effect will be set to zero.
        """
        centering = helpers.prep_centering(centering)
        features = helpers.prep_features(features, self.dim)

        # new implementation
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(
                s, centering, points_used_for_centering
            )
            self.is_fitted[s] = True

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        uncertainty: bool = False,
        centering: typing.Union[bool, str] = False,
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Notes:
            The standard deviation of the PDP is computed as:
            $$
            todo
            $$

        Parameters:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot, (T)
            uncertainty: whether to return the uncertainty measures
            centering: whether to center the plot

        Returns:
            the mean effect `y`, if `uncertainty=False` (default) or a tuple `(y, std, estimator_var)` otherwise

        Notes:
            * If `centering` is `False`, the PDP is not centered
            * If `centering` is `True` or `"zero_integral"`, the PDP is centered by subtracting the mean of the PDP.
            * If `centering` is `"zero_start"`, the PDP is centered by subtracting the value of the PDP at the first point.

        Notes:
            * If `uncertainty` is `False`, the PDP returns only the mean effect `y` at the given `xs`.
            * If `uncertainty` is `True`, the PDP returns `(y, std, estimator_var)` where:
                * `y` is the mean effect
                * `std` is the standard deviation of the mean effect
                * `estimator_var` is the variance of the mean effect estimator
        """
        centering = helpers.prep_centering(centering)
        if not self.is_fitted[feature]:
            self.fit(features=feature, centering=centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        yy = self.feature_effect["feature_" + str(feature)]["spline"](xs)

        if centering:
            norm_const = self.feature_effect["feature_" + str(feature)]["norm_const"]
            yy = yy - norm_const

        if uncertainty:
            yy_std = self.feature_effect["feature_" + str(feature)]["spline_std"](xs)
            return yy, yy_std, np.zeros_like(yy_std)
        else:
            return yy

    # TODO: implement
    def plot():
        pass

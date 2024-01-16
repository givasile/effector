import typing
from typing import Callable, List, Optional, Union, Tuple
import effector.visualization as vis
import effector.helpers as helpers
from effector.global_effect import GlobalEffectBase
import numpy as np
import shap
from scipy.interpolate import UnivariateSpline


class SHAPDependence(GlobalEffectBase):
    def __init__(
            self,
            data: np.ndarray,
            model: Callable,
            axis_limits: Optional[np.ndarray] = None,
            nof_instances: Union[int, str] = 100,
            avg_output: Optional[float] = None,
            feature_names: Optional[List[str]] = None,
            target_name: Optional[str] = None,
    ):
        """
        Constructor of the SHAPDependence class.

        Definition:
            The value of a coalition of $S$ features is estimated as:
            $$
            \hat{v}(S) = {1 \over N} \sum_{i=1}^N  f(x_S \cup x_C^i) - f(x^i)
            $$
            The value of a coalition $S$ quantifies what the values $\mathbf{x}_S$ of the features in $S$ contribute to the output of the model. It
            is the average (over all instances) difference on the output between setting features in $S$ to be $x_S$, i.e., $\mathbf{x} = (\mathbf{x}_S, \mathbf{x}_C^i)$ and leaving the instance as it is, i.e., $\mathbf{x}^i = (\mathbf{x}_S^i, \mathbf{x}_C^i)$.

            The contribution of a feature $j$ added to a coalition $S$ is estimated as:
            $$
            \hat{\Delta}_{S, j} = \hat{v}(S \cup \{j\}) - \hat{v}(S)
            $$

            The SHAP value of a feature $j$ with value $x_j$ is the average contribution of feature $j$ across all possible coalitions with a weight $w_{S, j}$:

            $$
            \hat{\phi}_j(x_j) = {1 \over N} \sum_{S \subseteq \{1, \dots, D\} \setminus \{j\}} w_{S, j} \hat{\Delta}_{S, j}
            $$

            where $w_{S, j}$ assures that the contribution of feature $j$ is the same for all coalitions of the same size. For example, there are $D-1$ ways for $x_j$ to enter a coalition of $|S| = 1$ feature, so $w_{S, j} = {1 \over D (D-1)}$ for each of them. In contrast, there is only one way for $x_j$ to enter a coaltion of $|S|=0$ (to be the first specified feature), so $w_{S, j} = {1 \over D}$.

            The SHAP Dependence Plot (SHAP-DP) is a spline $\hat{f}^{SDP}_j(x_j)$ fit to the dataset $\{(x_j^i, \hat{\phi}_j(x_j^i))\}_{i=1}^N$ using the `UnivariateSpline` function from `scipy.interpolate`.

        Notes:
            * The required parameters are `data` and `model`. The rest are optional.
            * SHAP values are computed using the `shap` package, using the class `Explainer`.
            * SHAP values are centered by default, i.e., the average SHAP value is subtracted from the SHAP values.
            * More details on the SHAP values can be found in the [original paper](https://arxiv.org/abs/1705.07874) and in the book [Interpreting Machine Learning Models with SHAP](https://christophmolnar.com/books/shap/)

        Args:
            data: the design matrix

                - shape: `(N,D)`
            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N,)`

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            nof_instances: maximum number of instances to be used for SHAP estimation.

                - use "all", for using all instances.
                - use an `int`, for using `nof_instances` instances.

            avg_output: The average output of the model.

                - use a `float`, to specify it manually
                - use `None`, to be inferred as `np.mean(model(data))`

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `                  ["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
        """
        self.nof_instances, self.indices = helpers.prep_nof_instances(
            nof_instances, data.shape[0]
        )
        data = data[self.indices, :]

        super(SHAPDependence, self).__init__(
            "SHAP DP", data, model, nof_instances, axis_limits, avg_output, feature_names, target_name
        )

    def _fit_feature(
        self,
        feature: int,
        centering: typing.Union[bool, str] = False,
        points_for_centering: int = 100,
    ) -> typing.Dict:

        # drop points outside of limits
        self.data = self.data[self.data[:, feature] >= self.axis_limits[0, feature]]
        self.data = self.data[self.data[:, feature] <= self.axis_limits[1, feature]]

        # compute shap values
        data = self.data
        shap_explainer = shap.Explainer(self.model, data)
        explanation = shap_explainer(data)

        # extract x and y pais
        yy = explanation.values[:, feature]
        xx = data[:, feature]

        # make xx monotonic
        idx = np.argsort(xx)
        xx = xx[idx]
        yy = yy[idx]

        # fit spline_mean to xx, yy pairs
        spline_mean = UnivariateSpline(xx, yy)

        # fit spline_mean to the sqrt of the residuals
        yy_std = np.abs(yy - spline_mean(xx))
        spline_std = UnivariateSpline(xx, yy_std)

        # compute norm constant
        if centering == "zero_integral":
            x_norm = np.linspace(xx[0], xx[-1], points_for_centering)
            y_norm = spline_mean(x_norm)
            norm_const = np.trapz(y_norm, x_norm) / (xx[-1] - xx[0])
        elif centering == "zero_start":
            norm_const = spline_mean(xx[0])
        else:
            norm_const = helpers.EMPTY_SYMBOL

        ret_dict = {
            "spline_mean": spline_mean,
            "spline_std": spline_std,
            "xx": xx,
            "yy": yy,
            "norm_const": norm_const,
        }
        return ret_dict

    def fit(
            self,
            features: Union[int, str, List] = "all",
            centering: Union[bool, str] = False,
            points_for_centering: Union[int, str] = 100,
    ) -> None:
        """Fit the SHAP Dependence Plot to the data.

        Notes:
            The SHAP Dependence Plot (SDP) $\hat{f}^{SDP}_j(x_j)$ is a spline fit to
            the dataset $\{(x_j^i, \hat{\phi}_j(x_j^i))\}_{i=1}^N$
            using the `UnivariateSpline` function from `scipy.interpolate`.

            The SHAP standard deviation, $\hat{\sigma}^{SDP}_j(x_j)$, is a spline fit            to the absolute value of the residuals, i.e., to the dataset $\{(x_j^i, |\hat{\phi}_j(x_j^i) - \hat{f}^{SDP}_j(x_j^i)|)\}_{i=1}^N$, using the `UnivariateSpline` function from `scipy.interpolate`.

        Args:
            features: the features to fit.
                - If set to "all", all the features will be fitted.
            centering:
                - If set to False, no centering will be applied.
                - If set to "zero_integral" or True, the integral of the feature effect will be set to zero.
                - If set to "zero_mean", the mean of the feature effect will be set to zero.

            points_for_centering: number of linspaced points along the feature axis used for centering.

                - If set to `all`, all the dataset points will be used.

        Notes:
            SHAP values are by default centered, i.e., $\sum_{i=1}^N \hat{\phi}_j(x_j^i) = 0$. This does not mean that the SHAP _curve_ is centered around zero; this happens only if the $s$-th feature of the dataset instances, i.e., the set $\{x_s^i\}_{i=1}^N$ is uniformly distributed along the $s$-th axis. So, use:

            * `centering=False`, to leave the SHAP values as they are.
            * `centering=True` or `centering=zero_integral`, to center the SHAP curve around the `y` axis.
            * `centering=zero_start`, to start the SHAP curve from `y=0`.

            SHAP values are expensive to compute.
            To speed up the computation consider using a subset of the dataset
            points for computing the SHAP values and for centering the spline.
            The default values (`points_for_fitting_spline=100`
            and `points_for_centering=100`) are a moderate choice.
        """
        centering = helpers.prep_centering(centering)
        features = helpers.prep_features(features, self.dim)

        # new implementation
        for s in features:
            self.feature_effect["feature_" + str(s)] = self._fit_feature(
                s, centering, points_for_centering
            )
            self.is_fitted[s] = True
            self.method_args["feature_" + str(s)] = {
                "centering": centering,
                "points_for_centering": points_for_centering,
            }

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        heterogeneity: bool = False,
        centering: typing.Union[bool, str] = False,
    ) -> typing.Union[np.ndarray, typing.Tuple[np.ndarray, np.ndarray]]:
        """Evaluate the effect of the s-th feature at positions `xs`.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the FE plot

              - `np.ndarray` of shape `(T,)`
            heterogeneity: whether to return the heterogeneity measures.

                  - if `heterogeneity=False`, the function returns the mean effect at the given `xs`
                  - If `heterogeneity=True`, the function returns `(y, std)` where `y` is the mean effect and `std` is the standard deviation of the mean effect

            centering: whether to center the plot

                - If `centering` is `False`, the SHAP curve is not centered
                - If `centering` is `True` or `zero_integral`, the SHAP curve is centered around the `y` axis.
                - If `centering` is `zero_start`, the SHAP curve starts from `y=0`.

        Returns:
            the mean effect `y`, if `heterogeneity=False` (default) or a tuple `(y, std, estimator_var)` otherwise
        """
        centering = helpers.prep_centering(centering)

        if self.refit(feature, centering):
            self.fit(features=feature, centering=centering)

        # Check if the lower bound is less than the upper bound
        assert self.axis_limits[0, feature] < self.axis_limits[1, feature]

        yy = self.feature_effect["feature_" + str(feature)]["spline_mean"](xs)

        if centering is not False:
            norm_const = self.feature_effect["feature_" + str(feature)]["norm_const"]
            yy = yy - norm_const

        if heterogeneity:
            yy_std = self.feature_effect["feature_" + str(feature)]["spline_std"](xs)
            return yy, yy_std
        else:
            return yy

    def plot(
        self,
        feature: int,
        heterogeneity: Union[bool, str] = False,
        centering: Union[bool, str] = False,
        nof_points: int = 30,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        nof_shap_values: Union[int, str] = "all",
        show_avg_output: bool = False,
        y_limits: Optional[List] = None,
    ) -> None:
        """
        Plot the SHAP Dependence Plot (SDP) of the s-th feature.

        Args:
            feature: index of the plotted feature
            heterogeneity: whether to output the heterogeneity of the SHAP values

                - If `heterogeneity` is `False`, no heterogeneity is plotted
                - If `heterogeneity` is `True` or `"std"`, the standard deviation of the shap values is plotted
                - If `heterogeneity` is `"shap_values"`, the shap values are scattered on top of the SHAP curve

            centering: whether to center the SDP

                - If `centering` is `False`, the SHAP curve is not centered
                - If `centering` is `True` or `zero_integral`, the SHAP curve is centered around the `y` axis.
                - If `centering` is `zero_start`, the SHAP curve starts from `y=0`.

            nof_points: number of points to evaluate the SDP plot
            scale_x: dictionary with keys "mean" and "std" for scaling the x-axis
            scale_y: dictionary with keys "mean" and "std" for scaling the y-axis
            nof_shap_values: number of shap values to show on top of the SHAP curve
            show_avg_output: whether to show the average output of the model
            y_limits: limits of the y-axis
        """
        heterogeneity = helpers.prep_confidence_interval(heterogeneity)

        x = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        # get the SHAP curve
        y = self.eval(feature, x, heterogeneity=False, centering=centering)
        y_std = (
            self.feature_effect["feature_" + str(feature)]["spline_std"](x)
            if heterogeneity == "std" or True
            else None
        )

        # get some SHAP values
        _, ind = helpers.prep_nof_instances(nof_shap_values, self.data.shape[0])
        yy = (
            self.feature_effect["feature_" + str(feature)]["yy"][ind]
            if heterogeneity == "shap_values"
            else None
        )
        if yy is not None and centering is not False:
            yy = yy - self.feature_effect["feature_" + str(feature)]["norm_const"]
        xx = (
            self.feature_effect["feature_" + str(feature)]["xx"][ind]
            if heterogeneity == "shap_values"
            else None
        )

        if show_avg_output:
            avg_output = helpers.prep_avg_output(self.data, self.model, self.avg_output, scale_y)
        else:
            avg_output = None

        vis.plot_shap(
            x,
            y,
            xx,
            yy,
            y_std,
            feature,
            heterogeneity=heterogeneity,
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            y_limits=y_limits
        )

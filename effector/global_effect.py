from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from effector import helpers, utils


class GlobalEffectBase(ABC):
    # the class-level centering default (R3): each subclass declares it once;
    # fit/eval/plot signatures converge on it during the homogenization
    DEFAULT_CENTERING: Union[bool, str] = False

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
        self.method_name = method_name.lower()
        self.model = model
        self.model_jac = model_jac

        self.dim = data.shape[1]

        # shared preprocessing: filter to axis_limits (or infer them), then
        # subsample nof_instances (helpers.prep_data)
        data, data_effect, axis_limits, self.nof_instances, self.indices = (
            helpers.prep_data(data, axis_limits, nof_instances, data_effect)
        )
        self.axis_limits: np.ndarray = axis_limits

        # store the data
        self.data: np.ndarray = data
        self.data_effect: Optional[np.ndarray] = data_effect

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
        **kwargs,
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
        **kwargs,
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

    @abstractmethod
    def _fit_feature(self, feature: int, **kwargs) -> dict:
        """Compute and return the method-specific payload for one feature
        (everything `eval`/`plot` need, except the normalization constant)."""
        raise NotImplementedError

    @abstractmethod
    def _eval_unnorm(
        self, feature: int, x: np.ndarray, heterogeneity: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """The method-specific evaluation kernel, over the stored (fitted)
        state: the *uncentered* mean effect at `x`, and — if `heterogeneity`
        — also the heterogeneity curve h(x) (a variance-like quantity in the
        method's own units, independent of any centering)."""
        raise NotImplementedError

    def _fit_loop(
        self,
        features: Union[int, str, list],
        centering: Union[bool, str],
        points_for_centering: int = 30,
        **fit_feature_kwargs,
    ) -> None:
        """The one fit skeleton every method shares (R1): normalize inputs,
        compute the per-feature payload, then the normalization constant."""
        features = helpers.prep_features(features, self.dim)
        centering = helpers.prep_centering(centering)
        for s in features:
            key = "feature_" + str(s)
            self.fit_args[key] = {
                "centering": centering,
                "points_for_centering": points_for_centering,
                **fit_feature_kwargs,
            }
            self.feature_effect[key] = self._fit_feature(s, **fit_feature_kwargs)
            self.feature_effect[key]["norm_const"] = (
                self._compute_norm_const(
                    s, method=centering, nof_points=points_for_centering
                )
                if centering is not False
                else None
            )
            self.is_fitted[s] = True

    def _compute_norm_const(
        self, feature: int, method: str = "zero_integral", nof_points: int = 30
    ) -> float:
        """Compute the normalization constant from the evaluation kernel:
        `zero_integral` = the mean over the feature interval, `zero_start` =
        the value at its left limit."""
        assert method in ["zero_integral", "zero_start"]

        def partial_eval(x):
            return self._eval_unnorm(feature, x, heterogeneity=False)

        start = self.axis_limits[0, feature]
        stop = self.axis_limits[1, feature]

        if method == "zero_integral":
            return utils.mean_1d_linspace(partial_eval, start, stop, nof_points)
        return partial_eval(np.array([start])).item()

    def eval_heter(self, feature: int, xs: np.ndarray) -> np.ndarray:
        """Evaluate the heterogeneity curve h(xs) of the `feature`-th feature.

        Notes:
            The values are *method-specific* (R2): the variance of the centered
            ICE curves (PDP), of the d-ICE curves (DerPDP), the per-bin variance
            of the local effects as a step function (ALE/RHALE), or the residual
            spline around the SHAP curve (ShapDP). They are variances — take a
            square root for a std-like band.

            There is deliberately no `centering` argument: heterogeneity is
            invariant to centering.

        Args:
            feature: index of feature of interest
            xs: the points to evaluate the heterogeneity at, `(T,)`

        Returns:
            the heterogeneity curve h(xs), `(T,)`, non-negative
        """
        if self.requires_refit(feature, centering=False):
            self.fit(features=feature)
        return self._eval_unnorm(feature, xs, heterogeneity=True)[1]

    def payload(self, feature: int) -> dict:
        """The method's raw fitted object for the `feature`-th feature — the
        honest method-specific state behind `eval`/`eval_heter` (bin effects
        and variances for (RH)ALE, splines and shap values for ShapDP, the
        normalization constants for PDP)."""
        if self.requires_refit(feature, centering=False):
            self.fit(features=feature)
        return dict(self.feature_effect["feature_" + str(feature)])

    def heter_score(self, feature: int) -> float:
        """The method-agnostic heterogeneity scalar of the `feature`-th
        feature: the mean of `eval_heter` over a 30-point grid on the feature's
        interval — the single quantity regional splitting (and the future
        interaction module) consumes."""
        xs = np.linspace(self.axis_limits[0, feature], self.axis_limits[1, feature], 30)
        return float(np.mean(self.eval_heter(feature, xs)))

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

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        centering: Union[None, bool, str] = None,
    ) -> np.ndarray:
        """Evaluate the mean effect of the `feature`-th feature at positions `xs`.

        Notes:
            This is the one evaluation method of every effect class (R1): it
            always returns the mean effect as a single `(T,)` array.
            Heterogeneity lives on its own surface — `eval_heter(feature, xs)`
            for the curve, `heter_score(feature)` for the scalar, and
            `payload(feature)` for the method's raw object.

        Args:
            feature: index of feature of interest
            xs: the points along the s-th axis to evaluate the effect at

              - `np.ndarray` of shape `(T, )`

            centering: whether to center the effect

                - `None` (default) uses the class default (`DEFAULT_CENTERING`)
                - `False`: no centering
                - `True` or `"zero_integral"`: center around the `y` axis
                - `"zero_start"`: the effect starts from `y=0`

        Returns:
            the mean effect `y` at the given `xs`, `(T,)`
        """
        centering = self.DEFAULT_CENTERING if centering is None else centering
        centering = helpers.prep_centering(centering)

        if self.requires_refit(feature, centering):
            self.fit(features=feature, centering=centering)

        if not self.axis_limits[0, feature] < self.axis_limits[1, feature]:
            raise ValueError(
                f"Feature {feature} has a degenerate axis interval "
                f"[{self.axis_limits[0, feature]}, {self.axis_limits[1, feature]}]"
            )

        y = self._eval_unnorm(feature, xs)
        if centering is not False:
            norm_const = self.feature_effect["feature_" + str(feature)]["norm_const"]
            # PDP stores a per-instance array (each ICE centers on its own);
            # the shift of the mean effect is its average
            y = y - (norm_const if np.ndim(norm_const) == 0 else np.mean(norm_const))
        return y

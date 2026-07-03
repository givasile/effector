import warnings
from typing import Callable, List, Optional, Union

import numpy as np

import effector.helpers as helpers
import effector.visualization as vis
from effector.global_effect_ale import ALE, RHALE
from effector.global_effect_pdp import PDP
from effector.global_effect_shap import ShapDP


class FeatureEffect:
    """Unified facade to compare global feature-effect methods on a single figure.

    `FeatureEffect` holds the shared ingredients (`data`, `model`, feature/target
    names, axis limits) once and lazily builds the underlying method objects
    (`PDP`, `ALE`, `RHALE`, `ShapDP`) on demand. Its `plot` overlays the mean
    effect of several methods for one feature on the same axis, so they can be
    compared directly.

    Notes:
        - All methods share the *same* background data: the data is filtered to
          `axis_limits` and subsampled to `nof_instances` once, then every method
          is built on that identical subset. The only difference between the
          curves is therefore the method itself.
        - The comparison is meaningful only for *centered* effects (each method
          uses a different reference level), so `plot` centers by default.
        - `DerPDP` is intentionally not part of the pool: it lives in derivative
          units and is not comparable with the output-unit methods.
    """

    # canonical name -> (class, needs model jacobian)
    _REGISTRY = {
        "pdp": (PDP, False),
        "ale": (ALE, False),
        "rhale": (RHALE, True),
        "shapdp": (ShapDP, False),
    }
    _ALIASES = {"shap": "shapdp", "shap_dp": "shapdp", "shap-dp": "shapdp"}
    _DISPLAY = {"pdp": "PDP", "ale": "ALE", "rhale": "RHALE", "shapdp": "SHAP-DP"}

    def __init__(
        self,
        data: np.ndarray,
        model: Callable,
        model_jac: Optional[Callable] = None,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 1_000,
        feature_names: Optional[List] = None,
        target_name: Optional[str] = None,
    ):
        """
        Args:
            data: the design matrix, shape `(N, D)`
            model: the black-box model, `Callable` `(N, D) -> (N,)`
            model_jac: the model Jacobian `(N, D) -> (N, D)`, optional. Needed by
                `RHALE`; if omitted, `RHALE` falls back to numerical differentiation.
            axis_limits: `(2, D)` array of per-feature limits, or `None` to infer
                from the data.
            nof_instances: number of instances shared across all methods.

                - use an `int` to subsample
                - use `"all"` to use every instance

            feature_names: list of feature names, or `None` for `["x_0", ...]`
            target_name: name of the target, or `None` for `"y"`
        """
        self.model = model
        self.model_jac = model_jac
        self.dim = data.shape[1]

        # shared preprocessing (helpers.prep_data): filter to axis limits (or
        # infer them), then subsample ONCE so every method sees the exact same
        # background data.
        self.data, _, self.axis_limits, _, _ = helpers.prep_data(
            data, axis_limits, nof_instances
        )

        self.feature_names = (
            helpers.get_feature_names(self.dim)
            if feature_names is None
            else feature_names
        )
        self.target_name = "y" if target_name is None else target_name

        # lazily instantiated method objects, keyed by canonical name
        self._methods: dict = {}

    def _canonical(self, name: str) -> str:
        key = self._ALIASES.get(name.lower(), name.lower())
        if key not in self._REGISTRY:
            raise ValueError(
                "Unknown method '{}'. Supported methods: {} (aliases: {}).".format(
                    name, sorted(self._REGISTRY), sorted(self._ALIASES)
                )
            )
        return key

    def _get_method(self, name: str, method_kwargs: Optional[dict] = None):
        """Build (and cache) the underlying effect object for `name`."""
        key = self._canonical(name)
        if key in self._methods:
            return self._methods[key]

        cls, needs_jac = self._REGISTRY[key]
        kwargs = dict(
            axis_limits=self.axis_limits,
            nof_instances="all",  # data is already subsampled in __init__
            feature_names=self.feature_names,
            target_name=self.target_name,
        )
        if method_kwargs:
            kwargs.update(method_kwargs)

        if needs_jac:
            if self.model_jac is None:
                warnings.warn(
                    "'{}' uses the model's jacobian, but `model_jac` was not passed "
                    "to FeatureEffect. Falling back to numerical differentiation "
                    "(slower and approximate) — pass `model_jac=...` for exact, "
                    "faster results.".format(self._DISPLAY[key]),
                    stacklevel=3,
                )
            obj = cls(self.data, self.model, self.model_jac, **kwargs)
        else:
            obj = cls(self.data, self.model, **kwargs)

        self._methods[key] = obj
        return obj

    def plot(
        self,
        feature: int,
        methods: Optional[List[str]] = None,
        centering: Union[bool, str] = True,
        nof_points: int = 100,
        scale_x: Optional[dict] = None,
        scale_y: Optional[dict] = None,
        y_limits: Optional[List] = None,
        show_avg_output: bool = False,
        method_kwargs: Optional[dict] = None,
        show_plot: bool = True,
    ):
        """Overlay the mean effect of several methods for one feature.

        Args:
            feature: index of the feature to plot
            methods: list of method names to compare. Supported: `"PDP"`, `"ALE"`,
                `"RHALE"`, `"ShapDP"` (alias `"SHAP"`). Defaults to
                `["PDP", "ALE", "RHALE"]` (`ShapDP` is opt-in as it is slower and
                needs the `shap` package).
            centering: how to center the curves. The comparison is meaningful only
                when centered, so `False` is coerced to `"zero_integral"`.

                - `True` / `"zero_integral"`: center around the `y` axis
                - `"zero_start"`: start each curve from `y=0`

            nof_points: size of the shared evaluation grid
            scale_x, scale_y: `None` or dict with keys `["mean", "std"]` to map the
                axes back to the original units
            y_limits: `None` or tuple, manual y-axis limits
            show_avg_output: whether to draw the model's average output as a line
            method_kwargs: optional `{method_name: {**constructor_kwargs}}` to
                customize individual methods (e.g. `{"ShapDP": {"nof_instances": 300}}`)
            show_plot: if `True`, show the figure; if `False`, return `(fig, ax)`
        """
        if methods is None:
            methods = ["PDP", "ALE", "RHALE"]

        centering = helpers.prep_centering(centering)
        if centering is False:
            warnings.warn(
                "Comparing methods without centering is not meaningful (each method "
                "uses a different reference level). Using centering='zero_integral'.",
                stacklevel=2,
            )
            centering = "zero_integral"

        # shared grid, shared across every method
        xs = np.linspace(
            self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
        )

        curves = {}
        for name in methods:
            mk = method_kwargs.get(name) if method_kwargs else None
            obj = self._get_method(name, mk)
            label = self._DISPLAY[self._canonical(name)]
            curves[label] = obj.eval(
                feature, xs, heterogeneity=False, centering=centering
            )

        avg_output = (
            helpers.prep_avg_output(self.data, self.model, None, scale_y)
            if show_avg_output
            else None
        )

        ret = vis.plot_effect_comparison(
            xs,
            feature,
            curves,
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            y_limits=y_limits,
            show_plot=show_plot,
        )
        if not show_plot:
            return ret

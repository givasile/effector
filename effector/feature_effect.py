import warnings
from typing import Callable, List, Optional, Union

import numpy as np

import effector.helpers as helpers
import effector.method_registry as method_registry
import effector.visualization as vis
from effector import ingestion


class FeatureEffect:
    """Unified facade to compare global feature-effect methods on a single figure.

    `FeatureEffect` holds the shared ingredients (`data`, `model`, feature/target
    names, axis limits) once and lazily builds the underlying method objects
    (`PDP`, `ALE`, `RHALE`, `ShapDP`) on demand. Its `eval` returns the mean
    effect of several methods on a shared grid, and its `plot` overlays them
    on the same axis, so they can be compared directly.

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

    # the comparison pool: every registry method in output units (R5 — the
    # per-method knowledge lives in effector.method_registry; DerPDP is in
    # derivative units, hence excluded)
    _POOL = ["pdp", "ale", "rhale", "shapdp"]

    def __init__(
        self,
        data,
        model: Callable,
        model_jac: Optional[Callable] = None,
        *,
        axis_limits: Optional[np.ndarray] = None,
        nof_instances: Union[int, str] = 10_000,
        schema: Optional[Union[ingestion.Schema, dict]] = None,
        random_state: Optional[int] = 21,
    ):
        """
        Args:
            data: the design matrix — a `(N, D)` numpy array or a pandas
                DataFrame (converted at the door, R10)
            model: the black-box model, `Callable` `(N, D) -> (N,)`; called with
                a reconstructed DataFrame when `data` is a DataFrame
            model_jac: the model Jacobian `(N, D) -> (N, D)`, optional. Needed by
                `RHALE`; if omitted, `RHALE` falls back to numerical differentiation.
            axis_limits: `(2, D)` array of per-feature limits, or `None` to infer
                from the data.
            nof_instances: number of instances shared across all methods.

                - use an `int` to subsample
                - use `"all"` to use every instance

            schema: input metadata (R10) — an `effector.Schema` or a plain `dict`
                with any of the keys `feature_names`, `feature_types`,
                `cat_limit`, `target_name`, `scale_x_list`, `scale_y`

                - omitted fields are inferred from the data (DataFrame dtypes,
                  numpy heuristics) or synthesized (`["x_0", ...]`, `"y"`)
                - explicit fields always win over inference

            random_state: seed for every internal random step (e.g. the shared
                `nof_instances` subsampling), inherited by every lazily built
                method object. Use an `int` (default: `21`) for reproducible
                output, or `None` for non-deterministic behavior.
        """
        self.random_state = random_state

        # the one door for data + metadata (R10)
        ing = ingestion.ingest(data, model, model_jac, schema=schema)
        data = ing.data
        self.model = ing.model
        self.model_jac = ing.model_jac
        self.feature_metadata = ing.meta
        self.dim = data.shape[1]

        # shared preprocessing (helpers.prep_data): filter to axis limits (or
        # infer them), then subsample ONCE so every method sees the exact same
        # background data.
        self.data, _, self.axis_limits, _, _ = helpers.prep_data(
            data, axis_limits, nof_instances, random_state=random_state
        )

        # flat mirrors of the resolved metadata
        self.feature_names = list(ing.meta.feature_names)
        self.feature_types = list(ing.meta.feature_types)
        self.cat_limit = ing.meta.cat_limit
        self.target_name = ing.meta.target_name
        self.scale_x_list = ing.meta.scale_x_list
        self.scale_y = ing.meta.scale_y

        # lazily instantiated method objects, keyed by canonical name
        self._methods: dict = {}

    def _canonical(self, name: str) -> str:
        key = method_registry.canonical(name)
        if key not in self._POOL:
            raise ValueError(
                "Method '{}' is not part of the comparison pool {} "
                "(d-PDP lives in derivative units and is not comparable).".format(
                    name, sorted(self._POOL)
                )
            )
        return key

    def _is_cat(self, feature: int) -> bool:
        return ingestion.is_categorical(self.feature_types[feature])

    def _levels(self, feature: int) -> np.ndarray:
        return np.unique(self.data[:, feature])

    def _supported_methods(self, feature: int, methods: List[str]) -> List[str]:
        """Drop methods whose capability matrix excludes this feature type (e.g.
        RHALE on a nominal feature), warning about what was skipped."""
        ftype = self.feature_types[feature]
        kept, dropped = [], []
        for name in methods:
            supported = method_registry.resolve(
                self._canonical(name)
            ).cls.SUPPORTED_FEATURE_TYPES
            (kept if ftype in supported else dropped).append(name)
        if dropped:
            warnings.warn(
                "Skipping {} for feature {!r} ({}): not supported for this "
                "feature type.".format(dropped, self.feature_names[feature], ftype),
                stacklevel=3,
            )
        if not kept:
            raise ValueError(
                "No requested method supports feature {!r} ({}).".format(
                    self.feature_names[feature], ftype
                )
            )
        return kept

    def _get_method(self, name: str, method_kwargs: Optional[dict] = None):
        """Build (and cache) the underlying effect object for `name`."""
        key = self._canonical(name)
        if key in self._methods:
            return self._methods[key]

        spec = method_registry.resolve(key)
        # explicit schema: sub-objects get the resolved metadata (types would
        # otherwise be re-inferred from the already-encoded numpy matrix)
        sub_schema = ingestion.Schema(
            feature_names=self.feature_names,
            feature_types=self.feature_types,
            cat_limit=self.cat_limit,
            target_name=self.target_name,
            scale_x_list=self.scale_x_list,
            scale_y=self.scale_y,
        )
        kwargs = dict(
            axis_limits=self.axis_limits,
            nof_instances="all",  # data is already subsampled in __init__
            schema=sub_schema,
            random_state=self.random_state,
        )
        if method_kwargs:
            kwargs.update(method_kwargs)

        if spec.needs_jac:
            if self.model_jac is None:
                warnings.warn(
                    "'{}' uses the model's jacobian, but `model_jac` was not passed "
                    "to FeatureEffect. Falling back to numerical differentiation "
                    "(slower and approximate) — pass `model_jac=...` for exact, "
                    "faster results.".format(spec.display_name),
                    stacklevel=3,
                )
            obj = spec.cls(self.data, self.model, self.model_jac, **kwargs)
        else:
            obj = spec.cls(self.data, self.model, **kwargs)

        self._methods[key] = obj
        return obj

    def eval(
        self,
        feature: int,
        xs: np.ndarray,
        methods: Optional[List[str]] = None,
        centering: Union[bool, str] = True,
        method_kwargs: Optional[dict] = None,
    ) -> dict:
        """Evaluate the mean effect of several methods on a shared grid.

        Args:
            feature: index of the feature
            xs: the grid to evaluate on, shape `(T,)`
            methods: list of method names. Supported: `"PDP"`, `"ALE"`,
                `"RHALE"`, `"ShapDP"` (alias `"SHAP"`). Defaults to
                `["PDP", "ALE", "RHALE"]` (`ShapDP` is opt-in as it is slower
                and needs the `shap` package).
            centering: how to center the curves (R3 vocabulary)
            method_kwargs: optional `{method_name: {**constructor_kwargs}}` to
                customize individual methods

        Returns:
            `{display_name: y}` with one `(T,)` mean-effect array per method
        """
        if methods is None:
            methods = ["PDP", "ALE", "RHALE"]
        methods = self._supported_methods(feature, methods)
        centering = helpers.prep_centering(centering)

        curves = {}
        for name in methods:
            mk = method_kwargs.get(name) if method_kwargs else None
            obj = self._get_method(name, mk)
            label = method_registry.resolve(self._canonical(name)).display_name
            curves[label] = obj.eval(feature, xs, centering=centering)
        return curves

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
        centering = helpers.prep_centering(centering)
        scale_x = helpers.resolve_scale(
            scale_x, self.scale_x_list[feature] if self.scale_x_list else None
        )
        scale_y = helpers.resolve_scale(scale_y, self.scale_y)
        if centering is False:
            warnings.warn(
                "Comparing methods without centering is not meaningful (each method "
                "uses a different reference level). Using centering='zero_integral'.",
                stacklevel=2,
            )
            centering = "zero_integral"

        # shared grid: observed levels for a categorical feature (evaluated only
        # at levels, R10), else a continuous linspace
        discrete = self._is_cat(feature)
        if discrete:
            xs = self._levels(feature)
        else:
            xs = np.linspace(
                self.axis_limits[0, feature], self.axis_limits[1, feature], nof_points
            )
        curves = self.eval(
            feature,
            xs,
            methods=methods,
            centering=centering,
            method_kwargs=method_kwargs,
        )

        avg_output = (
            helpers.prep_avg_output(self.data, self.model, None, scale_y)
            if show_avg_output
            else None
        )

        level_labels = None
        if discrete:
            name_map = (self.feature_metadata.category_names or {}).get(feature)
            if name_map is not None:
                level_labels = [name_map.get(float(v), f"{v:g}") for v in xs]

        return vis.plot_effect_comparison(
            xs,
            feature,
            curves,
            scale_x=scale_x,
            scale_y=scale_y,
            avg_output=avg_output,
            feature_names=self.feature_names,
            target_name=self.target_name,
            y_limits=y_limits,
            discrete=discrete,
            level_labels=level_labels,
            show_plot=show_plot,
        )

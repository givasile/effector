import typing
import numpy as np

import effector.space_partitioning
from effector.regional_effect import RegionalEffectBase
from effector import helpers
from effector.global_effect_pdp import PDP, DerPDP
from tqdm import tqdm

BIG_M = helpers.BIG_M


class RegionalPDPBase(RegionalEffectBase):
    def __init__(
        self,
        method_name: str,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable] = None,
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        self.y_ice = {}
        super(RegionalPDPBase, self).__init__(
            method_name,
            data,
            model,
            model_jac,
            None,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name,
        )

    def _create_heterogeneity_function(
        self,
        foi,
        min_points,
    ):
        def heter(active_indices) -> float:
            if np.sum(active_indices) < min_points:
                return BIG_M
            yy = self.y_ice["feature_" + str(foi)][active_indices.astype(bool), :]
            z = np.var(yy, axis=0)
            return np.mean(z)
        return heter




class RegionalPDP(RegionalPDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        """
        Initialize the Regional Effect method.

        Args:
            data: the design matrix, `ndarray` of shape `(N,D)`
            model: the black-box model, `Callable` with signature `f(x) -> y` where:

                - `x`: `ndarray` of shape `(N, D)`
                - `y`: `ndarray` of shape `(N)`

            axis_limits: Feature effect limits along each axis

                - `None`, infers them from `data` (`min` and `max` of each feature)
                - `array` of shape `(D, 2)`, manually specify the limits for each feature.

                !!! tip "When possible, specify the axis limits manually"

                    - they help to discard outliers and improve the quality of the fit
                    - `axis_limits` define the `.plot` method's x-axis limits; manual specification leads to better visualizations

                !!! tip "Their shape is `(2, D)`, not `(D, 2)`"

                    ```python
                    axis_limits = np.array([[0, 1, -1], [1, 2, 3]])
                    ```

            nof_instances: Max instances to use

                - `"all"`, uses all `data`
                - `int`, randomly selects `int` instances from `data`

                !!! tip "`10_000` (default), is a good balance between speed and accuracy"

            feature_types: The feature types.

                - `None`, infers them from data; if the number of unique values is less than `cat_limit`, it is considered categorical.
                - `['cat', 'cont', ...]`, manually specify the types of the features

            cat_limit: The minimum number of unique values for a feature to be considered categorical

                - if `feature_types` is manually specified, this parameter is ignored

            feature_names: The names of the features

                - `None`, defaults to: `["x_0", "x_1", ...]`
                - `["age", "weight", ...]` to manually specify the names of the features

            target_name: The name of the target variable

                - `None`, to keep the default name: `"y"`
                - `"price"`, to manually specify the name of the target variable
        """

        super(RegionalPDP, self).__init__(
            "pdp",
            data,
            model,
            None,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name,
        )

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union[str, None] = "best",
        points_for_centering: int = 30,
        points_for_mean_heterogeneity: int = 30,
        use_vectorized: bool = True,
    ):
        """
        Find subregions by minimizing the PDP-based heterogeneity.

        Args:
            features: for which features to search for subregions

                - use `"all"`, for all features, e.g. `features="all"`
                - use an `int`, for a single feature, e.g. `features=0`
                - use a `list`, for multiple features, e.g. `features=[0, 1, 2]`

            candidate_conditioning_features: list of features to consider as conditioning features

                - use `"all"`, for all features, e.g. `candidate_conditioning_features="all"`
                - use a `list`, for multiple features, e.g. `candidate_conditioning_features=[0, 1, 2]`
                - it means that for each feature in the `feature` list, the algorithm will consider applying a split
                conditioned on each feature in the `candidate_conditioning_features` list

            space_partitioner: the method to use for partitioning the space
            points_for_centering: number of equidistant points along the feature axis used for centering ICE plots
            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity
            use_vectorized: whether to use vectorized operations for the PDP and ICE curves


        """

        if isinstance(space_partitioner, str):
            space_partitioner = effector.space_partitioning.return_default(space_partitioner)

        assert space_partitioner.min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            # define the global method
            pdp = PDP(self.data, self.model, self.axis_limits, nof_instances="all")

            pdp.fit(
                features=feat,
                centering=True,
                points_for_centering=points_for_centering,
                use_vectorized=use_vectorized,
            )

            xx = np.linspace(self.axis_limits[:, feat][0], self.axis_limits[:, feat][1], points_for_mean_heterogeneity)
            y_ice = pdp.eval(
                    feature=feat,
                    xs=xx,
                    heterogeneity=True,
                    centering=True,
                    use_vectorized=use_vectorized,
                    return_all=True
                )
            self.y_ice["feature_" + str(feat)] = y_ice.T

            heter = self._create_heterogeneity_function(
                foi = feat,
                min_points=space_partitioner.min_points_per_subregion,
            )

            self._fit_feature(
                feat,
                heter,
                space_partitioner,
                candidate_conditioning_features,
            )

        all_arguments = locals()
        all_arguments.pop("self")

        # region splitting arguments are the first 8 arguments
        self.kwargs_subregion_detection = {k: all_arguments[k] for k in list(all_arguments.keys())[:3]}
        self.kwargs_subregion_detection["points_for_mean_heterogeneity"] = points_for_mean_heterogeneity

        # centering, points_for_centering, use_vectorized
        self.kwargs_fitting = {k:v for k,v in all_arguments.items() if k in ["centering", "points_for_centering", "use_vectorized"]}

    def plot(
        self,
        feature: int,
        node_idx: int,
        heterogeneity: bool = "ice",
        centering: typing.Union[bool, str] = False,
        nof_points: int = 30,
        scale_x_list: typing.Union[None, list] = None,
        scale_y: typing.Union[None, list] = None,
        nof_ice: int = 100,
        show_avg_output: bool = False,
        y_limits: typing.Union[None, list] = None,
        use_vectorized: bool = True,
    ):
        kwargs = locals()
        kwargs.pop("self")
        self._plot(kwargs)


class RegionalDerPDP(RegionalPDPBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Optional[callable] = None,
        nof_instances: typing.Union[int, str] = 10_000,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        target_name: typing.Union[str, None] = None,
    ):
        """
        Initialize the Regional Effect method.

        Args:
            data: the design matrix, `ndarray` of shape `(N,D)`
            model: the black-box model, `Callable` with signature `x -> y` where:

                - `x`: `ndarray` of shape `(N, D)`
                - `y`: `ndarray` of shape `(N)`

            model_jac: the black-box model's Jacobian, `Callable` with signature `x -> dy_dx` where:

                - `x`: `ndarray` of shape `(N, D)`
                - `dy_dx`: `ndarray` of shape `(N, D)`

            axis_limits: Feature effect limits along each axis

                - `None`, infers them from `data` (`min` and `max` of each feature)
                - `array` of shape `(D, 2)`, manually specify the limits for each feature.

                !!! tip "When possible, specify the axis limits manually"

                    - they help to discard outliers and improve the quality of the fit
                    - `axis_limits` define the `.plot` method's x-axis limits; manual specification leads to better visualizations

                !!! tip "Their shape is `(2, D)`, not `(D, 2)`"

                    ```python
                    axis_limits = np.array([[0, 1, -1], [1, 2, 3]])
                    ```

            nof_instances: Max instances to use

                - `"all"`, uses all `data`
                - `int`, randomly selects `int` instances from `data`

                !!! tip "`10_000` (default), is a good balance between speed and accuracy"

            feature_types: The feature types.

                - `None`, infers them from data; if the number of unique values is less than `cat_limit`, it is considered categorical.
                - `['cat', 'cont', ...]`, manually specify the types of the features

            cat_limit: The minimum number of unique values for a feature to be considered categorical

                - if `feature_types` is manually specified, this parameter is ignored

            feature_names: The names of the features

                - `None`, defaults to: `["x_0", "x_1", ...]`
                - `["age", "weight", ...]` to manually specify the names of the features

            target_name: The name of the target variable

                - `None`, to keep the default name: `"y"`
                - `"price"`, to manually specify the name of the target variable
        """

        super(RegionalDerPDP, self).__init__(
            "d-pdp",
            data,
            model,
            model_jac,
            nof_instances,
            axis_limits,
            feature_types,
            cat_limit,
            feature_names,
            target_name,
        )

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        candidate_conditioning_features: typing.Union["str", list] = "all",
        space_partitioner: typing.Union[str, None] = "best",
        points_for_mean_heterogeneity: int = 30,
        use_vectorized: bool = True,
    ):
        """
        Find subregions by minimizing the PDP-based heterogeneity.

        Args:
            features: for which features to search for subregions

                - use `"all"`, for all features, e.g. `features="all"`
                - use an `int`, for a single feature, e.g. `features=0`
                - use a `list`, for multiple features, e.g. `features=[0, 1, 2]`

            candidate_conditioning_features: list of features to consider as conditioning features

                - use `"all"`, for all features, e.g. `candidate_conditioning_features="all"`
                - use a `list`, for multiple features, e.g. `candidate_conditioning_features=[0, 1, 2]`
                - it means that for each feature in the `feature` list, the algorithm will consider applying a split
                conditioned on each feature in the `candidate_conditioning_features` list

            space_partitioner: the method to use for partitioning the space
            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity
            use_vectorized: whether to use vectorized operations for the PDP and ICE curves


        """

        if isinstance(space_partitioner, str):
            space_partitioner = effector.space_partitioning.return_default(space_partitioner)

        assert space_partitioner.min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            # define the global method
            pdp = DerPDP(self.data, self.model, self.model_jac, self.axis_limits, nof_instances="all")

            pdp.fit(
                features=feat,
                centering=False,
                use_vectorized=use_vectorized,
            )

            xx = np.linspace(self.axis_limits[:, feat][0], self.axis_limits[:, feat][1], points_for_mean_heterogeneity)
            y_ice = pdp.eval(
                    feature=feat,
                    xs=xx,
                    heterogeneity=True,
                    centering=False,
                    use_vectorized=use_vectorized,
                    return_all=True
                )
            self.y_ice["feature_" + str(feat)] = y_ice.T

            heter = self._create_heterogeneity_function(
                foi = feat,
                min_points=space_partitioner.min_points_per_subregion,
            )

            self._fit_feature(
                feat,
                heter,
                space_partitioner,
                candidate_conditioning_features,
            )

        all_arguments = locals()
        all_arguments.pop("self")

        # region splitting arguments are the first 8 arguments
        self.kwargs_subregion_detection = {k: all_arguments[k] for k in list(all_arguments.keys())[:3]}
        self.kwargs_subregion_detection["points_for_mean_heterogeneity"] = points_for_mean_heterogeneity

        # centering, points_for_centering, use_vectorized
        self.kwargs_fitting = {k:v for k,v in all_arguments.items() if k in ["centering", "points_for_centering", "use_vectorized"]}

    def plot(
        self,
        feature: int,
        node_idx: int = 0,
        heterogeneity: bool = "ice",
        centering: typing.Union[bool, str] = False,
        nof_points: int = 30,
        scale_x_list: typing.Union[None, list] = None,
        scale_y: typing.Union[None, list] = None,
        nof_ice: int = 100,
        show_avg_output: bool = False,
        dy_limits: typing.Union[None, list] = None,
        use_vectorized: bool = True,
    ):
        kwargs = locals()
        kwargs.pop("self")
        self._plot(kwargs)

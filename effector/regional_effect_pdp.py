import typing
import numpy as np
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

    def fit(
        self,
        features: typing.Union[int, str, list] = "all",
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.,
        max_depth: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
        centering: typing.Union[bool, str] = False,
        points_for_centering: int = 50,
        points_for_mean_heterogeneity: int = 50,
        use_vectorized: bool = True,
    ):
        """
        Find subregions by minimizing the PDP-based heterogeneity.

        Args:
            features: for which features to search for subregions

                - use `"all"`, for all features, e.g. `features="all"`
                - use an `int`, for a single feature, e.g. `features=0`
                - use a `list`, for multiple features, e.g. `features=[0, 1, 2]`

            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important

                - use a `float`, e.g. `heter_pcg_drop_thres=0.1`
                - The heterogeity drop is expressed as percentage ${(H_{\mathtt{before\_split}} - H_{\mathtt{after\_split}}) \over H_{\mathtt{before\_split}}}$

            heter_small_enough: heterogeneity threshold for a split to be considered already small enough

                - if the current split has an heterogeneity smaller than this value, it is not further split
                - use a `float`, e.g. `heter_small_enough=0.01`

            max_depth: maximum depth of the tree

            nof_candidate_splits_for_numerical: number of candidate splits for numerical features

                - use an `int`, e.g. `nof_candidate_splits_for_numerical=20`
                - The candidate splits are uniformly distributed between the minimum and maximum values of the feature
                - e.g. if range is [0, 1] and `nof_candidate_splits_for_numerical=3`, the candidate splits are [0.25, 0.5, 0.75]

            min_points_per_subregion: minimum number of points per subregion

                - use an `int`, e.g. `min_points_per_subregion=10`
                - if a subregion has less than `min_points_per_subregion` instances, it is discarded

            candidate_conditioning_features: list of features to consider as conditioning features

                - use `"all"`, for all features, e.g. `candidate_conditioning_features="all"`
                - use a `list`, for multiple features, e.g. `candidate_conditioning_features=[0, 1, 2]`
                - it means that for each feature in the `feature` list, the algorithm will consider applying a split
                conditioned on each feature in the `candidate_conditioning_features` list

            split_categorical_features: whether to find subregions for categorical features

               - It indicates whether to create a splitting tree for categorical features
               - It does not mean whether the conditioning feature can be categorical (it can be)

            centering: whether to center the PDP and ICE curves, before computing the heterogeneity

                - If `centering` is `False`, the PDP not centered
                - If `centering` is `True` or `zero_integral`, the PDP is centered around the `y` axis.
                - If `centering` is `zero_start`, the PDP starts from `y=0`.

            points_for_centering: number of equidistant points along the feature axis used for centering ICE plots
            points_for_mean_heterogeneity: number of equidistant points along the feature axis used for computing the mean heterogeneity
            use_vectorized: whether to use vectorized operations for the PDP and ICE curves


        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            # define the global method
            if self.method_name == "pdp":
                pdp = PDP(self.data, self.model, self.axis_limits, nof_instances="all")
            else:
                pdp = DerPDP(self.data, self.model, self.model_jac, self.axis_limits, nof_instances="all")

            pdp.fit(
                features=feat,
                centering=centering,
                points_for_centering=points_for_centering,
                use_vectorized=use_vectorized,
            )

            xx = np.linspace(self.axis_limits[:, feat][0], self.axis_limits[:, feat][1], points_for_mean_heterogeneity)
            y_ice = pdp.eval(
                    feature=feat,
                    xs=xx,
                    heterogeneity=True,
                    use_vectorized=use_vectorized,
                    return_all=True
                )
            self.y_ice["feature_" + str(feat)] = y_ice.T

            heter = self._create_heterogeneity_function(
                foi = feat,
                min_points=min_points_per_subregion,
            )

            self._fit_feature(
                feat,
                heter,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_depth,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )
            # todo add method args


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
        Constructor of the PDP class.

        Definition:
            Finds subregions by minimizing the PDP-based heterogeneity:
            $$
            H_{x_s} = {1 \over M} \sum_{j=1}^M h(x_s^j), \quad h(x_s) = {1 \over N} \sum_{i=1}^N ( ICE_c^i(x_s) - PDP_c(x_s) )^2
            $$

            where $x_s^j$ are an equally spaced grid of points in the axis, i.e.,  $[x_s^{\min}, x_s^{\max}]$.

        Notes:
            The required parameters are `data` and `model`. The rest are optional.

        Args:
            data: the design matrix

                - shape: `(N,D)`

            model: the black-box model. Must be a `Callable` with:

                - input: `ndarray` of shape `(N, D)`
                - output: `ndarray` of shape `(N,)`

            nof_instances: maximum number of instances to be used for the analysis.
                           The selection is done at the beginning of the analysis.
                           If there are less instances, all will be used.

                - use `"all"`, for using all instances.
                - use an `int`, for using `nof_instances` instances.

            axis_limits: The limits of the feature effect plot along each axis

                - use a `ndarray` of shape `(2, D)`, to specify them manually
                - use `None`, to be inferred from the data

            feature_types: The type of each feature

                - use `"cont"` for continuous and `"cat"` for categorical
                - e.g. `["cont", "cont", "cat", ... ]

            cat_limit: The number of individual values to consider a feature as categorical

                - if `feature_types` is provided, this argument remains unused
                - if `feature_types` is `None`, features with less than `cat_limit` unique values will
                be considered `categorical`, while the rest `numerical`

            feature_names: The names of the features

                - use a `list` of `str`, to specify the name manually. For example: `                  ["age", "weight", ...]`
                - use `None`, to keep the default names: `["x_0", "x_1", ...]`

            target_name: The name of the target variable

                - use a `str`, to specify it name manually. For example: `"price"`
                - use `None`, to keep the default name: `"y"`
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


class RegionalDerPDP(RegionalPDPBase):
    def __init__(
        self,
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

import numpy as np
import effector.binning_methods as binning_methods
import effector.helpers as helpers
import effector.utils as utils
from effector.regions import Regions
from effector.global_effect_rhale import RHALE
from effector.global_effect_pdp import PDP, PDPwithICE
import typing
from tqdm import tqdm


BIG_M = helpers.BIG_M


# base method for all regional effect methods
class RegionalEffect:
    def __init__(self,
                 axis_limits: np.ndarray,
                 feature_types: list[str],
                 feature_names: list[str]
                 ) -> None:
        """
        Constructor for the RegionalEffect class.


        """
        self.axis_limits: np.ndarray = axis_limits
        self.feature_types: typing.Union[list, None] = feature_types
        self.feature_names: typing.Union[list, None] = feature_names
        self.dim = self.axis_limits.shape[1]

        # state variables
        self.regions: typing.Dict[str, Regions] = {}
        self.splits_per_feature_full_depth: typing.Dict[str, list[dict]] = {}
        self.splts_per_feature_only_important: typing.Dict[str, list[dict]] = {}
        self.splits_per_feature_full_depth_found: np.ndarray = np.ones([self.dim]) < 0
        self.splits_per_feature_only_import_found: np.ndarray = np.ones([self.dim]) < 0

    def _fit_feature(self, *args, **kwargs):
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        raise NotImplementedError

    def plot_first_level(self, *args, **kwargs):
        raise NotImplementedError

    def describe_subregions(
        self,
        features,
        only_important=True,
        scale_x: typing.Union[None, list[dict]] = None
    ):
        features = helpers.prep_features(features, self.dim)
        for feature in features:
            if not self.splits_per_feature_full_depth_found[feature]:
                self._fit_feature(feature)

            feature_name = (
                self.feature_names[feature] if self.feature_names else feature
            )
            if only_important:
                print("Important splits for feature {}".format(feature_name))
                splits = self.splts_per_feature_only_important[
                    "feature_{}".format(feature)
                ]
            else:
                print("All splits for feature {}".format(feature_name))
                splits = self.splits_per_feature_full_depth[
                    "feature_{}".format(feature)
                ]

            for i, split in enumerate(splits):
                type_of_split_feature = self.feature_types[split["feature"]]
                foc_name = (
                    split["feature"]
                    if self.feature_names is None
                    else self.feature_names[split["feature"]]
                )
                print("- On feature {} ({})".format(foc_name, type_of_split_feature))

                x_start, x_stop = split["range"][0], split["range"][1]
                if scale_x is not None:
                    x_start = x_start * scale_x[split["feature"]]["std"] + scale_x[split["feature"]]["mean"]
                    x_stop = x_stop * scale_x[split["feature"]]["std"] + scale_x[split["feature"]]["mean"]
                range_formatted = "[{:.2f}, {:.2f}]".format(x_start, x_stop)
                print("  - Range: {}".format(range_formatted))

                candidate_splits_formatted = (
                    ", ".join(
                        ["{:.2f}".format(x) for x in split["candidate_split_positions"]]
                    )
                    if scale_x is None
                    else ", ".join(
                        [
                            "{:.2f}".format(
                                x * scale_x[split["feature"]]["std"]
                                + scale_x[split["feature"]]["mean"]
                            )
                            for x in split["candidate_split_positions"]
                        ]
                    )
                )
                print(
                    "  - Candidate split positions: {}".format(
                        candidate_splits_formatted
                    )
                )
                position_split_formatted = (
                    "{:.2f}".format(split["position"])
                    if scale_x is None
                    else "{:.2f}".format(
                        split["position"] * scale_x[split["feature"]]["std"]
                        + scale_x[split["feature"]]["mean"]
                    )
                )
                print("  - Position of split: {}".format(position_split_formatted))

                print(
                    "  - Heterogeneity before split: {:.2f}".format(
                        split["weighted_heter"] + split["weighted_heter_drop"]
                    )
                )
                print(
                    "  - Heterogeneity after split: {:.2f}".format(
                        split["weighted_heter"]
                    )
                )
                print(
                    "  - Heterogeneity drop: {:.2f} ({:.2f} %)".format(
                        split["weighted_heter_drop"],
                        (split["weighted_heter_drop"] / split["weighted_heter"] * 100),
                    )
                )

                nof_instances_before = (
                    sum(split["nof_instances"])
                    if i == 0
                    else splits[i - 1]["nof_instances"]
                )
                print(
                    "  - Number of instances before split: {}".format(
                        nof_instances_before
                    )
                )
                print(
                    "  - Number of instances after split: {}".format(
                        split["nof_instances"]
                    )
                )


class RegionalRHALE(RegionalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
    ):
        """
        Regional RHALE constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            model_jac: the black-box model Jacobian (N,D) -> (N,D)
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
        """
        self.data = data
        self.model = model
        self.model_jac = model_jac
        axis_limits = helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        feature_types = utils.get_feature_types(data, cat_limit) if feature_types is None else feature_types
        feature_names = helpers.get_feature_names(self.dim) if feature_names is None else feature_names

        self.cat_limit = cat_limit
        self.instance_effects = self.model_jac(self.data)

        super(RegionalRHALE, self).__init__(axis_limits, feature_types, feature_names)

    def _create_heterogeneity_function(self, foi, binning_method, min_points=10):
        binning_method = helpers.prep_binning_method(binning_method)

        def heter(data, instance_effects=None) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            rhale = RHALE(data, self.model, self.model_jac, None, instance_effects)
            try:
                rhale.fit(features=foi, binning_method=binning_method)
            except:
                return BIG_M

            # heterogeneity is the accumulated std at the end of the curve
            axis_limits = helpers.axis_limits_from_data(data)
            stop = np.array([axis_limits[:, foi][1]])
            _, z, _ = rhale.eval(feature=foi, xs=stop, uncertainty=True)
            return z.item()

        return heter

    def _fit_feature(
        self,
        feature: int,
        heter_pcg_drop_thres=0.1,
        heter_small_enough=0.1,
        binning_method: typing.Union[
            str,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional RHALE for a single feature.
        """
        heter = self._create_heterogeneity_function(
            feature, binning_method, min_points_per_subregion
        )

        # init Region Extractor
        regions = Regions(
            feature,
            heter,
            self.data,
            self.instance_effects,
            self.feature_types,
            self.cat_limit,
            candidate_conditioning_features,
            min_points_per_subregion,
            nof_candidate_splits_for_numerical,
            max_split_levels,
            heter_pcg_drop_thres,
            heter_small_enough,
            split_categorical_features,
        )

        self.regions["feature_{}".format(feature)] = regions
        splits = regions.search_all_splits()

        self.splits_per_feature_full_depth["feature_{}".format(feature)] = splits
        self.splits_per_feature_full_depth_found[feature] = True

        important_splits = regions.choose_important_splits()
        self.splts_per_feature_only_important[
            "feature_{}".format(feature)
        ] = important_splits
        self.splits_per_feature_only_import_found[feature] = True

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        binning_method: typing.Union[
            str,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional RHALE for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            binning_method: binning method to use
            max_split_levels: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            self._fit_feature(
                feat,
                heter_pcg_drop_thres,
                heter_small_enough,
                binning_method,
                max_split_levels,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )

    def plot_first_level(
        self,
        feature: int = 0,
        confidence_interval: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = False,
        scale_x_per_feature: typing.Union[None, list[dict[str, float]]] = None,
        scale_y: typing.Union[None, dict] = None,
    ):

        if not self.splits_per_feature_full_depth_found[feature]:
            self._fit_feature(feature)

        regions = self.regions["feature_{}".format(feature)]
        splits = self.splts_per_feature_only_important["feature_{}".format(feature)]

        if len(splits) == 0:
            print("No important splits found for feature {}".format(feature))
            return

        # split to two datasets
        foc = splits[0]["feature"]
        position = splits[0]["position"]
        type_of_split_feature = self.feature_types[foc]
        data_1, data_2 = regions.split_dataset(
            self.data, None, foc, position, type_of_split_feature
        )
        data_effect_1, data_effect_2 = regions.split_dataset(
            self.data, self.instance_effects, foc, position, type_of_split_feature
        )
        axis_limits = helpers.axis_limits_from_data(self.data)

        # create two RHALE objects and plot
        rhale_1 = RHALE(data_1, self.model, self.model_jac, axis_limits, data_effect_1)
        rhale_2 = RHALE(data_2, self.model, self.model_jac, axis_limits, data_effect_2)

        # plot the two RHALE objects
        foc_name = self.feature_names[foc]
        foi_name = self.feature_names[feature]

        position = (
            position
            if scale_x_per_feature is None
            else position * scale_x_per_feature[foc]["std"]
            + scale_x_per_feature[foc]["mean"]
        )

        title_1 = "Regional RHALE for {} ({} == {:.2f})".format(
            foi_name, foc_name, position
        )
        title_2 = "Regional RHALE for {} ({} != {:.2f})".format(
            foi_name, foc_name, position
        )
        rhale_1.plot(
            feature,
            confidence_interval,
            centering,
            scale_x_per_feature[feature],
            scale_y,
            title_1,
        )
        rhale_2.plot(
            feature,
            confidence_interval,
            centering,
            scale_x_per_feature[feature],
            scale_y,
            title_2,
        )


class RegionalPDP(RegionalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
    ):
        """
        Regional PDP constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
        """
        self.dim = data.shape[1]
        self.data = data
        self.model = model

        axis_limits = helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        feature_types = utils.get_feature_types(data, cat_limit) if feature_types is None else feature_types
        feature_names = helpers.get_feature_names(self.dim) if feature_names is None else feature_names

        self.cat_limit = cat_limit
        super(RegionalPDP, self).__init__(axis_limits, feature_types, feature_names)

    def _create_heterogeneity_function(self, foi, min_points=10):

        def heter(data) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            pdp = PDP(data, self.model, nof_instances=100)
            try:
                pdp.fit(features=foi)
            except:
                return BIG_M

            # heterogeneity is the mean heterogeneity over the curve
            axis_limits = helpers.axis_limits_from_data(data)

            xx = np.linspace(axis_limits[:, foi][0], axis_limits[:, foi][1], 10)
            try:
                _, z, _ = pdp.eval(feature=foi, xs=xx, uncertainty=True)
            except:
                return BIG_M
            return np.mean(z)

        return heter

    def _fit_feature(
        self,
        feature: int,
        heter_pcg_drop_thres=0.1,
        heter_small_enough=0.1,
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional RHALE for a single feature.
        """
        heter = self._create_heterogeneity_function(
            feature, min_points_per_subregion
        )

        # init Region Extractor
        regions = Regions(
            feature,
            heter,
            self.data,
            None,
            self.feature_types,
            self.cat_limit,
            candidate_conditioning_features,
            min_points_per_subregion,
            nof_candidate_splits_for_numerical,
            max_split_levels,
            heter_pcg_drop_thres,
            heter_small_enough,
            split_categorical_features,
        )

        self.regions["feature_{}".format(feature)] = regions
        splits = regions.search_all_splits()

        self.splits_per_feature_full_depth["feature_{}".format(feature)] = splits
        self.splits_per_feature_full_depth_found[feature] = True

        important_splits = regions.choose_important_splits()
        self.splts_per_feature_only_important[
            "feature_{}".format(feature)
        ] = important_splits
        self.splits_per_feature_only_import_found[feature] = True

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional PDP for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            max_split_levels: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
            split_categorical_features
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            self._fit_feature(
                feat,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_split_levels,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )


class RegionalPDPwithICE(RegionalEffect):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
        nof_instances: int = 100,
    ):
        """
        Regional PDP constructor.

        Args:
            data: X matrix (N,D).
            model: the black-box model (N,D) -> (N, )
            axis_limits: axis limits for the FE plot [2, D] or None. If None, axis limits are computed from the data.
            feature_types: list of feature types (categorical or numerical)
            cat_limit: the minimum number of unique values for a feature to be considered categorical
            feature_names: list of feature names
            nof_instances: number of instances to use for the ICE curves
        """
        self.dim = data.shape[1]
        self.data = data
        self.model = model
        self.nof_instances = nof_instances

        axis_limits = helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        feature_types = utils.get_feature_types(data, cat_limit) if feature_types is None else feature_types
        feature_names = helpers.get_feature_names(self.dim) if feature_names is None else feature_names

        self.cat_limit = cat_limit
        super(RegionalPDPwithICE, self).__init__(axis_limits, feature_types, feature_names)

    def _create_heterogeneity_function(self, foi, min_points=10):

        def heter(data) -> float:
            if data.shape[0] < min_points:
                return BIG_M

            pdp_ice = PDPwithICE(data, self.model, nof_instances=self.nof_instances)
            try:
                pdp_ice.fit(features=foi, centering=True, nof_points_centering=10)
            except:
                return BIG_M

            # heterogeneity is the mean heterogeneity over the curve
            axis_limits = helpers.axis_limits_from_data(data)

            xx = np.linspace(axis_limits[:, foi][0], axis_limits[:, foi][1], 30)
            try:
                _, z, _ = pdp_ice.eval(feature=foi, xs=xx, uncertainty=True)
            except:
                return BIG_M
            return np.mean(z)

        return heter

    def _fit_feature(
        self,
        feature: int,
        heter_pcg_drop_thres=0.1,
        heter_small_enough=0.1,
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional RHALE for a single feature.
        """
        heter = self._create_heterogeneity_function(
            feature, min_points_per_subregion
        )

        # init Region Extractor
        regions = Regions(
            feature,
            heter,
            self.data,
            None,
            self.feature_types,
            self.cat_limit,
            candidate_conditioning_features,
            min_points_per_subregion,
            nof_candidate_splits_for_numerical,
            max_split_levels,
            heter_pcg_drop_thres,
            heter_small_enough,
            split_categorical_features,
        )

        self.regions["feature_{}".format(feature)] = regions
        splits = regions.search_all_splits()

        self.splits_per_feature_full_depth["feature_{}".format(feature)] = splits
        self.splits_per_feature_full_depth_found[feature] = True

        important_splits = regions.choose_important_splits()
        self.splts_per_feature_only_important[
            "feature_{}".format(feature)
        ] = important_splits
        self.splits_per_feature_only_import_found[feature] = True

    def fit(
        self,
        features: typing.Union[int, str, list],
        heter_pcg_drop_thres: float = 0.1,
        heter_small_enough: float = 0.1,
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion: int = 10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
        split_categorical_features: bool = False,
    ):
        """
        Find the Regional PDP for a list of features.

        Args:
            features: list of features to fit
            heter_pcg_drop_thres: heterogeneity drop threshold for a split to be considered important
            heter_small_enough: heterogeneity threshold for a region to be considered homogeneous (splitting stops)
            max_split_levels: maximum number of splits to perform (depth of the tree)
            nof_candidate_splits_for_numerical: number of candidate splits to consider for numerical features
            min_points_per_subregion: minimum allowed number of points in a subregion (otherwise the split is not considered as valid)
            candidate_conditioning_features: list of features to consider as conditioning features for the candidate splits
            split_categorical_features
        """

        assert min_points_per_subregion >= 2, "min_points_per_subregion must be >= 2"
        features = helpers.prep_features(features, self.dim)
        for feat in tqdm(features):
            self._fit_feature(
                feat,
                heter_pcg_drop_thres,
                heter_small_enough,
                max_split_levels,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
                split_categorical_features,
            )
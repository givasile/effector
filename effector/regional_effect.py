import numpy as np
import effector.binning_methods as binning_methods
import effector.helpers as helpers
import effector.utils as utils
from effector.regions import Regions
from effector.rhale import RHALE
import typing


BIG_M = helpers.BIG_M


class RegionalRHALE:
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        model_jac: typing.Union[None, callable],
        axis_limits: typing.Union[None, np.ndarray] = None,
        feature_types: typing.Union[list, None] = None,
        cat_limit: typing.Union[int, None] = 10,
        feature_names: typing.Union[list, None] = None,
    ):
        self.dim = data.shape[1]
        self.data = data
        self.model = model
        self.model_jac = model_jac
        self.axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        self.feature_types = (
            utils.get_feature_types(data, cat_limit)
            if feature_types is None
            else feature_types
        )

        self.cat_limit = cat_limit
        self.instance_effects = self.model_jac(self.data) if self.model_jac else None
        self.feature_names = feature_names

        # init splits
        self.regions = {}
        self.splits = {}
        self.important_splits = {}
        self.splits_found: np.ndarray = np.ones([self.dim]) < 0
        self.important_splits_selected: np.ndarray = np.ones([self.dim]) < 0
        self.all_regions_found = False

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
            self.data,
            self.model,
            self.model_jac,
            heter,
            self.instance_effects,
            self.feature_types,
            self.cat_limit,
            candidate_conditioning_features,
            min_points_per_subregion,
            nof_candidate_splits_for_numerical,
            max_split_levels,
            heter_pcg_drop_thres,
        )

        self.regions["feature_{}".format(feature)] = regions
        splits = regions.search_all_splits()

        self.splits["feature_{}".format(feature)] = splits
        self.splits_found[feature] = True

        important_splits = regions.choose_important_splits(heter_thres=0.1)
        self.important_splits["feature_{}".format(feature)] = important_splits
        self.important_splits_selected[feature] = True

    def fit(
        self,
        features,
        heter_pcg_drop_thres=0.1,
        binning_method: typing.Union[
            str,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_split_levels: int = 2,
        nof_candidate_splits_for_numerical: int = 20,
        min_points_per_subregion=10,
        candidate_conditioning_features: typing.Union["str", list] = "all",
    ):

        features = helpers.prep_features(features, self.dim)
        for feat in features:
            self._fit_feature(
                feat,
                heter_pcg_drop_thres,
                binning_method,
                max_split_levels,
                nof_candidate_splits_for_numerical,
                min_points_per_subregion,
                candidate_conditioning_features,
            )

    def print_splits(
        self, features, only_important=True, scale_x: typing.Union[None, list] = None
    ):
        features = helpers.prep_features(features, self.dim)
        for feature in features:
            if not self.splits_found[feature]:
                self._fit_feature(feature)

            feature_name = (
                self.feature_names[feature] if self.feature_names else feature
            )
            if only_important:
                print("Important splits for feature {}".format(feature_name))
                splits = self.important_splits["feature_{}".format(feature)]
            else:
                print("All splits for feature {}".format(feature_name))
                splits = self.splits["feature_{}".format(feature)]

            for i, split in enumerate(splits):
                type_of_split_feature = self.feature_types[split["feature"]]
                foc_name = (
                    split["feature"]
                    if self.feature_names is None
                    else self.feature_names[split["feature"]]
                )
                print("- On feature {} ({})".format(foc_name, type_of_split_feature))

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

    def plot_first_level(
        self,
        feature: int = 0,
        confidence_interval: typing.Union[bool, str] = False,
        centering: typing.Union[bool, str] = False,
        scale_x: typing.Union[None, list] = None,
        scale_y: typing.Union[None, dict] = None,
    ):

        if not self.splits_found[feature]:
            self._fit_feature(feature)

        regions = self.regions["feature_{}".format(feature)]
        splits = self.important_splits["feature_{}".format(feature)]

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
        foc_name = foc if self.feature_names is None else self.feature_names[foc]
        foi_name = feature if self.feature_names is None else self.feature_names[feature]
        position = position if scale_x is None else position * scale_x[foc]["std"] + scale_x[foc]["mean"]

        title_1 = "Regional RHALE for {} (split at {} == {:.2f})".format(
            foi_name, foc_name, position
        )
        title_2 = "Regional RHALE for {} (split at {} != {:.2f})".format(
            foi_name, foc_name, position
        )
        rhale_1.plot(feature, confidence_interval, centering, scale_x[feature], scale_y, title_1)
        rhale_2.plot(feature, confidence_interval, centering, scale_x[feature], scale_y, title_2)

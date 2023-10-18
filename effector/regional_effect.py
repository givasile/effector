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
    ):
        self.dim = data.shape[1]
        self.data = data
        self.model = model
        self.model_jac = model_jac
        self.axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )

        self.regions = Regions(
            self.data, self.model, self.model_jac, feature_types, cat_limit
        )

    # def compile(self):
    #     if self.instance_effects is None:
    #         self.instance_effects = self.model_jac(self.data)
    #     else:
    #         pass

    def _create_heterogeneity_function(self, binning_method, min_points=10):
        binning_method = helpers.prep_binning_method(binning_method)

        def heter(foi, data, instance_effects=None) -> float:
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
        binning_method: typing.Union[
            str,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_levels: int = 2,
        min_points: int = 10,
        other_features: typing.Union["str", list] = "all",
        categorical_limit: int = 15,
    ) -> typing.Dict:
        """
        Find the Regional RHALE for a single feature.
        """
        heter = self._create_heterogeneity_function(binning_method, min_points=min_points)

        self.regions.find_splits_single_feature(
            feature, max_levels, nof_candidate_splits=10, method=heter, min_points=min_points
        )
        self.regions.choose_important_splits(features=feature, heter_thres=0.1, pcg=0.1)

    def fit(
        self,
        features,
        binning_method: typing.Union[
            None,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_levels: int = 2,
        other_features: typing.Union["str", list] = "all",
        categorical_limit: int = 15,
        min_points=10
    ):

        features = helpers.prep_features(features, self.dim)
        for feat in features:
            self._fit_feature(feat, binning_method, max_levels, min_points, other_features, categorical_limit)

        # heter = self._create_heterogeneity_function(binning_method, min_points=min_points)
        # self.regions.find_splits(
        #     features=features,
        #     nof_levels=max_levels, nof_candidate_splits=10, method=heter,
        #     min_points=min_points
        # )
        # self.regions.choose_important_splits(
        #     features=features, heter_thres=0.1, pcg=0.1
        # )

    def print_splits(self, features, only_important=True):
        features = helpers.prep_features(features, self.dim)
        for feature in features:
            self.regions.print_split(feature, only_important)

    def plot_first_level(self, feature):
        self.regions.plot_first_level(feature)

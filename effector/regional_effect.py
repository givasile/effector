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

        self.regions = Regions(self.data, self.model, self.model_jac, feature_types, cat_limit)


    # def compile(self):
    #     if self.instance_effects is None:
    #         self.instance_effects = self.model_jac(self.data)
    #     else:
    #         pass

    def _fit_feature(
        self,
        feature: int,
        binning_method: typing.Union[
            None,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_levels: int = 2,
        other_features: typing.Union["str", list] = "all",
        categorical_limit: int = 15,
    ) -> typing.Dict:
        """
        Find the Regional RHALE for a single feature.
        """

        self.regions.find_splits_single_feature(feature, max_levels, nof_candidate_splits=10, method="rhale")


        # TODO: check how to handle categorical features, maybe skip with a warning

        # if self.instance_effects is None:
        #     self.compile()
        #
        # foi, foc = feature, [i for i in range(self.dim) if i != feature]
        #
        # # define heterogeneity function
        # def rhale_heter(data, instance_effects):
        #     rhale = RHALE(data, self.model, None, None, instance_effects)
        #     try:
        #         rhale.fit(features=foi, binning_method=binning_method)
        #     except:
        #         return BIG_M
        #
        #     # heterogeneity is the accumulated std at the end of the curve
        #     axis_limits = helpers.axis_limits_from_data(data)
        #     stop = np.array([axis_limits[:, foi][1]])
        #     _, z, _ = rhale.eval(feature=foi, xs=stop, uncertainty=True)
        #     return z.item()
        #
        # assert max_levels <= len(
        #     foc
        # ), "nof_levels must be smaller than the number of features of conditioning"
        #
        # # initial heterogeneity
        # heter_init = rhale_heter(self.data, self.instance_effects)
        #
        # # find optimal split for each level
        # x_list = [self.data]
        # x_jac_list = [self.instance_effects]
        # splits = [
        #     {
        #         "heterogeneity": [heter_init],
        #         "weighted_heter": heter_init,
        #         "nof_instances": [len(self.data)],
        #         "split_i": -1,
        #         "split_j": -1,
        #         "foc": foc,
        #     }
        # ]
        #
        # return rhale_heter(self.data, self.instance_effects)

    def fit(self, features,
            binning_method: typing.Union[
            None,
            binning_methods.Fixed,
            binning_methods.DynamicProgramming,
            binning_methods.Greedy,
        ] = "greedy",
        max_levels: int = 2,
        other_features: typing.Union["str", list] = "all",
        categorical_limit: int = 15):

        self.regions.find_splits(nof_levels=max_levels, nof_candidate_splits=10, method="rhale")
        self.regions.choose_important_splits(heter_thres=.1, pcg=0.1)

    def print_splits(self, features, only_important=True):
        features = helpers.prep_features(features, self.dim)
        for feature in features:
            self.regions.print_split(feature, only_important)

    def plot(self, feature, *args):
        pass

import typing
import pythia.utils as utils
import pythia.visualization as vis
import numpy as np
from pythia import FeatureEffectBase
from pythia import helpers
from pythia import bin_estimation as be

empty_symbol = 1e10


class ALE(FeatureEffectBase):
    def __init__(
        self,
        data: np.ndarray,
        model: callable,
        axis_limits: typing.Union[None, np.ndarray] = None,
    ):
        """
        Initializes ALE.

        Parameters
        ----------
        data: [N, D] np.array, X matrix
        model: Callable [N, D] -> [N,], prediction function
        axis_limits: [2, D] np.ndarray or None, if None they will be auto computed from the data
        """
        # assertions
        assert data.ndim == 2

        # setters
        self.model = model
        self.data = data
        axis_limits = (
            helpers.axis_limits_from_data(data) if axis_limits is None else axis_limits
        )
        super(ALE, self).__init__(axis_limits)

        # init as None, it will get gradients after compile
        self.data_effect = None

    def compile(self):
        """Prepare everything for fitting, i.e., compute the gradients on data points.
        TODO add numerical approximation
        """
        self.data_effect = np.ones_like(self.data) * empty_symbol

    def _fit_feature(self, feat: int, params: typing.Dict = None) -> typing.Dict:
        """Fit a specific feature, using ALE.

        Parameters
        ----------
        feat: index of the feature
        params: Dict, with fitting-specific parameters
            - "nof_bins": int (default 100), how many bins to create

        Returns
        -------

        """
        params = helpers.prep_ale_fit_params(params)

        if self.data_effect is None:
            self.compile()

        # drop points outside of limits
        ind = np.logical_and(
            self.data[:, feat] >= self.axis_limits[0, feat],
            self.data[:, feat] <= self.axis_limits[1, feat],
        )
        data = self.data[ind, :]


        # Compute data effect
        limits, dx = np.linspace(
            self.axis_limits[0, feat],
            self.axis_limits[1, feat],
            num=params["nof_bins"] + 1, endpoint=True, retstep=True
        )

        data_effect = utils.compute_local_effects_at_bin_limits(
            self.data, self.model, limits, feat
        )
        self.data_effect[:, feat] = data_effect
        data_effect = self.data_effect

        # bin estimation
        bin_est = be.Fixed(
                data, data_effect, feature=feat, axis_limits=self.axis_limits
            )
        bin_est.find(nof_bins=params["nof_bins"], min_points=None)

        dale_params = utils.compute_ale_params_from_data(
            data[:, feat], data_effect[:, feat], bin_est.limits
        )

        dale_params["alg_params"] = params
        return dale_params

    def _eval_unnorm(self, x: np.ndarray, s: int, uncertainty: bool = False):
        params = self.feature_effect["feature_" + str(s)]
        y = utils.compute_accumulated_effect(
            x, limits=params["limits"], bin_effect=params["bin_effect"], dx=params["dx"]
        )
        if uncertainty:
            std = utils.compute_accumulated_effect(
                x,
                limits=params["limits"],
                bin_effect=np.sqrt(params["bin_variance"]),
                dx=params["dx"],
            )
            std_err = utils.compute_accumulated_effect(
                x,
                limits=params["limits"],
                bin_effect=np.sqrt(params["bin_estimator_variance"]),
                dx=params["dx"],
            )

            return y, std, std_err
        else:
            return y

    def plot(
        self,
        s: int = 0,
        error: typing.Union[None, str] = "std",
        scale_x=None,
        scale_y=None,
        savefig=False,
    ):
        """

        Parameters
        ----------
        s
        error:
        scale_x:
        scale_y:
        savefig:
        """
        vis.ale_plot(
            self.feature_effect["feature_" + str(s)],
            self.eval,
            s,
            error=error,
            scale_x=scale_x,
            scale_y=scale_y,
            savefig=savefig,
        )


# def compute_ale_parameters(
#     data: np.ndarray, model: np.ndarray, feature: int, alg_params: typing.Dict
# ) -> typing.Dict:
#     """Compute the ALE parameters for the s-th feature
#
#     Performs all actions to compute the parameters that are required for
#     the s-th feature ALE effect
#
#     Parameters
#     ----------
#     data
#     model
#     feature
#     k
#
#     Returns
#     -------
#
#     """
#     K = alg_params["nof_bins"] if "nof_bins" in alg_params.keys() else 30
#     min_points_per_bin = (
#         alg_params["min_points_per_bin"]
#         if "min_points_per_bin" in alg_params.keys()
#         else 10
#     )
#     limits = utils.create_fix_size_bins(data[:, feature], K)
#
#     # compute local data effects, based on the bins
#     data_effect = utils.compute_local_effects_at_bin_limits(
#         data, model, limits, feature
#     )
#
#     # compute parameters
#     ale_params = utils.compute_ale_params_from_data(
#         data[:, feature], data_effect, limits, min_points_per_bin
#     )
#     ale_params["method"] = "fixed-size"
#     ale_params["min_points_per_bin"] = min_points_per_bin
#     return ale_params


# class ALE:
#     def __init__(self, data: np.ndarray, model: typing.Callable):
#         self.data = data
#         self.model = model
#
#         self.data_effect = None
#         self.feature_effect = None
#         self.ale_params = None
#
#     @staticmethod
#     def _ale_func(params):
#         """Returns the DALE function on for the s-th feature.
#
#         Parameters
#         ----------
#         points: ndarray
#           The training-set points, shape: (N,D)
#         f: ndarray
#           The feature effect contribution of the training-set points, shape: (N,)
#         s: int
#           Index of the feature of interest
#         k: int
#           Number of bins
#
#         Returns
#         -------
#         dale_function: Callable
#           The dale_function on the s-th feature
#         parameters: Dict
#           - limits: ndarray (K+1,) with the bin limits
#           - bin_effects: ndarray (K,) with the effect of each bin
#           - dx: float, bin length
#           - z: float, the normalizer
#
#         """
#
#         def ale_function(x):
#             y = utils.compute_accumulated_effect(
#                 x,
#                 limits=params["limits"],
#                 bin_effect=params["bin_effect"],
#                 dx=params["dx"],
#             )
#             y -= params["z"]
#             estimator_var = utils.compute_accumulated_effect(
#                 x,
#                 limits=params["limits"],
#                 bin_effect=params["bin_estimator_variance"],
#                 dx=params["dx"],
#                 square=True,
#             )
#
#             var = utils.compute_accumulated_effect(
#                 x,
#                 limits=params["limits"],
#                 bin_effect=params["bin_variance"],
#                 dx=params["dx"],
#                 square=True,
#             )
#             return y, estimator_var, var
#
#         return ale_function
#
#     def compile(self):
#         pass
#
#     def fit(self, features: typing.Union[str, list] = "all", alg_params={}):
#         assert features == "all" or type(features) == list
#         if features == "all":
#             features = [i for i in range(self.data.shape[1])]
#
#         # fit and store dale parameters
#         funcs = {}
#         ale_params = {}
#         for s in features:
#             ale_params["feature_" + str(s)] = compute_ale_parameters(
#                 self.data, self.model, s, alg_params
#             )
#             funcs["feature_" + str(s)] = self._ale_func(ale_params["feature_" + str(s)])
#
#         # TODO change it to append, instead of overwriting
#         self.feature_effect = funcs
#         self.ale_params = ale_params
#
#     def eval(self, x: np.ndarray, s: int):
#         return self.feature_effect["feature_" + str(s)](x)
#
#     def plot(
#         self,
#         s: int = 0,
#         error="standard error",
#         block=False,
#         gt=None,
#         gt_bins=None,
#         savefig=None,
#     ):
#         title = "ALE: Effect of feature %d" % (s + 1)
#         vis.ale_plot(
#             self.ale_params["feature_" + str(s)],
#             self.eval,
#             s,
#             error=error,
#             min_points_per_bin=self.ale_params["feature_" + str(s)][
#                 "min_points_per_bin"
#             ],
#             title=title,
#             block=block,
#             gt=gt,
#             gt_bins=gt_bins,
#             savefig=savefig,
#         )
#
#     def plot_local_effects(self, s: int = 0, K: int = 10, limits=True, block=False):
#         data = self.data
#         model = self.model
#
#         # compute local effects
#         bin_limits = utils.create_fix_size_bins(data[:, s], K)
#
#         # compute local data effects, based on the bins
#         data_effect = utils.compute_local_effects_at_bin_limits(
#             data, model, bin_limits, s
#         )
#
#         # plot
#         xs = self.data[:, s]
#         if limits:
#             limits = bin_limits
#         vis.plot_local_effects(s, xs, data_effect, limits, block)
#
#
#
#

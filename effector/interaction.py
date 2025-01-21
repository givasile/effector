# import numpy as np
# import logging
# import typing
# from effector import helpers
# import effector
# from effector import DerPDP, PDP
# from effector.global_effect_pdp import (
#     pdp_1d_vectorized,
#     pdp_1d_non_vectorized,
#     pdp_nd_non_vectorized,
#     pdp_nd_vectorized,
# )
# import tqdm
#
#
# class HIndex:
#     """
#     H-Index interaction measure, as described in Friedman and Popescu (2008), https://arxiv.org/abs/0811.1679
#     """
#
#     empty_symbol = 1e10
#
#     def __init__(self, data, model, nof_instances=1000):
#         """
#
#         Args
#         ---
#             data (np.ndarray): The dataset, shape (nof_instances, nof_features)
#             model (function): The model to explain, takes as input data and returns predictions
#             nof_instances (int): The number of instances to use for the explanation
#         """
#         # setters
#         self.nof_instances, self.indices = helpers.prep_nof_instances(
#             nof_instances, data.shape[0]
#         )
#         self.data = data[self.indices]
#         self.model = model
#         self.nof_features = self.data.shape[1]
#
#         # init
#         self.interaction_matrix = (
#             np.ones((self.nof_features, self.nof_features)) * self.empty_symbol
#         )
#         self.one_vs_all_matrix = np.ones((self.nof_features)) * self.empty_symbol
#
#         # flags
#         self.fitted_interaction_matrix = False
#         self.fitted_one_vs_all_matrix = False
#
#     def fit(self, pairwise_matrix=True, one_vs_all=True):
#         if pairwise_matrix:
#             logging.info("\nH-Index interaction matrix: Start fitting")
#             for i in range(self.nof_features):
#                 print("Feature: ", i)
#                 self.interaction_matrix[i, i] = 0
#                 for j in range(i + 1, self.nof_features):
#                     self.eval_pairwise(i, j)
#             self.fitted_interaction_matrix = True
#             logging.info("H-Index interaction matrix: Fitting done")
#
#         if one_vs_all:
#             logging.info("\nH-Index one-vs-all matrix: Start fitting")
#             for i in range(self.nof_features):
#                 print("Feature: ", i)
#                 self.eval_one_vs_all(i)
#             self.fitted_one_vs_all_matrix = True
#             logging.info("H-Index one-vs-all matrix: Fitting done")
#
#     def plot(self, interaction_matrix=True, one_vs_all=True):
#         # if not fitted, fit
#         if not self.fitted_interaction_matrix and interaction_matrix:
#             self.fit(pairwise_matrix=True, one_vs_all=False)
#         if not self.fitted_one_vs_all_matrix and one_vs_all:
#             self.fit(pairwise_matrix=False, one_vs_all=True)
#
#         # plot
#         if interaction_matrix:
#             self._plot_interaction_matrix()
#         if one_vs_all:
#             self._plot_one_vs_all_matrix()
#
#     def eval_pairwise(self, feat_1, feat_2):
#         """Evaluate the interaction between feature 1 and feature 2
#
#         Args
#         ---
#             feat_1 (int): The index of feature 1
#             feat_2 (int): The index of feature 2
#
#         Returns
#         ---
#             float: The interaction between feature 1 and feature 2
#         """
#         if self.interaction_matrix[feat_1, feat_2] != self.empty_symbol:
#             return self.interaction_matrix[feat_1, feat_2]
#
#         x1 = self.data[:, feat_1]
#         x2 = self.data[:, feat_2]
#         pdp_1 = self._pdp_1d(feat_1, x1)
#         pdp_2 = self._pdp_1d(feat_2, x2)
#         pdp_12 = self._pdp_2d(feat_1, feat_2, x1, x2)
#
#         nom = np.mean(np.square(pdp_12 - pdp_1 - pdp_2))
#         denom = np.mean(np.square(pdp_12))
#         interaction = nom / denom
#
#         self.interaction_matrix[feat_1, feat_2] = interaction
#         self.interaction_matrix[feat_2, feat_1] = interaction
#         return interaction
#
#     def eval_one_vs_all(self, feat):
#         if self.one_vs_all_matrix[feat] != self.empty_symbol:
#             return self.one_vs_all_matrix[feat]
#
#         pdp_1 = self._pdp_1d(feat, self.data[:, feat])
#         pdp_minus_1 = self._pdp_minus_1d(
#             feat, self.data[:, [i for i in range(self.nof_features) if i != feat]]
#         )
#
#         yy = self.model(self.data)
#         nom = np.mean(np.square(yy - pdp_minus_1 - pdp_1))
#         denom = np.mean(np.square(yy))
#         interaction = nom / denom
#
#         self.one_vs_all_matrix[feat] = interaction
#         return interaction
#
#     def _pdp_1d(self, feature, x):
#         # eval normalized pdp for feature at x
#         yy = pdp_1d_vectorized(
#             self.model,
#             self.data,
#             x,
#             feature,
#             uncertainty=False,
#             model_returns_jac=False,
#         )
#         c = np.mean(yy)
#         return yy - c
#
#     def _pdp_2d(self, feature1, feature2, x1, x2):
#         features = [feature1, feature2]
#         x = np.stack([x1, x2], axis=-1)
#         yy = pdp_nd_vectorized(
#             self.model,
#             self.data,
#             x,
#             features,
#             uncertainty=False,
#             model_returns_jac=False,
#         )
#         c = np.mean(yy)
#         return yy - c
#
#     def _pdp_minus_1d(self, feature, x):
#         # features are all other features
#         features = [i for i in range(self.nof_features) if i != feature]
#         assert len(features) == x.shape[1]
#
#         yy = pdp_nd_vectorized(
#             self.model,
#             self.data,
#             x,
#             features,
#             uncertainty=False,
#             model_returns_jac=False,
#         )
#         c = np.mean(yy)
#         return yy - c
#
#     def _plot_interaction_matrix(self):
#         import matplotlib.pyplot as plt
#
#         plt.figure(figsize=(10, 10))
#         plt.title("Interaction Matrix")
#         # assign the mean to the diagonal values w
#         mu = np.mean(self.interaction_matrix)
#         for i in range(self.nof_features):
#             self.interaction_matrix[i, i] = mu
#
#         # plot as a heatmap
#         plt.imshow(self.interaction_matrix)
#         plt.colorbar()
#         # show grid values
#         for i in range(self.nof_features):
#             for j in range(self.nof_features):
#                 plt.text(
#                     j,
#                     i,
#                     round(self.interaction_matrix[i, j], 2),
#                     ha="center",
#                     va="center",
#                     color="w",
#                 )
#
#         plt.xticks(
#             np.arange(self.nof_features),
#             labels=["feature {}".format(i + 1) for i in range(self.nof_features)],
#         )
#         plt.yticks(
#             np.arange(self.nof_features),
#             labels=["feature {}".format(i + 1) for i in range(self.nof_features)],
#         )
#         plt.show()
#
#     def _plot_one_vs_all_matrix(self):
#         import matplotlib.pyplot as plt
#
#         # plot as horizontal bar chart
#         plt.figure(figsize=(10, 10))
#         plt.title("H-index for all features")
#         plt.barh(np.arange(self.nof_features), self.one_vs_all_matrix)
#         plt.yticks(
#             np.arange(self.nof_features),
#             labels=["feature {}".format(i + 1) for i in range(self.nof_features)],
#         )
#         plt.xlabel("H-index")
#         plt.show()
#
#
# class REPID:
#     """REPID definition of interaction between feature x_j and all the others, i.e. the variance of the dPDP plot"""
#
#     empty_symbol = 1e10
#
#     def __init__(self, data, model, model_jac, nof_instances):
#         self.nof_instances, self.indices = helpers.prep_nof_instances(
#             nof_instances, data.shape[0]
#         )
#         self.data = data[self.indices]
#         self.dim = self.data.shape[1]
#
#         self.model = model
#         self.model_jac = model_jac
#         self.nof_instances = nof_instances
#         self.nof_features = data.shape[1]
#
#         self.interaction_matrix = (
#             np.ones((self.nof_features, self.nof_features)) * self.empty_symbol
#         )
#         self.one_vs_all_matrix = np.ones(self.nof_features) * self.empty_symbol
#
#         self.fitted_interaction_matrix = False
#         self.fitted_one_vs_all_matrix = False
#
#     def fit(
#         self,
#         features: typing.Union[int, str, list] = "all",
#     ):
#         features = helpers.prep_features(features, self.dim)
#         for i, feat in enumerate(features):
#             print("Feature: ", feat)
#             self.eval_one_vs_all(feat)
#         self.fitted_one_vs_all_matrix = True
#
#     def eval_one_vs_all(self, feat):
#         if self.one_vs_all_matrix[feat] != self.empty_symbol:
#             return self.one_vs_all_matrix[feat]
#
#         axis_limits = effector.helpers.axis_limits_from_data(self.data)
#         pdp = PDP(self.data, self.model, axis_limits)
#         pdp.fit(feat)
#
#         # find the interaction index
#         start = axis_limits[:, feat][0]
#         stop = axis_limits[:, feat][1]
#         x = np.linspace(start, stop, 21)
#         x = 0.5 * (x[:-1] + x[1:])
#         mu, std, stderr = pdp.eval(
#             feature=feat, xs=x, uncertainty=True, centering="zero_start"
#         )  # "zero_integral")
#         interaction = np.mean(std)
#
#         # store the interaction index
#         self.one_vs_all_matrix[feat] = interaction
#         return interaction
#
#     def plot(self):
#         if self.fitted_one_vs_all_matrix is False:
#             self.fit()
#
#         import matplotlib.pyplot as plt
#
#         # plot as horizontal bar chart
#         plt.figure(figsize=(10, 10))
#         plt.title("REPID for all features")
#         plt.barh(np.arange(self.nof_features), self.one_vs_all_matrix)
#         plt.yticks(
#             np.arange(self.nof_features),
#             labels=["feature {}".format(i + 1) for i in range(self.nof_features)],
#         )
#         plt.xlabel("REPID index")
#         plt.show()

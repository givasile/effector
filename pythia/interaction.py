import numpy as np
from pythia import helpers
from pythia.pdp import pdp_1d_vectorized, pdp_1d_non_vectorized, pdp_nd_non_vectorized, pdp_nd_vectorized
import tqdm

class HIndex:
    empty_symbol = 1e10

    def __init__(self, data, model, nof_instances=1000):

        self.nof_instances, self.indices = helpers.prep_nof_instances(nof_instances, data.shape[0])
        self.data = data[self.indices]

        self.model = model
        self.nof_features = self.data.shape[1]

        self.interaction_matrix = np.ones((self.nof_features, self.nof_features)) * self.empty_symbol
        self.one_vs_all_matrix = np.ones((self.nof_features)) * self.empty_symbol

        self.fitted_interaction_matrix = False
        self.fitted_one_vs_all_matrix = False

    def fit(self, interaction_matrix=True, one_vs_all=True):
        print("Fit HIndex - interacation_matrix")
        if interaction_matrix:
            for i in range(self.nof_features):
                print("Feature: ", i)
                self.interaction_matrix[i, i] = 0
                for j in range(i + 1, self.nof_features):
                    self.pairwise(i, j)
            self.fitted_interaction_matrix = True

        print("Fit HIndex - one_vs_all matrix")
        if one_vs_all:
            for i in range(self.nof_features):
                print("Feature: ", i)
                self.one_vs_all(i)
            self.fitted_one_vs_all_matrix = True

    def plot(self, interaction_matrix=True, one_vs_all=True):
        if not self.fitted_interaction_matrix and interaction_matrix:
            self.fit(interaction_matrix=True, one_vs_all=False)
        if not self.fitted_one_vs_all_matrix and one_vs_all:
            self.fit(interaction_matrix=False, one_vs_all=True)

        if interaction_matrix:
            self._plot_interaction_matrix()
        if one_vs_all:
            self._plot_one_vs_all_matrix()


    def pairwise(self, feat_1, feat_2):
        if self.interaction_matrix[feat_1, feat_2] != self.empty_symbol:
            return self.interaction_matrix[feat_1, feat_2]

        x1 = self.data[:, feat_1]
        x2 = self.data[:, feat_2]
        pdp_1 = self.pdp_1d(feat_1, x1)
        pdp_2 = self.pdp_1d(feat_2, x2)
        pdp_12 = self.pdp_2d(feat_1, feat_2, x1, x2)

        nom = np.mean(np.square(pdp_12 - pdp_1 - pdp_2))
        denom = np.mean(np.square(pdp_12))
        interaction = nom / denom

        self.interaction_matrix[feat_1, feat_2] = interaction
        self.interaction_matrix[feat_2, feat_1] = interaction
        return interaction

    def one_vs_all(self, feat):
        if self.one_vs_all_matrix[feat] != self.empty_symbol:
            return self.one_vs_all_matrix[feat]

        pdp_1 = self.pdp_1d(feat, self.data[:, feat])
        pdp_minus_1 = self.pdp_minus_1d(feat, self.data[:, [i for i in range(self.nof_features) if i != feat]])

        yy = self.model(self.data)
        nom = np.mean(np.square(yy - pdp_minus_1 - pdp_1))
        denom = np.mean(np.square(yy))
        interaction = nom / denom

        self.one_vs_all_matrix[feat] = interaction
        return interaction

    def pdp_1d(self, feature, x):
        yy = pdp_1d_vectorized(self.model, self.data, x, feature, uncertainty=False, is_jac=False)
        c = np.mean(yy)
        return yy - c

    def pdp_2d(self, feature1, feature2, x1, x2):
        features = [feature1, feature2]
        x = np.stack([x1, x2], axis=-1)
        yy = pdp_nd_vectorized(self.model, self.data, x, features, uncertainty=False, is_jac=False)
        c = np.mean(yy)
        return yy - c

    def pdp_minus_1d(self, feature, x):
        # features are all other features
        features = [i for i in range(self.nof_features) if i != feature]
        assert len(features) == x.shape[1]

        yy = pdp_nd_vectorized(self.model, self.data, x, features, uncertainty=False, is_jac=False)
        c = np.mean(yy)
        return yy - c

    def _plot_interaction_matrix(self):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.title("Interaction Matrix")
        # assign the mean to the diagonal values w
        mu = np.mean(self.interaction_matrix)
        for i in range(self.nof_features):
            self.interaction_matrix[i, i] = mu

        # plot as a heatmap
        plt.imshow(self.interaction_matrix)
        plt.colorbar()
        # show grid values
        for i in range(self.nof_features):
            for j in range(self.nof_features):
                plt.text(j, i, round(self.interaction_matrix[i, j], 2), ha="center", va="center", color="w")

        plt.xticks(np.arange(self.nof_features), labels=["feature {}".format(i+1) for i in range(self.nof_features)])
        plt.yticks(np.arange(self.nof_features), labels=["feature {}".format(i+1) for i in range(self.nof_features)])
        plt.show()

    def _plot_one_vs_all_matrix(self):
        import matplotlib.pyplot as plt
        # plot as horizontal bar chart
        plt.figure(figsize=(10, 10))
        plt.title("H-index for all features")
        plt.barh(np.arange(self.nof_features), self.one_vs_all_matrix)
        plt.yticks(np.arange(self.nof_features), labels=["feature {}".format(i+1) for i in range(self.nof_features)])
        plt.xlabel("H-index")
        plt.show()

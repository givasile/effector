import sys, os
import timeit
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pythia
import pythia.regions as regions
import pythia.interaction as interaction
import matplotlib.pyplot as plt
# from nodegam.sklearn import NodeGAMClassifier, NodeGAMRegressor
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor

class RepidSimpleDist:
    """
    x1 ~ U(-1, 1)
    x2 ~ U(-1, 1)
    x3 ~ Bernoulli(0.5)
    """

    def __init__(self):
        self.D = 2
        self.axis_limits = np.array([[-1, 1], [-1, 1], [0, 1]]).T

    def generate(self, N):
        x1 = np.concatenate((np.array([-1]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x2 = np.concatenate((np.array([-1]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x3 = np.random.choice([0, 1], int(N), p=[0.5, 0.5])

        x = np.stack((x1, x2, x3), axis=-1)
        return x


class RepidSimpleModel:
    def __init__(self, a1=0.2, a2=-8, a3=8, a4=8):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4

    def predict(self, x):
        y = self.a1*x[:, 0] + self.a2*x[:, 1] + x[:, 2]

        cond = x[:, 0] > 0
        y[cond] += self.a3*x[cond, 1]

        cond = x[:, 2] == 0
        y[cond] += self.a4*x[cond, 1]

        eps = np.random.normal(loc=0, scale=0.1, size=y.shape[0])
        y += eps
        return y

    def jacobian(self, x):
        y = np.stack([self.a1*np.ones(x.shape[0]), self.a2*np.ones(x.shape[0]), np.ones(x.shape[0])], axis=-1)

        cond = x[:, 0] > 0
        y[cond, 1] += self.a3

        cond = x[:, 2] == 0
        y[cond, 1] += self.a4
        return y


np.random.seed(21)
dist = RepidSimpleDist()
model = RepidSimpleModel()

# generate data
X = dist.generate(N=1000)
Y = model.predict(X)


# check interactions
h_index = interaction.HIndex(data=X, model=model.predict, nof_instances=950)
# print(h_index.pairwise(0, 1))
# print(h_index.one_vs_all(0))

h_index.plot(interaction_matrix=True, one_vs_all=True)





# # find regions
# reg = pythia.regions.Regions(data=X, model=model.predict, model_jac=model.jacobian, cat_limit=25)
# reg.search_splits(nof_levels=2, nof_candidate_splits=20, criterion="rhale")
# opt_splits = reg.choose_important_splits(0.2)
#
# transf = pythia.regions.DataTransformer(splits=opt_splits)
# new_X = transf.transform(X)
#
# #
# seed = 21
# X_train, X_test, y_train, y_test = train_test_split(new_X, Y, test_size=0.20, random_state=seed)
# ebr = ExplainableBoostingRegressor()
# ebr.fit(X_train, y_train)
# print(ebr.score(X_test, y_test))

# for i in range(len(ebr.term_scores_)):
#     plt.plot(ebr.term_scores_[i])
#     plt.show()


# from interpret import show
# show(ebr.explain_global())

# # plot global effect
# feat = 1
# fe = pythia.PDP(data=X, model=model.predict, max_nof_instances=10000)
# fe.plot(feature=feat, uncertainty=True, centering=False, nof_points=100)
#
# fe = pythia.dPDP(data=X, model=model.predict, model_jac=model.jacobian, max_nof_instances=10000)
# fe.plot(feature=feat, uncertainty=True, nof_points=100)
#
# fe = pythia.RHALE(data=X, model=model.predict, model_jac=model.jacobian)
# fe.fit(features="all", binning_method="greedy", centering=True)
# fe.plot(feature=feat, uncertainty=True)
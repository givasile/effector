import sys, os
import timeit

import pandas as pd

sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import effector
import effector.partitioning as regions
import effector.interaction as interaction
import matplotlib.pyplot as plt
from nodegam.sklearn import NodeGAMClassifier, NodeGAMRegressor
import pandas as pd
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor


# hack to reload modules
import importlib
pythia = importlib.reload(effector)


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
        x4 = np.random.choice([0, 1], int(N), p=[0.5, 0.5])

        x = np.stack((x1, x2, x3, x4), axis=-1)
        return x


class RepidSimpleModel:
    def __init__(self, a1=100):
        self.a1 = a1

    def predict(self, x):
        """f(x) = a1*x2 if x1 > 0 and x3 == 0 else 0"""
        y = np.zeros_like(x[:, 0])

        cond = np.logical_and(x[:, 0] > 0, x[:, 2] == 0)
        y[cond] += self.a1*x[cond, 1]

        eps = np.random.normal(loc=0, scale=0.1, size=y.shape[0])
        y += eps
        return y

    def jacobian(self, x):
        """dfdx = [0, a1, 0] if x1 > 0 and x3 == 0 else [0, 0, 0]"""
        y = np.zeros_like(x)

        cond = np.logical_and(x[:, 0] > 0, x[:, 2] == 0)
        y[cond, 1] += self.a1
        return y


def plot_effect_ebm(ebm_model, ii):
    explanation = ebm_model.explain_global()
    plt.figure(figsize=(10, 6))
    xx = explanation.data(ii)["names"][:-1]
    yy = explanation.data(ii)["scores"]
    plt.xlim(-1, 1)
    plt.ylim(-4, 4)
    plt.plot(xx, yy)
    plt.show()


np.random.seed(21)
feature_names = ["x1", "x2", "x3"]
dist = RepidSimpleDist()
model = RepidSimpleModel()

# generate data
X = dist.generate(N=1000)
Y = model.predict(X)


# Our method
reg = pythia.regions.Regions(data=X, model=model.predict, model_jac=model.jacobian, categorical_limit=25)
reg.find_splits(nof_levels=2, nof_candidate_splits=20, method="rhale")
opt_splits = reg.choose_important_splits(0.2)

transf = pythia.regions.DataTransformer(splits=opt_splits)
new_X = transf.transform(X)

# split data
seed = 21
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_X, Y, test_size=0.20, random_state=seed)

# GAM to the original data
gam = ExplainableBoostingRegressor(interactions=0)
gam.fit(X_train, y_train)
print("GAM:", gam.score(X_test, y_test))

# fit a GAM to the transformed data
gam_subspaces = ExplainableBoostingRegressor(interactions=0)
gam_subspaces.fit(X_train_new, y_train_new)
print("GAM with subspaces:", gam_subspaces.score(X_test_new, y_test_new))

# GAM with interactions
gam_interactions = ExplainableBoostingRegressor()
gam_interactions.fit(X_train, y_train)
print("GAM with interactions:", gam_interactions.score(X_test, y_test))

# # with NodeGAM
# X_train_df = pd.DataFrame(X_train, columns=["x_1", "x_2", "x_3", "x_4"])
# model = NodeGAMRegressor(in_features=X_train.shape[1])
# model.fit(X_train_df, y_train)
# model.visualize(X_train_df)
# plt.show()
#
# # NodeGAME with subspaces
# X_train_new_df = pd.DataFrame(X_train_new, columns=["x_1", "x_2", "x_3", "x_4", "x_5", "x_6", "x_7"])
# model.fit(X_train_new_df, y_train)
# from nodegam.vis_utils import vis_GAM_effects
#
# vis_GAM_effects({
#     'nodegam': model.get_GAM_df(X_train_new_df),
# })
#
# plt.show()

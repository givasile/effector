import sys, os
import timeit
sys.path.append(os.path.dirname(os.getcwd()))
import numpy as np
import pythia
import pythia.regions as regions
import pythia.interaction as interaction
import matplotlib.pyplot as plt
from nodegam.sklearn import NodeGAMClassifier, NodeGAMRegressor
from sklearn.model_selection import train_test_split
from interpret.glassbox import ExplainableBoostingRegressor

# hack to reload modules
import importlib
pythia = importlib.reload(pythia)


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
    def __init__(self, a1=8):
        self.a1 = a1

    def predict(self, x):
        y = np.zeros_like(x[:, 0])

        cond = np.logical_and(x[:, 0] > 0, x[:, 2] == 1)
        y[cond] += self.a1*x[cond, 1]

        eps = np.random.normal(loc=0, scale=0.1, size=y.shape[0])
        y += eps
        return y

    def jacobian(self, x):
        y = np.zeros_like(x)

        cond = np.logical_and(x[:, 0] > 0, x[:, 2] == 1)
        y[cond, 1] = self.a1
        return y

def get_effect_from_ebm(ebm_model, ii):
    explanation = ebm_model.explain_global()
    xx = explanation.data(ii)["names"][:-1]
    yy = explanation.data(ii)["scores"]
    return xx, yy

def plot_effect_ebm(ebm_model, ii):
    xx, yy = get_effect_from_ebm(ebm_model, ii)
    plt.figure(figsize=(10, 6))
    plt.xlim(-1, 1)
    plt.ylim(-8, 8)
    plt.plot(xx, yy)
    plt.show()

def ground_truth_x2(xx):
    return 2*xx

def ground_truth_x2_reg_1(xx):
    return 8*xx

def ground_truth_x2_reg_2(xx):
    return 0*xx

np.random.seed(21)
feature_names = ["x1", "x2", "x3"]
dist = RepidSimpleDist()
model = RepidSimpleModel()

# generate data
X = dist.generate(N=1000)
Y = model.predict(X)

# # check interactions
# h_index = interaction.HIndex(data=X, model=model.predict, nof_instances=950)
# # print(h_index.eval_pairwise(0, 1))
# print(h_index.eval_one_vs_all(0))
# h_index.plot(interaction_matrix=False, one_vs_all=True)
#
# # check interactions with REPID (dPDP based method)
# repid_index = interaction.REPID(data=X, model=model.predict, model_jac=model.jacobian, nof_instances=950)
# print(repid_index.eval_one_vs_all(0))
# repid_index.plot()

# find regions
reg = pythia.regions.Regions(data=X, model=model.predict, model_jac=model.jacobian, cat_limit=25)
reg.find_splits(nof_levels=2, nof_candidate_splits=20, method="rhale")
opt_splits = reg.choose_important_splits(0.2)

transf = pythia.regions.DataTransformer(splits=opt_splits)
new_X = transf.transform(X)

# split data
seed = 21
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=seed)
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(new_X, Y, test_size=0.20, random_state=seed)


# fit a GAM to the transformed data
gam_subspaces = ExplainableBoostingRegressor(interactions=0)
gam_subspaces.fit(X_train_new, y_train_new)
print(gam_subspaces.score(X_test_new, y_test_new))

# plot for subspace whe x1 > 0 and x3 == 0
plt.figure()
plt.title("RAM fitted to region x1 > 0 and x3 == 1")
xx, yy = get_effect_from_ebm(gam_subspaces, 4)
# randomly select 100 points from the region from X_train
cond = np.logical_and(X_train[:, 0] > 0, X_train[:, 2] == 1)
rmse_ebm_subreg_1 = np.sqrt(np.mean((yy - ground_truth_x2_reg_1(np.array(xx)))**2))
idx = np.random.choice(np.where(cond)[0], size=50, replace=False)
plt.plot(X_train[idx, 1], y_train[idx], "x", color="black", alpha=0.5, label="data")
plt.plot(xx, yy, label="RAM - EBM (RMSE = {:.2f})".format(rmse_ebm_subreg_1), color="blue", linewidth=2, alpha=0.5, linestyle="--")
plt.plot(xx, ground_truth_x2_reg_1(np.array(xx)), label="ground truth", color="red", linewidth=2, alpha=0.5, linestyle="--")
plt.xlim(-1.1, 1.1)
plt.ylim(-8.5, 8.5)
plt.legend()
plt.savefig("figures/regional_gam_subreg_1.pdf", bbox_inches="tight")
plt.show()

# plot for the rest of the input space
plt.figure()
plt.title("RAM fitted to the rest of the input space")
xx, yy = get_effect_from_ebm(gam_subspaces, 1)
# randomly select 100 points from the region from X_train
cond = np.logical_or(X_train[:, 0] <= 0, X_train[:, 2] != 1)
# rmse for these points
rmse_ebm_subreg_2 = np.sqrt(np.mean((y_train_new[cond] - gam_subspaces.predict(X_train_new[cond]))**2))
idx = np.random.choice(np.where(cond)[0], size=50, replace=False)
plt.plot(X_train[idx, 1], y_train[idx], "x", color="black", alpha=0.5, label="data")
plt.plot(xx, yy, label="RAM - EBM (RMSE = {:.2f})".format(rmse_ebm_subreg_2), color="blue", linewidth=2, alpha=0.5, linestyle="--")
plt.plot(xx, ground_truth_x2_reg_2(np.array(xx)), label="ground truth (RMSE = 0.00)", color="red", linewidth=2, alpha=0.5, linestyle="--")
plt.xlim(-1.1, 1.1)
plt.ylim(-8.5, 8.5)
plt.legend()
plt.savefig("figures/regional_gam_subreg_2.pdf", bbox_inches="tight")
plt.show()

# fit a GAM to the initial data
ebm_no_interaction = ExplainableBoostingRegressor(feature_names, interactions=0)
ebm_no_interaction.fit(X_train, y_train)
print(ebm_no_interaction.score(X_train, y_train))
rmse_ebm_no_interaction = np.mean(np.square(y_train - ebm_no_interaction.predict(X_train)))
xx, yy = get_effect_from_ebm(ebm_no_interaction, 1)
plt.figure()
plt.title("Globally-fitted GAM")
idx = np.random.choice(X_train.shape[0], 100, replace=False)
plt.plot(X_train[idx, 1], y_train[idx], "x", color="black", alpha=0.5, label="data")
plt.plot(xx, yy, label="GAM - EBM (RMSE={:.2f})".format(rmse_ebm_no_interaction), color="blue", linewidth=2, alpha=0.5, linestyle="--")
plt.plot(xx, ground_truth_x2(np.array(xx)), label="ground truth (RMSE: 3.00) ", linestyle="--", color="red")
plt.legend()
plt.xlim(-1.1, 1.1)
plt.ylim(-8.5, 8.5)
plt.xlabel("$x_2$")
plt.ylabel("y")
plt.savefig("figures/global_GAM.pdf", bbox_inches="tight")
plt.show()


# fit a GAM with interactions to the initial data
ebm_interactions = ExplainableBoostingRegressor(feature_names)
ebm_interactions.fit(X_train, y_train)
print(ebm_interactions.score(X_test, y_test))
xx, yy = get_effect_from_ebm(ebm_interactions, 1)


# fit a NodeGAM to the transformed data
# gam_subspaces = NodeGAMRegressor(interactions=0)

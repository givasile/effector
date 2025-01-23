import effector
import numpy as np
import timeit

np.random.seed(21)

model = effector.models.DoubleConditionalInteraction()

N = 100_000
D = 3
M = 1_000

X = np.random.uniform(-1, 1, (N, D))

#%%
# ALE
ale = effector.RegionalALE(
    data=X,
    model=model.predict,
    feature_names=["x1", "x2", "x3"],
    nof_instances=N,
    axis_limits=np.array([[-1, 1], [-1, 1], [-1, 1]]),
    target_name="y"
)

ale.fit(
    features=0,
    max_depth=2
)

tree = ale.tree_full["feature_0"]
tree.show_full_tree()

scale_x_list = [
    {"mean": 10, "std":1},
    {"mean": 100, "std":1},
    {"mean": 1000, "std":1}
]

tree.show_full_tree(None, scale_x_list)


ale.summary(features=0, only_important=True, scale_x_list=scale_x_list)

ale.plot(
    feature=0,
    node_idx=0,
    heterogeneity=True
)

ale.plot(
    feature=0,
    node_idx=1,
    heterogeneity=True
)


ale.plot(
    feature=0,
    node_idx=3,
    heterogeneity=True
)

ale.plot(
    feature=0,
    node_idx=6,
    heterogeneity=True,
    scale_x_list=scale_x_list
)




# binning_method = effector.binning_methods.Fixed(10)
# ale.fit(features="all", binning_method=binning_method)
# y, h = ale.eval(feature=0, xs=np.linspace(-1, 1, 1000), heterogeneity=True, centering=True)
#
# import matplotlib.pyplot as plt
# plt.plot(np.linspace(-1, 1, 1000), h)
# plt.show()

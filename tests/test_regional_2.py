import numpy as np
import effector

np.random.seed(21)


N = 1000
T = 1000

# create data, model
data = np.stack(
    [
        np.random.uniform(-1, 1, N + 1),
        np.random.uniform(-1, 1, N + 1),
        np.random.randint(0, 2, N + 1)
    ],
    axis=1)


def model(x):
    y = np.zeros_like(x[:,0])
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind] = 5*x[ind, 0]
    return y


def model_jac(x):
    y = np.zeros_like(x)
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind,0] = 5
    return y

x = np.linspace(-1, 1, T)

# ground truth
regional_effect = effector.RegionalSHAP(data, model, nof_instances=1000)
regional_effect.fit("all",
                 heter_pcg_drop_thres=0.1,
                 heter_small_enough=0.1,
                 max_depth=2,
                 nof_candidate_splits_for_numerical=20,
                 min_points_per_subregion=10,
                 candidate_conditioning_features="all",
                 split_categorical_features=False)

regional_effect.describe_subregions("all")
regional_effect.print_tree(0)
regional_effect.print_level_stats(0)
regional_effect.plot(0, 5, heterogeneity=True)

# scale_x_per_feature ={
#     "feature_0": {"mean": 1, "std": 1},
#     "feature_1": {"mean": 2, "std": 2},
#     "feature_2": {"mean": 3, "std": 3}
# }

# regional_pdp.plot(0, 6, heterogeneity=True)

# # ground truth
# regional_ale = effector.RegionalRHALEBase(data, model, model_jac, nof_instances=1000)
# regional_ale.fit("all",
#                  heter_pcg_drop_thres=0.1,
#                  heter_small_enough=0.,
#                  max_split_levels=2,
#                  nof_candidate_splits_for_numerical=10,
#                  min_points_per_subregion=10,
#                  candidate_conditioning_features="all",
#                  split_categorical_features=False)

# # check the splitting
# regional_ale.describe_subregions(2)
# regional_ale.print_level_stats(1)
# regional_ale.print_tree(0)

# # plot/eval
# xs = np.linspace(-1, 1, 1000)
# heterogeneity = False
# y = regional_ale.eval(0, 0, xs, heterogeneity=heterogeneity)
# regional_ale.plot(0, 5, heterogeneity=True)

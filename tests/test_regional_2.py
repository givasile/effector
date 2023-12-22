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
    y[ind] = x[ind, 0]
    return y

def model_jac(x):
    y = np.zeros_like(x)
    ind = np.logical_and(x[:, 1] > 0, x[:, 2] == 0)
    y[ind,0] = 1
    return y

x = np.linspace(-1, 1, T)

# ground truth
regional_pdp = effector.RegionalPDP(data, model, nof_instances=1000)
regional_pdp.fit("all",
                 heter_pcg_drop_thres=0.1,
                 heter_small_enough=0.1,
                 max_split_levels=2,
                 nof_candidate_splits_for_numerical=10,
                 min_points_per_subregion=10,
                 candidate_conditioning_features="all",
                 split_categorical_features=False)

regional_pdp.describe_subregions("all")
regional_pdp.plot_first_level(0)
regional_pdp.print_tree(0)

x0_tree = regional_pdp.splits_full_depth_tree["feature_1"]


# ground truth
regional_ale = effector.RegionalRHALEBase(data, model, model_jac, nof_instances=1000)
regional_ale.fit("all",
                 heter_pcg_drop_thres=0.1,
                 heter_small_enough=0.,
                 max_split_levels=2,
                 nof_candidate_splits_for_numerical=10,
                 min_points_per_subregion=10,
                 candidate_conditioning_features="all",
                 split_categorical_features=False)

regional_ale.describe_subregions(0)

regional_ale.plot_first_level(0)

regional_ale.print_tree(1, only_important=False)
regional_ale.print_level_stats(0)

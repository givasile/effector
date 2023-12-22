import numpy as np
import effector

np.random.seed(21)


N = 1000
T = 1000

# create data, model
data = np.stack([np.random.uniform(-1, 1, N + 1), np.random.randint(0, 2, N + 1)], axis=1)
y = np.zeros_like(data[:, 0])

model = lambda x: np.where(x[:, 1] == 0, x[:, 0], -x[:, 0])
model_jac = lambda x: np.stack([np.where(x[:, 1] == 0, 1, -1), np.zeros_like(x[:, 1])], axis=1)
x = np.linspace(-1, 1, T)

# ground truth
regional_pdp = effector.RegionalPDP(data, model, nof_instances=100)
regional_pdp.fit("all",
                 heter_pcg_drop_thres=0.1,
                 heter_small_enough=0.1,
                 max_split_levels=1,
                 nof_candidate_splits_for_numerical=10,
                 min_points_per_subregion=10,
                 candidate_conditioning_features="all",
                 split_categorical_features=False)

regional_pdp.describe_subregions("all")
# regional_pdp.plot_first_level(0)
regional_pdp.print_tree(0)

x0_tree = regional_pdp.splits_full_depth_tree["feature_0"]

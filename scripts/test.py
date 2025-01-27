import effector
import numpy as np
import timeit
import time


np.random.seed(21)


def predict(x):
    time.sleep(0.01)
    model = effector.models.DoubleConditionalInteraction()
    return model.predict(x)

def jacobian(x):
    model = effector.models.DoubleConditionalInteraction()
    return model.jacobian(x)


N = 1_000
D = 5
M = 1_000

X = np.random.uniform(-1, 1, (N, D))

# # Global PDP
# pdp = effector.PDP(
#     data=X,
#     model=predict,
#     feature_names=["x1", "x2", "x3"],
#     nof_instances="all",
#     target_name="y"
# )

# pdp.fit(
#     features="all",
#     centering=True
# )

# tic = timeit.default_timer()
# pdp.eval(
#     feature=0,
#     xs=np.linspace(-1, 1, 100),
#     centering=True,
#     heterogeneity=True,
# )
# toc = timeit.default_timer()
# print(f"Global PDP: {toc - tic:.6f} sec")

# PDP
axis_limits = (np.ones((D, 2)) * np.array([-1, 1])).T

reg_pdp = effector.RegionalPDP(
    data=X,
    model=predict,
    nof_instances="all",
    axis_limits=axis_limits
)

reg_pdp.fit(
    features="all",
    heter_pcg_drop_thres=.2,
    heter_small_enough=0.,
    max_depth=3,
    nof_candidate_splits_for_numerical=51,
    min_points_per_subregion=10,
    use_vectorized=True,
    centering=True,
)

reg_pdp.summary(features="all")

##
reg_dpdp = effector.RegionalDerPDP(
    data=X,
    model=predict,
    model_jac=jacobian,
    nof_instances="all",
    axis_limits=axis_limits
)

reg_dpdp.fit(
    features="all",
    heter_pcg_drop_thres=.2,
    heter_small_enough=0.,
    max_depth=3,
    nof_candidate_splits_for_numerical=51,
    min_points_per_subregion=10,
    use_vectorized=True,
)

reg_pdp.summary(features="all")


#%%
# ALE
reg_ale = effector.RegionalALE(
    data=X,
    model=predict,
    nof_instances="all",
    axis_limits=axis_limits,
    target_name="y"
)

reg_ale.fit(
    features="all",
    heter_pcg_drop_thres=.2,
    heter_small_enough=0.,
    max_depth=2,
    nof_candidate_splits_for_numerical=11,
    min_points_per_subregion=10,
)

reg_ale.summary(features="all")


#%%# RHALE
reg_rhale = effector.RegionalRHALE(
    data=X,
    model=predict,
    model_jac=jacobian,
    nof_instances="all",
    axis_limits=axis_limits,
    target_name="y"
)

reg_rhale.fit(
    features="all",
    heter_pcg_drop_thres=.1,
    heter_small_enough=0.,
    max_depth=2,
    nof_candidate_splits_for_numerical=11,
    min_points_per_subregion=10,
    binning_method="greedy"
)

reg_rhale.summary(features="all")

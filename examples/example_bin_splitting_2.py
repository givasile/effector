import examples.utils as utils
path = utils.add_parent_path()
import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import example_models.distributions as dist
import example_models.models as models

savefig = True
folder_name =  "example_bin_splitting_2"
np.random.seed(23)

# gen dist
gen_dist = dist.Correlated1(D=2, x1_min=0, x1_max=1, x2_sigma=.4)
axis_limits = gen_dist.axis_limits

# model
model = models.SquareWithInteraction(b0=0, b1=4, b2=1, b3=1)

# fit ground-truth
X_0 = gen_dist.generate(N=1000000)
dale_gt = fe.DALE(data=X_0,
               model=model.predict,
               model_jac=model.jacobian,
               axis_limits=axis_limits)

alg_params = {"bin_method" : "fixed", "nof_bins" : 1000, "min_points_per_bin": 5}
dale_gt.fit(features=0, alg_params=alg_params)


# fit approx with auto binning
X = gen_dist.generate(N=500)
dale = fe.DALE(data=X,
               model=model.predict,
               model_jac=model.jacobian,
               axis_limits=axis_limits)
alg_params = {"bin_method" : "dp", "max_nof_bins" : 40, "min_points_per_bin": 10}
dale.fit(features=0, alg_params=alg_params)
if savefig:
    path2dir = os.path.join(path, folder_name)
    savepath = os.path.join(path2dir, "fig_1.pdf") if savefig else None
    dale.plot(savefig=savepath)
else:
    dale.plot()


# fit approximation
K_list = list(range(2,11)) + list(range(15, 101, 5))
stats_fixed = utils.measure_fixed_error(dale_gt,
                                        gen_dist,
                                        model,
                                        axis_limits,
                                        K_list,
                                        nof_iterations=10,
                                        nof_points=500)
stats_auto = utils.measure_auto_error(dale_gt,
                                      gen_dist,
                                      model,
                                      axis_limits,
                                      nof_iterations=10,
                                      nof_points=500)

# plots
path2dir = os.path.join(path, folder_name)
savepath = os.path.join(path2dir, "fig_2.pdf") if savefig else None
utils.plot_fixed_vs_auto(K_list,
                         stats_fixed["mu_err_mean"],
                         stats_fixed["mu_err_std"],
                         stats_auto["mu_err_mean"],
                         stats_auto["mu_err_mean"],
                         "mu error per bin",
                         savefig=savepath)

savepath = os.path.join(path2dir, "fig_3.pdf") if savefig else None
utils.plot_fixed_vs_auto(K_list,
                         stats_fixed["var_err_mean"],
                         stats_fixed["var_err_std"],
                         stats_auto["var_err_mean"],
                         stats_auto["var_err_mean"],
                         "var error per bin",
                         savefig=savepath)

savepath = os.path.join(path2dir, "fig_4.pdf") if savefig else None
utils.plot_fixed_vs_auto(K_list,
                         stats_fixed["rho_mean"],
                         stats_fixed["rho_std"],
                         stats_auto["rho_mean"],
                         stats_auto["rho_mean"],
                         "rho per bin",
                         savefig=savepath)

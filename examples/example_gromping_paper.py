import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import scipy.stats as sps
import scipy.integrate as integrate
import example_models.models as models
import example_models.distributions as dist


# define model
b0 = 0
b1 = 1
b2 = 1
b3 = 10
model = models.LinearWithInteraction(b0, b1, b2, b3)

# define distribution
D = 2
x1_min = 0
x1_max = 1
x2_sigma = 2.
gen_dist = dist.Correlated1(D, x1_min, x1_max, x2_sigma)


# generate points
X = gen_dist.generate(N=100000)

# dale = fe.DALE(X, model.predict, model.jacobian)
# dale.fit(features="all", method="fixed-size", alg_params={"nof_bins":20})
# dale.plot(error="std")


dale = fe.DALE(X, model.predict, model.jacobian)
dale.fit(features=0, alg_params={"bin_method": "fixed", "nof_bins":200})
dale.plot(s=0)
# dale.plot_local_effects()




# plt.figure()
# plt.title("PDP")
# xs = np.linspace(0, 1, 100)
# plt.plot(xs, pdp.eval_unnorm(xs, s), "b--", label="on dataset (unnorm)")
# plt.plot(xs, pdp.eval(xs, s), "b-", label="on dataset (norm)")
# plt.plot(xs, pdp_numerical.eval_unnorm(xs, s), "g--", label="numerical (unnorm)")
# plt.plot(xs, pdp_numerical.eval(xs, s), "g-", label="numerical (norm)")
# plt.plot(xs, y_gt_unnorm(xs), "r--", label="gt (unnorm)")
# plt.plot(xs, y_gt(xs), "r-", label="gt (norm)")
# plt.xlabel("x1")
# plt.ylabel("f_PDP")
# plt.legend()
# plt.show(block=False)

import numpy as np
import pythia
import example_models.distributions as dist
import example_models.models as models

np.random.seed(21)

# gen dist
gen_dist = dist.Correlated_3D_1(D=3, x1_min=0, x1_max=1, x2_sigma=0, x3_sigma=0.5)
data = gen_dist.generate(N = 1000)
axis_limits = gen_dist.axis_limits

# model
model = models.Example3(a1=1, a2=1, a=0)
model_pred = model.predict
model_jac = model.jacobian

# DALE with equal bin sizes
dale = pythia.DALE(data, model_pred, model_jac, axis_limits)
params = {"bin_method": "fixed", "nof_bins": 20, "min_points_per_bin": 5}
# params = {"bin_method": "greedy", "max_nof_bins": 20, "min_points_per_bin": 5}
# params = {"bin_method": "dp", "max_nof_bins": 20, "min_points_per_bin": 5}
dale.fit(features=0, params=params)
y, var, stderr = dale.eval(x=np.linspace(axis_limits[0, 0], axis_limits[1, 0], 100),
                           s=0,
                           uncertainty=True)
dale.plot(s=0, error="std")

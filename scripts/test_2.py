import numpy as np
import effector

np.random.seed(21)

X_test = np.random.uniform(-1, 1, (1000, 3))
axis_limits = np.stack([[-1]*3, [1]*3], axis=0)

predict = effector.models.DoubleConditionalInteraction().predict
jacobian = effector.models.DoubleConditionalInteraction().jacobian

reg_pdp = effector.RegionalPDP(X_test, predict, axis_limits=axis_limits)
reg_pdp.summary(features=0)

xs = np.linspace(-1, 1, 10)
reg_pdp.eval(feature=0, node_idx=0, xs=xs)
reg_pdp.plot(feature=0, node_idx=1, heterogeneity="ice", nof_points=10, nof_ice=5)

import effector
import numpy as np
import timeit

# set up
dim = 15
X = effector.datasets.IndependentUniform(dim=dim, low=-1, high=1).generate_data(1000, seed=21)
model = effector.models.ConditionalInteraction()
predict = model.predict
jacobian = model.jacobian
Y = predict(X)
axis_limits = np.array([[-1, 1]] * dim).T

# global shap
tic = timeit.default_timer()
shap_dp = effector.ShapDP(X, predict, axis_limits=axis_limits)
shap_dp.fit(features="all")
toc = timeit.default_timer()
print("Global SHAP: ", toc - tic)
# Output: Global SHAP:  5.139942132002034

# global shap
tic = timeit.default_timer()
shap_dp = effector.ShapDP(X, predict, axis_limits=axis_limits, backend="shapiq")
shap_dp.fit(features="all")
toc = timeit.default_timer()
print("Global SHAP: ", toc - tic)
# Output: Global SHAP:  171.6272543039995

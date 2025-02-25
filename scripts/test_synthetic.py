import effector
import numpy as np
import timeit

# set up
X = effector.datasets.IndependentUniform(dim=15, low=-1, high=1).generate_data(1000, seed=21)
model = effector.models.ConditionalInteraction()
predict = model.predict
jacobian = model.jacobian
Y = predict(X)
axis_limits = np.array([[-1, 1]] * 15).T

# global shap
tic = timeit.default_timer()
shap_dp = effector.ShapDP(X, predict, axis_limits=axis_limits)
shap_dp.fit(features="all")
toc = timeit.default_timer()
print("Global SHAP: ", toc - tic)


# global shap
tic = timeit.default_timer()
shap_dp = effector.ShapDP(X, predict, axis_limits=axis_limits, backend="shapiq")
shap_dp.fit(features="all")
toc = timeit.default_timer()
print("Global SHAP: ", toc - tic)



shap_dp.plot(0)




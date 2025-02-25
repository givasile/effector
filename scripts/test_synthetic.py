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

shap_explainer_kwargs = {"masker": X[:50]}
shap_dp.fit(features="all", shap_explainer_kwargs=shap_explainer_kwargs)
toc = timeit.default_timer()

shap_dp.plot(0)
print("Global SHAP: ", toc - tic)
# Output: Global SHAP:  5.139942132002034


# regional shap
tic = timeit.default_timer()
r_method = effector.RegionalShapDP(X, predict, axis_limits=axis_limits)
r_method.fit(features=0, shap_explainer_kwargs=shap_explainer_kwargs)
toc = timeit.default_timer()

r_method.summary(0)

r_method.plot(0, 1)
r_method.plot(0, 2)
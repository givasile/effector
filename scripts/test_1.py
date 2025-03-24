import effector
import numpy as np
import timeit

# set up
dim = 3
X = effector.datasets.IndependentUniform(dim=dim, low=-1, high=1).generate_data(1000, seed=21)
model = effector.models.ConditionalInteraction()
predict = model.predict
jacobian = model.jacobian
Y = predict(X)
axis_limits = np.array([[-1, 1]] * dim).T

# global shap
tic = timeit.default_timer()
reg_pdp = effector.RegionalPDP(X, predict, axis_limits=axis_limits)
reg_pdp.fit(features="all", space_partitioner="best")
reg_pdp.summary("all")
toc = timeit.default_timer()

tic = timeit.default_timer()
reg_pdp = effector.RegionalPDP(X, predict, axis_limits=axis_limits)
reg_pdp.fit(features="all", space_partitioner="best_level_wise")
reg_pdp.summary("all")
toc = timeit.default_timer()



# reg_pdp.plot(2, 2, centering=True)




# pdp = effector.PDP(X, predict, axis_limits=axis_limits)
# pdp.fit(features="all", centering=True)
# pdp.plot(2)

from tests.test_functional import TestExample
import feature_effect as fe

N = 100
K = 10
X = TestExample.generate_samples(n=N, seed=21)

# generic estimator
est = fe.Estimator(data=X, model=TestExample.f, model_jac=TestExample.f_der)
est.fit(features=[0, 1], method="all", nof_bins=K)
est.plot(feature=0, method="all")

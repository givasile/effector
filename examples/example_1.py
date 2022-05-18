from tests.test_functional import TestExample1
import feature_effect as fe

N = 10
K = 50
X = TestExample1.generate_samples(n=N, seed=21)

#
dale = fe.DALE(data=X, model=TestExample1.f, model_jac=TestExample1.f_der)
dale.fit(features=[0, 1], k=K, method="variable-size")
dale.plot(0)


# generic estimator
# est = fe.Estimator(data=X, model=TestExample1.f, model_jac=TestExample1.f_der)
# est.fit(features=[0, 1], method="all", nof_bins=K)
# est.plot(feature=0, method="all")

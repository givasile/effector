# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import matplotlib.pyplot as plt
import numpy as np
from tests.test_functional import TestExample
import feature_effect as fe

# DALE
N = 5
K = 3
X = TestExample.generate_samples(N=N, seed=21)

# ALE
ale_inst = fe.ALE(points=X, f=TestExample.f)
ale_inst.fit(features=[0, 1], k=K)
ale_inst.plot(s=0, block=False)

# DALE
dale_inst = fe.DALE(data=X, model=TestExample.f, model_jac=TestExample.f_der)
dale_inst.fit(features=[0, 1], k=K)
dale_inst.plot(s=0, block=False)


# generic estimator
est = fe.Estimator(data=X, model=TestExample.f, model_jac=TestExample.f_der)
est.fit(features=[0, 1], method="all", nof_bins=K)
est.plot(feature=0, method="DALE")
est.plot(feature=0, method="ALE")
est.plot(feature=0, method="PDP")
est.plot(feature=0, method="MPlot")

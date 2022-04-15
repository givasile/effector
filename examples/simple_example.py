import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from  tests.test_functional import TestExample
import mdale

# DALE
N = 1000
K = 100

# prediction
samples = TestExample.generate_samples(N=N, seed=21)
X_der = TestExample.f_der(samples)
dalef = mdale.dale.DALE(f=TestExample.f, f_der=TestExample.f_der)
dalef.fit(samples, features=[0, 1], k=K, effects=X_der)
dalef.plot(s=0)


import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.test_functional import TestExample
import mdale.ale as ale
import mdale.dale as dale

# DALE
N = 1000
K = 100

# prediction
samples = TestExample.generate_samples(N=N, seed=21)
dale = dale.DALE(points=samples, f=TestExample.f, f_der=TestExample.f_der)
dale.fit(features=[0, 1], k=K)
dale.plot(s=0, block=False)

ale = ale.ALE(points=samples, f=TestExample.f)
ale.fit(features=[0, 1], k=K)
ale.plot(s=0, block=False)

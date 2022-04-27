import os
import sys
import matplotlib.pyplot as plt
import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tests.test_functional import TestExample
import mdale.ale as ale
import mdale.dale as dale

# DALE
N = 5
K = 3
X = TestExample.generate_samples(N=N, seed=21)

# ALE
ale_inst = ale.ALE(points=X, f=TestExample.f)
ale_inst.fit(features=[0, 1], k=K)
ale_inst.plot(s=0, block=False)

# DALE
dale_inst = dale.DALE(points=X, f=TestExample.f, f_der=TestExample.f_der)
dale_inst.fit(features=[0, 1], k=K)
dale_inst.plot(s=0, block=False)

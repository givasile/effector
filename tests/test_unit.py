import numpy as np
import os
import sys

# hack to import mdale
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mdale.dale


def test_compute_bin_effects_1():
    # predict
    N = 100
    points = np.ones([N]) - .5
    point_effects = np.ones_like(points)*10
    limits = np.array([0, 1, 2.])
    predict_bin_effects = mdale.dale.compute_bin_effects(points, point_effects, limits)

    # ground-truth
    gt_bin_effects = np.array([10., np.NAN])
    assert np.array_equal(gt_bin_effects, predict_bin_effects, equal_nan=True)


if __name__ == "__main__":
    test_compute_bin_effects_1()

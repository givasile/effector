import numpy as np
import time
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
    predict = mdale.dale.compute_bin_effects(points, point_effects, limits)

    # ground-truth
    gt = np.array([10., np.NaN])
    assert np.array_equal(gt, predict, equal_nan=True)


def test_compute_accumulated_effect_1():
    # predict
    xs = np.array([-1., -.5, 0., 0.5, 1., 1.5, 2., 2.5, 3.])
    limits = np.array([0, 1, 2.])
    bin_effects = np.array([1., 1.])
    dx = 1.
    predict = mdale.dale.compute_accumulated_effect(xs, limits, bin_effects, dx)
    gt = np.array([0., 0., 0., .5, 1., 1.5, 2., 2., 2.])
    assert np.array_equal(predict, gt)


if __name__ == "__main__":
    test_compute_bin_effects_1()
    test_compute_accumulated_effect_1()

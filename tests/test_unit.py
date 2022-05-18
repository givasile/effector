import pytest
import numpy as np
import feature_effect.utils as utils
import feature_effect.bin_estimation as be


def test_create_bins_1():
    """Test data with one point.

    Notes
    =====
    It must raise AssertionError
    """
    k = 10
    data = np.array([1.])
    with pytest.raises(AssertionError):
        utils.create_fix_size_bins(data, k)


def test_create_bins_2():
    """Test data with only one discrete point.

    Notes
    =====
    It must raise AssertionError. The data must contain at least two
    distinct values per feature.
    """
    k = 10
    data = np.ones(1000)
    with pytest.raises(AssertionError):
        utils.create_fix_size_bins(data, k)


def test_create_bins_3():
    """Test data with enough points.

    Notes
    =====
    The function must return 10 equal-sized bins.
    """
    k = 4
    data = np.array([0, 1])
    limits, dx = utils.create_fix_size_bins(data, k=k)
    assert dx == .25
    limits_gt = np.array([0., .25, .5, .75, 1.])
    assert np.array_equal(limits_gt, limits)


def test_compute_data_effect_1():
    """Test data effects for small bins.
    """
    data = np.array([[1, 2], [2, 3.]])
    model = lambda x: np.sum(x, axis=1)

    # feature 0
    dx = 1.
    limits = np.array([1., 2.])
    data_effect = utils.compute_data_effect(data, model, limits, dx, feature=0)
    data_effect_gt = np.array([1., 1.])
    assert np.allclose(data_effect_gt, data_effect, atol=1.e-5)

    # feature 1
    dx = 1.
    limits = np.array([2., 3.])
    data_effect = utils.compute_data_effect(data, model, limits, dx, feature=1)
    data_effect_gt = np.array([1., 1.])
    assert np.allclose(data_effect_gt, data_effect, atol=1.e-5)


def test_compute_data_effect_2():
    """Test data effects for big bins.
    """
    data = np.array([[1, 2], [2, 3.]])
    model = lambda x: np.sum(x, axis=1)

    # feature 0
    dx = 100.
    limits = np.array([0., 100.])
    data_effect = utils.compute_data_effect(data, model, limits, dx, feature=0)
    data_effect_gt = np.array([1., 1.])
    assert np.allclose(data_effect_gt, data_effect, atol=1.e-5)

    # feature 1
    dx = 100.
    limits = np.array([0., 100.])
    data_effect = utils.compute_data_effect(data, model, limits, dx, feature=1)
    data_effect_gt = np.array([1., 1.])
    assert np.allclose(data_effect_gt, data_effect, atol=1.e-5)


def test_compute_bin_effect_mean_1():
    n = 100
    data = np.ones([n]) - .5
    data_effect = np.ones_like(data)*10
    limits = np.array([0, 1, 2.])

    # predict
    bin_effects, points_per_bin = utils.compute_bin_effect_mean(data, data_effect, limits)

    # ground-truth
    bin_effects_gt = np.array([10., np.NaN])
    assert np.array_equal(bin_effects_gt, bin_effects, equal_nan=True)
    points_per_bin_gt = np.array([n, 0])
    assert np.array_equal(points_per_bin_gt, points_per_bin)


def test_compute_bin_effect_variance_1():
    data = np.ones(2) - .5
    data_effect = np.array([1., 2.])
    limits = np.array([0, 1, 2.])
    mean_effect = np.array([1.5, np.NaN])

    bin_variance, bin_estimator_variance = utils.compute_bin_effect_variance(data, data_effect, limits, mean_effect)

    bin_variance_gt = np.array([.25, np.NaN])
    assert np.allclose(bin_variance_gt, bin_variance, equal_nan=True)

    bin_estimator_variance_gt = np.array([.125, np.NaN])
    assert np.allclose(bin_estimator_variance_gt, bin_estimator_variance, equal_nan=True)


def test_fill_nans_1():
    bin_effect = np.array([1., np.NaN, 2.])
    pred = utils.fill_nans(bin_effect)
    gt = np.array([1., 1.5, 2.])
    assert np.allclose(pred, gt)


def test_fill_nans_2():
    bin_effect = np.array([1., np.NaN, np.NaN, np.NaN, 2.])
    pred = utils.fill_nans(bin_effect)
    gt = np.array([1., 1.25, 1.5, 1.75, 2.])
    assert np.allclose(pred, gt)


def test_fill_nans_3():
    bin_effect = np.array([0.5, 1., np.NaN, np.NaN, np.NaN])
    pred = utils.fill_nans(bin_effect)
    gt = np.array([0.5, 1., 1., 1., 1.])
    assert np.allclose(pred, gt)


def test_compute_accumulated_effect_1():
    # predict
    x = np.array([-1., -.5, 0., 0.5, 1., 1.5, 2., 2.5, 3.])
    limits = np.array([0, 1, 2.])
    bin_effect = np.array([1., 1.])
    dx = 1.
    predict = utils.compute_accumulated_effect(x, limits, bin_effect, dx)
    gt = np.array([0., 0., 0., .5, 1., 1.5, 2., 2., 2.])
    assert np.array_equal(predict, gt)


def test_find_first_nan_bin_1():
    x = np.array([0, np.NaN])
    pred = utils.find_first_nan_bin(x)
    assert pred == 1


def test_find_first_nan_bin_2():
    x = np.array([0, 1])
    pred = utils.find_first_nan_bin(x)
    assert pred is None


def test_compute_normalizer_1():
    x = np.array([0.5, 1.5])
    limits = np.array([0, 1, 2])
    bin_effect = np.array([10, 10])
    dx = 1

    pred = utils.compute_normalizer(x, limits, bin_effect, dx)
    gt = 10
    assert np.allclose(pred, gt)


class TestBinEstimator:
    def test_1(self):
        np.random.seed(21)
        N1 = 1000
        data = np.expand_dims(
            np.concatenate((np.random.uniform(low=0., high=.25, size=N1),
                            np.random.uniform(low=0.25, high=.5, size=N1),
                            np.random.uniform(low=0.5, high=.75, size=N1),
                            np.random.uniform(low=0.75, high=1., size=N1))
                           ),
            axis=-1
        )

        data_effect = np.expand_dims(
            np.concatenate((1*np.ones(N1) + np.random.normal(size=N1)*0.1,
                            -2*np.ones(N1) + np.random.normal(size=N1)*0.1,
                            2*np.ones(N1) + np.random.normal(size=N1)*0.1,
                            -1*np.ones(N1) + np.random.normal(size=N1)*0.1)
                           ),
            axis=-1
        )

        bin_estimator = be.BinEstimator(data, data_effect, None, feature=0, K=70)
        limits, dx_list = bin_estimator.solve_dp()

        assert np.allclose(limits, np.array([0., .25, .5, .75, 1]), atol=.1)
        assert np.allclose(dx_list, np.ones(4)*.25, atol=.1)


    def test_2(self):
        np.random.seed(21)
        N1 = 100
        data = np.expand_dims(
            np.concatenate((np.random.uniform(low=0., high=.1, size=N1),
                            np.random.uniform(low=0.1, high=.6, size=N1),
                            np.random.uniform(low=0.6, high=.61, size=N1),
                            np.random.uniform(low=0.61, high=.8, size=N1))
                           ),
            axis=-1
        )

        data_effect = np.expand_dims(
            np.concatenate((1*np.ones(N1) + np.random.normal(size=N1)*0.1,
                            -2*np.ones(N1) + np.random.normal(size=N1)*0.1,
                            2*np.ones(N1) + np.random.normal(size=N1)*0.1,
                            -1*np.ones(N1) + np.random.normal(size=N1)*0.1)
                           ),
            axis=-1
        )

        bin_estimator = be.BinEstimator(data, data_effect, None, feature=0, K=70)
        limits, dx_list = bin_estimator.solve_dp()

        assert np.allclose(limits, np.array([0., .1, .6, .61, .8]), atol=.1)
        assert np.allclose(dx_list, np.array([.1, .5, .01, .19]), atol=.1)


if __name__ == "__main__":
    test_create_bins_1()
    test_create_bins_2()
    test_create_bins_3()

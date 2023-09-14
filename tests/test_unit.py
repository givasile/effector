import numpy as np
import effector.utils as utils


def test_compute_data_effect_1():
    """Test data effects for small bins."""
    data = np.array([[1, 2], [2, 3.0]])
    model = lambda x: np.sum(x, axis=1)

    # feature 0
    dx = 1.0
    limits = np.array([1.0, 2.0])
    data_effect = utils.compute_local_effects_at_bin_limits(
        data, model, limits, feature=0
    )
    data_effect_gt = np.array([1.0, 1.0])
    assert np.allclose(data_effect_gt, data_effect, atol=1.0e-5)

    # feature 1
    dx = 1.0
    limits = np.array([2.0, 3.0])
    data_effect = utils.compute_local_effects_at_bin_limits(
        data, model, limits, feature=1
    )
    data_effect_gt = np.array([1.0, 1.0])
    assert np.allclose(data_effect_gt, data_effect, atol=1.0e-5)


def test_compute_data_effect_2():
    """Test data effects for big bins."""
    data = np.array([[1, 2], [2, 3.0]])
    model = lambda x: np.sum(x, axis=1)

    # feature 0
    dx = 100.0
    limits = np.array([0.0, 100.0])
    data_effect = utils.compute_local_effects_at_bin_limits(
        data, model, limits, feature=0
    )
    data_effect_gt = np.array([1.0, 1.0])
    assert np.allclose(data_effect_gt, data_effect, atol=1.0e-5)

    # feature 1
    dx = 100.0
    limits = np.array([0.0, 100.0])
    data_effect = utils.compute_local_effects_at_bin_limits(
        data, model, limits, feature=1
    )
    data_effect_gt = np.array([1.0, 1.0])
    assert np.allclose(data_effect_gt, data_effect, atol=1.0e-5)


def test_compute_bin_effect_mean_1():
    n = 100
    data = np.ones([n]) - 0.5
    data_effect = np.ones_like(data) * 10
    limits = np.array([0, 1, 2.0])

    # predict
    bin_effects, points_per_bin = utils.compute_bin_effect(data, data_effect, limits)

    # ground-truth
    bin_effects_gt = np.array([10.0, np.NaN])
    assert np.array_equal(bin_effects_gt, bin_effects, equal_nan=True)
    points_per_bin_gt = np.array([n, 0])
    assert np.array_equal(points_per_bin_gt, points_per_bin)


def test_compute_bin_effect_variance_1():
    data = np.ones(4) * 0.5
    data_effect = np.array([1.0, 2.0, 3.0, 4.0])
    limits = np.array([0, 1, 2.0])
    mean_effect = np.array([np.mean(data_effect), np.NaN])

    bin_variance, bin_estimator_variance = utils.compute_bin_variance(
        data, data_effect, limits, mean_effect
    )

    bin_variance_gt = np.array([np.var(data_effect), np.NaN])
    assert np.allclose(bin_variance_gt, bin_variance, equal_nan=True)

    bin_estimator_variance_gt = np.array([np.var(data_effect) / data.shape[0], np.NaN])
    assert np.allclose(
        bin_estimator_variance_gt, bin_estimator_variance, equal_nan=True
    )


def test_fill_nans_1():
    bin_effect = np.array([1.0, np.NaN, 2.0])
    pred = utils.fill_nans(bin_effect)
    gt = np.array([1.0, 1.5, 2.0])
    assert np.allclose(pred, gt)


def test_fill_nans_2():
    bin_effect = np.array([1.0, np.NaN, np.NaN, np.NaN, 2.0])
    pred = utils.fill_nans(bin_effect)
    gt = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
    assert np.allclose(pred, gt)


def test_fill_nans_3():
    bin_effect = np.array([0.5, 1.0, np.NaN, np.NaN, np.NaN])
    pred = utils.fill_nans(bin_effect)
    gt = np.array([0.5, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(pred, gt)


def test_compute_accumulated_effect_1():
    # predict
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    limits = np.array([0, 1.5, 2.0])
    bin_effect = np.array([1.0, -1.0])
    dx = np.array([1.5, 0.5])
    predict = utils.compute_accumulated_effect(x, limits, bin_effect, dx)
    gt = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 1.0, 1.0, 1.0])
    assert np.array_equal(predict, gt)


def test_compute_accumulated_effect_2():
    # predict
    x = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    limits = np.array([0, 1.5, 2.0])
    bin_effect = np.array([1.0, 1.0])
    dx = np.array([1.5, 0.5])
    predict = utils.compute_accumulated_effect(x, limits, bin_effect, dx)
    gt = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.5, 2, 2, 2])
    assert np.array_equal(predict, gt)


# def test_compute_normalizer_1():
#     limits = np.array([0, 1])
#     bin_effect = np.array([10])
#     x = np.linspace(limits[0], limits[-1], 10000)
#     pred = utils.compute_normalizer(limits, bin_effect, x)
#     gt = 5
#     assert np.allclose(pred, gt)
#
#
# def test_compute_normalizer_2():
#     limits = np.array([0, 1, 2])
#     bin_effect = np.array([1, 3])
#     x = np.linspace(limits[0], limits[-1], 10000)
#     pred = utils.compute_normalizer(limits, bin_effect, x)
#     gt = 1.5
#     assert np.allclose(pred, gt, atol=1e-2)
#
#
# def test_compute_normalizer_3():
#     limits = np.array([0, 1, 2, 3, 4])
#     bin_effect = np.array([1, -1, -1, 1])
#     x = np.linspace(limits[0], limits[-1], 10000)
#     pred = utils.compute_normalizer(limits, bin_effect, x)
#     gt = 0.0
#     assert np.allclose(pred, gt)


# class TestBinEstimator:
#     def test_1(self):
#         np.random.seed(21)
#         N1 = 1000
#         data = np.expand_dims(
#             np.concatenate((np.random.uniform(low=0., high=.25, size=N1),
#                             np.random.uniform(low=0.25, high=.5, size=N1),
#                             np.random.uniform(low=0.5, high=.75, size=N1),
#                             np.random.uniform(low=0.75, high=1., size=N1))
#                            ),
#             axis=-1
#         )
#
#         data_effect = np.expand_dims(
#             np.concatenate((1*np.ones(N1) + np.random.normal(size=N1)*0.1,
#                             -2*np.ones(N1) + np.random.normal(size=N1)*0.1,
#                             2*np.ones(N1) + np.random.normal(size=N1)*0.1,
#                             -1*np.ones(N1) + np.random.normal(size=N1)*0.1)
#                            ),
#             axis=-1
#         )
#
#         bin_estimator = be.BinEstimatorDP(data, data_effect, None, feature=0, K=70)
#         limits, dx_list = bin_estimator.solve_dp()
#
#         assert np.allclose(limits, np.array([0., .25, .5, .75, 1]), atol=.1)
#         assert np.allclose(dx_list, np.ones(4)*.25, atol=.1)
#
#
#     def test_2(self):
#         np.random.seed(21)
#         N1 = 100
#         data = np.expand_dims(
#             np.concatenate((np.random.uniform(low=0., high=.1, size=N1),
#                             np.random.uniform(low=0.1, high=.6, size=N1),
#                             np.random.uniform(low=0.6, high=.61, size=N1),
#                             np.random.uniform(low=0.61, high=.8, size=N1))
#                            ),
#             axis=-1
#         )
#
#         data_effect = np.expand_dims(
#             np.concatenate((1*np.ones(N1) + np.random.normal(size=N1)*0.1,
#                             -2*np.ones(N1) + np.random.normal(size=N1)*0.1,
#                             2*np.ones(N1) + np.random.normal(size=N1)*0.1,
#                             -1*np.ones(N1) + np.random.normal(size=N1)*0.1)
#                            ),
#             axis=-1
#         )
#
#         bin_estimator = be.BinEstimatorDP(data, data_effect, None, feature=0, K=70)
#         limits, dx_list = bin_estimator.solve_dp()
#
#         assert np.allclose(limits, np.array([0., .1, .6, .61, .8]), atol=.1)
#         assert np.allclose(dx_list, np.array([.1, .5, .01, .19]), atol=.1)

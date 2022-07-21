import pytest
import numpy as np
from scipy.stats import norm
import feature_effect.utils as utils
import matplotlib.pyplot as plt
import feature_effect as fe
import numdifftools

np.random.seed(21)

class TestCase1:

    @staticmethod
    def model(x, params):
        """f(x1, x2) = a + b*x1 + x1x2
        """
        ind_1 = np.logical_and(x[:, 0] >= params[0]["from"], x[:, 0] < params[0]["to"])
        ind_2 = np.logical_and(x[:, 0] >= params[1]["from"], x[:, 0] < params[1]["to"])
        ind_3 = np.logical_and(x[:, 0] >= params[2]["from"], x[:, 0] < params[2]["to"])
        ind_4 = np.logical_and(x[:, 0] >= params[3]["from"], x[:, 0] <= params[3]["to"])

        y = params[0]["b"] * x[:, 0] + params[0]["a"] + x[:, 0] * x[:, 1]
        y[ind_2] = params[1]["b"] * (x[ind_2, 0] - params[1]["from"]) + params[1]["a"] + x[ind_2,0] * x[ind_2, 1]
        y[ind_3] = params[2]["b"] * (x[ind_3, 0] - params[2]["from"]) + params[2]["a"] + x[ind_3,0] * x[ind_3, 1]
        y[ind_4] = params[3]["b"] * (x[ind_4, 0] - params[3]["from"]) + params[3]["a"] + x[ind_4,0] * x[ind_4, 1]

        return y

    @staticmethod
    def model_jac(x, params):
        """df/dx1 = b*x1 + x2
           df/dx2 = x1
        """
        ind_1 = np.logical_and(x[:, 0] >= params[0]["from"], x[:, 0] < params[0]["to"])
        ind_2 = np.logical_and(x[:, 0] >= params[1]["from"], x[:, 0] < params[1]["to"])
        ind_3 = np.logical_and(x[:, 0] >= params[2]["from"], x[:, 0] < params[2]["to"])
        ind_4 = np.logical_and(x[:, 0] >= params[3]["from"], x[:, 0] <= params[3]["to"])

        y = np.ones_like(x)
        y[ind_1, 0] = params[0]["b"] + x[ind_1, 1]
        y[ind_2, 0] = params[1]["b"] + x[ind_2, 1]
        y[ind_3, 0] = params[2]["b"] + x[ind_3, 1]
        y[ind_4, 0] = params[3]["b"] + x[ind_4, 1]

        y[ind_1, 1] = x[ind_1, 0]
        y[ind_2, 1] = x[ind_2, 0]
        y[ind_3, 1] = x[ind_3, 0]
        y[ind_4, 1] = x[ind_4, 0]
        return y

    @staticmethod
    def generate_samples(N, noise_level):
        """x1 ~ U(0,1)
           x2 ~ N(0, noise_level)
        """
        eps = 1e-03
        stop = 5
        x1 = np.concatenate((np.array([0.]),
                             np.random.uniform(0., 1., size=int(N-2)),
                             np.array([1.]))
                            )

        x2 = np.random.normal(loc=0, scale=noise_level, size=(int(x1.shape[0])))
        x = np.stack((x1, x2), axis=-1)
        return x


    # test case 1 - only five points
    def create_data(self, params):
        x1 = np.array([0., 0.2, 0.8, 1.])
        x2 = np.random.normal(loc=0, scale=0.01, size=(int(x1.shape[0])))
        x = np.stack((x1, x2), axis=-1)
        y = self.model(x, params)
        y_grad = self.model_jac(x, params)
        return x, y_grad


    def test_min_points_2(self):
        """Must create two bins
        """
        params = [{"a": 0., "b":10, "from": 0., "to": .25},
                  {"a": 3., "b":-10., "from": .25, "to": .5},
                  {"a": 0., "b": 5., "from": .5, "to": .75},
                  {"a": 1., "b":-5., "from": .75, "to": 1}]
        x, y_grad = self.create_data(params)

        # test Greedy
        min_points = 2
        est = fe.bin_estimation.Greedy(x, y_grad, feature=0)
        limits_Greedy = est.solve(min_points)

        assert limits_Greedy.size == 3
        assert np.allclose(0, limits_Greedy[0])
        assert 0.2 <= limits_Greedy[1] <= .8
        assert np.allclose(1., limits_Greedy[2])

        # test DP
        min_points = 2
        est = fe.bin_estimation.BinEstimatorDP(x, y_grad, feature=0)
        limits_DP = est.solve(min_points, K=10)

        assert limits_DP.size == 3
        assert np.allclose(0, limits_DP[0])
        assert 0.2 <= limits_DP[1] <= .8
        assert np.allclose(1., limits_DP[2])


    def test_min_points_3(self):
        params = [{"a": 0., "b":10, "from": 0., "to": .25},
                  {"a": 3., "b":-10., "from": .25, "to": .5},
                  {"a": 0., "b": 5., "from": .5, "to": .75},
                  {"a": 1., "b":-5., "from": .75, "to": 1}]
        x, y_grad = self.create_data(params)
        gt_limits = np.array([0, 1.])

        min_points = 3
        est = fe.bin_estimation.Greedy(x, y_grad, feature=0)
        limits_Greedy = est.solve(min_points)
        assert np.allclose(gt_limits, limits_Greedy)

        min_points = 3
        est = fe.bin_estimation.BinEstimatorDP(x, y_grad, feature=0)
        limits_DP = est.solve(min_points)
        assert np.allclose(gt_limits, limits_DP)


    def test_min_points_4(self):
        params = [{"a": 0., "b":10, "from": 0., "to": .25},
                  {"a": 3., "b":-10., "from": .25, "to": .5},
                  {"a": 0., "b": 5., "from": .5, "to": .75},
                  {"a": 1., "b":-5., "from": .75, "to": 1}]
        x, y_grad = self.create_data(params)
        gt_limits = np.array([0, 1.])

        min_points = 4
        est = fe.bin_estimation.Greedy(x, y_grad, feature=0)
        limits_Greedy = est.solve(min_points)
        assert np.allclose(gt_limits, limits_Greedy)

        min_points = 4
        est = fe.bin_estimation.BinEstimatorDP(x, y_grad, feature=0)
        limits_DP = est.solve(min_points)
        assert np.allclose(gt_limits, limits_DP)

    def test_min_points_5(self):
        params = [{"a": 0., "b":10, "from": 0., "to": .25},
                  {"a": 3., "b":-10., "from": .25, "to": .5},
                  {"a": 0., "b": 5., "from": .5, "to": .75},
                  {"a": 1., "b":-5., "from": .75, "to": 1}]
        x, y_grad = self.create_data(params)

        min_points = 5
        est = fe.bin_estimation.Greedy(x, y_grad, feature=0)
        limits_Greedy = est.solve(min_points)
        assert limits_Greedy is False

        min_points = 5
        est = fe.bin_estimation.BinEstimatorDP(x, y_grad, feature=0)
        limits_DP = est.solve(min_points)
        assert limits_DP is False


    def test_many_points(self):
        params = [{"a": 0., "b":10, "from": 0., "to": .25},
                  {"a": 3., "b":-10., "from": .25, "to": .5},
                  {"a": 0., "b": 5., "from": .5, "to": .75},
                  {"a": 1., "b":-5., "from": .75, "to": 1}]

        tol = .05
        N = 1e4
        noise_level = 0
        x = self.generate_samples(N, noise_level)
        y = self.model(x, params)
        y_grad = self.model_jac(x, params)
        gt_limits = np.array([0., .25, .5, .75, 1])

        # test Greedy
        min_points = 10
        est = fe.bin_estimation.Greedy(x, y_grad, feature=0)
        limits_Greedy = est.solve(min_points)
        assert  np.sum(np.logical_and(limits_Greedy >= 0 - tol, limits_Greedy <= 0 + tol, )) >= 1
        assert  np.sum(np.logical_and(limits_Greedy >= .25 - tol, limits_Greedy <= .25 + tol, )) >= 1
        assert  np.sum(np.logical_and(limits_Greedy >= .5 - tol, limits_Greedy <= .5 + tol, )) >= 1
        assert  np.sum(np.logical_and(limits_Greedy >= .75 - tol, limits_Greedy <= .75 + tol, )) >= 1
        assert  np.sum(np.logical_and(limits_Greedy >= 1. - tol, limits_Greedy <= 1. + tol, )) >= 1

        # test DP
        min_points = 10
        est = fe.bin_estimation.BinEstimatorDP(x, y_grad, feature=0)
        limits_DP = est.solve(min_points)
        assert  np.sum(np.logical_and(limits_DP >= 0 - tol, limits_DP <= 0 + tol, )) >= 1
        assert  np.sum(np.logical_and(limits_DP >= .25 - tol, limits_DP <= .25 + tol, )) >= 1
        assert  np.sum(np.logical_and(limits_DP >= .5 - tol, limits_DP <= .5 + tol, )) >= 1
        assert  np.sum(np.logical_and(limits_DP >= .75 - tol, limits_DP <= .75 + tol, )) >= 1
        assert  np.sum(np.logical_and(limits_DP >= 1. - tol, limits_DP <= 1. + tol, )) >= 1

import matplotlib.pyplot as plt
import numpy as np
import pythia as fe
import scipy.stats as sps
import scipy.integrate as integrate
import pytest
from functools import partial
import example_models.models as models
import example_models.distributions as dist
import pythia.binning_methods

np.random.seed(21)


class TestCase1:
    def _create_cond(self, x, i, s):
        par = self.params
        if x.ndim >= 2:
            return np.logical_and(x[:, s] >= par[i]["from"], x[:, s] <= par[i]["to"])
        elif x.ndim == 1:
            return np.logical_and(x >= par[i]["from"], x <= par[i]["to"])

    def _create_func(self, i, func):
        par = self.params
        return partial(func, a=par[i]["a"], b=par[i]["b"], x0=par[i]["from"])

    def create_model_data(self):
        params = [
            {"a": 0.0, "b": 10, "from": 0.0, "to": 0.25},
            {"a": 2.5, "b": -10.0, "from": 0.25, "to": 0.5},
            {"a": 0.0, "b": 5.0, "from": 0.5, "to": 0.75},
            {"a": 1.25, "b": -5.0, "from": 0.75, "to": 1},
        ]
        self.params = params

        model = models.PiecewiseLinear(params)

        D = 2
        x1_min = 0
        x1_max = 1
        x2_sigma = 0.1
        gen_dist = dist.Uncorrelated1(D, x1_min, x1_max, x2_sigma)

        X = gen_dist.generate(N=10000)
        X_jac = model.jacobian(X)

        def pdp_correct(x):
            def mu(x, a, b, x0):
                return a + b * (x - x0)

            par = self.params
            condlist = [self._create_cond(x, i, s=0) for i in range(4)]
            funclist = [self._create_func(i, mu) for i in range(4)]

            y = np.zeros(x.shape[0])
            for i, cond in enumerate(condlist):
                y[cond] = funclist[i](x[cond])
            return y

        self.pdp_gt = pdp_correct

        def dale_mean_correct(x):
            def mu(x, a, b, x0):
                return b

            par = self.params
            condlist = [self._create_cond(x, i, s=0) for i in range(4)]
            funclist = [self._create_func(i, mu) for i in range(4)]

            y = np.zeros(x.shape[0])
            for i, cond in enumerate(condlist):
                y[cond] = funclist[i](x[cond])
            return y

        self.dale_mean = dale_mean_correct

        def dale_mean_int_correct(x):
            def mu(x, a, b, x0):
                return a + b * (x - x0)

            par = self.params
            condlist = [self._create_cond(x, i, s=0) for i in range(4)]
            funclist = [self._create_func(i, mu) for i in range(4)]

            y = np.zeros(x.shape[0])
            for i, cond in enumerate(condlist):
                y[cond] = funclist[i](x[cond])
            return y

        self.dale_mean_int = dale_mean_int_correct

        self.dale_var = lambda x: np.ones(x.shape[0]) * x2_sigma
        self.dale_var_int = lambda x: x2_sigma * x

        return model, gen_dist, X, X_jac

    def test_pdp(self):
        model, gen_dist, X, X_jac = self.create_model_data()

        # pdp monte carlo approximation
        s = 0
        pdp = fe.PDP(data=X, model=model.predict, axis_limits=gen_dist.axis_limits)
        pdp.fit(features=0)

        pdp_gt = pythia.pdp.PDPGroundTruth(self.pdp_gt, gen_dist.axis_limits)
        pdp_gt.fit(features=0)

        xs = np.linspace(gen_dist.axis_limits[0, 0], gen_dist.axis_limits[1, 0], 100)
        y1 = pdp.eval(xs, feature=0)
        y2 = pdp_gt.eval(xs, feature=0)
        assert np.allclose(y1, y2, rtol=0.1, atol=0.1)

    def test_dale(self):
        model, gen_dist, X, X_jac = self.create_model_data()

        dale = fe.DALE(data=X, model=model.predict, model_jac=model.jacobian)
        binning = pythia.binning_methods.Fixed(nof_bins=20)
        dale.fit(features=0, binning_method=binning)

        dale_gt = pythia.dale.DALEGroundTruth(
            self.dale_mean,
            self.dale_mean_int,
            self.dale_var,
            self.dale_var_int,
            gen_dist.axis_limits,
        )
        dale_gt.fit(features=0)

        dale_gt_bins = pythia.dale.DALEBinsGT(
            self.dale_mean, self.dale_var, gen_dist.axis_limits
        )
        binning = pythia.binning_methods.Fixed(nof_bins=20)
        dale_gt_bins.fit(features=0, binning_method=binning)

        xs = np.linspace(0, 1, 100)
        y1 = dale_gt.eval(xs, feature=0)
        y2 = dale_gt_bins.eval(xs, feature=0)
        y3 = dale.eval(xs, feature=0)

        assert np.allclose(y1, y2, rtol=0.1, atol=0.1)
        assert np.allclose(y1, y3, rtol=0.1, atol=0.1)

    def _bin_limit_in_region(self, limits, point, tol):
        return (
            np.sum(
                np.logical_and(
                    limits >= point - tol,
                    limits <= point + tol,
                )
            )
            >= 1
        )

    def test_bin_greedy(self):
        tol = 0.1
        gt_list = [0.0, 0.25, 0.5, 0.75, 1]
        model, gen_dist, X, X_jac = self.create_model_data()

        # test greedy GT
        greedy = fe.bin_estimation.GreedyGT(
            self.dale_mean, self.dale_var, gen_dist.axis_limits, feature=0
        )

        greedy.find(n_max=20)
        for i, point in enumerate(gt_list):
            assert self._bin_limit_in_region(greedy.limits, point, tol)

        # test Greedy on points
        greedy = fe.bin_estimation.Greedy(
            X, X_jac, feature=0, axis_limits=gen_dist.axis_limits
        )
        greedy.find()
        for i, point in enumerate(gt_list):
            assert self._bin_limit_in_region(greedy.limits, point, tol)

    def test_bin_dp(self):
        tol = 0.05
        gt_list = [0.0, 0.25, 0.5, 0.75, 1]
        model, gen_dist, X, X_jac = self.create_model_data()

        # test greedy GT
        dp = fe.bin_estimation.DPGT(
            self.dale_mean, self.dale_var, gen_dist.axis_limits, feature=0
        )
        dp.find(k_max=20)
        for i, point in enumerate(gt_list):
            assert self._bin_limit_in_region(dp.limits, point, tol)

        # test Greedy on points
        dp = fe.bin_estimation.DP(X, X_jac, feature=0, axis_limits=gen_dist.axis_limits)
        dp.find()
        for i, point in enumerate(gt_list):
            assert self._bin_limit_in_region(dp.limits, point, tol)


# case1 = TestCase1()
# model, gen_dist, X, X_jac = case1.create_model_data()

# greedy = fe.bin_estimation.GreedyGroundTruth(model.ale_mu, model.ale_var,
#                                              gen_dist.axis_limits, feature=0)
# greedy.solve()
# print(greedy.limits)

import matplotlib.pyplot as plt
import numpy as np
import pythia as fe
import scipy.stats as sps
import scipy.integrate as integrate
import pytest
import example_models.models as models
import example_models.distributions as dist

np.random.seed(21)

class TestCase1:
    def create_model_data(self):
        # define model and distribution
        b0 = 1
        b1 = -10
        b2 = 20
        b3 = 100
        model = models.LinearWithInteraction(b0=b0, b1=b1, b2=b2, b3=b3)

        D = 2
        x1_min = 0
        x1_max = 1
        x2_sigma = .1
        gen_dist = dist.Correlated1(D, x1_min, x1_max, x2_sigma)

        # generate points
        X = gen_dist.generate(N=10000)

        # ground truth
        self.pdp_gt = lambda x: (b1 + b3*.5)*x
        self.dale_mean = lambda x: b1 + b3*x
        self.dale_mean_int = lambda x: b1*x + b3/2*x**2
        self.dale_var = lambda x: b3 * x2_sigma
        self.dale_var_int = lambda x: b3 * x2_sigma * x

        return model, gen_dist, X

    def test_pdp(self):
        model, gen_dist, X = self.create_model_data()

        # pdp monte carlo approximation
        s = 0
        pdp = fe.PDP(data=X, model=model.predict, axis_limits=gen_dist.axis_limits)
        pdp.fit(features=0)

        # pdp numerical approximation
        # p_xc = gen_dist.pdf_x2
        # pdp_numerical = fe.PDPNumerical(p_xc, model.predict, gen_dist.axis_limits, s=0, D=2)
        # pdp_numerical.fit(features=0)

        # pdp ground truth
        pdp_gt = fe.PDPGroundTruth(self.pdp_gt, gen_dist.axis_limits)
        pdp_gt.fit(features=0)

        xs = np.linspace(gen_dist.axis_limits[0, 0], gen_dist.axis_limits[1, 0], 100)
        y1 = pdp.eval(xs, s=0)
        y2 = pdp_gt.eval(xs, s=0)
        assert np.allclose(y1, y2, rtol=0.1, atol=0.1)


    def test_dale(self):
        model, gen_dist, X = self.create_model_data()

        s = 0
        dale = fe.DALE(data=X, model=model.predict, model_jac=model.jacobian)
        dale.fit(features=0, binning_params={"bin_method": "fixed", "nof_bins": 20})

        dale_gt = fe.DALEGroundTruth(self.dale_mean, self.dale_mean_int, self.dale_var,
                                     self.dale_var_int, gen_dist.axis_limits)
        dale_gt.fit(features=0)

        dale_gt_bins = fe.DALEBinsGT(self.dale_mean, self.dale_var, gen_dist.axis_limits)
        dale_gt_bins.fit(features=0, binning_params={"bin_method": "fixed", "nof_bins": 20})

        xs = np.linspace(0, 1, 100)
        y1 = dale_gt.eval(xs, s=0)
        y2 = dale_gt_bins.eval(xs, s=0)
        y3 = dale.eval(xs, s=0)
        assert np.allclose(y1, y2, rtol=0.1, atol=0.1)
        assert np.allclose(y1, y3, rtol=0.1, atol=0.1)



# case1 = TestCase1()
# print(case1.test_pdp())
# print(case1.test_dale_bin_gt())
# model, gen_dist, X = case1.create_model_data()

# s = 0
# dale = fe.DALE(data=X, model=model.predict, model_jac=model.jacobian)
# dale.fit(features=[0], method="fixed-size", alg_params={"nof_bins": 20})

# # dale ground truth
# # dale_gt = fe.DALEGroundTruth(case1.dale_mean, case1.dale_mean_int, case1.dale_var,
# #                              case1.dale_var_int, gen_dist.axis_limits)
# # dale_gt.fit(features=0)

# dale_gt_bins = fe.DALEBinsGT(case1.dale_mean, case1.dale_var, gen_dist.axis_limits)
# dale_gt_bins.fit(features=0, alg_params={"bin_method": "fixed", "nof_bins": 20})

# xs = np.linspace(0, 1, 100)
# y1 = dale_gt_bins.eval(xs, s=0)
# y2 = dale.eval(xs, s=0)[0]
# # y3 = dale_gt.eval(xs, s=0)
# # assert np.allclose(y1, y2, rtol=0.1, atol=0.1)

# plt.figure()
# plt.plot(xs, y1, label="dale gt bins")
# plt.plot(xs, y2, label="dale monte carlo")
# # plt.plot(xs, y3, label="gt")
# plt.legend()
# plt.show(block=False)

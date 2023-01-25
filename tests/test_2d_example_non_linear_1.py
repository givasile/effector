import numpy as np
import pythia
import example_models.models as models
import example_models.distributions as dist

np.random.seed(21)


class TestCase1:
    def create_model_data(self):
        # define model and distribution
        b0 = 5
        b1 = 100
        b2 = -100
        b3 = -10
        model = models.SquareWithInteraction(b0=b0, b1=b1, b2=b2, b3=b3)

        D = 2
        x1_min = 0
        x1_max = 1
        x2_sigma = 1.0
        gen_dist = dist.Correlated1(D, x1_min, x1_max, x2_sigma)

        # generate points
        X = gen_dist.generate(N=10000)

        # ground truth
        self.pdp_gt = (
            lambda x: b0 + b1 * x**2 + b2 * (x2_sigma + 0.5**2) + b3 * 0.5 * x
        )
        self.dale_mean = lambda x: 2 * b1 * x + b3 * x
        self.dale_mean_int = lambda x: b1 * x**2 + b3 / 2 * x**2
        self.dale_var = lambda x: b3 * x2_sigma
        self.dale_var_int = lambda x: b3 * x2_sigma * x

        return model, gen_dist, X

    def test_pdp(self):
        model, gen_dist, X = self.create_model_data()

        # pdp monte carlo approximation
        s = 0
        pdp = pythia.PDP(data=X, model=model.predict, axis_limits=gen_dist.axis_limits)
        pdp.fit(features=0)

        pdp_gt = pythia.pdp.PDPGroundTruth(self.pdp_gt, gen_dist.axis_limits)
        pdp_gt.fit(features=0)

        xs = np.linspace(gen_dist.axis_limits[0, 0], gen_dist.axis_limits[1, 0], 100)
        y1 = pdp.eval(feature=0, x=xs)
        y2 = pdp_gt.eval(feature=0, x=xs)

        xs = np.linspace(0, 1, 100)
        assert np.allclose(
            pdp.eval(feature=0, x=xs), pdp_gt.eval(feature=0, x=xs), rtol=0.1, atol=0.1
        )

    def test_dale_fixed_bins(self):
        model, gen_dist, X = self.create_model_data()

        s = 0
        dale = pythia.DALE(data=X, model=model.predict, model_jac=model.jacobian)
        binning = pythia.binning_methods.Fixed(nof_bins=20)
        dale.fit(features=0, binning_method=binning)

        dale_gt = pythia.dale.DALEGroundTruth(
            self.dale_mean_int,
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
        y1 = dale_gt.eval(feature=0, x=xs)
        y2 = dale_gt_bins.eval(feature=0, x=xs)
        y3 = dale.eval(feature=0, x=xs)
        assert np.allclose(y1, y2, rtol=0.1, atol=0.1)
        assert np.allclose(y1, y3, rtol=0.1, atol=0.1)


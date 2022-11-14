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
        b0 = 5
        b1 = 100
        b2 = -100
        b3 = -10
        model = models.LinearWithInteraction3D(b0=b0, b1=b1, b2=b2, b3=b3)

        D = 3
        x1_min = 0
        x1_max = 1
        x2_sigma = .1
        x3_sigma = .1
        gen_dist = dist.Correlated_3D_1(D, x1_min, x1_max, x2_sigma, x3_sigma)

        # generate points
        X = gen_dist.generate(N=10000)

        return model, gen_dist, X

    @pytest.mark.slow
    def test_pdp(self):
        model, gen_dist, X = self.create_model_data()

        # pdp monte carlo approximation
        s = 0
        pdp = fe.PDP(data=X, model=model.predict, axis_limits=gen_dist.axis_limits)

        # pdp numerical approximation
        p_xc = gen_dist.pdf_x2_x3
        pdp_numerical = fe.PDPNumerical(p_xc, model.predict, gen_dist.axis_limits, s=0, D=3)

        xs = np.linspace(0, 1, 5)
        y1 = pdp.eval_unnorm(xs, s=0)
        y2 = pdp_numerical.eval_unnorm(xs, s=0)
        assert np.allclose(y1[1:] - y1[:-1], y2[1:] - y2[:-1], rtol=0.1, atol=0.1)

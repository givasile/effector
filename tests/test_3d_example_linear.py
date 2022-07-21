import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import scipy.stats as sps
import scipy.integrate as integrate
import pytest

np.random.seed(21)

class OpaqueModel:
    def __init__(self, b0, b1, b2, b3):
        self.b0 = b0
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

    def predict(self, x):
        y = self.b0 + self.b1*x[:, 0] + self.b2*x[:, 1] + self.b3*x[:, 0]*x[:, 1] + 100*x[:, 2]
        return y

    def jacobian(self, x):
        df_dx1 = self.b1 + self.b3 * x[:, 1]
        df_dx2 = self.b2 + self.b3 * x[:, 0]
        df_dx3 = np.ones([x.shape[0]])*100
        return np.stack([df_dx1, df_dx2], axis=-1)


class GenerativeDistribution:

    def __init__(self, D, x1_min, x1_max, x2_sigma, x3_sigma):
        self.D = D
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_sigma = x2_sigma
        self.x3_sigma = x3_sigma

        self.axis_limits = np.array([[0, 1],
                                     [-4*x2_sigma, 1 + 4*x2_sigma],
                                     [-4*x3_sigma, 4*x3_sigma]]).T

    def generate(self, N):
        x1 = np.concatenate((np.array([0]),
                             np.random.uniform(0., 1., size=int(N)),
                             np.array([1])))
        x2 = np.random.normal(loc=x1, scale=self.x2_sigma)
        x3 = np.random.normal(loc=np.zeros_like(x1), scale=self.x3_sigma)
        x = np.stack((x1, x2, x3), axis=-1)
        return x

    # define all PDFs
    def pdf_x1(self, x1):
        x1_dist = sps.uniform(loc=self.x1_min, scale=self.x1_max - self.x1_min)
        return x1_dist.pdf(x1)

    def pdf_x2(self, x2):
        x2_dist = sps.norm(loc=.5, scale=self.x2_sigma)
        return x2_dist.pdf(x2)

    def pdf_x3(self, x3):
        x3_dist = sps.norm(loc=0, scale=self.x3_sigma)
        return x3_dist.pdf(x3)

    def pdf_x2_x3(self, x2, x3):
        return self.pdf_x2(x2) * self.pdf_x3(x3)


class TestCase1:
    def create_model_data(self):
        # define model and distribution
        b0 = 5
        b1 = 100
        b2 = -100
        b3 = -10
        model = OpaqueModel(b0=b0, b1=b1, b2=b2, b3=b3)

        D = 2
        x1_min = 0
        x1_max = 1
        x2_sigma = .1
        x3_sigma = .1
        gen_dist = GenerativeDistribution(D, x1_min, x1_max, x2_sigma, x3_sigma)

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

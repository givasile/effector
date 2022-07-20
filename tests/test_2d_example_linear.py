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
        y = self.b0 + self.b1*x[:, 0] + self.b2*x[:, 1] + self.b3*x[:, 0]*x[:, 1]
        return y

    def jacobian(self, x):
        df_dx1 = self.b1 + self.b3 * x[:, 1]
        df_dx2 = self.b2 + self.b3 * x[:, 0]
        return np.stack([df_dx1, df_dx2], axis=-1)

    def plot(self, X):
        x1 = np.linspace(-.5, 1.5, 30)
        x2 = np.linspace(-.5, 1.5, 30)
        XX, YY = np.meshgrid(x1, x2)
        x = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = self.predict(x)
        ZZ = Z.reshape([30, 30])

        plt.figure()
        plt.contourf(XX, YY, ZZ, levels=100)
        if X is not None:
            plt.plot(X[:, 0], X[:, 1], "ro")
        plt.colorbar()
        plt.show(block=True)


class GenerativeDistribution:

    def __init__(self, D, x1_min, x1_max, x2_sigma):
        self.D = D
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_sigma = x2_sigma

        self.axis_limits = np.array([[0, 1], [-4*x2_sigma, 1 + 4 * x2_sigma]]).T

    def generate(self, N):

        x1 = np.concatenate((np.array([0]),
                             np.random.uniform(0., 1., size=int(N)),
                             np.array([1])))
        x2 = np.random.normal(loc=x1, scale=self.x2_sigma)
        x = np.stack((x1, x2), axis=-1)
        return x

    # define all PDFs
    def pdf_x1(self, x1):
        x1_dist = sps.uniform(loc=self.x1_min, scale=self.x1_max - self.x1_min)
        return x1_dist.pdf(x1)

    def pdf_x2(self, x2):
        x2_dist = sps.norm(loc=.5, scale=self.x2_sigma)
        return x2_dist.pdf(x2)

    def pdf_x2_given_x1(self, x2, x1):
        x2_dist = sps.norm(loc=x1, scale=self.x2_sigma)
        return x2_dist.pdf(x2)

    def pdf_x1_x2(self, x1, x2):
        return self.pdf_x2_given_x1(x2, x1) * self.pdf_x1(x1)



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
        x2_sigma = 10.
        gen_dist = GenerativeDistribution(D, x1_min, x1_max, x2_sigma)

        # generate points
        X = gen_dist.generate(N=10000)

        # ground truth
        y_gt_unnorm = lambda x: (b1 + b3*.5)*x
        y_gt = lambda x: y_gt_unnorm(x) - 0.5*(b1 + b3*.5)

        return model, gen_dist, X, y_gt_unnorm, y_gt


    def test_pdp(self):
        model, gen_dist, X, y_gt_unnorm, y_gt = self.create_model_data()


        # pdp monte carlo approximation
        s = 0
        pdp = fe.PDP(data=X, model=model.predict, axis_limits=gen_dist.axis_limits)
        pdp.fit(features=0)

        # pdp numerical approximation
        p_xc = gen_dist.pdf_x2
        pdp_numerical = fe.PDPNumerical(p_xc, model.predict, gen_dist.axis_limits, s=0, D=2)
        pdp_numerical.fit(features=0)

        xs = np.linspace(0, 1, 100)
        assert np.allclose(pdp.eval(xs, s=0), y_gt(xs), rtol=0.1, atol=0.1)
        assert np.allclose(pdp_numerical.eval(xs, s=0), y_gt(xs), rtol=0.1, atol=0.1)


def create_model_data():
    # define model and distribution
    b0 = 5
    b1 = 100
    b2 = -100
    b3 = -10
    model = OpaqueModel(b0=b0, b1=b1, b2=b2, b3=b3)

    D = 2
    x1_min = 0
    x1_max = 1
    x2_sigma = 10.
    gen_dist = GenerativeDistribution(D, x1_min, x1_max, x2_sigma)

    # generate points
    X = gen_dist.generate(N=10000)

    # ground truth
    y_gt_unnorm = lambda x: (b1 + b3*.5)*x
    y_gt = lambda x: y_gt_unnorm(x) - 0.5*(b1 + b3*.5)

    return model, gen_dist, X, y_gt_unnorm, y_gt



# main part
model, gen_dist, X, y_gt_unnorm, y_gt = create_model_data()

s = 0
pdp = fe.PDP(data=X, model=model.predict, axis_limits=gen_dist.axis_limits)
pdp.fit(features=0)

# pdp numerical approximation
p_xc = gen_dist.pdf_x2
pdp_numerical = fe.PDPNumerical(p_xc, model.predict, gen_dist.axis_limits, s=0, D=2)
pdp_numerical.fit(features=0)

xs = np.linspace(0, 1, 100)
assert np.allclose(pdp.eval(xs, s=0), y_gt(xs), rtol=0.1, atol=0.1)
assert np.allclose(pdp_numerical.eval(xs, s=0), y_gt(xs), rtol=0.1, atol=0.1)



# plot all together
plt.figure()
plt.title("PDP")
xs = np.linspace(0, 1, 100)
plt.plot(xs, pdp.eval(xs, s), "b-", label="on dataset (norm)")
plt.plot(xs, pdp_numerical.eval(xs, s), "g-", label="numerical (norm)")
plt.plot(xs, y_gt(xs), "r-", label="gt (norm)")
plt.xlabel("x1")
plt.ylabel("f_PDP")
plt.legend()
plt.show(block=False)

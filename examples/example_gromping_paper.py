import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import scipy.stats as sps


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


class GenerativeDistribution:

    def __init__(self, D):
        self.D = D

    def generate(self, N, noise_level):

        x1 = np.concatenate((np.array([0]),
                             np.random.uniform(0., 1., size=int(N)),
                             np.array([1])))
        x2 = np.random.normal(loc=x1, scale=noise_level)
        x = np.stack((x1, x2), axis=-1)
        return x

    def pdf(self, x):
        # p(x_1)
        x1_dist = sps.uniform(loc=0, scale=1)
        pdf_x1 = x1_dist.pdf(x[:, 0])

        # p(x_2|x_1)
        x2_dist = sps.norm(loc=x[:, 0], scale=1)
        pdf_x2_given_x1 = x2_dist.pdf(x[:, 1])

        # p(x) = p(x2|x1)*p(x1)
        pdf_x = pdf_x1 * pdf_x2_given_x1
        return pdf_x

    def pdf_x2_given_x1(self, x, x1):
        x2_dist = sps.norm(loc=x1, scale=1)
        return x2_dist.pdf(x)

    def plot(self, X=None):
        if self.D == 2:
            x1 = np.linspace(-.5, 1.5, 30)
            x2 = np.linspace(-.5, 1.5, 30)
            XX, YY = np.meshgrid(x1, x2)
            x = np.vstack([XX.ravel(), YY.ravel()]).T
            Z = model.predict(x)
            ZZ = Z.reshape([30, 30])

            plt.figure()
            plt.contourf(XX, YY, ZZ, levels=100)
            if X is not None:
                plt.plot(X[:, 0], X[:, 1], "ro")
            plt.colorbar()
            plt.show(block=True)


# define model and distribution
model = OpaqueModel(b0=0, b1=1, b2=1, b3=100)
gen_dist = GenerativeDistribution(D=2)

# generate data
X = gen_dist.generate(N=1000, noise_level=.1)
# gen_dist.plot(X)

# pdp = fe.PDP(data=X, model=model.predict)
# pdp.plot(feature=0)

# est = fe.Estimator(data=X, model=model.predict, model_jac=model.jacobian)
# est.fit()
# est.plot(feature=0)
# plt.figure()

# plt.cont

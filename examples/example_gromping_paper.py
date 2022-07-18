import matplotlib.pyplot as plt
import numpy as np
import feature_effect as fe
import scipy.stats as sps
import scipy.integrate as integrate

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

    def plot_pdf(self, dist="pdf_x1_x2"):
        if dist=="pdf_x1_x2" or dist=="pdf_x2_given_x1":
            x1 = np.linspace(-.5, 1.5, 100)
            x2 = np.linspace(-.5, 1.5, 100)
            XX, YY = np.meshgrid(x1, x2)
            z = []
            for i in range(x2.shape[0]):
                if dist=="pdf_x1_x2":
                    z.append(self.pdf_x1_x2(x1, x2[i]))
                elif dist=="pdf_x2_given_x1":
                    z.append(self.pdf_x2_given_x1(x2[i], x1))
            Z = np.array(z)
            plt.figure()
            plt.contourf(XX, YY, Z, levels=100)
            plt.colorbar()
            if dist=="pdf_x1_x2":
                plt.title("p(x1, x2)")
            elif dist=="pdf_x2_given_x1":
                plt.title("p(x2|x1)")
            plt.xlabel("x1")
            plt.ylabel("x2")
            plt.show(block=False)
        elif dist=="pdf_x1":
            x = np.linspace(-.5, 1.5, 100)
            y = self.pdf_x1(x)
            plt.figure()
            plt.plot(x, y, "b--")
            plt.title("PDF of x1")
            plt.ylabel("p(x1)")
            plt.xlabel("x1")
            plt.show(block=False)
        elif dist=="pdf_x2":
            x = np.linspace(-.5, 1.5, 100)
            y = self.pdf_x2(x)
            plt.figure()
            plt.plot(x, y, "b--")
            plt.title("PDF of x2")
            plt.ylabel("p(x2)")
            plt.xlabel("x2")
            plt.show(block=False)

# define model and distribution
b0 = 0
b1 = 1
b2 = 1
b3 = 100
model = OpaqueModel(b0=b0, b1=b1, b2=b2, b3=b3)

D = 2
x1_min = 0
x1_max = 1
x2_sigma = .2
gen_dist = GenerativeDistribution(D, x1_min, x1_max, x2_sigma)

# # plot distributions
# gen_dist.plot_pdf("pdf_x1_x2")
# gen_dist.plot_pdf("pdf_x2_given_x1")
# gen_dist.plot_pdf("pdf_x1")
# gen_dist.plot_pdf("pdf_x2")

# generate points
X = gen_dist.generate(N=100)
# model.plot(X)


#
#
#

s = 0
pdp = fe.PDP(data=X, model=model.predict)
# pdp.plot(s)

p_xc = gen_dist.pdf_x2
pdp_1 = fe.PDPNumerical(p_xc, model.predict, D=2, start=0, stop=1)
# pdp_1.plot(s)


# y_gt = pdp.eval_gt(xs, func=lambda x: (b1 + b3*.5)*xs)


plt.figure()
plt.title("PDP")
# plt.plot(xs, y_gt, "r-", label="closed_form")
# plt.plot(xs, y_num, "g--", label="numerical approx")
xs = np.linspace(0, 1, 100)
plt.plot(xs, pdp.eval_unnorm(xs, s), "b--", label="on dataset (unnorm)")
plt.plot(xs, pdp.eval(xs, s), "b-", label="on dataset (norm)")
plt.plot(xs, pdp_1.eval_unnorm(xs, s), "r--", label="numerical (unnorm)")
plt.plot(xs, pdp.eval(xs, s), "r-", label="numerical (norm)")
plt.xlabel("x1")
plt.ylabel("f_PDP")
plt.legend()
plt.show(block=False)

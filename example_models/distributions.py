import numpy as np
import scipy.stats as sps


class GenerativeBase:
    def __init__(self):
        raise NotImplementedError

    def generate(self, N: int):
        raise NotImplementedError


class Correlated1(GenerativeBase):
    """
    x1 ~ U(x1_min, x1_max)
    x2 ~ x1 + N(0, x2_sigma)
    """
    def __init__(self, D, x1_min, x1_max, x2_sigma):
        self.D = D
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_sigma = x2_sigma

        self.axis_limits = np.array([[0, 1], [-4*x2_sigma, 1 + 4 * x2_sigma]]).T

    def generate(self, N):

        x1 = np.concatenate((np.array([self.x1_min]),
                             np.random.uniform(0., 1., size=int(N)),
                             np.array([self.x1_max])))
        x2 = np.random.normal(loc=x1, scale=self.x2_sigma)
        x = np.stack((x1, x2), axis=-1)
        return x

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


class Uncorrelated1(GenerativeBase):
    """
    x1 ~ U(x1_min, x1_max)
    x2 ~ N(0, x2_sigma)
    """

    def __init__(self, D, x1_min, x1_max, x2_sigma):
        self.D = D
        self.x1_min = x1_min
        self.x1_max = x1_max
        self.x2_sigma = x2_sigma

        self.axis_limits = np.array([[0, 1], [-4*x2_sigma, 4 * x2_sigma]]).T

    def generate(self, N):
        x1 = np.concatenate((np.array([0]),
                             np.random.uniform(0., 1., size=int(N)),
                             np.array([1])))
        x2 = np.random.normal(loc=np.zeros_like(x1), scale=self.x2_sigma)
        x = np.stack((x1, x2), axis=-1)
        return x

    def pdf_x2(self, x2):
        x2_dist = sps.norm(loc=0, scale=self.x2_sigma)
        return x2_dist.pdf(x2)


class Correlated_3D_1(GenerativeBase):
    """
    x1 ~ U(x1_min, x1_max)
    x2 ~ N(x1, x2_sigma)
    x3 ~ N(0, x3_sigma)
    """
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


class RepidSimpleDist(GenerativeBase):
    """
    x1 ~ U(-1, 1)
    x2 ~ U(-1, 1)
    """

    def __init__(self):
        self.D = 2
        self.axis_limits = np.array([[-1, 1], [-1, 1], [0, 1]]).T

    def generate(self, N):
        x1 = np.concatenate((np.array([0]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x2 = np.concatenate((np.array([0]),
                             np.random.uniform(-1, 1., size=int(N-2)),
                             np.array([1])))
        x3 = np.random.choice([0, 1], int(N), p=[0.5, 0.5])

        x = np.stack((x1, x2, x3), axis=-1)
        return x
import os
import sys
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mdale.dale as dale
import mdale.utils as utils
import pandas as pd
import PyALE

# set global seed
np.random.seed(21)


class TestExample:
    """
    Artificial Example 1 of the paper, the goal is to:
    (a) compare how pdp, m-plots, ALE compute the feature effect
    (b) show that ALE plots provide the correct estimation
    (c) show the equivalence between ALE and DALE with dense sampling

    Black box function;
    y = 1 - x1 - x2    if x1 + x2 <= 1
    y = 0              otherwise
    """

    @staticmethod
    def generate_samples(N: int) -> np.array:
        """Generates N samples

        :param N: nof samples
        :returns: (N, 2)

        """
        x1 = np.random.uniform(size=(N))
        x2 = x1 + np.random.normal(size=(N))*0
        return np.stack([x1, x2]).T

    @staticmethod
    def f(x: np.array) -> np.array:
        """Black box function;
        y = 1 - x1 - x2    if x1 + x2 <= 1
        y = 0              otherwise

        :param x: (N, 2)
        :returns: (N,)

        """
        y = 1 - x[:,0] - x[:,1]
        y[x[:,0] + x[:,1] > 1] = 0
        return y

    @staticmethod
    def f_der(x):
        """Jacobian matrix on f

        :param x: (N,2)
        :returns: (N,2)

        """
        y = - np.ones_like(x)
        y[x[:,0] + x[:,1] > 1, :] = 0
        return y

    @staticmethod
    def pdp(x):
        """Ground-truth PDP

        :param x: (N,)
        :returns: (N,)

        """
        return (1 - x)**2 / 2

    @staticmethod
    def mplot(x):
        """Ground-truth M-Plot

        :param x: (N,)
        :returns: (N,)

        """
        y = 1 - 2*x
        y[x > 0.5] = 0
        return y

    @staticmethod
    def ale(x):
        """Ground-truth ALE

        :param x: (N,)
        :returns: (N,)

        """
        y = -x + 0.375
        y[x>0.5] = - .125
        return y

    def test_pdp(self):
        N = 1000
        samples = self.generate_samples(N)

        x = np.linspace(0, 1, 1000)
        y_pred = np.array([utils.pdp(self.f, samples, xx, i=0) for xx in x])
        y_gt = self.pdp(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

    def test_mplot(self):
        N = 1000
        K = 100
        samples = self.generate_samples(N)
        tau = (np.max(samples) - np.min(samples)) / K

        x = np.linspace(0, 1, 1000)
        y_pred = np.array([utils.mplot(self.f, samples, xx, i=0, tau=tau) for xx in x])
        y_gt = self.mplot(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

    def test_ale(self):
        N = 1000
        K = 100
        samples = self.generate_samples(N)
        tau = (np.max(samples) - np.min(samples)) / K

        x = np.linspace(0, 1, 1000)
        x, y = utils.ale(samples, self.f, s=0, K=K)
        # y_pred = np.array([utils.mplot(self.f, samples, xx, i=0, tau=tau) for xx in x])
        # y_gt = self.mplot(x)
        # np.allclose(y_pred, y_gt, atol=1.e-3)

    def test_dale(self):
        N = 1000
        K = 100
        samples = self.generate_samples(N)
        tau = (np.max(samples) - np.min(samples)) / K
        X_der = self.f_der(samples)

        # dale
        x = np.linspace(0, 1, 1000)

        # feature 1
        f1 = dale.create_dale_function(samples, X_der, s=0, K=K)
        y_pred = f1(x)
        y_gt = self.ale(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

        # feature 2
        f2 = dale.create_dale_function(samples, X_der, s=1, K=K)
        y_pred = f2(x)
        y_gt = self.ale(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

    def test_dale_2(self):
        N = 1000
        K = 100
        samples = self.generate_samples(N)
        tau = (np.max(samples) - np.min(samples)) / K
        X_der = self.f_der(samples)

        # dale
        x = np.linspace(0, 1, 1000)

        # feature 1
        y_pred = dale.compute_dale(x, s=0, k=K, points=samples, effects=X_der)
        y_gt = self.ale(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

        # feature 2
        y_pred = dale.compute_dale(x, s=1, k=K, points=samples, effects=X_der)
        y_gt = self.ale(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)


# # set global seed
# np.random.seed(21)

# test_example = TestExample()
# X = test_example.generate_samples(N = 1000)
# X_df = pd.DataFrame(X, columns=["feat_" + str(i) for i in range(X.shape[-1])])
# class model():
#     def __init__(self, f, X_df):
#         self.predict = self.func
#         self.f_bb = f

#     def func(self, X_df):
#         return self.f_bb(X_df.to_numpy())

# model_bb = model(test_example.f, X_df)
# s = 0
# feature_type = "auto"
# K = 50
# ale_computation = PyALE.ale(X_df, model_bb, feature= ["feat_" + str(s)],
#                             feature_type=feature_type,
#                             grid_size=K,
#                             plot=False)

# K = 100
# # dale
# x = np.linspace(0, 1, 1000)
# X_der = test_example.f_der(X)

# # feature 1
# f1 = ale2.create_ale_gradients(X, X_der, s=0, K=K)
# y_pred = f1(x)
# y_gt = test_example.ale(x)
# np.allclose(y_pred, y_gt, atol=1.e-2)


# # ale2.plot(X, X_der, s=0, K=K)

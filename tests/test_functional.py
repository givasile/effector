import numpy as np
import feature_effect as fe

# set global seed
np.random.seed(21)


class TestExample:
    """Simple example, where ground-truth can be computed in closed-form.

    Notes
    -----
    The black-box function is

    .. math:: x_1 + x_2 \leq 1

    then

    .. math:: y = y = 1 - x_1 - x_2

    otherwise

    .. math:: 0

    """

    @staticmethod
    def generate_samples(n: int, seed: int = None) -> np.array:
        """Generate N samples

        Parameters
        ----------
        n: int
          nof samples
        seed: int or None
          seed for generating samples

        Returns
        -------
        y: ndarray, shape: [N,2]
          the samples
        """
        if seed is not None:
            np.random.seed(seed)

        x1 = np.random.uniform(size=n)
        x2 = x1
        return np.stack([x1, x2]).T

    @staticmethod
    def f(x: np.array) -> np.array:
        """Evaluate the black-box function;

        Parameters
        ----------
        x: ndarray
          array with the points to evaluate

        Returns
        -------
        y: ndarray with shape (N,)
         array with function evaluation on x
          
        """
        
        y = 1 - x[:, 0] - x[:, 1]
        y[x[:, 0] + x[:, 1] > 1] = 0
        return y

    @staticmethod
    def f_der(x):
        """Evaluate the Jacobian of the black-box function.

        :param x: (N,2)
        :returns: (N,2)

        Parameters
        ----------
        x: ndarray with shape (N,2)
          array with the points to evaluate the Jacobian
          
        Returns
        -------
        y: ndarray with shape (N,2)
         array with the Jacobian on points x
        """
        y = - np.ones_like(x)
        y[x[:, 0] + x[:, 1] > 1, :] = 0
        return y

    @staticmethod
    def pdp(x):
        """Ground-truth PDP

        :param x: (N,)
        :returns: (N,)

        Parameters
        ----------
        x: ndarray (N,)
          array the points to evaluate the PDP function

        Returns
        -------
        y: ndarray with shape (N,)
          the PDP effect evaluation

        """
        return (1 - x)**2 / 2

    @staticmethod
    def mplot(x):
        """Ground-truth MPlot

        :param x: (N,)
        :returns: (N,)

        Parameters
        ----------
        x: ndarray (N,)
          array the points to evaluate the MPlot

        Returns
        -------
        y: ndarray with shape (N,)
          the PDP effect evaluation

        """
        y = 1 - 2*x
        y[x > 0.5] = 0
        return y

    @staticmethod
    def ale(x):
        """Ground-truth ALE

        :param x: (N,)
        :returns: (N,)

        Parameters
        ----------
        x: ndarray (N,)
          array the points to evaluate the ALE function

        Returns
        -------
        y: ndarray with shape (N,)
          the ALE effect evaluation

        """
        y = -x + 0.375
        y[x > 0.5] = - .125
        return y

    def test_pdp(self):
        """Test PDP approximation is close, i.e. tolerance=10^-2, to the ground-truth.
        """
        N = 1000
        samples = self.generate_samples(N)

        pdp = fe.PDP(data=samples, model=self.f)
        x = np.linspace(0, 1, 1000)
        y_pred = pdp.eval(x, feature=0)
        y_gt = self.pdp(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

    def test_mplot(self):
        """Test MPlot approximation is close, i.e. tolerance=10^-2, to the ground-truth.
        """
        N = 1000
        K = 100
        samples = self.generate_samples(N)
        tau = (np.max(samples) - np.min(samples)) / K

        mplot = fe.MPlot(data=samples, model=self.f)
        x = np.linspace(0, 1, 1000)
        y_pred = mplot.eval(x, feature=0, tau=tau)
        y_gt = self.mplot(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

    def test_ale_functional(self):
        """Test ALE approximation is close, i.e. tolerance=10^-2, to the ground-truth.
        """
        N = 1000
        K = 100
        samples = self.generate_samples(N)

        x = np.linspace(0, 1, 1000)

        # feature 1
        y_pred = fe.ale(x, data=samples, model=self.f, s=0, k=K)
        y_gt = self.ale(x)
        np.allclose(y_pred, y_gt, atol=1.e-3)

        # feature 2
        y_pred = fe.ale(x, data=samples, model=self.f, s=1, k=K)
        y_gt = self.ale(x)
        np.allclose(y_pred, y_gt, atol=1.e-3)

    def test_ale_class(self):
        # generate data
        N = 1000
        K = 100
        samples = self.generate_samples(N)

        # prediction
        ale = fe.ALE(data=samples, model=self.f)
        ale.fit(features=[0, 1], k=K)

        # feature 1
        x = np.linspace(0, 1, 1000)
        pred = ale.eval(x, s=0)
        gt = self.ale(x)
        assert np.allclose(pred, gt, atol=1.e-2)

        # feature 2
        pred = ale.eval(x, s=1)
        gt = self.ale(x)
        assert np.allclose(pred, gt, atol=1.e-2)

    def test_dale_functional(self):
        N = 1000
        K = 100
        samples = self.generate_samples(N)
        X_der = self.f_der(samples)

        # dale
        x = np.linspace(0, 1, 1000)

        # feature 1
        y_pred, _ = fe.dale(x, points=samples, point_effects=X_der, s=0, k=K)
        y_gt = self.ale(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

        # feature 2
        y_pred, _ = fe.dale(x, points=samples, point_effects=X_der, s=1, k=K)
        y_gt = self.ale(x)
        assert np.allclose(y_pred, y_gt, atol=1.e-2)

    def test_dale_class(self):
        # generate data
        N = 1000
        K = 100
        samples = self.generate_samples(N)
        # X_der = self.f_der(samples)

        # prediction
        dale = fe.DALE(data=samples, model=self.f, model_jac=self.f_der)
        dale.fit(features=[0, 1], k=K)

        # feature 1
        x = np.linspace(0, 1, 1000)
        pred, _ = dale.eval(x, s=0)
        gt = self.ale(x)
        assert np.allclose(pred, gt, atol=1.e-2)

        # feature 2
        pred, _ = dale.eval(x, s=1)
        gt = self.ale(x)
        assert np.allclose(pred, gt, atol=1.e-2)


if __name__ == "__main__":
    test_example = TestExample()
    test_example.test_pdp()
    test_example.test_mplot()
    test_example.test_ale_functional()
    test_example.test_ale_class()
    test_example.test_dale_functional()
    test_example.test_dale_class()

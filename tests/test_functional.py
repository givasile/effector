import matplotlib.pyplot as plt
import numpy as np
import pythia as fe

# set global seed
import pythia.binning_methods

np.random.seed(21)


class TestExample1:
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

    atol = 1.0e-2
    n = 100000
    k = 100

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
        y = -np.ones_like(x)
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
        return (1 - x) ** 2 / 2

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
        y = 1 - 2 * x
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
        y[x > 0.5] = -0.125
        return y

    def test_pdp(self):
        """Test PDP approximation is close, i.e. tolerance=10^-2, to the ground-truth."""
        n = 10000
        samples = self.generate_samples(n)

        pdp = fe.PDP(data=samples, model=self.f)
        x = np.linspace(0, 1, 1000)
        y_pred = pdp._eval_unnorm(x, s=0)
        y_gt = self.pdp(x)
        assert np.allclose(y_pred, y_gt, atol=1.0e-2)

    def test_mplot(self):
        """Test MPlot approximation is close, i.e. tolerance=10^-2, to the ground-truth."""
        samples = self.generate_samples(self.n)
        tau = (np.max(samples) - np.min(samples)) / self.k

        mplot = fe.MPlot(data=samples, model=self.f)
        x = np.linspace(0, 1, 1000)
        y_pred = mplot.eval(x, feature=0, tau=tau)
        y_gt = self.mplot(x)
        assert np.allclose(y_pred, y_gt, atol=1.0e-2)

    def test_dale(self):
        # generate data
        samples = self.generate_samples(self.n)

        # DALE - fixed size
        dale = fe.DALE(data=samples, model=self.f, model_jac=self.f_der)
        binning = pythia.binning_methods.Fixed(nof_bins=self.k)
        dale.fit(binning_method=binning)

        x = np.linspace(0, 1, 1000)
        pred = dale.eval(x, feature=0)
        gt = self.ale(x)
        assert np.allclose(pred, gt, atol=1.0e-2)

        pred = dale.eval(x, feature=1)
        gt = self.ale(x)
        assert np.allclose(pred, gt, atol=1.0e-2)

        # DALE variable-size
        binning = pythia.binning_methods.DynamicProgramming(max_nof_bins=30, min_points_per_bin=10)
        dale.fit(binning_method=binning)

        x = np.linspace(0, 1, 1000)
        pred = dale.eval(x, feature=0)
        gt = self.ale(x)
        assert np.allclose(pred, gt, atol=1.0e-2)

        pred = dale.eval(x, feature=1)
        gt = self.ale(x)
        assert np.allclose(pred, gt, atol=1.0e-2)


class TestExample2:
    """Simple example, where ground-truth can be computed in closed-form.

    Notes
    -----
    The black-box function is

    .. math:: f(x_1, x_2) = x_1^2 + x_1^2x_2

    """

    atol = 1.0e-2
    n = 100000
    k = 100

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
        x2 = np.random.normal(size=n)
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

        return x[:, 0] ** 2 + x[:, 0] ** 2 * x[:, 1]

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

        return np.stack([2 * x[:, 0] * (1 + x[:, 1]), x[:, 0] ** 2], axis=-1)

    @staticmethod
    def ale_1(x):
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
        z = 1 / 3
        return x**2 - z

    @staticmethod
    def ale_2(x):
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
        z = 0
        return x / 3 - z

    # def test_dale(self):
    #     # generate data
    #     samples = self.generate_samples(self.n)

    #     # prediction
    #     dale = fe.DALE(data=samples, model=self.f, model_jac=self.f_der)
    #     dale.fit(alg_params={"nof_bins": self.k})

    #     x = np.linspace(0, 1, 1000)
    #     pred = dale.eval(x, s=0)
    #     gt = self.ale_1(x)
    #     assert np.allclose(pred, gt, atol=self.atol)

    #     pred = dale.eval(x, s=1)
    #     gt = self.ale_2(x)
    #     assert np.allclose(pred, gt, atol=self.atol)

    # # DALE variable-size
    # dale.fit(method="variable-size", alg_params={"max_nof_bins": 30, "min_points_per_bin": 10})

    # x = np.linspace(0, 1, 1000)
    # pred, _, _ = dale.eval(x, s=0)
    # gt = self.ale_1(x)
    # assert np.allclose(pred, gt, atol=1.e-2)

    # pred, _, _ = dale.eval(x, s=1)
    # gt = self.ale_2(x)
    # assert np.allclose(pred, gt, atol=1.e-2)

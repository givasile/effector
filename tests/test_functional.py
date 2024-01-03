import numpy as np
import effector
import effector.bin_splitting


np.random.seed(21)


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

        x1 = np.concatenate((np.array([0]),
                             np.random.uniform(0, 1, size=int(n-2)),
                             np.array([1])))
        x2 = np.concatenate((np.array([0]),
                             np.random.uniform(0, 1, size=int(n-2)),
                             np.array([1])))
        return np.stack([x1, x2]).T

    @staticmethod
    def f(x: np.array) -> np.array:
        return 3 + 2 * x[:, 0] - 4 * x[:, 1]

    @staticmethod
    def f_der(x):
        return np.stack([np.ones(x.shape[0]) * 2, np.ones(x.shape[0]) * -4], axis=-1)

    @staticmethod
    def ale_1(x):
        return 2*x - 1

    @staticmethod
    def ale_2(x):
        return -4*x + 2

    def test_dale(self):
        # generate data
        samples = self.generate_samples(self.n)

        # prediction
        dale = effector.RHALE(data=samples, model=self.f, model_jac=self.f_der)
        fixed = effector.binning_methods.Fixed(nof_bins=self.k, min_points_per_bin=0)
        dale.fit(features="all", binning_method=fixed, centering=True)

        x = np.linspace(0, 1, 1000)
        pred = dale.eval(feature=0, xs=x, centering=True)
        gt = self.ale_1(x)
        assert np.allclose(pred, gt, atol=self.atol)

        pred = dale.eval(feature=1, xs=x, centering=True)
        gt = self.ale_2(x)
        assert np.allclose(pred, gt, atol=self.atol)

        # DALE variable-size
        dp = effector.binning_methods.DynamicProgramming(max_nof_bins=20, min_points_per_bin=10, cat_limit=1)
        dale.fit(binning_method=dp, centering=True)

        pred = dale.eval(feature=0, xs=x, centering=True)
        gt = self.ale_1(x)
        assert np.allclose(pred, gt, atol=1.e-2)

        pred = dale.eval(feature=1, xs=x, centering=True)
        gt = self.ale_2(x)
        assert np.allclose(pred, gt, atol=1.e-2)


class TestBinEstimation:
    """
    Tests only whether the solution is valid, not if it is the optimal.
    """

    @staticmethod
    def model(x, par):
        """f(x1, x2) = a + b*x1 + x1x2"""
        ind_1 = np.logical_and(x[:, 0] >= par[0]["from"], x[:, 0] < par[0]["to"])
        ind_2 = np.logical_and(x[:, 0] >= par[1]["from"], x[:, 0] < par[1]["to"])
        ind_3 = np.logical_and(x[:, 0] >= par[2]["from"], x[:, 0] < par[2]["to"])
        ind_4 = np.logical_and(x[:, 0] >= par[3]["from"], x[:, 0] <= par[3]["to"])

        y = par[0]["b"] * x[:, 0] + par[0]["a"] + x[:, 0] * x[:, 1]
        y[ind_2] = (
            par[1]["b"] * (x[ind_2, 0] - par[1]["from"])
            + par[1]["a"]
            + x[ind_2, 0] * x[ind_2, 1]
        )
        y[ind_3] = (
            par[2]["b"] * (x[ind_3, 0] - par[2]["from"])
            + par[2]["a"]
            + x[ind_3, 0] * x[ind_3, 1]
        )
        y[ind_4] = (
            par[3]["b"] * (x[ind_4, 0] - par[3]["from"])
            + par[3]["a"]
            + x[ind_4, 0] * x[ind_4, 1]
        )

        return y

    @staticmethod
    def model_jac(x, params):
        """df/dx1 = b*x1 + x2
        df/dx2 = x1
        """
        ind_1 = np.logical_and(x[:, 0] >= params[0]["from"], x[:, 0] < params[0]["to"])
        ind_2 = np.logical_and(x[:, 0] >= params[1]["from"], x[:, 0] < params[1]["to"])
        ind_3 = np.logical_and(x[:, 0] >= params[2]["from"], x[:, 0] < params[2]["to"])
        ind_4 = np.logical_and(x[:, 0] >= params[3]["from"], x[:, 0] <= params[3]["to"])

        y = np.ones_like(x)
        y[ind_1, 0] = params[0]["b"] + x[ind_1, 1]
        y[ind_2, 0] = params[1]["b"] + x[ind_2, 1]
        y[ind_3, 0] = params[2]["b"] + x[ind_3, 1]
        y[ind_4, 0] = params[3]["b"] + x[ind_4, 1]

        y[ind_1, 1] = x[ind_1, 0]
        y[ind_2, 1] = x[ind_2, 0]
        y[ind_3, 1] = x[ind_3, 0]
        y[ind_4, 1] = x[ind_4, 0]
        return y

    @staticmethod
    def generate_samples(N, noise_level):
        """x1 ~ U(0,1)
        x2 ~ N(0, noise_level)
        """
        # eps = 1e-03
        # stop = 5
        x1 = np.concatenate(
            (
                np.array([0.0]),
                np.random.uniform(0.0, 1.0, size=int(N - 2)),
                np.array([1.0]),
            )
        )

        x2 = np.random.normal(loc=0, scale=noise_level, size=(int(x1.shape[0])))
        x = np.stack((x1, x2), axis=-1)
        return x

    @staticmethod
    def create_4_data_points():
        # creates 4 points
        x1 = np.array([0.0, 0.2, 0.8, 1.0])
        x2 = np.random.normal(loc=0, scale=0.01, size=(int(x1.shape[0])))
        x = np.stack((x1, x2), axis=-1)

        y_grad = np.array([[10, 10, -10, -10], [10, 10, -10, -10]]).T
        axis_limits = np.stack([x.min(axis=0), x.max(axis=0)])
        return x, y_grad, axis_limits

    def create_many_points(self):
        params = [
            {"a": 0.0, "b": 10, "from": 0.0, "to": 0.25},
            {"a": 3.0, "b": -10.0, "from": 0.25, "to": 0.5},
            {"a": 0.0, "b": 5.0, "from": 0.5, "to": 0.75},
            {"a": 1.0, "b": -5.0, "from": 0.75, "to": 1},
        ]

        N = 1e4
        noise_level = 0
        x = self.generate_samples(N, noise_level)
        y_grad = self.model_jac(x, params)
        axis_limits = np.stack([x.min(axis=0), x.max(axis=0)])
        return x, y_grad, axis_limits

    def test_min_points_2(self):
        """Tests with min_points = 2 => two bins must be created"""
        x, y_grad, axis_limits = self.create_4_data_points()

        # test Greedy
        min_points = 2
        est = effector.bin_splitting.Greedy(
            x, y_grad, feature=0, axis_limits=axis_limits
        )
        limits_greedy = est.find(init_nof_bins=100, discount=.3, min_points=min_points, cat_limit=1)

        assert limits_greedy.size == 3
        assert np.allclose(0, limits_greedy[0])
        assert np.allclose(1.0, limits_greedy[-1])

        # test DP
        min_points = 2
        est = effector.bin_splitting.DP(x, y_grad, feature=0, axis_limits=axis_limits)
        limits_dp = est.find(max_nof_bins=10, min_points=min_points, cat_limit=1)

        assert limits_dp.size == 3
        assert np.allclose(0, limits_dp[0])
        assert 0.2 <= limits_dp[1] <= 0.8
        assert np.allclose(1.0, limits_dp[2])

    def test_min_points_3(self):
        x, y_grad, axis_limits = self.create_4_data_points()
        gt_limits = np.array([0, 1.0])

        min_points = 3
        est = effector.bin_splitting.Greedy(
            x, y_grad, feature=0, axis_limits=axis_limits
        )
        limits_greedy = est.find(init_nof_bins=100, discount=1.05, min_points=min_points, cat_limit=1)
        assert np.allclose(gt_limits, limits_greedy)

        min_points = 3
        est = effector.bin_splitting.DP(x, y_grad, feature=0, axis_limits=axis_limits)
        limits_dp = est.find(max_nof_bins=10, min_points=min_points, cat_limit=1)
        assert np.allclose(gt_limits, limits_dp)

    def test_min_points_4(self):
        x, y_grad, axis_limits = self.create_4_data_points()
        gt_limits = np.array([0, 1.0])

        min_points = 4
        est = effector.bin_splitting.Greedy(
            x, y_grad, feature=0, axis_limits=axis_limits
        )
        limits_greedy = est.find(init_nof_bins=100, discount=1.05, min_points=min_points, cat_limit=1)
        assert np.allclose(gt_limits, limits_greedy)

        min_points = 4
        est = effector.bin_splitting.DP(x, y_grad, feature=0, axis_limits=axis_limits)
        limits_dp = est.find(max_nof_bins=10, min_points=min_points, cat_limit=1)
        assert np.allclose(gt_limits, limits_dp)

    def test_min_points_5(self):
        x, y_grad, axis_limits = self.create_4_data_points()

        min_points = 5
        est = effector.bin_splitting.Greedy(
            x, y_grad, feature=0, axis_limits=axis_limits
        )
        limits_Greedy = est.find(min_points, cat_limit=1)
        assert limits_Greedy is False

        min_points = 5
        est = effector.bin_splitting.DP(x, y_grad, feature=0, axis_limits=axis_limits)
        limits_DP = est.find(min_points=min_points, cat_limit=1)
        assert limits_DP is False

    def test_many_points(self):
        x, y_grad, axis_limits = self.create_many_points()
        tol = 0.05

        # test Greedy
        min_points = 10
        est = effector.bin_splitting.Greedy(
            x, y_grad, feature=0, axis_limits=axis_limits
        )
        limits_greedy = est.find(min_points, cat_limit=1)
        assert (
            np.sum(
                np.logical_and(
                    limits_greedy >= 0 - tol,
                    limits_greedy <= 0 + tol,
                )
            )
            >= 1
        )
        assert (
            np.sum(
                np.logical_and(
                    limits_greedy >= 0.25 - tol,
                    limits_greedy <= 0.25 + tol,
                )
            )
            >= 1
        )
        assert (
            np.sum(
                np.logical_and(
                    limits_greedy >= 0.5 - tol,
                    limits_greedy <= 0.5 + tol,
                )
            )
            >= 1
        )
        assert (
            np.sum(
                np.logical_and(
                    limits_greedy >= 0.75 - tol,
                    limits_greedy <= 0.75 + tol,
                )
            )
            >= 1
        )
        assert (
            np.sum(
                np.logical_and(
                    limits_greedy >= 1.0 - tol,
                    limits_greedy <= 1.0 + tol,
                )
            )
            >= 1
        )

        # test DP
        min_points = 10
        est = effector.bin_splitting.DP(x, y_grad, feature=0, axis_limits=axis_limits)
        limits_dp = est.find(min_points, cat_limit=1)
        assert (
            np.sum(
                np.logical_and(
                    limits_dp >= 0 - tol,
                    limits_dp <= 0 + tol,
                )
            )
            >= 1
        )
        assert (
            np.sum(
                np.logical_and(
                    limits_dp >= 0.25 - tol,
                    limits_dp <= 0.25 + tol,
                )
            )
            >= 1
        )
        assert (
            np.sum(
                np.logical_and(
                    limits_dp >= 0.5 - tol,
                    limits_dp <= 0.5 + tol,
                )
            )
            >= 1
        )
        assert (
            np.sum(
                np.logical_and(
                    limits_dp >= 0.75 - tol,
                    limits_dp <= 0.75 + tol,
                )
            )
            >= 1
        )
        assert (
            np.sum(
                np.logical_and(
                    limits_dp >= 1.0 - tol,
                    limits_dp <= 1.0 + tol,
                )
            )
            >= 1
        )

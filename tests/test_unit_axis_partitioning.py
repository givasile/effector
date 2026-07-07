import numpy as np
import pytest

import effector


class TestBinEstimation:
    """
    Tests only whether the solution is valid, not if it is the optimal.
    """

    @pytest.fixture(autouse=True)
    def _seed(self):
        np.random.seed(21)

    @staticmethod
    def model(x, par):
        """f(x1, x2) = a + b*x1 + x1x2"""
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
        est = effector.axis_partitioning.Greedy(
            init_nof_bins=100, discount=0.3, min_points_per_bin=min_points
        )
        limits_greedy = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])

        assert limits_greedy.size == 3
        assert np.allclose(0, limits_greedy[0])
        assert np.allclose(1.0, limits_greedy[-1])

        # test DP
        min_points = 2
        est = effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=10, min_points_per_bin=min_points
        )
        limits_dp = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])

        assert limits_dp.size == 3
        assert np.allclose(0, limits_dp[0])
        assert 0.2 <= limits_dp[1] <= 0.8
        assert np.allclose(1.0, limits_dp[2])

    def test_min_points_3(self):
        x, y_grad, axis_limits = self.create_4_data_points()
        gt_limits = np.array([0, 1.0])

        min_points = 3
        est = effector.axis_partitioning.Greedy(
            init_nof_bins=100, discount=1.05, min_points_per_bin=min_points
        )
        limits_greedy = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])
        assert np.allclose(gt_limits, limits_greedy)

        min_points = 3
        est = effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=10, min_points_per_bin=min_points
        )
        limits_dp = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])
        assert np.allclose(gt_limits, limits_dp)

    def test_min_points_4(self):
        x, y_grad, axis_limits = self.create_4_data_points()
        gt_limits = np.array([0, 1.0])

        min_points = 4
        est = effector.axis_partitioning.Greedy(
            init_nof_bins=100, discount=1.05, min_points_per_bin=min_points
        )
        limits_greedy = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])
        assert np.allclose(gt_limits, limits_greedy)

        min_points = 4
        est = effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=10, min_points_per_bin=min_points
        )
        limits_dp = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])
        assert np.allclose(gt_limits, limits_dp)

    def test_min_points_5(self):
        x, y_grad, axis_limits = self.create_4_data_points()

        min_points = 5
        est = effector.axis_partitioning.Greedy(
            init_nof_bins=100, discount=1.05, min_points_per_bin=min_points
        )
        limits_greedy = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])
        assert limits_greedy is False

        min_points = 5
        est = effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=10, min_points_per_bin=min_points
        )
        limits_dp = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])
        assert limits_dp is False

    def test_many_points(self):
        x, y_grad, axis_limits = self.create_many_points()
        tol = 0.05

        # test Greedy
        min_points = 10
        est = effector.axis_partitioning.Greedy(
            init_nof_bins=100, discount=1.05, min_points_per_bin=min_points
        )
        limits_greedy = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])
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
        est = effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=10, min_points_per_bin=min_points
        )
        limits_dp = est.find_limits(x[:, 0], y_grad[:, 0], axis_limits[:, 0])
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


class TestFixed:
    """Spec for the Fixed binning method (PLAN II §3.2; B6 territory)."""

    def test_limits_are_exact_linspace(self):
        rng = np.random.default_rng(21)
        x = rng.uniform(0, 1, 100)
        est = effector.axis_partitioning.Fixed(nof_bins=4)
        limits = est.find_limits(x, None, np.array([0.0, 1.0]))
        np.testing.assert_allclose(limits, np.linspace(0.0, 1.0, 5))

    def test_min_points_violation_returns_false(self):
        # 3 points, 4 bins, min 2 per bin: no valid binning exists
        x = np.array([0.05, 0.1, 0.9])
        est = effector.axis_partitioning.Fixed(nof_bins=4, min_points_per_bin=2)
        assert est.find_limits(x, None, np.array([0.0, 1.0])) is False

    def test_single_unique_value_returns_false(self):
        x = np.ones(50) * 0.3
        est = effector.axis_partitioning.Fixed(nof_bins=4)
        assert est.find_limits(x, None, np.array([0.3, 0.3])) is False


class TestConstantEffectMerging:
    """Greedy/DP on constant-effect data must merge to few bins (behavioral pin)."""

    def _data(self):
        rng = np.random.default_rng(21)
        x = np.sort(rng.uniform(0, 1, 1000))
        y_grad = np.ones_like(x) * 5.0  # constant effect: nothing to separate
        return x, y_grad

    def test_greedy_merges(self):
        x, y_grad = self._data()
        est = effector.axis_partitioning.Greedy(
            init_nof_bins=20, min_points_per_bin=2, discount=0.3
        )
        limits = est.find_limits(x, y_grad, np.array([0.0, 1.0]))
        assert limits is not False
        assert len(limits) <= 5  # 20 initial bins collapse to a handful

    def test_dp_merges(self):
        x, y_grad = self._data()
        est = effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=20, min_points_per_bin=2, discount=0.3
        )
        limits = est.find_limits(x, y_grad, np.array([0.0, 1.0]))
        assert limits is not False
        assert len(limits) <= 5


# ---------------------------------------------------------------------------
# Golden oracle + edge-case coverage (PR-1 of the axis_partitioning rework).
# The golden values below are byte-exact outputs captured from the code as of
# this commit; they FREEZE current behavior so the later template/efficiency
# refactors can be proven drift-free with `assert_array_equal` (not allclose).
# ---------------------------------------------------------------------------


def _greedy():
    return effector.axis_partitioning.Greedy(
        init_nof_bins=20, min_points_per_bin=2, discount=0.3
    )


def _dp():
    return effector.axis_partitioning.DynamicProgramming(
        max_nof_bins=20, min_points_per_bin=2, discount=0.3
    )


def _fixed():
    return effector.axis_partitioning.Fixed(nof_bins=4, min_points_per_bin=2)


def _case_4pt():
    x = np.array([0.0, 0.2, 0.8, 1.0])
    g = np.array([10.0, 10.0, -10.0, -10.0])
    return x, g, np.array([0.0, 1.0])


def _case_1k():
    rng = np.random.default_rng(21)
    x = np.sort(rng.uniform(0.0, 1.0, 1000))
    g = np.piecewise(
        x,
        [x < 0.25, (x >= 0.25) & (x < 0.5), (x >= 0.5) & (x < 0.75), x >= 0.75],
        [10.0, -10.0, 5.0, -5.0],
    )
    return x, g, np.array([0.0, 1.0])


def _case_const():
    rng = np.random.default_rng(99)
    x = np.sort(rng.uniform(0.0, 1.0, 1000))
    g = np.ones_like(x) * 5.0
    return x, g, np.array([0.0, 1.0])


def _case_unique():
    x = np.ones(50) * 0.3
    g = np.ones(50) * 4.0
    return x, g, np.array([0.3, 0.3])


_CASES = {
    "4pt": _case_4pt,
    "1k": _case_1k,
    "const": _case_const,
    "unique": _case_unique,
}

# (id, method_factory, case_name, expected) — `False` or the exact edge list.
_GOLDEN = [
    ("greedy-4pt", _greedy, "4pt", [0.0, 0.75, 1.0]),
    ("dp-4pt", _dp, "4pt", [0.0, 0.25, 1.0]),
    ("fixed-4pt", _fixed, "4pt", False),
    ("greedy-1k", _greedy, "1k", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("dp-1k", _dp, "1k", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("fixed-1k", _fixed, "1k", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("greedy-const", _greedy, "const", [0.0, 1.0]),
    ("dp-const", _dp, "const", [0.0, 1.0]),
    ("fixed-const", _fixed, "const", [0.0, 0.25, 0.5, 0.75, 1.0]),
    ("greedy-unique", _greedy, "unique", False),
    ("dp-unique", _dp, "unique", False),
    ("fixed-unique", _fixed, "unique", False),
]


class TestGoldenFindLimits:
    """Byte-exact behavioral freeze across {Greedy, DP, Fixed} × cases."""

    @pytest.mark.parametrize(
        "method_factory, case_name, expected",
        [(m, c, e) for _, m, c, e in _GOLDEN],
        ids=[i for i, _, _, _ in _GOLDEN],
    )
    def test_golden_find_limits(self, method_factory, case_name, expected):
        x, g, axis_limits = _CASES[case_name]()
        data_effect = None if method_factory is _fixed else g
        limits = method_factory().find_limits(x, data_effect, axis_limits)
        if expected is False:
            assert limits is False
        else:
            np.testing.assert_array_equal(limits, np.array(expected))


class TestAxisPartitioningEdgeCases:
    """Gap-fillers for branches the existing suite never exercised."""

    def test_greedy_single_unique_value_returns_false(self):
        # _none_valid_binning cond_1 (len(unique)==1) for Greedy (only Fixed
        # covered this before).
        x = np.ones(50) * 0.3
        g = np.ones(50) * 4.0
        assert _greedy().find_limits(x, g, np.array([0.3, 0.3])) is False

    def test_dp_single_unique_value_returns_false(self):
        x = np.ones(50) * 0.3
        g = np.ones(50) * 4.0
        assert _dp().find_limits(x, g, np.array([0.3, 0.3])) is False

    def test_adapt_for_categorical_rewrites_bin_count_greedy(self):
        # Greedy's candidate-bin knob is a method param (init_nof_bins)
        method = effector.axis_partitioning.Greedy(init_nof_bins=20)
        adapted = effector.axis_partitioning.adapt_for_categorical(method, nof_levels=5)
        assert adapted.params["init_nof_bins"] == 4  # nof_levels - 1
        # deepcopy independence: the original is untouched
        assert method.params["init_nof_bins"] == 20

    def test_adapt_for_categorical_rewrites_bin_count_dp(self):
        # DP's candidate-bin knob is a constraint (max_nof_bins)
        method = effector.axis_partitioning.DynamicProgramming(max_nof_bins=20)
        adapted = effector.axis_partitioning.adapt_for_categorical(method, nof_levels=5)
        assert adapted.constraints.max_nof_bins == 4
        assert method.constraints.max_nof_bins == 20

    def test_axis_limits_none_branch(self):
        # axis_limits=None => xs_min/xs_max derive from data.min()/max().
        x = np.linspace(0.1, 0.9, 100)
        limits = _fixed().find_limits(x, None, None)
        assert limits[0] == x.min()
        assert limits[-1] == x.max()
        np.testing.assert_allclose(limits, np.linspace(0.1, 0.9, 5))

    def test_dp_max_nof_bins_1_returns_single_bin(self):
        # the max_nof_bins==1 shortcut -> 2-edge [xs_min, xs_max].
        rng = np.random.default_rng(7)
        x = np.sort(rng.uniform(0, 1, 500))
        g = np.sin(10 * x)
        est = effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=1, min_points_per_bin=2
        )
        np.testing.assert_array_equal(
            est.find_limits(x, g, np.array([0.0, 1.0])), np.array([0.0, 1.0])
        )

    def test_greedy_last_bin_merge_back(self):
        # zero-variance body keeps bins open; a high-variance, under-filled final
        # cell forces a close at 0.95 whose bin (<min_points) is merged back into
        # the previous one (Greedy L234-235). A naive result would be
        # [0, 0.95, 1.0] with an under-filled last bin.
        body_x = np.linspace(0.0, 0.9, 300)
        x = np.concatenate([body_x, np.array([0.97, 0.99])])
        g = np.concatenate([np.zeros(300), np.array([50.0, -50.0])])
        est = effector.axis_partitioning.Greedy(
            init_nof_bins=20, min_points_per_bin=5, discount=0.3
        )
        limits = est.find_limits(x, g, np.array([0.0, 1.0]))
        np.testing.assert_array_equal(limits, np.array([0.0, 1.0]))

    def test_greedy_dp_agree_on_clean_step(self):
        # A single sharp step at 0.5: both optimizers must recover the same split.
        rng = np.random.default_rng(3)
        x = np.sort(rng.uniform(0, 1, 2000))
        g = np.where(x < 0.5, 8.0, -8.0)
        ax = np.array([0.0, 1.0])
        gr = _greedy().find_limits(x, g, ax)
        dp = _dp().find_limits(x, g, ax)
        np.testing.assert_array_equal(gr, dp)
        np.testing.assert_array_equal(gr, np.array([0.0, 0.5, 1.0]))

    def test_fixed_min_points_none_no_longer_raises(self):
        # AP-3: Fixed(min_points_per_bin=None) used to raise TypeError in
        # _none_valid_binning (`size < None`). Now None == "no minimum" and it
        # returns the exact uniform grid.
        x = np.linspace(0.0, 1.0, 100)
        est = effector.axis_partitioning.Fixed(nof_bins=4, min_points_per_bin=None)
        limits = est.find_limits(x, None, np.array([0.0, 1.0]))
        np.testing.assert_allclose(limits, np.linspace(0.0, 1.0, 5))
        assert est.no_binning_reason is None


class TestNoBinningReason:
    """The `False` outcome carries a machine-readable reason (folded into PR-2)."""

    _Reason = effector.axis_partitioning.NoBinningReason

    def test_greedy_single_unique_reason(self):
        est = _greedy()
        assert (
            est.find_limits(np.ones(50) * 0.3, np.ones(50), np.array([0.3, 0.3]))
            is False
        )
        assert est.no_binning_reason is self._Reason.SINGLE_UNIQUE_VALUE

    def test_dp_single_unique_reason(self):
        est = _dp()
        assert (
            est.find_limits(np.ones(50) * 0.3, np.ones(50), np.array([0.3, 0.3]))
            is False
        )
        assert est.no_binning_reason is self._Reason.SINGLE_UNIQUE_VALUE

    def test_greedy_too_few_points_reason(self):
        # 4 points, min_points=5 -> data.size < min_points
        x, g, ax = _case_4pt()
        est = effector.axis_partitioning.Greedy(init_nof_bins=100, min_points_per_bin=5)
        assert est.find_limits(x, g, ax) is False
        assert est.no_binning_reason is self._Reason.TOO_FEW_POINTS

    def test_dp_too_few_points_reason(self):
        x, g, ax = _case_4pt()
        est = effector.axis_partitioning.DynamicProgramming(
            max_nof_bins=10, min_points_per_bin=5
        )
        assert est.find_limits(x, g, ax) is False
        assert est.no_binning_reason is self._Reason.TOO_FEW_POINTS

    def test_fixed_grid_underfilled_reason(self):
        # 3 points, 4 fixed bins, min 2 per bin -> a bin is under-filled
        x = np.array([0.05, 0.1, 0.9])
        est = effector.axis_partitioning.Fixed(nof_bins=4, min_points_per_bin=2)
        assert est.find_limits(x, None, np.array([0.0, 1.0])) is False
        assert est.no_binning_reason is self._Reason.FIXED_GRID_UNDERFILLED

    def test_reason_resets_on_success(self):
        # a failing call then a succeeding call on the same instance: no stale reason
        est = _fixed()
        est.find_limits(np.ones(50) * 0.3, None, np.array([0.3, 0.3]))
        assert est.no_binning_reason is not None
        est.find_limits(np.linspace(0, 1, 100), None, np.array([0.0, 1.0]))
        assert est.no_binning_reason is None

    def test_raise_if_no_binning_surfaces_reason(self):
        import effector.utils as utils

        est = _greedy()
        limits = est.find_limits(np.ones(50) * 0.3, np.ones(50), np.array([0.3, 0.3]))
        with pytest.raises(ValueError, match="all points share a single value"):
            utils.raise_if_no_binning(limits, feature=0, binning_method=est)

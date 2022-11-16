import typing

import numpy as np
import pythia.utils as utils
import pythia.utils_integrate as integrate
import matplotlib.pyplot as plt

# global (script-level) variables
big_M = 1.0e10


class BinBase:
    big_M = big_M

    def __init__(self,
                 feature: int,
                 xs_min: float,
                 xs_max: float,
                 data: typing.Union[None, np.ndarray],
                 data_effect: typing.Union[None, np.ndarray],
                 mu: typing.Union[None, callable],
                 sigma: typing.Union[None, callable],
                 axis_limits: typing.Union[None, np.ndarray] = None
                 ):
        """Initializer.

        Parameters
        ----------
        feature: feature index
        xs_min: min value on axis
        xs_max: max value on axis
        data: np.ndarray with X if binning is based on data, else, None
        data_effect: np.ndarray with dy/dX if binning is based on data, else, None
        mu: mu(xs) function
        sigma: sigma(xs) function if binning
        axis_limits: np.ndarray (2, D) or None, if it given it gets priority over xs_min, xs_max
        """
        # set immediately
        self.feature: int = feature
        self.xs_min: float = xs_min
        self.xs_max: float = xs_max

        self.data = data
        self.data_effect = data_effect
        self.mu = mu
        self.sigma = sigma
        self.axis_limits = axis_limits

        # set after execution of method `find`

        # min_points:
        # None -> .find() not executed or we don't care about min_points,
        # int -> min points per bin
        self.min_points: typing.Union[None, int] = None

        # limits:
        #  - None -> .find() not executed yet,
        #  - False -> .find() executed and didn't find acceptable solution
        #  - np.ndarray -> the limits
        self.limits: typing.Union[None, False, np.ndarray] = None

    def _bin_loss(self, start: float, stop: float):
        """Cost of creating the bin with limits [start, stop].

        If the bin contains less points than the specified min_points (min_points is not None)
        then big_M is passed as return value.

        Returns:
            cost of creating the particular bin
        """
        return NotImplementedError

    def _bin_valid(self, start: float, stop: float):
        """Whether the bin is valid.

        Returns a boolean value
        """
        return NotImplementedError

    def _none_bin_possible(self):
        """True, if non bin is possible

        Returns a boolean value
        """
        return NotImplementedError

    def _one_bin_possible(self):
        """True, if only one bin is possible

        Returns a boolean value
        """
        return NotImplementedError

    def find(self, *args):
        """ Finds the optimal bins
        If it is possible to find a set of optimal bins, it returns the limits as np.ndarray
        If it is not, returns False

        """
        return NotImplementedError

    def plot(self, s=0, block=False):
        assert self.limits is not None
        limits = self.limits

        plt.figure()
        plt.title("Bin splitting for feature %d" % (s + 1))
        if self.data is not None:
            xs = self.data[:, self.feature]
            dy_dxs = self.data_effect[:, self.feature]
            plt.plot(xs, dy_dxs, "bo", label="local effects")
        elif self.mu is not None:
            xs = np.linspace(self.xs_min, self.xs_max, 1000)
            dy_dxs = self.mu(xs)
            plt.plot(xs, dy_dxs, "b-", label="mu(x)")

        y_min = np.min(dy_dxs)
        y_max = np.max(dy_dxs)
        plt.vlines(limits, ymin=y_min, ymax=y_max, linestyles="dashed", label="bins")
        plt.xlabel("x_%d" % (s+1))
        plt.ylabel("dy/dx_%d" % (s+1))
        plt.legend()
        plt.show(block=block)


class GreedyBase(BinBase):
    def __init__(self,
                 feature: int,
                 xs_min: float,
                 xs_max: float,
                 data,
                 data_effect,
                 mu,
                 sigma,
                 axis_limits):
        super(GreedyBase, self).__init__(feature, xs_min, xs_max, data, data_effect, mu, sigma, axis_limits)

    def find(self,
             min_points: typing.Union[None, int] = 10,
             n_max: int = 100,
             fact: float = 1.05) -> typing.Union[np.ndarray, bool]:
        """Finds the optimal bins using the Greedy method.

        Parameters
        ----------
        min_points: min points per bin
        n_max: initial (maximum) number of bins, before merging
        fact: factor for deciding to close the window or not

        Returns
        -------
        (K, ) np.ndarray if a solution is feasible, False otherwise
        """
        assert (
            min_points >= 2
        ), "set min_points > 2. Greedy method needs at least two points per bin to estimate the variance."
        xs_min = self.xs_min
        xs_max = self.xs_max
        self.min_points = min_points

        if self._none_bin_possible():
            self.limits = False
        elif self._one_bin_possible():
            self.limits = np.array([self.xs_min, self.xs_max])
        else:
            # limits with high resolution
            limits, _ = np.linspace(
                xs_min, xs_max, num=n_max + 1, endpoint=True, retstep=True
            )
            # bin merging
            i = 0
            merged_limits = [limits[0]]
            while i < n_max:
                # left limit is always the last item of merged_limits
                left_lim = merged_limits[-1]

                # choose whether to close the bin
                if i == n_max - 1:
                    # if last bin, close it
                    close_bin = True
                else:
                    # bin_1_loss: cost if close in this bin
                    bin_1_loss = self._bin_loss(left_lim, limits[i + 1])
                    bin_1_valid = self._bin_valid(left_lim, limits[i + 1])

                    # bin_2_loss: cost if close after the next bin
                    bin_2_loss = self._bin_loss(left_lim, limits[i + 2])
                    bin_2_valid = self._bin_valid(left_lim, limits[i + 2])

                    if bin_1_valid and bin_2_valid:
                        # if both bins valid
                        if bin_1_loss == 0.0 and bin_2_loss > 0:
                            # if first zero, second positive -> close
                            close_bin = True
                        elif bin_1_loss == 0.0 and bin_2_loss == 0:
                            # if both zero, keep it (we could close as well)
                            close_bin = False
                        else:
                            # if both positive, compare and decide
                            close_bin = False if (bin_2_loss / bin_1_loss <= fact) else True
                    else:
                        # if either invalid, keep open
                        close_bin = False

                if close_bin:
                    merged_limits.append(limits[i + 1])
                i += 1

            # if last bin is without enough points, merge it with the previous
            if not self._bin_valid(merged_limits[-2], merged_limits[-1]):
                merged_limits = merged_limits[:-2] + merged_limits[-1:]

            # store result
            self.limits = np.array(merged_limits)
        return self.limits


class Greedy(GreedyBase):
    """
    Greedy bin splitting based on datapoints, i.e. data, data_effect.
    """
    def __init__(self,
                 data: np.ndarray,
                 data_effect: np.ndarray,
                 feature: int,
                 axis_limits: typing.Union[None, np.ndarray]
                 ):
        """

        Parameters
        ----------
        data: (N, D) np.ndarray: input X
        data_effect: (N, D) np.ndarray, jacobian
        feature: column
        axis_limits: (2, D) np.ndarray or None, if None limits is given by data points
        """
        xs_min = (
            data[:, feature].min() if axis_limits is None else axis_limits[0, feature]
        )
        xs_max = (
            data[:, feature].max() if axis_limits is None else axis_limits[1, feature]
        )
        super(Greedy, self).__init__(feature, xs_min, xs_max, data, data_effect, None, None, axis_limits)

    def _bin_loss(self, start, stop):
        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        _, effect_1 = utils.filter_points_belong_to_bin(
            xs, dy_dxs, np.array([start, stop])
        )
        loss = np.var(effect_1) if effect_1.size >= self.min_points else self.big_M
        return loss

    def _bin_valid(self, start, stop):
        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        _, effect_1 = utils.filter_points_belong_to_bin(
            xs, dy_dxs, np.array([start, stop])
        )
        valid = effect_1.size >= self.min_points
        return valid

    def _none_bin_possible(self):
        dy_dxs = self.data_effect[:, self.feature]
        return dy_dxs.size < self.min_points

    def _one_bin_possible(self):
        dy_dxs = self.data_effect[:, self.feature]
        return self.min_points <= dy_dxs.size < 2 * self.min_points


class GreedyGT(GreedyBase):
    """
    Greedy bin splitting based on callables, i.e. mean, var.
    """

    def __init__(
            self,
            mean: callable,
            var: callable,
            axis_limits: np.ndarray,
            feature: int
    ):
        self.axis_limits: np.ndarray = axis_limits

        xs_min = axis_limits[0, feature]
        xs_max = axis_limits[1, feature]
        super(GreedyGT, self).__init__(feature, xs_min, xs_max, None, None, mean, var, axis_limits)

    def _bin_loss(self, start, stop):
        mu_bin = integrate.integrate_1d_linspace(self.mu, start, stop) / (
            stop - start
        )

        def mean_var(x):
            return (self.mu(x) - mu_bin) ** 2

        var1 = integrate.integrate_1d_linspace(self.sigma, start, stop)
        var1 = var1 / (stop - start)
        var2 = integrate.integrate_1d_linspace(mean_var, start, stop)
        var2 = var2 / (stop - start)
        return var1 + var2

    def _bin_valid(self, start, stop):
        return True

    def _none_bin_possible(self):
        return False

    def _one_bin_possible(self):
        return False


class DPBase(BinBase):
    def __init__(self,
                 feature,
                 xs_min,
                 xs_max,
                 data,
                 data_effect,
                 mu,
                 sigma,
                 axis_limits):

        # self.dx_list = None
        self.matrix = None
        self.argmatrix = None

        super(DPBase, self).__init__(feature, xs_min, xs_max, data, data_effect, mu, sigma, axis_limits)

    def _index_to_position(self, index_start, index_stop, K):
        dx = (self.xs_max - self.xs_min) / K
        start = self.xs_min + index_start * dx
        stop = self.xs_min + index_stop * dx
        return start, stop

    def _cost_of_move(self, index_before, index_next, K):
        """Compute the cost of move.

        Computes the cost for moving from the index of the previous bin (index_before)
        to the index of the next bin (index_next).
        """

        big_M = self.big_M
        if index_before > index_next:
            cost = big_M
        elif index_before == index_next:
            cost = 0
        else:
            start, stop = self._index_to_position(index_before, index_next, K)
            cost = self._bin_loss(start, stop)
        return cost

    def _argmatrix_to_limits(self, K):
        assert self.argmatrix is not None
        argmatrix = self.argmatrix
        dx = (self.xs_max - self.xs_min) / K

        lim_indices = [int(argmatrix[-1, -1])]
        for j in range(K - 2, 0, -1):
            lim_indices.append(int(argmatrix[int(lim_indices[-1]), j]))
        lim_indices.reverse()

        lim_indices.insert(0, 0)
        lim_indices.append(argmatrix.shape[-1])

        # remove identical bins
        lim_indices_1 = []
        before = np.nan
        for i, lim in enumerate(lim_indices):
            if before != lim:
                lim_indices_1.append(lim)
                before = lim

        limits = self.xs_min + np.array(lim_indices_1) * dx
        dx_list = np.array(
            [limits[i + 1] - limits[i] for i in range(limits.shape[0] - 1)]
        )
        return limits, dx_list

    def find(self, min_points: int = 10, k_max: int = 30):
        """

        Parameters
        ----------
        min_points: minimum points per bin
        k_max: maximum number of bins

        Returns
        -------

        """
        self.min_points = min_points
        big_M = self.big_M
        nof_limits = k_max + 1
        nof_bins = k_max

        if self._none_bin_possible():
            self.limits = False
        elif k_max == 1:
            self.limits = np.array([self.xs_min, self.xs_max])
            self.dx_list = np.array([self.xs_max - self.xs_min])
        else:
            # init matrices
            matrix = np.ones((nof_limits, nof_bins)) * big_M
            argmatrix = np.ones((nof_limits, nof_bins)) * np.nan

            # init first bin_index
            bin_index = 0
            for lim_index in range(nof_limits):
                matrix[lim_index, bin_index] = self._cost_of_move(
                    bin_index, lim_index, k_max
                )

            # for all other bins
            for bin_index in range(1, k_max):
                for lim_index_next in range(k_max + 1):
                    # find best solution
                    tmp = []
                    for lim_index_before in range(k_max + 1):
                        tmp.append(
                            matrix[lim_index_before, bin_index - 1]
                            + self._cost_of_move(
                                lim_index_before, lim_index_next, k_max
                            )
                        )
                    # store best solution
                    matrix[lim_index_next, bin_index] = np.min(tmp)
                    argmatrix[lim_index_next, bin_index] = np.argmin(tmp)

            # store solution matrices
            self.matrix = matrix
            self.argmatrix = argmatrix

            # find indices
            self.limits, _ = self._argmatrix_to_limits(k_max)

        return self.limits


class DP(DPBase):
    def __init__(self, data, data_effect, feature, axis_limits):
        xs_min = (
            data[:, feature].min() if axis_limits is None else axis_limits[0, feature]
        )
        xs_max = (
            data[:, feature].max() if axis_limits is None else axis_limits[1, feature]
        )
        self.nof_points = data.shape[0]
        self.feature = feature
        super(DP, self).__init__(feature, xs_min, xs_max, data, data_effect, None, None, axis_limits)

    def _none_bin_possible(self):
        dy_dxs = self.data_effect[:, self.feature]
        return dy_dxs.size < self.min_points

    def _bin_valid(self, start: float, stop: float) -> bool:
        data = self.data[:, self.feature]
        data_effect = self.data_effect[:, self.feature]
        data, data_effect = utils.filter_points_belong_to_bin(
            data, data_effect, np.array([start, stop])
        )
        if data_effect.size < self.min_points:
            valid = False
        else:
            valid = True
        return valid

    def _bin_loss(self, start, stop):
        min_points = self.min_points
        data = self.data[:, self.feature]
        data_effect = self.data_effect[:, self.feature]
        data, data_effect = utils.filter_points_belong_to_bin(
            data, data_effect, np.array([start, stop])
        )

        # compute cost
        if data_effect.size < min_points:
            cost = self.big_M
            cost_var = self.big_M
        else:
            # cost = np.std(data_effect) * (stop-start) / np.sqrt(data_effect.size)
            discount_for_more_points = 1 - 0.3 * (data_effect.size / self.nof_points)
            cost = np.var(data_effect) * (stop - start) * discount_for_more_points
            cost_var = np.var(data_effect)
        return cost


class DPGT(DPBase):
    def __init__(
        self, mean: callable, var: callable, axis_limits: np.ndarray, feature: int
    ):
        self.axis_limits = axis_limits
        xs_min = axis_limits[0, feature]
        xs_max = axis_limits[1, feature]
        super(DPGT, self).__init__(feature, xs_min, xs_max, None, None, mean, var, axis_limits)

    def _none_bin_possible(self):
        return False

    def _bin_valid(self, start, stop):
        return True

    def _bin_loss(self, start, stop):
        mu_bin = integrate.integrate_1d_linspace(self.mu, start, stop) / (
            stop - start
        )
        mean_var = lambda x: (self.mu(x) - mu_bin) ** 2
        var1 = integrate.integrate_1d_linspace(self.sigma, start, stop)
        var1 = var1 / (stop - start)
        var2 = integrate.integrate_1d_linspace(mean_var, start, stop)
        var2 = var2 / (stop - start)
        total_var = var1 + var2

        # cost = np.std(data_effect) * (stop-start) / np.sqrt(data_effect.size)
        bin_length_pcg = (stop - start) / (self.xs_max - self.xs_min)
        discount_for_more_points = 1 - 0.3 * bin_length_pcg
        cost = total_var * (stop - start) * discount_for_more_points
        return cost


class FixedBase(BinBase):
    def __init__(self,
                 feature,
                 xs_min,
                 xs_max,
                 data,
                 data_effect,
                 mu,
                 sigma,
                 axis_limits):
        super(FixedBase, self).__init__(feature, xs_min, xs_max, data, data_effect, mu, sigma, axis_limits)

    def _bin_valid(self, start, stop):
        return NotImplementedError

    def find(self, nof_bins: int, min_points: typing.Union[None, int] = None, enforce_bin_creation: bool = True):
        self.min_points = min_points
        self.K = nof_bins

        if min_points is not None:
            assert (min_points >= 2), "We need at least two points per bin to estimate the variance"

        limits, dx = np.linspace(
            self.xs_min, self.xs_max, num=nof_bins + 1, endpoint=True, retstep=True
        )
        if enforce_bin_creation:
            self.limits = limits
            return limits

        valid_binning = True
        for i in range(nof_bins):
            start = limits[i]
            stop = limits[i + 1]
            if not self._bin_valid(start, stop):
                valid_binning = False

        if not valid_binning:
            self.limits = False
        else:
            self.limits = limits
        return self.limits


class Fixed(FixedBase):
    def __init__(self, data, data_effect, feature, axis_limits):
        xs_min = (
            data[:, feature].min() if axis_limits is None else axis_limits[0, feature]
        )
        xs_max = (
            data[:, feature].max() if axis_limits is None else axis_limits[1, feature]
        )
        super(Fixed, self).__init__(feature, xs_min, xs_max, data, data_effect, None, None, axis_limits)

    def _bin_valid(self, start, stop):
        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        _, effect_1 = utils.filter_points_belong_to_bin(
            xs, dy_dxs, np.array([start, stop])
        )
        valid = effect_1.size >= self.min_points
        return valid

    def _cost_of_bin(self, start, stop):
        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        _, effect_1 = utils.filter_points_belong_to_bin(
            xs, dy_dxs, np.array([start, stop])
        )
        return np.var(effect_1), np.var(effect_1)


class FixedGT(FixedBase):
    def __init__(self, mean: callable, var: callable, axis_limits, feature):
        self.axis_limits = axis_limits
        xs_min = axis_limits[0, feature]
        xs_max = axis_limits[1, feature]
        super(FixedGT, self).__init__(feature, xs_min, xs_max, None, None, mean, var, axis_limits)

    def _bin_valid(self, start, stop):
        return True

    def _cost_of_bin(self, start, stop):
        mu_bin = integrate.integrate_1d_quad(self.mu, start, stop) / (
            stop - start
        )
        mean_var = lambda x: (self.mu(x) - mu_bin) ** 2
        var1 = integrate.integrate_1d_quad(self.sigma, start, stop)
        var1 = var1 / (stop - start)
        var2 = integrate.integrate_1d_quad(mean_var, start, stop)
        var2 = var2 / (stop - start)
        total_var = var1 + var2
        return total_var, total_var

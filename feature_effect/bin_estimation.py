import numpy as np
import feature_effect.utils as utils
import feature_effect.utils_integrate as integrate
import matplotlib.pyplot as plt


class BinEstimator:
    big_M = 1.e+10

    def __init__(self, data, data_effect, feature):
        # set immediately
        self.data = data
        self.data_effect = data_effect
        self.feature = feature

        # will be set after execution
        self.min_points = None
        self.limits = None
        self.statistics = None


    def compute_statistics(self):
        assert self.limits is not None
        assert self.min_points is not None

        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        limits = self.limits
        min_points = self.min_points
        self.statistics = utils.compute_fe_parameters(xs, dy_dxs, limits, min_points)
        return self.statistics


    def plot(self, s=0, block=False):
        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        limits = self.limits

        plt.figure()
        plt.title("Local effect for feature " + str(s+1))
        plt.plot(xs, dy_dxs, "bo", label="local effects")
        if limits is not None:
            plt.vlines(limits, ymin=np.min(dy_dxs), ymax=np.max(dy_dxs))
        plt.show(block=block)



class GreedyBase:
    def __init__(self, feature, xs_min, xs_max):
        self.feature = feature
        self.big_M = 1e9
        self.xs_min = xs_min
        self.xs_max = xs_max

    def bin_loss(self, start, stop):
        return NotImplementedError

    def bin_valid(self, start, stop):
        return NotImplementedError

    def none_bin_possible(self):
        return NotImplementedError

    def one_bin_possible(self):
        return NotImplementedError

    def solve(self, min_points=10, K=1000):
        assert min_points >= 2, "We need at least two points per bin to estimate the variance"
        xs_min = self.xs_min
        xs_max = self.xs_max
        self.min_points = min_points

        # TODO make sure there are enough points to fulfill the constraint
        if self.none_bin_possible():
            self.limits = False
        elif self.one_bin_possible():
            self.limits = np.array([self.xs_min, self.xs_max])
        else:
            # limits with high resolution
            limits, _ = np.linspace(xs_min, xs_max, num=K+1, endpoint=True, retstep=True)
            # bin merging
            i = 0
            merged_limits = [limits[0]]
            while (i < K):
                # left limit is always the last item of merged_limits
                left_lim = merged_limits[-1]

                # choose whether to close the bin
                if i == K - 1:
                    close_bin = True
                else:
                    # compare the added loss from closing or keep open in a greedy manner
                    loss_1 = self.bin_loss(left_lim, limits[i+1])
                    loss_2 = self.bin_loss(left_lim, limits[i+2])

                    if loss_1 == self.big_M or loss_1 == 0:
                        close_bin = False
                    else:
                        close_bin = False if (loss_2 / loss_1 <= 1.05) else True

                if close_bin:
                    merged_limits.append(limits[i+1])
                i += 1

            # if last bin is without enough points, merge it with the previous
            if not self.bin_valid(merged_limits[-2], merged_limits[-1]):
                if len(merged_limits) > 2:
                    merged_limits = merged_limits[:-2] + merged_limits[-1:]

            self.limits = np.array(merged_limits)
        return self.limits


class Greedy(GreedyBase):
    def __init__(self, data, data_effect, feature):
        self.data = data
        self.data_effect = data_effect
        xs_min = data[:, feature].min()
        xs_max = data[:, feature].max()
        super(Greedy, self).__init__(feature, xs_min, xs_max)

    def bin_loss(self, start, stop):
        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        _, effect_1 = utils.filter_points_belong_to_bin(xs, dy_dxs, np.array([start, stop]))
        loss = np.var(effect_1) if effect_1.size >= self.min_points else self.big_M
        return loss

    def bin_valid(self, start, stop):
        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        _, effect_1 = utils.filter_points_belong_to_bin(xs, dy_dxs, np.array([start, stop]))
        valid = effect_1.size >= self.min_points
        return valid

    def none_bin_possible(self):
        dy_dxs = self.data_effect[:, self.feature]
        return dy_dxs.size < self.min_points

    def one_bin_possible(self):
        dy_dxs = self.data_effect[:, self.feature]
        return self.min_points <= dy_dxs.size < 2*self.min_points


class GreedyGroundTruth(GreedyBase):
    def __init__(self, mean: callable, var: callable, axis_limits: np.ndarray, feature: int):
        self.mean = mean
        self.var = var
        self.axis_limits = axis_limits
        xs_min = axis_limits[0, feature]
        xs_max = axis_limits[1, feature]
        super(GreedyGroundTruth, self).__init__(feature, xs_min, xs_max)


    def bin_loss(self, start, stop):
        mu_bin = integrate.normalization_constant_1D(self.mean, start, stop) / (stop-start)
        mean_var = lambda x: (self.mean(x) - mu_bin)**2
        var1 = integrate.normalization_constant_1D(self.var, start, stop)
        var1 = var1 / (stop-start)
        var2 = integrate.normalization_constant_1D(mean_var, start, stop)
        var2 = var2 / (stop-start)
        return var1 + var2

    def bin_valid(self, start, stop):
        return True

    def none_bin_possible(self):
        return False

    def one_bin_possible(self):
        return False



class DPBase:
    def __init__(self, feature, xs_min, xs_max):
        self.big_M = 1.e9
        self.feature = feature
        self.xs_min = xs_min
        self.xs_max = xs_max

        self.limits = None
        self.dx_list = None
        self.matrix = None
        self.argmatrix = None

    def _cost_of_bin(self, start, stop, min_points):
        raise NotImplementedError

    def _none_bin_possible(self):
        return NotImplementedError

    def _index_to_position(self, index_start, index_stop, K):
        dx = (self.xs_max - self.xs_min) / K
        start = self.xs_min + index_start * dx
        stop = self.xs_min + index_stop * dx
        return start, stop

    def _cost_of_move(self, index_before, index_next, min_points, K):
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
            cost = self._cost_of_bin(start, stop, min_points)
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
        dx_list = np.array([limits[i+1] - limits[i] for i in range(limits.shape[0]-1)])
        return limits, dx_list


    def solve(self, min_points=10, K=30):
        self.min_points = min_points
        big_M = self.big_M
        nof_limits = K + 1
        nof_bins = K

        if self._none_bin_possible():
            self.limits = False
        elif K == 1:
            self.limits = np.array([self.xs_min, self.xs_max])
            self.dx_list = np.array([self.xs_max - self.xs_min])
        else:
            # init matrices
            matrix = np.ones((nof_limits, nof_bins)) * big_M
            argmatrix = np.ones((nof_limits, nof_bins)) * np.nan

            # init first bin_index
            bin_index = 0
            for lim_index in range(nof_limits):
                matrix[lim_index, bin_index] = self._cost_of_move(bin_index, lim_index, min_points, K)

            # for all other bins
            for bin_index in range(1, K):
                for lim_index_next in range(K + 1):

                    # find best solution
                    tmp = []
                    for lim_index_before in range(K + 1):
                        tmp.append(matrix[lim_index_before, bin_index - 1] + self._cost_of_move(lim_index_before, lim_index_next, min_points, K))

                    # store best solution
                    matrix[lim_index_next, bin_index] = np.min(tmp)
                    argmatrix[lim_index_next, bin_index] = np.argmin(tmp)

            # store solution matrices
            self.matrix = matrix
            self.argmatrix = argmatrix

            # find indices
            self.limits, self.dx_list = self._argmatrix_to_limits(K)

        return self.limits


class DP(DPBase):
    def __init__(self, data, data_effect, feature):
        self.data = data
        self.data_effect = data_effect
        xs_min = np.min(data[:, feature])
        xs_max = np.max(data[:, feature])
        self.nof_points = data.shape[0]
        self.feature = feature
        super(DP, self).__init__(feature, xs_min, xs_max)

    def _none_bin_possible(self):
        dy_dxs = self.data_effect[:, self.feature]
        return dy_dxs.size < self.min_points

    def _cost_of_bin(self, start, stop, min_points):
        data = self.data[:, self.feature]
        data_effect = self.data_effect[:, self.feature]
        data, data_effect = utils.filter_points_belong_to_bin(data,
                                                              data_effect,
                                                              np.array([start, stop]))

        # compute cost
        if data_effect.size < min_points:
            cost = self.big_M
        else:
            # cost = np.std(data_effect) * (stop-start) / np.sqrt(data_effect.size)
            discount_for_more_points = (1 - .3*(data_effect.size / self.nof_points))
            cost = np.var(data_effect) * (stop-start) * discount_for_more_points
        return cost


class DPGroundTruth(DPBase):
    def __init__(self, mean: callable, var: callable, axis_limits: np.ndarray, feature: int):
        self.mean = mean
        self.var = var
        self.axis_limits = axis_limits
        xs_min = axis_limits[0, feature]
        xs_max = axis_limits[1, feature]
        super(DPGroundTruth, self).__init__(feature, xs_min, xs_max)

    def _none_bin_possible(self):
        return False

    def _cost_of_bin(self, start, stop, min_points):
        mu_bin = integrate.normalization_constant_1D(self.mean, start, stop) / (stop-start)
        mean_var = lambda x: (self.mean(x) - mu_bin)**2
        var1 = integrate.normalization_constant_1D(self.var, start, stop)
        var1 = var1 / (stop-start)
        var2 = integrate.normalization_constant_1D(mean_var, start, stop)
        var2 = var2 / (stop-start)
        total_var = var1 + var2

        # cost = np.std(data_effect) * (stop-start) / np.sqrt(data_effect.size)
        bin_length_pcg = (stop - start) / (self.xs_max - self.xs_min)
        discount_for_more_points = (1 - .3*bin_length_pcg)
        cost = total_var * (stop-start) * discount_for_more_points
        return cost


class FixedSizeBase:
    def __init__(self, feature, xs_min, xs_max):
        self.feature = feature
        self.big_M = 1e9
        self.xs_min = xs_min
        self.xs_max = xs_max

    def bin_valid(self, start, stop):
        return NotImplementedError

    def solve(self, min_points, K):
        self.min_points = min_points
        self.K = K
        assert min_points >= 2, "We need at least two points per bin to estimate the variance"

        limits, dx = np.linspace(self.xs_min, self.xs_max, num=K + 1, endpoint=True, retstep=True)
        valid_binning = True
        for i in range(K):
            start = limits[i]
            stop = limits[i+1]
            if not self.bin_valid(start, stop):
                valid_binning = False

        if not valid_binning:
            self.limits = False
        else:
            self.limits = limits
        return self.limits


class FixedSize(FixedSizeBase):
    def __init__(self, data, data_effect, feature):
        self.data = data
        self.data_effect = data_effect
        xs_min = data[:, feature].min()
        xs_max = data[:, feature].max()
        super(FixedSize, self).__init__(feature, xs_min, xs_max)

    def bin_valid(self, start, stop):
        xs = self.data[:, self.feature]
        dy_dxs = self.data_effect[:, self.feature]
        _, effect_1 = utils.filter_points_belong_to_bin(xs, dy_dxs, np.array([start, stop]))
        valid = effect_1.size >= self.min_points
        return valid


class FixedSizeGT(FixedSizeBase):
    def __init__(self, axis_limits, feature):
        self.axis_limits = axis_limits
        xs_min = axis_limits[0, feature]
        xs_max = axis_limits[1, feature]
        super(FixedSizeGT, self).__init__(feature, xs_min, xs_max)


    def bin_valid(self, start, stop):
        return True

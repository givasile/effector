import numpy as np
import feature_effect.utils as utils


class BinEstimatorDP:
    def __init__(self, data, data_effect, feature, K):
        self.x_min = np.min(data[:, feature])
        self.x_max = np.max(data[:, feature])
        self.K = K
        self.dx = (self.x_max - self.x_min) / K
        self.big_M = 1.e+10
        self.data = data
        self.nof_points = data.shape[0]
        self.data_effect = data_effect
        self.feature = feature

        self.limits = None
        self.dx_list = None
        self.matrix = None
        self.argmatrix = None

    def _cost_of_bin(self, start, stop, min_points):
        data = self.data[:, self.feature]
        data_effect = self.data_effect[:, self.feature]
        data, data_effect = utils.filter_points_belong_to_bin(data,
                                                              data_effect,
                                                              np.array([start, stop]))

        big_cost = 1e+10

        # compute cost
        if data_effect.size < min_points:
            cost = big_cost
        else:
            # cost = np.std(data_effect) * (stop-start) / np.sqrt(data_effect.size)
            discount_for_more_points = (1 - .3*(data_effect.size / self.nof_points))
            cost = np.var(data_effect) * (stop-start) * discount_for_more_points
        return cost

    def _index_to_position(self, index_start, index_stop):
        start = self.x_min + index_start * self.dx
        stop = self.x_min + index_stop * self.dx
        return start, stop

    def _cost_of_move(self, index_before, index_next, min_points):
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
            start, stop = self._index_to_position(index_before, index_next)
            cost = self._cost_of_bin(start, stop, min_points)

        return cost

    def _argmatrix_to_limits(self):
        assert self.argmatrix is not None
        argmatrix = self.argmatrix
        dx = self.dx
        x_min = self.x_min

        lim_indices = [int(argmatrix[-1, -1])]
        for j in range(self.K - 2, 0, -1):
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

        limits = x_min + np.array(lim_indices_1) * dx
        dx_list = np.array([limits[i+1] - limits[i] for i in range(limits.shape[0]-1)])
        return limits, dx_list

    def solve_dp(self, min_points):
        K = self.K
        big_M = self.big_M
        nof_limits = K + 1
        nof_bins = K

        if K == 1:
            self.limits = np.array([self.x_min, self.x_max])
            self.dx_list = np.array([self.x_max - self.x_min])
        else:
            # init matrices
            matrix = np.ones((nof_limits, nof_bins)) * big_M
            argmatrix = np.ones((nof_limits, nof_bins)) * np.nan

            # init first bin_index
            bin_index = 0
            for lim_index in range(nof_limits):
                matrix[lim_index, bin_index] = self._cost_of_move(bin_index, lim_index, min_points)

            # for all other bins
            for bin_index in range(1, K):
                for lim_index_next in range(K + 1):

                    # find best solution
                    tmp = []
                    for lim_index_before in range(K + 1):
                        tmp.append(matrix[lim_index_before, bin_index - 1] + self._cost_of_move(lim_index_before, lim_index_next, min_points))

                    # store best solution
                    matrix[lim_index_next, bin_index] = np.min(tmp)
                    argmatrix[lim_index_next, bin_index] = np.argmin(tmp)

            # store solution matrices
            self.matrix = matrix
            self.argmatrix = argmatrix

            # find indices
            self.limits, self.dx_list = self._argmatrix_to_limits()

        return self.limits

import typing

import numpy as np
import effector.utils as utils
import effector.helpers as helpers
import matplotlib.pyplot as plt


class Base:
    big_M = helpers.BIG_M

    def __init__(
        self,
        name: str,
        method_args: typing.Dict[str, typing.Any],
    ):
        """Initializer.

        Parameters
        ----------
        feature: feature index
        data: np.ndarray with X if binning is based on data, else, None
        data_effect: np.ndarray with dy/dX if binning is based on data, else, None
        axis_limits: np.ndarray (2, D) or None, if None, then axis_limits are set to [xs_min, xs_max]
        """
        self.name = name

        # set immediately
        self.xs_min = None
        self.xs_max = None
        self.data = None
        self.data_effect = None

        # arguments passed to method find
        self.method_args: typing.Dict[str, typing.Any] = method_args

        # set in method find
        self.method_outputs: typing.Dict[str, typing.Any] = {}

        # limits: None if not set, False if no binning is possible, np.ndarray (N, 2) if binning is possible
        self.limits: typing.Union[None, False, np.ndarray] = None

    def _bin_cost(self, start, stop, discount):
        min_points = self.method_args["min_points"]
        nof_points = self.data.shape[0]
        data = self.data
        data_effect = self.data_effect
        data, data_effect = utils.filter_points_in_bin(
            data, data_effect, np.array([start, stop])
        )

        # compute cost
        thres = max(min_points, 2)
        if data_effect.size < thres:
            cost = self.big_M
        else:
            discount_for_more_points = 1 - discount * (data_effect.size / nof_points)
            cost = np.var(data_effect) * (stop - start) * discount_for_more_points
        return cost

    def _bin_valid(self, start, stop):
        """Check if creating a bin with limits [start, stop] is valid.

        Returns:
            Boolean, True if the bin is valid, False otherwise
        """
        min_points = self.method_args["min_points"]
        if min_points is None:
            return True

        xs = self.data
        # dy_dxs = self.data_effect[:, self.feature]
        filtered_points, _ = utils.filter_points_in_bin(
            xs, None, np.array([start, stop])
        )
        valid = filtered_points.size >= min_points
        return valid

    def _none_valid_binning(self):
        """Check if it is impossible to create any binning

        Returns:
            Boolean, True if it is impossible to create any binning, False otherwise
        """
        # if there is only one unique value, then it is impossible to create any binning
        cond_1 = len(np.unique(self.data)) == 1

        # if there are less than min_points, then it is impossible to create any binning
        cond_2 = self.data.size < self.method_args["min_points"]

        # if either is true, then it is impossible to create any binning
        return cond_1 or cond_2

    def _only_one_bin_possible(self):
        """Check if the only possible binning is all points in one bin

        Returns:
            Boolean, True if the only possible binning is all points in one bin, False otherwise
        """
        min_points = self.method_args["min_points"]
        # if xs is categorical, then only one bin is possible
        is_categorical = np.allclose(self.xs_min, self.xs_max)

        # if xs is not categorical, then one bin is possible if there are enough points only in one bin
        dy_dxs = self.data_effect
        enough_for_one_bin = min_points <= dy_dxs.size < 2 * min_points
        return is_categorical or enough_for_one_bin

    # def _is_categorical(self, cat_limit):
    #     """Check if the feature is categorical, i.e. has less than cat_limit unique values, and if so, set the limits
    #
    #     Returns:
    #         Boolean, True if the feature is categorical, False otherwise
    #     """
    #     # if unique values are leq 10, then it is categorical
    #     is_cat = len(np.unique(self.data)) <= cat_limit
    #     # if only one unique value, then it is categorical and set the limits
    #     if len(np.unique(self.data)) == 1:
    #         self.limits = False
    #
    #     if is_cat:
    #         # set unique values as the center of the bins
    #         uniq = np.sort(np.unique(self.data))
    #         dx = [uniq[i + 1] - uniq[i] for i in range(len(uniq) - 1)]
    #         lims = np.array(
    #             [uniq[0] - dx[0] / 2]
    #             + [uniq[i] + dx[i] / 2 for i in range(len(uniq) - 1)]
    #             + [uniq[-1] + dx[-1] / 2]
    #         )
    #
    #         # if all limits are valid, then set them
    #         if np.all(
    #             [self._bin_valid(lims[i], lims[i + 1]) for i in range(len(lims) - 1)]
    #         ):
    #             self.limits = lims
    #         else:
    #             self.limits = False
    #     return is_cat

    def _preprocess_find(self, data, data_effect, axis_limits):
        self.xs_min: float = axis_limits[0] if axis_limits is not None else data.min()
        self.xs_max: float = axis_limits[1] if axis_limits is not None else data.max()
        self.data: np.ndarray = data
        self.data_effect: np.ndarray = data_effect

    def find(self, data, data_effect, axis_limits):
        """Find the optimal binning for the feature."""
        return NotImplementedError

    def plot(self, feature=0, block=False):
        assert self.limits is not None
        assert self.data is not None
        assert self.data_effect is not None

        limits = self.limits

        plt.figure()
        plt.title("Bin splitting for feature %d" % (feature + 1))
        xs = self.data
        dy_dxs = self.data_effect
        plt.plot(xs, dy_dxs, "bo", label="local effects")
        y_min = np.min(dy_dxs)
        y_max = np.max(dy_dxs)
        plt.vlines(limits, ymin=y_min, ymax=y_max, linestyles="dashed", label="bins")
        plt.xlabel("x_%d" % (feature + 1))
        plt.ylabel("dy/dx_%d" % (feature + 1))
        plt.legend()
        plt.show(block=block)


class Greedy(Base):
    """
    Greedy binning algorithm
    """

    def __init__(
        self,
        init_nof_bins: int = 20,
        min_points_per_bin: int = 2,
        discount: float = 0.3,
        cat_limit: int = 10,
    ):
        assert min_points_per_bin >= 2, "min_points_per_bin should be at least 2"
        method_args = {
            "init_nof_bins": init_nof_bins,
            "min_points": min_points_per_bin,
            "discount": discount,
            "cat_limit": cat_limit,
        }
        super(Greedy, self).__init__("greedy", method_args)


    def find_limits(self, data, data_effect, axis_limits) -> typing.Union[np.ndarray, bool]:

        self._preprocess_find(data, data_effect, axis_limits)
        xs_min = self.xs_min
        xs_max = self.xs_max
        init_nof_bins = self.method_args["init_nof_bins"]
        discount = self.method_args["discount"]
        cat_limit = self.method_args["cat_limit"]

        if self._none_valid_binning():
            self.limits = False
        # elif self._is_categorical(cat_limit):
        #     return self.limits
        elif self._only_one_bin_possible():
            self.limits = np.array([self.xs_min, self.xs_max])
        else:
            # limits with high resolution
            limits, _ = np.linspace(
                xs_min, xs_max, num=init_nof_bins + 1, endpoint=True, retstep=True
            )

            # merging
            i = 0
            merged_limits = [limits[0]]
            while i < init_nof_bins:
                # left limit is the last item of the merged_limits list
                left_lim = merged_limits[-1]

                # choose whether to close the bin
                if i == init_nof_bins - 1:
                    # if last bin, close it
                    close_bin = True
                else:
                    # bin_1, the bin if I close it here
                    bin_1_loss = self._bin_cost(left_lim, limits[i + 1], discount)
                    bin_1_valid = self._bin_valid(left_lim, limits[i + 1])

                    # bin_2: the bin if I close it in the next limit
                    bin_2_loss = self._bin_cost(left_lim, limits[i + 2], discount)
                    bin_2_valid = self._bin_valid(left_lim, limits[i + 2])

                    # if both bins valid
                    if bin_1_valid and bin_2_valid:
                        # if first zero, second positive -> close
                        if bin_1_loss == 0.0 and bin_2_loss > 0:
                            close_bin = True
                        # if both zero, keep it (we could close as well)
                        elif bin_1_loss == 0.0 and bin_2_loss == 0:
                            close_bin = False
                        # if both positive, compare and decide
                        else:
                            close_bin = False if bin_2_loss <= bin_1_loss else True
                    else:
                        # if either invalid, keep open
                        close_bin = False

                # if close_bin, then add the next limit to the merged_limits
                if close_bin:
                    merged_limits.append(limits[i + 1])

                i += 1

            # if last bin is without enough points, merge it with the previous
            if not self._bin_valid(merged_limits[-2], merged_limits[-1]):
                merged_limits = merged_limits[:-2] + merged_limits[-1:]

            # store result
            self.limits = np.array(merged_limits)
            self.method_outputs = {"limits": self.limits}
        return self.limits


class DynamicProgramming(Base):

    def __init__(self, max_nof_bins: int = 20, min_points_per_bin: int = 2., discount: float = 0.3, cat_limit: int = 10):
        assert min_points_per_bin >= 2, "min_points_per_bin should be at least 2"
        method_args = {
            "max_nof_bins": max_nof_bins,
            "min_points": min_points_per_bin,
            "discount": discount,
            "cat_limit": cat_limit,
        }
        super(DynamicProgramming, self).__init__("dynamic_programming", method_args)

    def _index_to_position(self, index_start, index_stop, K):
        dx = (self.xs_max - self.xs_min) / K
        start = self.xs_min + index_start * dx
        stop = self.xs_min + index_stop * dx
        return start, stop

    def _cost_of_move(self, index_before, index_next, K, discount):
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
            cost = self._bin_cost(start, stop, discount)
        return cost

    def _argmatrix_to_limits(self, K):
        assert (
            "argmatrix" in self.method_outputs
        ), "argmatrix not found in method_outputs"
        argmatrix = self.method_outputs["argmatrix"]
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

    def find_limits(self, data, data_effect, axis_limits):
        """Find the optimal binning."""
        self._preprocess_find(data, data_effect, axis_limits)
        max_nof_bins = self.method_args["max_nof_bins"]
        min_points = self.method_args["min_points"]
        discount = self.method_args["discount"]
        cat_limit = self.method_args["cat_limit"]

        self.min_points = min_points
        big_M = self.big_M
        nof_limits = max_nof_bins + 1
        nof_bins = max_nof_bins

        # if is categorical, then only one bin is possible
        if self._none_valid_binning():
            self.limits = False
        # elif self._is_categorical(cat_limit):
        #     return self.limits
        elif self._only_one_bin_possible():
            self.limits = np.array([self.xs_min, self.xs_max])
            self.dx_list = np.array([self.xs_max - self.xs_min])
        elif max_nof_bins == 1:
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
                    bin_index, lim_index, max_nof_bins, discount
                )

            # for all other bins
            for bin_index in range(1, max_nof_bins):
                for lim_index_next in range(max_nof_bins + 1):
                    # find best solution
                    tmp = []
                    for lim_index_before in range(max_nof_bins + 1):
                        tmp.append(
                            matrix[lim_index_before, bin_index - 1]
                            + self._cost_of_move(
                                lim_index_before, lim_index_next, max_nof_bins, discount
                            )
                        )
                    # store best solution
                    matrix[lim_index_next, bin_index] = np.min(tmp)
                    argmatrix[lim_index_next, bin_index] = np.argmin(tmp)

            # find indices
            self.method_outputs = {"matrix": matrix, "argmatrix": argmatrix}
            self.limits, _ = self._argmatrix_to_limits(max_nof_bins)
            self.method_outputs["limits"] = self.limits
        return self.limits


class Fixed(Base):
    def __init__(self, nof_bins: int = 20, min_points_per_bin=0., cat_limit: int = 10):
        method_args = {
            "nof_bins": nof_bins,
            "min_points": min_points_per_bin,
            "cat_limit": cat_limit,
        }
        super(Fixed, self).__init__("fixed", method_args)

    def find_limits(self, data, data_effect, axis_limits):
        self._preprocess_find(data, data_effect, axis_limits)
        nof_bins = self.method_args["nof_bins"]
        min_points = self.method_args["min_points"]
        cat_limit = self.method_args["cat_limit"]

        if self._none_valid_binning():
            self.limits = False

        # if self._is_categorical(cat_limit):
        #     return self.limits

        limits, dx = np.linspace(
            self.xs_min, self.xs_max, num=nof_bins + 1, endpoint=True, retstep=True
        )
        if min_points is not None:
            limits_valid = all(
                self._bin_valid(limits[i], limits[i + 1]) for i in range(nof_bins)
            )
            self.limits = limits if limits_valid else False
        else:
            self.limits = limits

        return self.limits


def return_default(method):
    assert (
        method in ["greedy", "dp", "fixed"]
        or isinstance(method, Fixed)
        or isinstance(method, DynamicProgramming)
        or isinstance(method, Greedy)
    )

    if isinstance(method, Base):
        return method

    if method == "greedy":
        return Greedy()
    elif method == "dp":
        return DynamicProgramming()
    elif method == "fixed":
        return Fixed()
    else:
        raise ValueError("Unknown method")

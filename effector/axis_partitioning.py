import copy
import enum
import typing

import matplotlib.pyplot as plt
import numpy as np

import effector.helpers as helpers
import effector.theme as theme
import effector.utils as utils


class NoBinningReason(enum.Enum):
    """Machine-readable reason an optimizer returned ``False`` from
    ``find_limits`` — the *why* behind the outcome, so a higher-level caller can
    branch on it and ``utils.raise_if_no_binning`` can render a specific message.
    Each member's ``.value`` is the human-readable sentence."""

    SINGLE_UNIQUE_VALUE = "all points share a single value"
    TOO_FEW_POINTS = "fewer points than min_points"
    FIXED_GRID_UNDERFILLED = "the fixed uniform grid leaves a bin under min_points"


class Base:
    """A standalone 1D bin optimizer.

    This module is deliberately self-contained: it knows nothing about the rest
    of the package. It works purely on 1D points ``x`` and an optional per-point
    value ``y``, and produces bin edges over a range ``x_lims`` that keep the
    within-bin variance of ``y`` low. Callers translate their own quantities
    (feature columns, local effects, SHAP values, ...) into this ``x``/``y``
    vocabulary at the boundary.
    """

    big_M = helpers.BIG_M

    def __init__(
        self,
        name: str,
        method_args: typing.Dict[str, typing.Any],
    ):
        """Initializer.

        Parameters
        ----------
        name: the method's canonical name (e.g. "fixed", "greedy")
        method_args: the method's hyperparameters; `find_limits` reads them
            (all subclasses store at least `min_points`)
        """
        self.name = name

        # set in _preprocess_find
        self.x_min = None
        self.x_max = None
        self.x = None
        self.y = None

        # arguments passed to method find
        self.method_args: typing.Dict[str, typing.Any] = method_args

        # set in method find
        self.method_outputs: typing.Dict[str, typing.Any] = {}

        # limits: None if not set, False if no binning is possible, np.ndarray (K+1,) otherwise
        self.limits: typing.Union[None, bool, np.ndarray] = None

        # when `find_limits` returns False, why (see NoBinningReason); None otherwise
        self.no_binning_reason: typing.Optional[NoBinningReason] = None

    # ------------------------------------------------------------------ #
    # The template method: shared preprocessing + degenerate guards + a   #
    # single subclass hook (`_search`). Subclasses implement `_search`    #
    # (and optionally override `_collapse_to_one_bin`); they do NOT       #
    # override `find_limits`.                                             #
    # ------------------------------------------------------------------ #
    def find_limits(self, x, y, x_lims) -> typing.Union[np.ndarray, bool]:
        """Find bin edges for a 1D set of points.

        Contract
        --------
        Inputs:
            x: 1D float array `(N,)` — the points to bin.
            y: 1D float array `(N,)` — a per-point value whose within-bin
                variance drives bin cost, or `None` for optimizers that do not
                use it (e.g. `Fixed`).
            x_lims: `[min, max]` length-2 sequence, or `None` to derive them from
                `x.min()/max()`.

        Returns:
            An ascending float array `(K+1,)` of bin edges with `edges[0] ==
            x_min` and `edges[-1] == x_max`; **or** the literal `False` when no
            binning satisfies the method's constraints. A degenerate but valid
            "one bin" result is the 2-element `[x_min, x_max]` — that is *not* a
            failure.

        Failure is a *fact, not a severity*: the optimizer never raises. It
        returns `False` and records `self.no_binning_reason` (a
        `NoBinningReason`). The caller decides what to do with it — the global
        effect methods treat it as fatal via `utils.raise_if_no_binning`, while
        the regional subregion search treats it as a normal "skip this
        candidate".
        """
        self._preprocess_find(x, y, x_lims)

        reason = self._none_valid_binning()
        if reason is not None:
            self.no_binning_reason = reason
            self.limits = False
        elif self._collapse_to_one_bin():
            self.limits = np.array([self.x_min, self.x_max])
        else:
            self.limits = self._search()
        return self.limits

    def _collapse_to_one_bin(self) -> bool:
        """Policy hook: should the whole range collapse into a single bin?
        Default = yes when only one bin is possible. `Fixed` overrides this to
        `False` (it never collapses); `DynamicProgramming` also folds in its
        `max_nof_bins == 1` shortcut."""
        return self._only_one_bin_possible()

    def _search(self) -> typing.Union[np.ndarray, bool]:
        """Subclass kernel: the actual edge search, run only once the shared
        guards have passed. Returns ascending edges, or `False` (setting
        `self.no_binning_reason`) if the method's own constraints fail."""
        raise NotImplementedError

    def _bin_cost(self, start, stop, discount):
        min_points = self.method_args["min_points"]
        nof_points = self.x.shape[0]
        x, y = utils.filter_points_in_bin(self.x, self.y, np.array([start, stop]))

        # compute cost
        thres = max(min_points, 2)
        if y.size < thres:
            cost = self.big_M
        else:
            discount_for_more_points = 1 - discount * (y.size / nof_points)
            cost = np.var(y) * (stop - start) * discount_for_more_points
        return cost

    def _bin_valid(self, start, stop):
        """Check if creating a bin with limits [start, stop] is valid.

        Returns:
            Boolean, True if the bin is valid, False otherwise
        """
        min_points = self.method_args["min_points"]
        if min_points is None:
            return True

        filtered_points, _ = utils.filter_points_in_bin(
            self.x, None, np.array([start, stop])
        )
        valid = filtered_points.size >= min_points
        return valid

    def _none_valid_binning(self) -> typing.Optional[NoBinningReason]:
        """Why no binning at all is possible, or `None` if some binning exists.

        A `NoBinningReason` member is truthy, so `if self._none_valid_binning():`
        still reads as a predicate while also carrying the reason.
        """
        # if there is only one unique value, no binning is possible
        if len(np.unique(self.x)) == 1:
            return NoBinningReason.SINGLE_UNIQUE_VALUE

        # if there are fewer than min_points, no binning is possible
        if self.x.size < self.method_args["min_points"]:
            return NoBinningReason.TOO_FEW_POINTS

        return None

    def _only_one_bin_possible(self):
        """Check if the only possible binning is all points in one bin

        Returns:
            Boolean, True if the only possible binning is all points in one bin, False otherwise
        """
        min_points = self.method_args["min_points"]
        # if x is categorical (single point on the axis), only one bin is possible
        is_categorical = np.allclose(self.x_min, self.x_max)

        # otherwise, one bin is possible if there are enough points for exactly one
        enough_for_one_bin = min_points <= self.y.size < 2 * min_points
        return is_categorical or enough_for_one_bin

    def _preprocess_find(self, x, y, x_lims):
        # reset per-call state; normalize the min_points contract so subclasses
        # and guards never see None (0 == "no minimum"). This also removes the
        # latent `x.size < None` TypeError for `Fixed(min_points_per_bin=None)`.
        self.no_binning_reason = None
        if self.method_args.get("min_points") is None:
            self.method_args["min_points"] = 0

        self.x_min: float = x_lims[0] if x_lims is not None else x.min()
        self.x_max: float = x_lims[1] if x_lims is not None else x.max()
        self.x: np.ndarray = x
        self.y: np.ndarray = y

    def plot(self, feature=0, block=False):
        assert self.limits is not None
        assert self.x is not None
        assert self.y is not None

        limits = self.limits

        plt.figure()
        plt.title("Bin splitting for feature %d" % feature)
        plt.plot(
            self.x,
            self.y,
            color=theme.active().MEAN,
            marker="o",
            linestyle="none",
            label="local effects",
        )
        y_min = np.min(self.y)
        y_max = np.max(self.y)
        plt.vlines(limits, ymin=y_min, ymax=y_max, linestyles="dashed", label="bins")
        plt.xlabel("x_%d" % feature)
        plt.ylabel("dy/dx_%d" % feature)
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

    def _search(self) -> np.ndarray:
        x_min = self.x_min
        x_max = self.x_max
        init_nof_bins = self.method_args["init_nof_bins"]
        discount = self.method_args["discount"]

        # limits with high resolution
        limits, _ = np.linspace(
            x_min, x_max, num=init_nof_bins + 1, endpoint=True, retstep=True
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
        result = np.array(merged_limits)
        self.method_outputs = {"limits": result}
        return result


class DynamicProgramming(Base):
    def __init__(
        self,
        max_nof_bins: int = 20,
        min_points_per_bin: int = 2,
        discount: float = 0.3,
        cat_limit: int = 10,
    ):
        assert min_points_per_bin >= 2, "min_points_per_bin should be at least 2"
        method_args = {
            "max_nof_bins": max_nof_bins,
            "min_points": min_points_per_bin,
            "discount": discount,
            "cat_limit": cat_limit,
        }
        super(DynamicProgramming, self).__init__("dynamic_programming", method_args)

    def _collapse_to_one_bin(self) -> bool:
        # DP's dedicated single-bin shortcut folds into the collapse policy: an
        # explicit request for one bin yields the same [x_min, x_max] as the
        # generic "only one bin possible" case.
        return self._only_one_bin_possible() or self.method_args["max_nof_bins"] == 1

    def _index_to_position(self, index_start, index_stop, K):
        dx = (self.x_max - self.x_min) / K
        start = self.x_min + index_start * dx
        stop = self.x_min + index_stop * dx
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
        assert "argmatrix" in self.method_outputs, (
            "argmatrix not found in method_outputs"
        )
        argmatrix = self.method_outputs["argmatrix"]
        dx = (self.x_max - self.x_min) / K

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

        limits = self.x_min + np.array(lim_indices_1) * dx
        dx_list = np.array(
            [limits[i + 1] - limits[i] for i in range(limits.shape[0] - 1)]
        )
        return limits, dx_list

    def _search(self) -> np.ndarray:
        max_nof_bins = self.method_args["max_nof_bins"]
        discount = self.method_args["discount"]

        big_M = self.big_M
        nof_limits = max_nof_bins + 1
        nof_bins = max_nof_bins

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
        limits, _ = self._argmatrix_to_limits(max_nof_bins)
        self.method_outputs["limits"] = limits
        return limits


class Fixed(Base):
    def __init__(
        self, nof_bins: int = 20, min_points_per_bin: int = 0, cat_limit: int = 10
    ):
        method_args = {
            "nof_bins": nof_bins,
            "min_points": min_points_per_bin,
            "cat_limit": cat_limit,
        }
        super(Fixed, self).__init__("fixed", method_args)

    def _collapse_to_one_bin(self) -> bool:
        # Fixed honors the requested bin count exactly: it never collapses to a
        # single bin — it returns the uniform grid or False (B6). Opting out here
        # also keeps `_only_one_bin_possible` (which reads y.size) off the path,
        # since Fixed is called with y=None.
        return False

    def _search(self) -> typing.Union[np.ndarray, bool]:
        nof_bins = self.method_args["nof_bins"]

        limits, _ = np.linspace(
            self.x_min, self.x_max, num=nof_bins + 1, endpoint=True, retstep=True
        )
        if not all(self._bin_valid(limits[i], limits[i + 1]) for i in range(nof_bins)):
            self.no_binning_reason = NoBinningReason.FIXED_GRID_UNDERFILLED
            return False
        return limits


def adapt_for_categorical(method: Base, nof_levels: int) -> Base:
    """A copy of `method` whose candidate bin edges land exactly on the
    integer level codes 0..K-1, so Greedy/DP merging over the K-1 transitions
    becomes *adaptive level grouping* (method_semantics.md, RHALE-ordinal)."""
    method = copy.deepcopy(method)
    nof_transitions = nof_levels - 1
    for key in ("init_nof_bins", "max_nof_bins", "nof_bins"):
        if key in method.method_args:
            method.method_args[key] = nof_transitions
    return method


# the single alias table for binning-method strings (R6): validation and
# resolution both read it, so they cannot disagree
VALID_METHODS = {
    "fixed": Fixed,
    "greedy": Greedy,
    "dp": DynamicProgramming,
}


def return_default(method):
    if isinstance(method, Base):
        return method

    if not isinstance(method, str) or method not in VALID_METHODS:
        raise ValueError(
            f"Unknown binning method: {method!r}; valid options are "
            f"{sorted(VALID_METHODS)} or an axis_partitioning class instance"
        )

    return VALID_METHODS[method]()

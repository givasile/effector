import copy
import dataclasses
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
    TOO_FEW_POINTS = "fewer points than min_points_per_bin"
    FIXED_GRID_UNDERFILLED = (
        "the fixed uniform grid leaves a bin under min_points_per_bin"
    )


@dataclasses.dataclass
class Constraints:
    """Feasibility rules any valid 1D partition must obey — algorithm-agnostic.

    Kept separate from each optimizer's *method parameters* (which shape the
    objective and the search, e.g. ``discount``, ``init_nof_bins``): constraints
    say what a *valid* partition is, params say how a given method looks for one.
    The shared ``Base`` guards read only these.
    """

    min_points_per_bin: int = 0  # a bin with fewer points is invalid (0 == no minimum)
    max_nof_bins: typing.Optional[int] = None  # never emit more (None == unbounded)


class Base:
    """A standalone 1D bin optimizer.

    This module is deliberately self-contained: it knows nothing about the rest
    of the package. It works purely on 1D points ``x`` and an optional per-point
    value ``y``, and produces bin edges over a range ``x_lims`` that keep the
    within-bin variance of ``y`` low. Callers translate their own quantities
    (feature columns, local effects, SHAP values, ...) into this ``x``/``y``
    vocabulary at the boundary — including feature types: the binner always
    operates on continuous positions (ordinal features reach it as integer level
    codes via `adapt_for_categorical`; nominal features never reach it).
    """

    big_M = helpers.BIG_M

    def __init__(
        self,
        name: str,
        constraints: Constraints,
        params: typing.Dict[str, typing.Any],
    ):
        """Initializer.

        Parameters
        ----------
        name: the method's canonical name (e.g. "fixed", "greedy")
        constraints: the feasibility rules (min points per bin, max #bins)
        params: the method's own parameters (objective/search knobs)
        """
        self.name = name

        # constraints (shared vocabulary) vs params (method-specific)
        self.constraints = constraints
        # normalize the min_points contract so guards never see None (0 == "no
        # minimum"); also removes the latent Fixed(min_points_per_bin=None) crash
        if self.constraints.min_points_per_bin is None:
            self.constraints.min_points_per_bin = 0
        self.params: typing.Dict[str, typing.Any] = params

        # set in _preprocess_find
        self.x_min = None
        self.x_max = None
        self.x = None
        self.y = None

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
            binning satisfies the constraints. A degenerate but valid "one bin"
            result is the 2-element `[x_min, x_max]` — that is *not* a failure.

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

    def _set_candidate_bin_count(self, k: int) -> None:
        """Set how many candidate bins the method considers. Used by
        `adapt_for_categorical` to run a continuous binner over the K-1 ordinal
        level transitions. Each optimizer maps `k` to its own bin-count knob."""
        raise NotImplementedError

    def _bin_cost(self, start, stop, discount):
        min_points = self.constraints.min_points_per_bin
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
        min_points = self.constraints.min_points_per_bin
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

        # if there are fewer than min_points_per_bin, no binning is possible
        if self.x.size < self.constraints.min_points_per_bin:
            return NoBinningReason.TOO_FEW_POINTS

        return None

    def _only_one_bin_possible(self):
        """Check if the only possible binning is all points in one bin

        Returns:
            Boolean, True if the only possible binning is all points in one bin, False otherwise
        """
        min_points = self.constraints.min_points_per_bin
        # if x is categorical (single point on the axis), only one bin is possible
        is_categorical = np.allclose(self.x_min, self.x_max)

        # otherwise, one bin is possible if there are enough points for exactly one
        enough_for_one_bin = min_points <= self.y.size < 2 * min_points
        return is_categorical or enough_for_one_bin

    def _preprocess_find(self, x, y, x_lims):
        # reset per-call state
        self.no_binning_reason = None

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
    ):
        assert min_points_per_bin >= 2, "min_points_per_bin should be at least 2"
        constraints = Constraints(min_points_per_bin=min_points_per_bin)
        params = {"init_nof_bins": init_nof_bins, "discount": discount}
        super(Greedy, self).__init__("greedy", constraints, params)

    def _set_candidate_bin_count(self, k: int) -> None:
        self.params["init_nof_bins"] = k

    def _search(self) -> np.ndarray:
        x_min = self.x_min
        x_max = self.x_max
        init_nof_bins = self.params["init_nof_bins"]
        discount = self.params["discount"]

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
    ):
        assert min_points_per_bin >= 2, "min_points_per_bin should be at least 2"
        constraints = Constraints(
            min_points_per_bin=min_points_per_bin, max_nof_bins=max_nof_bins
        )
        params = {"discount": discount}
        super(DynamicProgramming, self).__init__(
            "dynamic_programming", constraints, params
        )

    def _set_candidate_bin_count(self, k: int) -> None:
        self.constraints.max_nof_bins = k

    def _collapse_to_one_bin(self) -> bool:
        # DP's dedicated single-bin shortcut folds into the collapse policy: an
        # explicit request for one bin yields the same [x_min, x_max] as the
        # generic "only one bin possible" case.
        return self._only_one_bin_possible() or self.constraints.max_nof_bins == 1

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

    def _bin_cost_matrix(self, max_nof_bins, discount):
        """`cost[i, j]` = cost of a single bin spanning limit-index i..j, for all
        i, j at once, in O(N + K^2).

        Points are assigned to the K uniform grid cells and per-cell count / Σy /
        Σy^2 are prefix-summed, so any bin's mean and variance are O(1). This
        replaces the O(K^2 · N) matrix build (one O(N) `filter_points_in_bin` +
        `np.var` per bin) with a single O(N) pass.

        Boundary note: the grid cells are HALF-OPEN `[edge_m, edge_{m+1})`, unlike
        `filter_points_in_bin` which is inclusive on both ends. The two agree on
        every input except one where a point lands *exactly* on an interior grid
        edge — measure-zero for continuous data, and impossible on the
        integer-code categorical grid (positions are half-integers). Variance is
        `E[y^2] - E[y]^2`, clamped at 0 to absorb floating-point noise (`np.var`
        is non-negative by construction).
        """
        big_M = self.big_M
        nof_limits = max_nof_bins + 1
        nof_points = self.x.shape[0]
        thres = max(self.constraints.min_points_per_bin, 2)
        dx = (self.x_max - self.x_min) / max_nof_bins

        cell = np.clip(((self.x - self.x_min) / dx).astype(int), 0, max_nof_bins - 1)
        count = np.bincount(cell, minlength=max_nof_bins).astype(float)
        sum_y = np.bincount(cell, self.y, minlength=max_nof_bins)
        sum_y2 = np.bincount(cell, self.y * self.y, minlength=max_nof_bins)
        cum_count = np.concatenate([[0.0], np.cumsum(count)])
        cum_sum_y = np.concatenate([[0.0], np.cumsum(sum_y)])
        cum_sum_y2 = np.concatenate([[0.0], np.cumsum(sum_y2)])

        # n[i, j] / s[i, j] / q[i, j] over the points in cells [i, j)
        n = cum_count[None, :] - cum_count[:, None]
        s = cum_sum_y[None, :] - cum_sum_y[:, None]
        q = cum_sum_y2[None, :] - cum_sum_y2[:, None]
        idx = np.arange(nof_limits)
        width = (idx[None, :] - idx[:, None]) * dx
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = s / n
            var = np.maximum(q / n - mean * mean, 0.0)
        cost = var * width * (1.0 - discount * n / nof_points)
        # under-filled bins (n < thres, which also covers i >= j) cost big_M...
        cost = np.where(n < thres, big_M, cost)
        # ...except a zero-width bin (i == j), which costs 0
        np.fill_diagonal(cost, 0.0)
        return cost

    def _search(self) -> np.ndarray:
        max_nof_bins = self.constraints.max_nof_bins
        discount = self.params["discount"]

        big_M = self.big_M
        nof_limits = max_nof_bins + 1
        nof_bins = max_nof_bins

        cost = self._bin_cost_matrix(max_nof_bins, discount)

        # init matrices
        matrix = np.ones((nof_limits, nof_bins)) * big_M
        argmatrix = np.ones((nof_limits, nof_bins)) * np.nan

        # first bin: cost of a single bin from index 0 to each limit
        matrix[:, 0] = cost[0, :]

        # for all other bins: matrix[next, b] = min_before matrix[before, b-1] +
        # cost[before, next]. `argmin(axis=0)` scans ascending `before` (the
        # tie-break the original `np.argmin(tmp)` used).
        for bin_index in range(1, max_nof_bins):
            prev = matrix[:, bin_index - 1][:, None] + cost
            matrix[:, bin_index] = prev.min(axis=0)
            argmatrix[:, bin_index] = prev.argmin(axis=0)

        # find indices
        self.method_outputs = {"matrix": matrix, "argmatrix": argmatrix}
        limits, _ = self._argmatrix_to_limits(max_nof_bins)
        self.method_outputs["limits"] = limits
        return limits


class Fixed(Base):
    def __init__(self, nof_bins: int = 20, min_points_per_bin: int = 0):
        constraints = Constraints(min_points_per_bin=min_points_per_bin)
        params = {"nof_bins": nof_bins}
        super(Fixed, self).__init__("fixed", constraints, params)

    def _set_candidate_bin_count(self, k: int) -> None:
        self.params["nof_bins"] = k

    def _collapse_to_one_bin(self) -> bool:
        # Fixed honors the requested bin count exactly: it never collapses to a
        # single bin — it returns the uniform grid or False (B6). Opting out here
        # also keeps `_only_one_bin_possible` (which reads y.size) off the path,
        # since Fixed is called with y=None.
        return False

    def _search(self) -> typing.Union[np.ndarray, bool]:
        nof_bins = self.params["nof_bins"]

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
    method._set_candidate_bin_count(nof_levels - 1)
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

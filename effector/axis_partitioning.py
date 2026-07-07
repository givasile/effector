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
        name: the method's canonical name (e.g. "fixed", "agglomerative")
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
                use it (e.g. `Fixed`, `Quantile`).
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

    def _bin_cost_matrix(self, nof_cells, discount):
        """`cost[i, j]` = cost of a single bin spanning grid-edge i..j, for all
        i, j at once, over a uniform grid of `nof_cells` cells, in O(N + K^2).

        Points are assigned to the K uniform grid cells and per-cell count / Σy /
        Σy^2 are prefix-summed, so any bin's mean and variance are O(1). Shared by
        `DynamicProgramming` (optimal search) and `Agglomerative` (greedy merge).

        Boundary note: the grid cells are HALF-OPEN `[edge_m, edge_{m+1})`, unlike
        `filter_points_in_bin` which is inclusive on both ends. The two agree on
        every input except one where a point lands *exactly* on an interior grid
        edge — measure-zero for continuous data, and impossible on the
        integer-code categorical grid (positions are half-integers). Variance is
        `E[y^2] - E[y]^2`, clamped at 0 to absorb floating-point noise (`np.var`
        is non-negative by construction).
        """
        big_M = self.big_M
        nof_limits = nof_cells + 1
        nof_points = self.x.shape[0]
        thres = max(self.constraints.min_points_per_bin, 2)
        dx = (self.x_max - self.x_min) / nof_cells

        cell = np.clip(((self.x - self.x_min) / dx).astype(int), 0, nof_cells - 1)
        count = np.bincount(cell, minlength=nof_cells).astype(float)
        sum_y = np.bincount(cell, self.y, minlength=nof_cells)
        sum_y2 = np.bincount(cell, self.y * self.y, minlength=nof_cells)
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


class Agglomerative(Base):
    """Bottom-up agglomerative binning.

    Start from a fine uniform grid of `init_nof_bins` cells and repeatedly remove
    the interior boundary whose removal reduces the total cost the most, stopping
    when no removal helps (under-filled bins carry `big_M`, so they are merged
    away first). A genuine greedy — order-independent and driven by the same
    variance×width objective `DynamicProgramming` optimizes, but only locally
    optimal. O(N + K^2). (Replaces the old left-to-right `Greedy` sweep.)
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
        super(Agglomerative, self).__init__("agglomerative", constraints, params)

    def _set_candidate_bin_count(self, k: int) -> None:
        self.params["init_nof_bins"] = k

    def _search(self) -> np.ndarray:
        K = self.params["init_nof_bins"]
        discount = self.params["discount"]
        cost = self._bin_cost_matrix(K, discount)  # (K+1, K+1), cost[i, j] for i<j

        # `kept` = the grid-edge indices still acting as bin boundaries. Greedily
        # drop the interior edge whose removal most reduces the total cost:
        #   Δ(remove e between bins [a,e] and [e,b]) = cost[a,b] - cost[a,e] - cost[e,b]
        kept = list(range(K + 1))
        while len(kept) > 2:
            best_m, best_delta = None, None
            for m in range(1, len(kept) - 1):
                a, e, b = kept[m - 1], kept[m], kept[m + 1]
                delta = cost[a, b] - cost[a, e] - cost[e, b]
                if best_delta is None or delta < best_delta:
                    best_delta, best_m = delta, m
            # merge while it does not INCREASE the total cost (delta <= 0): the
            # discount rewards fusing similar bins, and equal-cost regions (e.g.
            # constant effect) are collapsed rather than left as spurious splits.
            # A genuine variance jump gives delta > 0 and halts merging.
            if best_delta is None or best_delta > 0.0:
                break
            kept.pop(best_m)

        dx = (self.x_max - self.x_min) / K
        limits = self.x_min + np.array(kept) * dx
        self.method_outputs = {"limits": limits}
        return limits


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


class Quantile(Base):
    """Equal-frequency binning: edges at data quantiles so every bin holds ~the
    same number of points. Like `Fixed` it ignores `y`, but it adapts the edge
    positions to the `x` distribution — robust for skewed features where a
    uniform grid wastes bins. O(N log N). Under-filled bins (only with heavy
    ties) are merged into a neighbor to honor `min_points_per_bin`."""

    def __init__(self, nof_bins: int = 20, min_points_per_bin: int = 0):
        constraints = Constraints(min_points_per_bin=min_points_per_bin)
        params = {"nof_bins": nof_bins}
        super(Quantile, self).__init__("quantile", constraints, params)

    def _set_candidate_bin_count(self, k: int) -> None:
        self.params["nof_bins"] = k

    def _collapse_to_one_bin(self) -> bool:
        # y-agnostic: collapse only when the axis itself is degenerate (single
        # point). Avoids `_only_one_bin_possible`, which reads y.size.
        return bool(np.allclose(self.x_min, self.x_max))

    def _search(self) -> np.ndarray:
        nof_bins = self.params["nof_bins"]
        min_points = self.constraints.min_points_per_bin

        edges = np.quantile(self.x, np.linspace(0.0, 1.0, nof_bins + 1))
        edges[0], edges[-1] = self.x_min, self.x_max
        edges = np.unique(edges)  # collapse ties / discrete duplicates

        # honor min_points: merge the least-populated bin into a neighbor until
        # all bins are valid (equal-frequency rarely needs this)
        while min_points > 0 and edges.size > 2:
            counts = np.array(
                [
                    np.sum((self.x >= edges[i]) & (self.x <= edges[i + 1]))
                    for i in range(edges.size - 1)
                ]
            )
            if counts.min() >= min_points:
                break
            b = int(np.argmin(counts))
            drop = b + 1 if b < edges.size - 2 else b  # never an endpoint
            edges = np.delete(edges, drop)

        self.method_outputs = {"limits": edges}
        return edges


def adapt_for_categorical(method: Base, nof_levels: int) -> Base:
    """A copy of `method` whose candidate bin edges land exactly on the
    integer level codes 0..K-1, so Agglomerative/DP merging over the K-1
    transitions becomes *adaptive level grouping* (method_semantics.md,
    RHALE-ordinal)."""
    method = copy.deepcopy(method)
    method._set_candidate_bin_count(nof_levels - 1)
    return method


# `Greedy` is retained as a deprecated alias for `Agglomerative` (the old
# left-to-right sweep was renamed and replaced by the proper agglomerative
# algorithm). Existing `Greedy(...)` / `"greedy"` code keeps working.
Greedy = Agglomerative


# the single alias table for binning-method strings (R6): validation and
# resolution both read it, so they cannot disagree
VALID_METHODS = {
    "fixed": Fixed,
    "agglomerative": Agglomerative,
    "quantile": Quantile,
    "dp": DynamicProgramming,
    "greedy": Agglomerative,  # deprecated alias
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

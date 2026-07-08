import copy
import dataclasses
import typing

import numpy as np

from effector import helpers, ingestion, proposers
from effector.partition import Partition, Region
from effector.rules import Rule

BIG_M = helpers.BIG_M


@dataclasses.dataclass
class SplitEvaluation:
    """The winner of one exhaustive candidate scan: the selected
    `CandidateSplit` plus its children, flattened parent-major
    (``[for parent in parents for condition in candidate.conditions]``)."""

    candidate: typing.Optional[proposers.CandidateSplit]
    child_masks: list
    child_heters: list
    child_counts: list
    weighted_heter: float


class Base:
    def __init__(
        self,
        name: str,
        min_heterogeneity_decrease_pcg: float = 0.1,
        heter_small_enough: float = 0.001,
        max_depth: int = 2,
        min_samples_leaf: int = 10,
        numerical_features_grid_size: int = 20,
        search_partitions_when_categorical: bool = True,
        categorical_proposer="one_vs_rest",
        continuous_proposer="threshold",
    ):
        """Shared configuration of the space partitioners.

        Args:
            min_heterogeneity_decrease_pcg: Minimum percentage of heterogeneity decrease to accept a split.

                ??? Example "Example"
                    - `0.1`: if the heterogeneity before any split is 1, the heterogeneity after the first split must be at most 0.9 to be accepted. Otherwise, no split will be accepted.

            heter_small_enough: When heterogeneity is smaller than this value, no more splits are performed.

                ??? Note "Default is `0.001`"
                    Value 0.001 is small enough for most cases.
                    It is advisable to set this value to a small number to avoid unnecessary splits.

                ??? Note "Custom value"
                    If you know a priori that a specific heterogeneity value is small enough,
                    you can set this parameter to a higher value than the default.

            max_depth: Maximum number of splits to perform

                ??? Note "Default is `2`"
                    2 splits already create 4 subregions, i.e. 4 regional plots per feature, which are already enough.
                    Setting this value to a higher number will increase the number of subregions and plots, which may be too much for the user to analyze.

            min_samples_leaf: Minimum number of instances per subregion

                ??? Note "Default is `10`"
                    If a subregion has less than 10 instances, it may not be representative enough to be analyzed.

            numerical_features_grid_size: Number of candidate split positions for numerical features

                ??? Note "Default is `20`"
                    For numerical features, the algorithm will create a grid of 20 equally spaced values between the minimum and maximum values of the feature.

            search_partitions_when_categorical: Whether to search for partitions when the feature is categorical

                ??? warning "refers to a categorical feature of interest"
                    This argument asks whether to search for partitions when the feature of interest is categorical.
                    If the feature of interest is numerical, the algorithm will always search for partitions and will consider
                    categorical features for conditioning.

                ??? Note "Default is `False`"
                    It is difficult to compute the heterogeneity for categorical features, so by default, the algorithm will not search for partitions when the feature of interest is categorical.

            categorical_proposer: How candidate splits on categorical conditioning features are enumerated

                ??? Note "Options"
                    - `"one_vs_rest"` (default): one level vs. all others, per observed level
                    - `"subsets"`: every binary subset-vs-complement split
                    - `"ordered"`: contiguous cuts after ordering the levels (natural for ordinal, similarity seriation for nominal)
                    - `"multiway"`: one k-way candidate with one child per level
                    - a proposer instance (anything exposing `propose(ctx, foc)`), e.g. `effector.proposers.CategoricalOrdered(order=[...])`

            continuous_proposer: How candidate splits on continuous conditioning features are enumerated

                ??? Note "Options"
                    - `"threshold"` (default): binary splits on an interior grid of `numerical_features_grid_size` positions
                    - `"quantiles"`: one k-way candidate per child count, split at the marginal quantiles
                    - a proposer instance, e.g. `effector.proposers.ContinuousQuantiles(max_children=3)`

        """
        self.name = helpers.camel_to_snake(name)

        self.min_points_per_subregion = min_samples_leaf
        self.nof_candidate_splits_for_numerical = numerical_features_grid_size
        self.max_split_levels = max_depth
        self.heter_pcg_drop_thres = min_heterogeneity_decrease_pcg
        self.heter_small_enough = heter_small_enough
        self.split_categorical_features = search_partitions_when_categorical

        # all methods will set these attributes
        self.feature = None  # feature of interest
        self.foi = None  # feature of interest
        self.data = None  # dataset (N, D)
        self.dim = None  # dimensionality of the dataset (= D)
        self.heter_func = None  # heterogeneity function (callable (mask) -> float)
        self.axis_limits = (
            None  # axis limits (min and max for each feature), shape (2, D)
        )
        self.feature_types = None  # feature types (continuous/ordinal/nominal)
        self.cat_limit = None  # categorical limit
        self.feature_names = None  # feature names
        self.target_name = None  # target name
        self.foc_types = None  # feature-of-conditioning types (three-way taxonomy)
        self.candidate_conditioning_features = None  # candidate conditioning features
        self.ctx = None  # proposers.SearchContext, built by compile()

        # candidate enumeration: feature type -> proposer (the extension seam;
        # the kwargs are sugar over it, assigning a custom factory still works)
        self.proposer_factory = proposers.make_proposer_factory(
            categorical_proposer, continuous_proposer
        )

    def compile(
        self,
        feature: int,
        data: np.ndarray,
        heter_func: callable,
        axis_limits: np.ndarray,
        feature_types: typing.Union[list, None] = None,
        cat_limit: int = 10,
        candidate_conditioning_features: typing.Union[str, list] = "all",
        feature_names: typing.Union[None, list] = None,
        target_name: typing.Union[None, str] = None,
    ):
        "Tidy up the input data."
        self.feature = feature
        self.foi = feature
        self.data = data
        self.dim = self.data.shape[1]
        self.heter_func = heter_func
        self.axis_limits = axis_limits
        self.cat_limit = cat_limit
        self.feature_names = feature_names
        self.target_name = target_name

        self.candidate_conditioning_features = helpers.prep_conditioning_features(
            candidate_conditioning_features, feature, self.dim
        )

        self.feature_types = (
            ingestion.infer_feature_types(data, cat_limit)
            if feature_types is None
            else feature_types
        )
        self.foc_types = [
            self.feature_types[i] for i in self.candidate_conditioning_features
        ]

        self.ctx = proposers.SearchContext(
            data=self.data,
            axis_limits=self.axis_limits,
            feature_types=tuple(self.feature_types),
            numerical_grid_size=self.nof_candidate_splits_for_numerical,
        )

    def find_regions(
        self,
        feature,
        data,
        score_fn,
        *,
        axis_limits,
        feature_types,
        cat_limit,
        candidate_conditioning_features,
        feature_names,
        target_name,
    ) -> Partition:
        """Finder protocol: given a ``mask -> float`` ``score_fn`` plus the data
        and metadata needed to PROPOSE candidate splits, return a `Partition`.

        Any object exposing this method (returning a `Partition`) is a valid
        region finder. ``score_fn`` is RAW — it may raise ``ValueError`` or
        return nan; this adapter owns the min-points and degeneracy guard, so the
        ``BIG_M`` vocabulary lives here and never leaks into the effect.
        """
        if self.min_points_per_subregion < 2:
            raise ValueError("min_points_per_subregion must be >= 2")

        from effector import utils  # local import for the except tuple

        def guarded(active_indices):
            mask = active_indices.astype(bool)
            if mask.sum() < self.min_points_per_subregion:
                return BIG_M
            try:
                score = score_fn(mask)
            except (utils.AllBinsHaveAtMostOnePointError, ValueError):
                return BIG_M
            return score if np.isfinite(score) else BIG_M

        # compile() mutates the partitioner in place; work on a copy so the
        # caller's instance stays reusable (RC5 non-mutation contract).
        worker = copy.deepcopy(self)
        worker.compile(
            feature,
            data,
            guarded,
            axis_limits,
            feature_types,
            cat_limit,
            candidate_conditioning_features,
            feature_names,
            target_name,
        )
        partition = worker.fit()

        if partition is None:
            # BestLevelWise no-search path: a root-only Partition.
            n = data.shape[0]
            root = Region(
                idx=0,
                name=feature_names[feature],
                rule=Rule({}),
                heterogeneity=float(guarded(np.ones(n))),
                nof_instances=n,
                weight=1.0,
                level=0,
                parent_idx=None,
                mask=np.ones(n, dtype=bool),
            )
            return Partition(
                [root],
                feature=feature,
                feature_name=feature_names[feature],
                finder_name=self.name,
                feature_names=feature_names,
            )

        return partition

    def fit(self) -> typing.Optional[Partition]:
        """Find the subregions."""
        raise NotImplementedError

    def _make_region(self, *, idx, rule, mask, heter, level, parent_idx) -> Region:
        """One rule-primary `Region`; name, count, and weight derive from the
        rule and the mask (Region.idx must equal the insertion position)."""
        feature_name = self.feature_names[self.feature]
        name = (
            feature_name
            if rule.is_root
            else f"{feature_name} | {rule.format(self.feature_names)}"
        )
        nof_instances = int(np.sum(mask))
        return Region(
            idx=idx,
            name=name,
            rule=rule,
            heterogeneity=float(heter),
            nof_instances=nof_instances,
            weight=nof_instances / mask.shape[0],
            level=level,
            parent_idx=parent_idx,
            mask=np.asarray(mask).astype(bool),
        )

    def _propose_candidates(self) -> list:
        """Enumerate every candidate split: candidate-conditioning features in
        order, each feature's proposals in the proposer's (ascending) order —
        the tie-breaking order of the argmin below."""
        candidates = []
        for foc_type, foc in zip(self.foc_types, self.candidate_conditioning_features):
            candidates.extend(self.proposer_factory(foc_type).propose(self.ctx, foc))
        return candidates

    def _evaluate_splits(
        self, before_split_active_indices_list: list
    ) -> typing.Optional[SplitEvaluation]:
        """The shared split search (§2.7): exhaustive scan over every proposed
        candidate, applied to *each* set of active indices in the list (one
        set = node-wise, a whole level = level-wise), and return the candidate
        minimizing the Laplace-weighted heterogeneity (`None` when no
        candidates exist — a degenerate configuration)."""
        heter_func = self.heter_func
        data = self.data

        candidates = self._propose_candidates()
        if not candidates:
            return None

        weighted_heter = np.full(len(candidates), BIG_M, dtype=float)
        evaluated = []
        for k, candidate in enumerate(candidates):
            condition_masks = [c.contains(data) for c in candidate.conditions]
            child_masks = [
                np.logical_and(active_indices, m)
                for active_indices in before_split_active_indices_list
                for m in condition_masks
            ]
            child_heters = [heter_func(m) for m in child_masks]

            # weights analogous to the populations in each split
            populations = np.array([np.sum(m) for m in child_masks])
            child_weights = (populations + 1) / (np.sum(populations + 1))

            weighted_heter[k] = np.sum(child_weights * np.array(child_heters))
            evaluated.append((child_masks, child_heters, populations))

        # the candidate with the minimum weighted heterogeneity (argmin takes
        # the first minimum: ties resolve to the earliest-enumerated candidate)
        best = int(np.argmin(weighted_heter))
        child_masks, child_heters, populations = evaluated[best]
        return SplitEvaluation(
            candidate=candidates[best],
            child_masks=child_masks,
            child_heters=child_heters,
            child_counts=[int(p) for p in populations],
            weighted_heter=weighted_heter[best],
        )


class Best(Base):
    """Node-wise recursive partitioning: find the best split for each node,
    recurse into the children (see `Base.__init__` for the parameters)."""

    def __init__(
        self,
        min_heterogeneity_decrease_pcg: float = 0.1,
        heter_small_enough: float = 0.001,
        max_depth: int = 2,
        min_samples_leaf: int = 10,
        numerical_features_grid_size: int = 20,
        search_partitions_when_categorical: bool = True,
        categorical_proposer="one_vs_rest",
        continuous_proposer="threshold",
    ):
        super().__init__(
            "Best",
            min_heterogeneity_decrease_pcg,
            heter_small_enough,
            max_depth,
            min_samples_leaf,
            numerical_features_grid_size,
            search_partitions_when_categorical,
            categorical_proposer,
            continuous_proposer,
        )

    def fit(self) -> Partition:
        root_mask = np.ones((self.data.shape[0]))
        root_heter = self.heter_func(root_mask)
        root_rule = Rule({})
        self._regions = [
            self._make_region(
                idx=0,
                rule=root_rule,
                mask=root_mask,
                heter=root_heter,
                level=0,
                parent_idx=None,
            )
        ]
        self._recursive_split(
            parent_idx=0,
            parent_rule=root_rule,
            parent_mask=root_mask,
            parent_heter=root_heter,
            level=0,
        )
        return Partition(
            self._regions,
            feature=self.feature,
            feature_name=self.feature_names[self.feature],
            finder_name=self.name,
            feature_names=self.feature_names,
        )

    def _recursive_split(
        self, *, parent_idx, parent_rule, parent_mask, parent_heter, level
    ) -> None:
        """Recursively split a region, appending children pre-order (each
        child is added and fully expanded before its sibling)."""

        # if any of the following, stop before splitting
        conditions = [
            level >= self.max_split_levels,  # Max split levels reached
            np.sum(parent_mask) < self.min_points_per_subregion,  # Not enough points,
            parent_heter < self.heter_small_enough,  # Heterogeneity already small
        ]

        if any(conditions):
            return None

        # find the best split
        split = self._evaluate_splits([parent_mask])
        if split is None:
            return None

        # weighted heterogeneity of the best split
        weights = split.child_counts / np.sum(split.child_counts)
        heter_after = np.sum(weights * np.array(split.child_heters))
        heter_before = parent_heter

        heter_drop_pcg = (heter_before - heter_after) / heter_before
        if heter_drop_pcg < self.heter_pcg_drop_thres:
            return None

        for condition, child_mask, child_heter in zip(
            split.candidate.conditions, split.child_masks, split.child_heters
        ):
            child_rule = parent_rule.refine(condition)
            child_idx = len(self._regions)
            self._regions.append(
                self._make_region(
                    idx=child_idx,
                    rule=child_rule,
                    mask=child_mask,
                    heter=child_heter,
                    level=level + 1,
                    parent_idx=parent_idx,
                )
            )
            self._recursive_split(
                parent_idx=child_idx,
                parent_rule=child_rule,
                parent_mask=child_mask,
                parent_heter=child_heter,
                level=level + 1,
            )


class BestLevelWise(Base):
    """Level-wise partitioning: find the single best split for each level
    (applied to every node of that level at once), then keep the prefix of
    levels whose heterogeneity drop is large enough (see `Base.__init__` for
    the parameters)."""

    def __init__(
        self,
        min_heterogeneity_decrease_pcg: float = 0.1,
        heter_small_enough: float = 0.001,
        max_depth: int = 2,
        min_samples_leaf: int = 10,
        numerical_features_grid_size: int = 20,
        search_partitions_when_categorical: bool = True,
        categorical_proposer="one_vs_rest",
        continuous_proposer="threshold",
    ):
        super().__init__(
            "best_level_wise",
            min_heterogeneity_decrease_pcg,
            heter_small_enough,
            max_depth,
            min_samples_leaf,
            numerical_features_grid_size,
            search_partitions_when_categorical,
            categorical_proposer,
            continuous_proposer,
        )

        # init splits
        self.splits: list = []
        self.important_splits: list = []

        # state variable
        self.split_found: bool = False
        self.important_splits_selected: bool = False

    def fit(self) -> typing.Optional[Partition]:
        self._search_all_splits()
        self._choose_important_splits()
        return self._splits_to_partition(self.important_splits)

    def _search_all_splits(self):
        """
        Iterate over all features of conditioning and choose the best split for each level in a greedy fashion.
        """
        if (
            ingestion.is_categorical(self.feature_types[self.feature])
            and not self.split_categorical_features
        ):
            self.splits = []
        else:
            if self.max_split_levels > len(self.candidate_conditioning_features):
                self.max_split_levels = len(self.candidate_conditioning_features)

            active_indices = np.ones((self.data.shape[0]))
            heter_init = self.heter_func(active_indices)
            # level-0 pseudo-entry: the unsplit root as a one-child "split"
            splits = [
                SplitEvaluation(
                    candidate=None,
                    child_masks=[active_indices],
                    child_heters=[heter_init],
                    child_counts=[len(self.data)],
                    weighted_heter=heter_init,
                )
            ]

            for lev in range(self.max_split_levels):
                # TODO: check this, as it seems redundant;
                # if any subregion had less than min_points, the
                # specific split should not have been selected
                if any(
                    [
                        np.sum(x) < self.min_points_per_subregion
                        for x in splits[-1].child_masks
                    ]
                ):
                    break

                # find optimal split
                new_split = self._evaluate_splits(splits[-1].child_masks)
                if new_split is None:
                    break
                splits.append(new_split)
            self.splits = splits

        # update state
        self.split_found = True
        return self.splits

    def _choose_important_splits(self):
        assert self.split_found, "No splits found for feature {}".format(self.feature)

        # if split is empty, skip
        if len(self.splits) == 0:
            optimal_splits = []
        # if initial heterogeneity is BIG_M, skip
        elif self.splits[0].weighted_heter == BIG_M:
            optimal_splits = []
        # if initial heterogeneity is small right from the beginning, skip
        elif self.splits[0].weighted_heter <= self.heter_small_enough:
            optimal_splits = []
        else:
            splits = self.splits

            # accept split if heterogeneity drops over `heter_pcg_drop_thres`
            heter = np.array([s.weighted_heter for s in splits])
            heter_drop = (heter[:-1] - heter[1:]) / heter[:-1]
            split_valid = heter_drop > self.heter_pcg_drop_thres

            # accept split if heterogeneity is not already small enough
            heter_not_too_small = heter[:-1] > self.heter_small_enough
            split_valid = np.logical_and(split_valid, heter_not_too_small)

            # if all are negative, return nothing
            if np.sum(split_valid) == 0:
                optimal_splits = []
            # if all are positive, return all
            elif np.sum(split_valid) == len(split_valid):
                optimal_splits = splits[1:]
            else:
                # find first negative split
                first_negative = np.where(~split_valid)[0][0]

                # if first negative is the first split, return nothing
                if first_negative == 0:
                    optimal_splits = []
                else:
                    optimal_splits = splits[1 : first_negative + 1]

        # update state variable
        self.important_splits_selected = True
        self.important_splits = optimal_splits
        return optimal_splits

    def _splits_to_partition(self, splits) -> typing.Optional[Partition]:
        """Materialize the selected splits as a `Partition`, level by level
        (BFS insertion order). Each level's `SplitEvaluation.child_masks` is
        parent-major (`j // k` indexes the parent, `j % k` the condition), so
        every child's rule refines its parent's rule with the candidate's
        `j % k`-th condition. Returns `None` when no search ran (categorical
        feature of interest without `search_partitions_when_categorical`)."""
        if len(self.splits) == 0:
            return None

        root_rule = Rule({})
        regions = [
            self._make_region(
                idx=0,
                rule=root_rule,
                mask=np.ones((self.data.shape[0])),
                heter=self.splits[0].child_heters[0],
                level=0,
                parent_idx=None,
            )
        ]
        # (region idx, rule) of every node in the current deepest level
        parent_slots = [(0, root_rule)]

        for lev, split in enumerate(splits):
            k = len(split.candidate.conditions)
            new_slots = []
            for j, (child_mask, child_heter) in enumerate(
                zip(split.child_masks, split.child_heters)
            ):
                parent_idx, parent_rule = parent_slots[j // k]
                child_rule = parent_rule.refine(split.candidate.conditions[j % k])
                child_idx = len(regions)
                regions.append(
                    self._make_region(
                        idx=child_idx,
                        rule=child_rule,
                        mask=child_mask,
                        heter=child_heter,
                        level=lev + 1,
                        parent_idx=parent_idx,
                    )
                )
                new_slots.append((child_idx, child_rule))
            parent_slots = new_slots

        return Partition(
            regions,
            feature=self.feature,
            feature_name=self.feature_names[self.feature],
            finder_name=self.name,
            feature_names=self.feature_names,
        )


def return_default(partitioner_name):
    if partitioner_name == "best":
        return Best()
    elif partitioner_name == "best_level_wise":
        return BestLevelWise()
    else:
        raise ValueError("Partitioner not found")

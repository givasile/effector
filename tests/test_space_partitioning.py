from effector.space_partitioning import *
import numpy as np


def test_space_partitioning():
    np.random.seed(0)
    N = 1000
    D = 3
    # Generate features uniformly in [0, 10].
    X = np.random.uniform(0, 10, size=(N, D))

    # Create a target variable y with four groups.
    # Group 1: x2 < 3 and x3 < 5  -> label 0
    # Group 2: x2 < 3 and x3 >= 5 -> label 1
    # Group 3: x2 >= 3 and x2 < 5 -> label 2
    # Group 4: x2 >= 3 and x2 >= 5 -> label 3
    y = np.empty(N, dtype=int)
    for i in range(N):
        if X[i, 1] < 3:
            y[i] = 0 if X[i, 1] < 1.5 else 1
        else:
            y[i] = 2 if X[i, 1] < 5 else 3

    # Define a heterogeneity function (Gini impurity) that uses the target y.
    def heterogeneity(mask):
        indices = np.where(mask)[0]
        if len(indices) < 50:
            return 10000000000
        labels = y[indices]
        classes, counts = np.unique(labels, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p**2)

    def parent_heter_lower(node, is_lower):
        if not is_lower:
            return False
        if node.parent_node is None:
            return is_lower

        return parent_heter_lower(
            node.parent_node,
            node.info["weighted_heterogeneity"]
            <= node.parent_node.info["weighted_heterogeneity"],
        )

    # Set axis limits (min and max for each feature).
    axis_limits = np.array([[0, 10], [0, 10], [0, 10]]).T

    # We want to allow splits on x1 and x2. To do so, we choose the primary feature as x3 (index 2)
    # and explicitly pass candidate conditioning features [0, 1].
    best = Best(
        min_heterogeneity_decrease_pcg=0.1,
        heter_small_enough=0.0,
        max_depth=2,
        min_samples_leaf=10,
        numerical_features_grid_size=20,
        search_partitions_when_categorical=False,
    )

    best.compile(
        feature=0,  # primary feature (x3) -- not used for splitting in this test.
        data=X,
        heter_func=heterogeneity,
        axis_limits=axis_limits,
        candidate_conditioning_features=[0, 1, 2],
        feature_names=["x1", "x2", "x3"],
        target_name="y",
    )
    tree = best.fit()

    # tree.show_full_tree()

    assert tree is not None

    heter_decreasing_per_level = all([parent_heter_lower(n, True) for n in tree.nodes])
    assert heter_decreasing_per_level

    ############################

    best_level_wise = BestLevelWise(
        min_heterogeneity_decrease_pcg=0.1,
        heter_small_enough=0.0,
        max_depth=2,
        min_samples_leaf=10,
        numerical_features_grid_size=20,
        search_partitions_when_categorical=False,
    )

    best_level_wise.compile(
        feature=0,  # primary feature (x3) -- not used for splitting in this test.
        data=X,
        heter_func=heterogeneity,
        axis_limits=axis_limits,
        candidate_conditioning_features=[0, 1, 2],
        feature_names=["x1", "x2", "x3"],
        target_name="y",
    )
    tree = best_level_wise.fit()

    assert tree is not None

    heter_decreasing_per_level = all([parent_heter_lower(n, True) for n in tree.nodes])
    assert heter_decreasing_per_level

from effector.tree import Tree, Node

def test_tree_get_level_stats():
    tree = Tree()
    info1 = {
        "heterogeneity": 0.5,
        "weight": 1.0,
        "nof_instances": 100,
        "feature": 0,
    }
    info2 = {
        "heterogeneity": 0.3,
        "weight": 0.5,
        "nof_instances": 50,
        "feature": 1,
        "foc_name": "feat1",
        "comparison": ">=",
        "position": 3.0,
        "feature_type": "cont",
        "candidate_split_positions": [0.0, 1.0, 2.0, 3.0, 4., 5.],
        "range": [0, 5],
    }
    info3 = {
        "heterogeneity": 0.2,
        "weight": 0.5,
        "nof_instances": 50,
        "feature": 1,
        "foc_name": "feat1",
        "comparison": "<",
        "position": 3.0,
        "feature_type": "cont",
        "candidate_split_positions": [0.0, 1.0, 2.0, 3.0, 4., 5.],
        "range": [0, 5],
    }
    tree.add_node("root", None, info1, level=0)
    tree.add_node("child1", "root", info2, level=1)
    tree.add_node("child2", "root", info3, level=1)

    stats0 = tree.get_level_stats(0)
    stats1 = tree.get_level_stats(1)
    stats2 = tree.get_level_stats(2)

    tree.show_level_stats()
    tree.show_full_tree()

    scale_x_list=[
        {"mean": 3, "std":2},
        {"mean": 3, "std":2},
        {"mean": 3, "std":2}
    ]
    tree.show_full_tree(scale_x_list=scale_x_list)



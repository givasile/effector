from effector.tree import Tree, Node

def test_tree_get_level_stats():
    tree = Tree()
    info1 = {
        "heterogeneity": 0.5,
        "weight": 1.0,
        "nof_instances": 100,
        "level": 0,
    }
    info2 = {
        "heterogeneity": 0.3,
        "weight": 0.5,
        "nof_instances": 50,
        "feature": 1,
        "foc_name": "x2",
        "comparison": ">=",
        "position": 3.0,
        "feature_type": "cont",
        "candidate_split_positions": [0.0, 1.0, 2.0, 3.0, 4., 5.],
        "range": [0, 5],
        "level": 1,
    }
    info3 = {
        "heterogeneity": 0.2,
        "weight": 0.5,
        "nof_instances": 50,
        "feature": 1,
        "foc_name": "x2",
        "comparison": "<",
        "position": 3.0,
        "feature_type": "cont",
        "candidate_split_positions": [0.0, 1.0, 2.0, 3.0, 4., 5.],
        "range": [0, 5],
        "level": 1,
    }
    tree.add_node("x1", None, info1)
    tree.add_node("x2", "x1", info2)
    tree.add_node("x3", "x1", info3)

    scale_x_list=[
        {"mean": 3, "std":2},
        {"mean": 3, "std":3},
        {"mean": 3, "std":2}
    ]

    tree.show_full_tree(scale_x_list)
    tree.show_level_stats()

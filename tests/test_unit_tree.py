"""Unit layer for effector.tree (PLAN II §3.2): replaces the print-only
test_tree.py with real asserts — derived node stats, exact display strings,
level stats, and the error paths."""

import numpy as np
import pytest

from effector.tree import Tree


def _build_tree():
    tree = Tree()
    tree.add_node(
        "x1",
        None,
        {
            "heterogeneity": 0.5,
            "active_indices": np.array([True, True, True, True, True]),
            "level": 0,
        },
    )
    tree.add_node(
        "x1 | x2 ≥ 3.0",
        "x1",
        {
            "heterogeneity": 0.3,
            "foc_index": 1,
            "foc_name": "x2",
            "comparison": ">=",
            "active_indices": np.array([True, False, True, False, True]),
            "foc_split_position": 3.0,
            "foc_type": "cont",
            "level": 1,
        },
    )
    tree.add_node(
        "x1 | x2 < 3.0",
        "x1",
        {
            "heterogeneity": 0.2,
            "foc_index": 1,
            "foc_name": "x2",
            "comparison": "<",
            "active_indices": np.array([False, True, False, True, False]),
            "foc_split_position": 3.0,
            "foc_type": "cont",
            "level": 1,
        },
    )
    return tree


def test_node_stats_derived_from_active_indices():
    tree = _build_tree()
    root = tree.get_root()
    assert root.info["nof_instances"] == 5
    assert root.info["weight"] == 1.0

    left = tree.get_node_by_idx(1)
    assert left.info["nof_instances"] == 3
    assert left.info["weight"] == pytest.approx(0.6)
    assert left.info["weighted_heterogeneity"] == pytest.approx(0.3 * 0.6)

    right = tree.get_node_by_idx(2)
    assert right.info["nof_instances"] == 2
    assert right.info["weight"] == pytest.approx(0.4)


def test_lookup_helpers():
    tree = _build_tree()
    assert tree.get_root().name == "x1"
    assert tree.get_node_by_name("x1 | x2 < 3.0").idx == 2
    assert tree.get_node_by_idx(99) is None
    assert [c.idx for c in tree.get_children("x1")] == [1, 2]


def test_create_node_name():
    tree = _build_tree()
    root = tree.get_root()
    assert tree.create_node_name("x2", None) == "x2"
    assert tree.create_node_name("x2", root, ">=", 3.0) == "x1 | x2 ≥ 3.0"
    child = tree.get_node_by_idx(1)
    assert (
        tree.create_node_name("x3", child, "==", 0.0) == "x1 | x2 ≥ 3.0 and x3 = 0.0"
    )


def test_create_node_name_requires_comp_and_pos():
    tree = _build_tree()
    with pytest.raises(AssertionError):
        tree.create_node_name("x2", tree.get_root())


def test_set_display_name_unscaled_and_scaled():
    tree = _build_tree()
    assert tree.set_display_name("x1 | x2 ≥ 3.0", None) == "x1 | x2 ≥ 3.00"
    assert tree.set_display_name("x1 | x2 ≥ 3.0", None, full=False) == "x2 ≥ 3.00"

    scale_x_list = [
        {"mean": 3, "std": 2},
        {"mean": 3, "std": 3},
        {"mean": 3, "std": 2},
    ]
    # pos_scaled = std * pos + mean = 3 * 3 + 3 = 12
    assert tree.set_display_name("x1 | x2 ≥ 3.0", scale_x_list) == "x1 | x2 ≥ 12.00"


def test_get_level_stats():
    tree = _build_tree()
    assert tree.get_level_stats(0)["heterogeneity"] == pytest.approx(0.5)
    # weighted: 0.3 * 0.6 + 0.2 * 0.4
    assert tree.get_level_stats(1)["heterogeneity"] == pytest.approx(0.26)


def test_show_methods_smoke(capsys):
    tree = _build_tree()
    tree.show_full_tree()
    tree.show_level_stats()
    out = capsys.readouterr().out
    assert "Full Tree Structure" in out
    assert "Tree Summary" in out


def test_missing_required_info_key_raises():
    tree = Tree()
    with pytest.raises(KeyError):
        tree.add_node("x1", None, {"heterogeneity": 0.5, "level": 0})


def test_unknown_parent_raises():
    tree = _build_tree()
    with pytest.raises(ValueError):
        tree.add_node("orphan", "nonexistent", {})

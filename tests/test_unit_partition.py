"""Unit layer for effector.partition — pure numpy, no models.

Builds Region/Partition objects by hand and pins the container protocol, label
rendering (scaled/unscaled + every glyph), the byte-for-byte show() output, the
to_dict() roundtrip, and the mask()-returns-a-copy contract.
"""

import numpy as np
import pytest

from effector.partition import Partition, Region


def _root(n=6):
    return Region(
        idx=0,
        name="x0",
        mask=np.ones(n, dtype=bool),
        heterogeneity=0.50,
        nof_instances=n,
        weight=1.0,
        level=0,
        parent_idx=None,
    )


def _two_level_partition():
    n = 8
    m_root = np.ones(n, dtype=bool)
    m_left = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    m_right = ~m_left
    root = _root(n)
    left = Region(
        idx=1,
        name="x0 | x1 <= 3.00",
        mask=m_left,
        heterogeneity=0.30,
        nof_instances=int(m_left.sum()),
        weight=float(m_left.sum() / n),
        level=1,
        parent_idx=0,
        foc_index=1,
        foc_name="x1",
        foc_type="numerical",
        foc_split_position=3.0,
        comparison="<=",
    )
    right = Region(
        idx=2,
        name="x0 | x1 > 3.00",
        mask=m_right,
        heterogeneity=0.20,
        nof_instances=int(m_right.sum()),
        weight=float(m_right.sum() / n),
        level=1,
        parent_idx=0,
        foc_index=1,
        foc_name="x1",
        foc_type="numerical",
        foc_split_position=3.0,
        comparison=">",
    )
    return Partition(
        [root, left, right], feature=0, feature_name="x0", finder_name="best"
    )


def test_container_protocol():
    part = _two_level_partition()
    assert len(part) == 3
    assert [r.idx for r in part] == [0, 1, 2]
    assert part[1].foc_name == "x1"
    with pytest.raises(IndexError):
        _ = part[99]


def test_constructor_validates_root():
    n = 4
    bad_root = Region(
        idx=0, name="x0", mask=np.ones(n, dtype=bool), heterogeneity=0.1,
        nof_instances=n, weight=0.5, level=0, parent_idx=None,
    )
    with pytest.raises(ValueError):
        Partition([bad_root], feature=0, feature_name="x0", finder_name="best")


def test_leaves():
    part = _two_level_partition()
    assert [r.idx for r in part.leaves] == [1, 2]
    single = Partition([_root()], feature=0, feature_name="x0", finder_name="best")
    assert [r.idx for r in single.leaves] == [0]


def test_label_root_and_conditions():
    part = _two_level_partition()
    assert part.label(0) == "x0"
    assert part.label(1) == "x0 | x1 ≤ 3.00"
    assert part.label(2) == "x0 | x1 > 3.00"


def test_label_glyphs():
    # exercise all four comparison glyphs via hand-built single-split partitions
    n = 4
    for comparison, glyph in [(">=", "≥"), ("<=", "≤"), ("!=", "≠"), ("==", "=")]:
        child = Region(
            idx=1, name="c", mask=np.array([1, 1, 0, 0], dtype=bool),
            heterogeneity=0.1, nof_instances=2, weight=0.5, level=1, parent_idx=0,
            foc_index=1, foc_name="x1", foc_type="numerical",
            foc_split_position=2.0, comparison=comparison,
        )
        part = Partition(
            [_root(n), child], feature=0, feature_name="x0", finder_name="best"
        )
        assert part.label(1) == f"x0 | x1 {glyph} 2.00"


def test_label_scaled():
    part = _two_level_partition()
    # scale feature 1 by mean=10, std=2 -> 3.0 becomes 16.00
    scale = [None, {"mean": 10.0, "std": 2.0}]
    assert part.label(1, scale_x_list=scale) == "x0 | x1 ≤ 16.00"


def test_show_tree(capsys):
    part = _two_level_partition()
    part.show()
    out = capsys.readouterr().out
    assert "Feature 0 - Full partition tree:" in out
    assert "🌳 Full Tree Structure:" in out
    assert "─" * 23 in out
    assert "x0 🔹 [id: 0 | heter: 0.50 | inst: 8 | w: 1.00]" in out
    assert "x1 ≤ 3.00 🔹 [id: 1 | heter: 0.30 | inst: 4 | w: 0.50]" in out
    assert "Feature 0 - Statistics per tree level:" in out
    assert "🌳 Tree Summary:" in out
    assert "Level 0🔹heter: 0.50" in out


def test_show_single_region(capsys):
    part = Partition([_root()], feature=0, feature_name="x0", finder_name="best")
    part.show()
    out = capsys.readouterr().out
    assert "No splits found for feature 0" in out


def test_to_dict_roundtrip_fields():
    part = _two_level_partition()
    d = part.to_dict()
    assert d["feature"] == 0
    assert d["feature_name"] == "x0"
    assert d["finder"] == "best"
    assert len(d["regions"]) == 3
    r1 = d["regions"][1]
    assert r1["comparison"] == "<="
    assert r1["foc_split_position"] == 3.0
    assert r1["mask"] == part[1].mask.tolist()
    assert "_effect" not in d


def test_mask_returns_copy():
    part = _two_level_partition()
    m = part.mask(1)
    m[:] = False
    assert part[1].mask.any()  # original untouched

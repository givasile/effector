"""Unit layer for effector.partition — pure numpy, no models.

Builds Region/Partition objects by hand (plus a stub effect for the bind /
from_rules contracts) and pins the container protocol, rule-derived label
rendering, the show()/show_axes() output, the to_dict() v2 roundtrip, the
partition invariant, and the mask()-returns-a-copy contract.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import effector
from effector.partition import Partition, Region
from effector.rules import Interval, LevelSet, Rule

FEATURE_NAMES = ["x0", "x1", "x2"]


def _region(idx, rule, mask, heter, level=1, parent_idx=0, weight=None, name="r"):
    n = int(mask.sum())
    return Region(
        idx=idx,
        name=name,
        rule=rule,
        heterogeneity=heter,
        nof_instances=n,
        weight=weight if weight is not None else n / mask.shape[0],
        level=level,
        parent_idx=parent_idx,
        mask=mask,
    )


def _root(n=6):
    return Region(
        idx=0,
        name="x0",
        rule=Rule({}),
        heterogeneity=0.50,
        nof_instances=n,
        weight=1.0,
        level=0,
        parent_idx=None,
        mask=np.ones(n, dtype=bool),
    )


def _two_level_partition():
    n = 8
    m_left = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    m_right = ~m_left
    regions = [
        _root(n),
        _region(
            1, Rule({1: Interval(hi=3.0)}), m_left, 0.30, name="x0 where x1 < 3.00"
        ),
        _region(
            2, Rule({1: Interval(lo=3.0)}), m_right, 0.20, name="x0 where x1 ≥ 3.00"
        ),
    ]
    return Partition(
        regions,
        feature=0,
        feature_name="x0",
        finder_name="best",
        feature_names=FEATURE_NAMES,
    )


class _StubEffect:
    """The duck-typed slice of an effect that bind/from_rules consume."""

    def __init__(self, data, heter=0.1):
        self.data = data
        self.dim = data.shape[1]
        self.scale_x_list = None
        self.feature_names = FEATURE_NAMES[: data.shape[1]]
        self.feature_types = ["continuous"] * data.shape[1]
        self.feature_metadata = SimpleNamespace(category_names=None)
        self._heter = heter

    def heter_score(self, feature, mask=None, rule=None):
        return self._heter


def _matching_data():
    """Data on which the fixture's rules reproduce the fixture's masks."""
    data = np.zeros((8, 2))
    data[:, 1] = [0.0, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]  # x1 < 3 -> first 4
    return data


def test_container_protocol():
    part = _two_level_partition()
    assert len(part) == 3
    assert [r.idx for r in part] == [0, 1, 2]
    assert part[1].rule[1] == Interval(hi=3.0)
    with pytest.raises(IndexError):
        _ = part[99]


def test_constructor_validates_root():
    n = 4
    bad_weight = Region(
        idx=0,
        name="x0",
        rule=Rule({}),
        heterogeneity=0.1,
        nof_instances=n,
        weight=0.5,
        level=0,
        parent_idx=None,
        mask=np.ones(n, dtype=bool),
    )
    with pytest.raises(ValueError, match="weight"):
        Partition([bad_weight], feature=0, feature_name="x0", finder_name="best")
    bad_rule = Region(
        idx=0,
        name="x0",
        rule=Rule({1: Interval(hi=3.0)}),
        heterogeneity=0.1,
        nof_instances=n,
        weight=1.0,
        level=0,
        parent_idx=None,
        mask=np.ones(n, dtype=bool),
    )
    with pytest.raises(ValueError, match="root rule"):
        Partition([bad_rule], feature=0, feature_name="x0", finder_name="best")


def test_constructor_invariant_overlap_and_gap():
    n = 8
    root = _root(n)
    m_left = np.array([1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)  # overlaps m_right
    m_right = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=bool)
    with pytest.raises(ValueError, match="partition the root"):
        Partition(
            [
                root,
                _region(1, Rule({1: Interval(hi=3.0)}), m_left, 0.1),
                _region(2, Rule({1: Interval(lo=3.0)}), m_right, 0.1),
            ],
            feature=0,
            feature_name="x0",
            finder_name="best",
        )
    m_gap = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)  # rows 2-3 uncovered
    with pytest.raises(ValueError, match="partition the root"):
        Partition(
            [
                root,
                _region(1, Rule({1: Interval(hi=3.0)}), m_gap, 0.1),
                _region(2, Rule({1: Interval(lo=3.0)}), m_right, 0.1),
            ],
            feature=0,
            feature_name="x0",
            finder_name="best",
        )


def test_leaves():
    part = _two_level_partition()
    assert [r.idx for r in part.leaves] == [1, 2]
    single = Partition([_root()], feature=0, feature_name="x0", finder_name="best")
    assert [r.idx for r in single.leaves] == [0]


def test_label_root_and_conditions():
    part = _two_level_partition()
    assert part.label(0) == "x0"
    assert part.label(1) == "x0 where x1 < 3.00"
    assert part.label(2) == "x0 where x1 ≥ 3.00"


def test_label_rule_shapes():
    # every rule shape renders through the same format boundary
    n = 4
    cases = [
        (Rule({1: Interval(lo=2.0)}), "x0 where x1 ≥ 2.00"),
        (Rule({1: Interval(hi=2.0)}), "x0 where x1 < 2.00"),
        (Rule({1: Interval(1.0, 2.0)}), "x0 where 1.00 ≤ x1 < 2.00"),
        (Rule({2: LevelSet([2])}), "x0 where x2 = 2.00"),
        (Rule({2: LevelSet([0, 1])}), "x0 where x2 ∈ {0.00, 1.00}"),
        (
            Rule({1: Interval(hi=2.0), 2: LevelSet([0])}),
            "x0 where (x1 < 2.00) and (x2 = 0.00)",
        ),
    ]
    for rule, expected in cases:
        child = _region(1, rule, np.array([1, 1, 0, 0], dtype=bool), 0.1)
        sibling = _region(
            2, Rule({1: Interval(lo=99.0)}), np.array([0, 0, 1, 1], dtype=bool), 0.1
        )
        part = Partition(
            [_root(n), child, sibling],
            feature=0,
            feature_name="x0",
            finder_name="best",
            feature_names=FEATURE_NAMES,
        )
        assert part.label(1) == expected


def test_label_scaled():
    part = _two_level_partition()
    # scale feature 1 by mean=10, std=2 -> 3.0 becomes 16.00
    scale = [None, {"mean": 10.0, "std": 2.0}, None]
    assert part.label(1, scale_x_list=scale) == "x0 where x1 < 16.00"


def test_show_tree(capsys):
    part = _two_level_partition()
    part.show()
    out = capsys.readouterr().out
    assert "Feature 0 - Full partition tree:" in out
    assert "🌳 Full Tree Structure:" in out
    assert "─" * 23 in out
    assert "x0 🔹 [id: 0 | heter: 0.50 | inst: 8 | w: 1.00]" in out
    assert "x1 < 3.00 🔹 [id: 1 | heter: 0.30 | inst: 4 | w: 0.50]" in out
    assert "x1 ≥ 3.00 🔹 [id: 2 | heter: 0.20 | inst: 4 | w: 0.50]" in out
    assert "Feature 0 - Statistics per tree level:" in out
    assert "🌳 Tree Summary:" in out
    assert "Level 0🔹heter: 0.50" in out


def test_show_own_condition_diffs_against_parent(capsys):
    # a level-2 node re-splitting the same feature prints only its own refinement
    n = 8
    m_left = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    m_ll = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool)
    m_lr = np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=bool)
    regions = [
        _root(n),
        _region(1, Rule({1: Interval(hi=3.0)}), m_left, 0.3),
        _region(2, Rule({1: Interval(lo=3.0)}), ~m_left, 0.2),
        _region(3, Rule({1: Interval(hi=1.5)}), m_ll, 0.1, level=2, parent_idx=1),
        _region(4, Rule({1: Interval(1.5, 3.0)}), m_lr, 0.1, level=2, parent_idx=1),
    ]
    part = Partition(
        regions,
        feature=0,
        feature_name="x0",
        finder_name="best",
        feature_names=FEATURE_NAMES,
    )
    part.show()
    out = capsys.readouterr().out
    assert "x1 < 1.50 🔹 [id: 3" in out
    assert "1.50 ≤ x1 < 3.00 🔹 [id: 4" in out


def test_show_single_region(capsys):
    part = Partition([_root()], feature=0, feature_name="x0", finder_name="best")
    part.show()
    out = capsys.readouterr().out
    assert "No splits found for feature 0" in out


def test_show_axes_one_axis(capsys):
    part = _two_level_partition()
    part.show_axes()
    out = capsys.readouterr().out
    assert "Feature 0 - Partition along 1 axis:" in out
    assert "x1 < 3.00 🔹 [id: 1 | heter: 0.30 | inst: 4 | w: 0.50]" in out
    assert "x1 ≥ 3.00 🔹 [id: 2 | heter: 0.20 | inst: 4 | w: 0.50]" in out


def test_show_axes_two_axis_grid(capsys):
    n = 8
    masks = {
        (0, 0): np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        (0, 1): np.array([0, 0, 1, 1, 0, 0, 0, 0], dtype=bool),
        (1, 0): np.array([0, 0, 0, 0, 1, 1, 0, 0], dtype=bool),
        (1, 1): np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype=bool),
    }
    iv = {0: Interval(hi=3.0), 1: Interval(lo=3.0)}
    ls = {0: LevelSet([0]), 1: LevelSet([1])}
    regions = [_root(n)] + [
        _region(
            k + 1,
            Rule({1: iv[i], 2: ls[j]}),
            masks[(i, j)],
            0.1,
            level=2,
            parent_idx=0,
        )
        for k, (i, j) in enumerate(masks)
    ]
    part = Partition(
        regions,
        feature=0,
        feature_name="x0",
        finder_name="best",
        feature_names=FEATURE_NAMES,
    )
    part.show_axes()
    out = capsys.readouterr().out
    assert "Feature 0 - Partition along 2 axes:" in out
    assert "x2 = 0.00" in out and "x2 = 1.00" in out
    assert "x1 < 3.00" in out and "x1 ≥ 3.00" in out
    assert "[id: 1 | heter: 0.10 | inst: 2]" in out


def test_show_axes_falls_back_to_tree(capsys):
    # leaves conditioning on three features -> tree print
    n = 8
    m1 = np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=bool)
    regions = [
        _root(n),
        _region(
            1,
            Rule({0: Interval(hi=1.0), 1: Interval(hi=2.0), 2: LevelSet([0])}),
            m1,
            0.1,
        ),
        _region(2, Rule({1: Interval(lo=99.0)}), ~m1, 0.1),
    ]
    part = Partition(
        regions,
        feature=0,
        feature_name="x0",
        finder_name="best",
        feature_names=FEATURE_NAMES,
    )
    part.show_axes()
    out = capsys.readouterr().out
    assert "Full partition tree" in out


def test_to_dict_v2_roundtrip():
    part = _two_level_partition()
    d = part.to_dict()
    assert d["schema_version"] == 2
    assert d["feature"] == 0
    assert d["feature_name"] == "x0"
    assert d["feature_names"] == FEATURE_NAMES
    assert d["finder"] == "best"
    assert len(d["regions"]) == 3
    r1 = d["regions"][1]
    assert "mask" not in r1
    assert r1["rule"] == {
        "conditions": [
            {
                "feature": 1,
                "kind": "interval",
                "lo": None,
                "hi": 3.0,
                "lo_closed": False,
                "hi_closed": False,
            }
        ]
    }
    assert "_effect" not in d

    restored = Partition.from_dict(d)
    assert restored.label(1) == part.label(1)
    assert restored[1].rule == part[1].rule
    assert restored[1].mask is None
    with pytest.raises(RuntimeError, match="bind"):
        restored.mask(1)


def test_from_dict_rejects_v1():
    with pytest.raises(ValueError, match="schema_version 2"):
        Partition.from_dict({"feature": 0, "regions": [{"mask": [True]}]})


def test_bind_recomputes_and_verifies():
    part = Partition.from_dict(_two_level_partition().to_dict())
    effect = _StubEffect(_matching_data())
    part.bind(effect)
    assert part.mask(1).tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    # bound sugar now works at the mask level
    assert part[2].mask.sum() == 4


def test_bind_raises_on_different_data():
    part = Partition.from_dict(_two_level_partition().to_dict())
    wrong = _matching_data()
    wrong[:, 1] = 100.0  # every x1 >= 3 -> counts cannot match
    with pytest.raises(ValueError, match="nof_instances"):
        part.bind(_StubEffect(wrong))


def test_bind_raises_on_mask_mismatch():
    part = _two_level_partition()  # masks present
    with pytest.raises(ValueError, match="different data"):
        part.bind(_StubEffect(np.zeros((8, 2))))  # rules select everything/nothing


def test_from_rules_happy_path_and_validation():
    effect = _StubEffect(_matching_data(), heter=0.25)
    part = Partition.from_rules(
        [Rule({1: Interval(hi=3.0)}), Rule({1: Interval(lo=3.0)})],
        effect=effect,
        feature=0,
    )
    assert len(part) == 3
    assert part.finder_name == "user"
    assert part[1].nof_instances == 4 and part[1].heterogeneity == 0.25
    assert part.label(1) == "x0 where x1 < 3.00"
    assert part.mask(2).sum() == 4
    # string rules parse with the effect's metadata
    part2 = Partition.from_rules(["x1 < 3", "x1 >= 3"], effect=effect, feature=0)
    assert part2[1].rule == part[1].rule
    # non-covering rules violate the partition invariant
    with pytest.raises(ValueError, match="partition the root"):
        Partition.from_rules(
            [Rule({1: Interval(hi=1.0)}), Rule({1: Interval(lo=3.0)})],
            effect=effect,
            feature=0,
        )


def test_mask_returns_copy():
    part = _two_level_partition()
    m = part.mask(1)
    m[:] = False
    assert part[1].mask.any()  # original untouched


# ---------------------------------------------------------------------------
# integer-valued features, conditions(), tree_lines()
# ---------------------------------------------------------------------------


def _integer_effect():
    """x0's effect flips on x1 (integer hours 0..23) and x2 (a 0/1 flag)."""
    rng = np.random.default_rng(0)
    n = 600
    data = np.column_stack(
        [
            rng.uniform(-1, 1, n),
            rng.integers(0, 24, n).astype(float),
            rng.integers(0, 2, n).astype(float),
        ]
    )

    def model(x):
        return x[:, 0] * np.where(x[:, 1] < 6.9, 1.0, -1.0) + x[:, 0] * x[:, 2]

    fx = effector.PDP(
        data,
        model,
        nof_instances="all",
        schema={
            "feature_names": ["x", "hr", "flag"],
            "feature_types": ["continuous", "continuous", "ordinal"],
        },
    )
    fx.fit("all")
    return fx


def test_integer_features_print_as_the_integers_they_admit():
    fx = _integer_effect()
    part = Partition.from_rules(
        [
            Rule({1: Interval(hi=6.9)}),
            Rule({1: Interval(6.9, 14.95), 2: LevelSet([0])}),
            Rule({1: Interval(6.9, 14.95), 2: LevelSet([1])}),
            Rule({1: Interval(lo=14.95)}),
        ],
        effect=fx,
        feature=0,
    )
    assert part.label(1) == "x where hr ≤ 6"
    assert part.label(2) == "x where (7 ≤ hr ≤ 14) and (flag = 0)"
    assert part.label(4) == "x where hr ≥ 15"
    assert part.conditions(2) == ["7 ≤ hr ≤ 14", "flag = 0"]
    assert part.conditions(0) == []
    # the stamp survives serialization: an unbound partition prints the same
    unbound = Partition.from_dict(part.to_dict())
    assert unbound.label(1) == "x where hr ≤ 6"


def test_continuous_features_keep_their_decimals():
    fx = _integer_effect()
    part = Partition.from_rules(
        [Rule({0: Interval(hi=0.0)}), Rule({0: Interval(lo=0.0)})],
        effect=fx,
        feature=1,
    )
    # and a threshold at zero never prints as -0.00
    assert part.label(1) == "hr where x < 0.00"
    assert Rule({0: Interval(hi=-0.0)}).format(["x"]) == "x < 0.00"


def test_tree_lines_is_a_plain_tree_with_leaf_counts():
    fx = _integer_effect()
    part = fx.find_regions(0, finder=effector.space_partitioning.Best(max_depth=2))
    lines = part.tree_lines()
    assert lines[0] == "x"
    body = lines[1:]
    assert len(body) == len(part) - 1
    assert all(line.lstrip("│ ").startswith(("├─ ", "└─ ")) for line in body)
    counts = [line for line in body if " n = " in line]
    assert len(counts) == len(part.leaves)
    total = sum(int(line.rsplit("n = ", 1)[1].replace(",", "")) for line in counts)
    assert total == 600
    # the counts are aligned in one column
    assert len({line.index(" n = ") for line in counts}) == 1
    assert "heter" not in "\n".join(lines)
    assert all(" n = " not in line for line in part.tree_lines(counts=False))

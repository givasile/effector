"""Unit layer for effector.rules — the predicate algebra.

Pure numpy; no models, no effects. Pins the algebra's contracts: interval
closedness semantics, level-set membership, rule normalization, the
rule->mask site, serialization round-trips, formatting glyphs, and the
string parser.
"""

import numpy as np
import pytest

from effector.rules import Condition, Interval, LevelSet, Rule, subset_from_dict

# ---------------------------------------------------------------------------
# Interval
# ---------------------------------------------------------------------------


def test_interval_defaults_are_half_open():
    iv = Interval(hi=3.0)
    col = np.array([2.9, 3.0, 3.1])
    assert iv.contains(col).tolist() == [True, False, False]  # x < t
    iv = Interval(lo=3.0)
    assert iv.contains(col).tolist() == [False, True, True]  # x >= t


@pytest.mark.parametrize(
    "lo_closed, hi_closed, expected",
    [
        (True, True, [True, True, True]),
        (True, False, [True, True, False]),
        (False, True, [False, True, True]),
        (False, False, [False, True, False]),
    ],
)
def test_interval_flag_combinations(lo_closed, hi_closed, expected):
    iv = Interval(1.0, 3.0, lo_closed, hi_closed)
    assert iv.contains(np.array([1.0, 2.0, 3.0])).tolist() == expected


def test_interval_none_bounds_normalize_to_inf():
    assert Interval(None, 3.0) == Interval(-np.inf, 3.0)
    assert Interval(3.0, None) == Interval(3.0, np.inf)
    full = Interval()
    assert full.contains(np.array([-1e300, 0.0, 1e300])).all()


def test_interval_infinite_bound_forces_open_flag():
    # [-inf, t) must equal (-inf, t) — containment at -inf is impossible
    assert Interval(-np.inf, 3.0, lo_closed=True) == Interval(
        -np.inf, 3.0, lo_closed=False
    )
    assert hash(Interval(None, 3.0, lo_closed=True)) == hash(
        Interval(-np.inf, 3.0, lo_closed=False)
    )


def test_interval_nan_raises():
    with pytest.raises(ValueError, match="NaN"):
        Interval(np.nan, 3.0)


@pytest.mark.parametrize(
    "iv, empty",
    [
        (Interval(3.0, 1.0), True),  # lo > hi
        (Interval(3.0, 3.0), True),  # point, hi open by default
        (Interval(3.0, 3.0, True, True), False),  # closed point [t, t]
        (Interval(1.0, 3.0), False),
    ],
)
def test_interval_is_empty(iv, empty):
    assert iv.is_empty is empty


def test_interval_intersect_nesting_and_overlap():
    assert Interval(1.0, 10.0).intersect(Interval(3.0, 5.0)) == Interval(3.0, 5.0)
    assert Interval(1.0, 4.0).intersect(Interval(3.0, 6.0)) == Interval(3.0, 4.0)
    assert Interval(1.0, 2.0).intersect(Interval(3.0, 4.0)).is_empty


def test_interval_intersect_tied_bounds_merge_flags():
    a = Interval(1.0, 3.0, lo_closed=True, hi_closed=True)
    b = Interval(1.0, 3.0, lo_closed=True, hi_closed=False)
    merged = a.intersect(b)
    assert merged.hi_closed is False and merged.lo_closed is True


def test_interval_closed_point_intersection():
    # [1, 3] ∩ [3, 5] = the closed point [3, 3]
    a = Interval(1.0, 3.0, True, True)
    b = Interval(3.0, 5.0, True, True)
    merged = a.intersect(b)
    assert not merged.is_empty
    assert merged.contains(np.array([3.0])).tolist() == [True]


def test_interval_to_dict_round_trip():
    for iv in [
        Interval(hi=3.0),
        Interval(lo=-1.5, hi=2.5, lo_closed=False, hi_closed=True),
        Interval(),
    ]:
        d = iv.to_dict()
        assert d["kind"] == "interval"
        assert Interval.from_dict(d) == iv
        assert subset_from_dict(d) == iv
    # infinite bounds serialize as None (JSON-safe)
    assert Interval(hi=3.0).to_dict()["lo"] is None


def test_interval_format_glyphs():
    assert Interval(hi=3.0).format("x") == "x < 3.00"
    assert Interval(hi=3.0, hi_closed=True).format("x") == "x ≤ 3.00"
    assert Interval(lo=3.0).format("x") == "x ≥ 3.00"
    assert Interval(lo=3.0, lo_closed=False).format("x") == "x > 3.00"
    assert Interval(1.0, 3.0).format("x") == "1.00 ≤ x < 3.00"
    assert Interval(3.0, 3.0, True, True).format("x") == "x = 3.00"
    assert Interval(3.0, 1.0).format("x") == "x ∈ ∅"
    assert Interval().format("x") == "x ∈ (-∞, ∞)"


def test_interval_format_scales_both_bounds():
    scale = {"mean": 10.0, "std": 2.0}
    assert Interval(3.0, 5.0).format("x", scale=scale) == "16.00 ≤ x < 20.00"
    assert Interval(hi=3.0).format("x", scale=scale) == "x < 16.00"


# ---------------------------------------------------------------------------
# LevelSet
# ---------------------------------------------------------------------------


def test_levelset_contains():
    ls = LevelSet([0, 2])
    assert ls.contains(np.array([0.0, 1.0, 2.0, 3.0])).tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_levelset_eq_hash_order_insensitive():
    assert LevelSet([1, 2]) == LevelSet([2.0, 1.0])
    assert hash(LevelSet([1, 2])) == hash(LevelSet([2.0, 1.0]))


def test_levelset_intersect_and_empty():
    assert LevelSet([0, 1, 2]).intersect(LevelSet([1, 2, 3])) == LevelSet([1, 2])
    assert LevelSet([0]).intersect(LevelSet([1])).is_empty
    assert LevelSet([]).is_empty


def test_levelset_interval_mixed_intersection():
    # symmetric: interval filters the levels
    ls, iv = LevelSet([0, 1, 2, 3]), Interval(1.0, 3.0)  # [1, 3)
    assert ls.intersect(iv) == LevelSet([1, 2])
    assert iv.intersect(ls) == LevelSet([1, 2])


def test_levelset_round_trip():
    ls = LevelSet([3, 1])
    d = ls.to_dict()
    assert d == {"kind": "levels", "levels": [1.0, 3.0]}
    assert LevelSet.from_dict(d) == ls
    assert subset_from_dict(d) == ls


def test_levelset_format():
    assert LevelSet([0]).format("x") == "x = 0.00"
    assert LevelSet([1, 0]).format("x") == "x ∈ {0.00, 1.00}"
    many = LevelSet(range(24))
    assert many.format("x") == "x ∈ {0.00, 1.00, 2.00, …} (24 levels)"
    assert LevelSet([]).format("x") == "x ∈ ∅"


def test_levelset_format_category_names():
    names = {0.0: "winter", 1.0: "spring"}
    assert LevelSet([0]).format("season", level_names=names) == "season = winter"
    assert (
        LevelSet([0, 1]).format("season", level_names=names)
        == "season ∈ {winter, spring}"
    )
    # uncovered level falls back to numeric
    assert LevelSet([5]).format("season", level_names=names) == "season = 5.00"


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------


def test_condition_validation_and_contains():
    with pytest.raises(ValueError, match="non-negative int"):
        Condition(-1, Interval(hi=3.0))
    X = np.array([[0.0, 5.0], [0.0, 1.0]])
    assert Condition(1, Interval(hi=3.0)).contains(X).tolist() == [False, True]


def test_condition_format_resolves_metadata():
    cond = Condition(1, Interval(hi=3.0))
    assert cond.format() == "x_1 < 3.00"
    assert cond.format(feature_names=["a", "temp"]) == "temp < 3.00"
    scale_x_list = [None, {"mean": 10.0, "std": 2.0}]
    assert (
        cond.format(feature_names=["a", "temp"], scale_x_list=scale_x_list)
        == "temp < 16.00"
    )
    cat = Condition(0, LevelSet([1]))
    assert (
        cat.format(feature_names=["season", "b"], category_names={0: {1.0: "spring"}})
        == "season = spring"
    )


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------


def test_rule_root():
    root = Rule({})
    assert root.is_root and not root.is_empty
    assert root.format() == ""
    X = np.random.default_rng(0).normal(size=(7, 3))
    assert root.contains(X).all()


def test_rule_normalizes_same_feature_conditions():
    r = Rule([Condition(0, Interval(lo=3.0)), Condition(0, Interval(hi=5.0))])
    assert r.features == (0,)
    assert r[0] == Interval(3.0, 5.0)


def test_rule_drops_full_interval():
    assert Rule({0: Interval()}) == Rule({})
    assert Rule({0: Interval()}).is_root


def test_rule_contains_equals_and_of_conditions():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 4))
    X[:, 2] = rng.integers(0, 3, size=50)
    r = Rule({0: Interval(hi=0.5), 2: LevelSet([0, 2])})
    expected = (X[:, 0] < 0.5) & np.isin(X[:, 2], [0.0, 2.0])
    assert np.array_equal(r.contains(X), expected)


def test_rule_contains_validates_input():
    r = Rule({5: Interval(hi=1.0)})
    with pytest.raises(ValueError, match="2D"):
        r.contains(np.zeros(3))
    with pytest.raises(ValueError, match="feature 5"):
        r.contains(np.zeros((3, 2)))


def test_rule_intersect_and_refine():
    a = Rule({0: Interval(lo=1.0)})
    b = Rule({0: Interval(hi=4.0), 1: LevelSet([0])})
    both = a.intersect(b)
    assert both[0] == Interval(1.0, 4.0) and both[1] == LevelSet([0])
    assert a.refine(Condition(1, LevelSet([0]))) == Rule(
        {0: Interval(lo=1.0), 1: LevelSet([0])}
    )


def test_rule_is_empty_propagates():
    assert Rule({0: Interval(3.0, 1.0)}).is_empty
    assert Rule({0: Interval(lo=5.0)}).intersect(Rule({0: Interval(hi=1.0)})).is_empty


def test_rule_eq_hash_order_insensitive_format_order_preserving():
    a = Rule({0: Interval(hi=3.0), 1: LevelSet([0])})
    b = Rule({1: LevelSet([0]), 0: Interval(hi=3.0)})
    assert a == b and hash(a) == hash(b)
    # two or more conditions are parenthesized; a single one stays bare
    assert a.format() == "(x_0 < 3.00) and (x_1 = 0.00)"
    assert b.format() == "(x_1 = 0.00) and (x_0 < 3.00)"
    assert Rule({0: Interval(hi=3.0)}).format() == "x_0 < 3.00"


def test_rule_round_trip():
    r = Rule({0: Interval(hi=3.0), 2: LevelSet([0, 2])})
    d = r.to_dict()
    assert d["conditions"][0] == {
        "feature": 0,
        "kind": "interval",
        "lo": None,
        "hi": 3.0,
        "lo_closed": False,
        "hi_closed": False,
    }
    assert Rule.from_dict(d) == r


def test_rule_repr():
    assert repr(Rule({})) == "Rule(<root>)"
    assert repr(Rule({0: Interval(hi=3.0)})) == "Rule(x_0 < 3.00)"


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

NAMES = ["temp", "hr", "season"]
TYPES = ["continuous", "ordinal", "nominal"]
LEVELS = {1: np.arange(24.0), 2: np.array([0.0, 1.0, 2.0, 3.0])}
CATNAMES = {2: {0.0: "winter", 1.0: "spring", 2.0: "summer", 3.0: "fall"}}


def parse(text, **overrides):
    kwargs = dict(
        feature_names=NAMES,
        feature_types=TYPES,
        levels=LEVELS,
        category_names=CATNAMES,
    )
    kwargs.update(overrides)
    return Rule.parse(text, **kwargs)


def test_parse_continuous_ops():
    assert parse("temp < 3") == Rule({0: Interval(hi=3.0)})
    assert parse("temp <= 3") == Rule({0: Interval(hi=3.0, hi_closed=True)})
    assert parse("temp > 3") == Rule({0: Interval(lo=3.0, lo_closed=False)})
    assert parse("temp >= 3") == Rule({0: Interval(lo=3.0)})


def test_parse_numeric_formats():
    assert parse("temp < -1.5") == Rule({0: Interval(hi=-1.5)})
    assert parse("temp < 1e-3") == Rule({0: Interval(hi=0.001)})
    assert parse("temp < 3") == parse("temp < 3.0")


def test_parse_categorical_ops():
    assert parse("season == 2") == Rule({2: LevelSet([2])})
    assert parse("season = 2") == Rule({2: LevelSet([2])})  # '=' alias
    assert parse("hr in {7, 8}") == Rule({1: LevelSet([7, 8])})
    assert parse("season != 0") == Rule({2: LevelSet([1, 2, 3])})


def test_parse_category_names_resolve():
    assert parse("season == winter") == Rule({2: LevelSet([0])})
    assert parse("season != 'winter'") == Rule({2: LevelSet([1, 2, 3])})
    assert parse("season in {winter, fall}") == Rule({2: LevelSet([0, 3])})


def test_parse_conjunction_and_case_tolerance():
    r = parse("temp < 3 AND season == winter")
    assert r == Rule({0: Interval(hi=3.0), 2: LevelSet([0])})
    r = parse("temp >= 1 and temp < 3")
    assert r == Rule({0: Interval(1.0, 3.0)})


def test_parse_errors():
    with pytest.raises(ValueError, match="unknown feature"):
        parse("windspeed < 3")
    with pytest.raises(ValueError, match="continuous"):
        parse("temp == 3")
    with pytest.raises(ValueError, match="categorical"):
        parse("season < 2")
    with pytest.raises(ValueError, match="contradictory"):
        parse("temp < 1 and temp > 3")
    with pytest.raises(ValueError, match="cannot parse"):
        parse("temp is small")
    with pytest.raises(ValueError, match="cannot resolve"):
        parse("season == monsoon")
    with pytest.raises(ValueError, match="levels"):
        parse("season != 0", levels=None)
    with pytest.raises(ValueError, match="types were not provided"):
        parse("season == 2", feature_types=None)
    with pytest.raises(ValueError, match="empty level set"):
        parse("season in {}")

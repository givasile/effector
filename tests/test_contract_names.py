"""Contract tests — feature names on every verb (the api shell).

Every public verb resolves `feature` given as a name to the same result as the
integer index: `pdp.plot("a") ≡ pdp.plot(0)`. Names come from the schema
(`feature_names`); without declared names the synthesized `x_0…` work the same
way. Unknown names raise a ValueError that lists the available names (R9);
bools are rejected as always.
"""

import numpy as np
import pytest

from tests.conftest import GLOBAL_NAMES, eval_mean, make_global

NAMES = ["alpha", "beta", "gamma"]
SCHEMA = {"feature_names": NAMES}


def _make(name, data):
    return make_global(name, data, schema=SCHEMA)


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_eval_by_name_equals_by_index(name, global_data):
    m = _make(name, global_data)
    xs = np.linspace(-0.8, 0.8, 7)
    np.testing.assert_array_equal(
        eval_mean(m, "beta", xs), eval_mean(m, 1, xs)
    )


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_scalar_verbs_by_name(name, global_data):
    m = _make(name, global_data)
    xs = np.linspace(-0.8, 0.8, 7)
    assert m.importance("alpha") == m.importance(0)
    assert m.heter_score("gamma") == m.heter_score(2)
    np.testing.assert_array_equal(m.eval_heter("beta", xs), m.eval_heter(1, xs))
    assert m.payload("alpha").keys() == m.payload(0).keys()


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_fit_with_names(name, global_data):
    m = _make(name, global_data)
    m.fit(features=["alpha", "gamma"])
    assert m.is_fitted[0] and m.is_fitted[2]
    assert not m.is_fitted[1]


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_plot_by_name(name, global_data):
    m = _make(name, global_data)
    m.plot("beta", show_plot=False)


def test_find_regions_by_name(global_data):
    m = _make("pdp", global_data)
    part_by_name = m.find_regions("beta")
    part_by_idx = m.find_regions(1)
    assert part_by_name.to_dict() == part_by_idx.to_dict()


def test_find_regions_conditioning_by_name(global_data):
    m = _make("pdp", global_data)
    part = m.find_regions("beta", candidate_conditioning_features=["alpha"])
    assert part.to_dict() == m.find_regions(
        1, candidate_conditioning_features=[0]
    ).to_dict()


def test_synthesized_names_resolve(global_data):
    m = make_global("pdp", global_data)  # no schema -> x_0, x_1, x_2
    xs = np.linspace(-0.8, 0.8, 5)
    np.testing.assert_array_equal(eval_mean(m, "x_1", xs), eval_mean(m, 1, xs))


def test_unknown_name_lists_available(global_data):
    m = _make("pdp", global_data)
    with pytest.raises(ValueError, match=r"alpha.*beta.*gamma"):
        m.importance("delta")


def test_bool_feature_rejected(global_data):
    m = _make("pdp", global_data)
    with pytest.raises(TypeError, match="bool"):
        m.importance(True)


def test_out_of_range_index_rejected(global_data):
    m = _make("pdp", global_data)
    with pytest.raises(ValueError, match="out of range"):
        m.heter_score(3)


def test_fit_unknown_name_rejected(global_data):
    m = _make("pdp", global_data)
    with pytest.raises(ValueError, match="Unknown feature"):
        m.fit(features=["alpha", "delta"])

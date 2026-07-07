"""TEMPORARY cutover safety net (deleted in the Regional* removal commit).

While both paths coexist, prove the new `find_regions` reproduces the old
`Regional*` tree byte-for-byte on the gated model: same region count, same
masks, same heterogeneities, same split metadata per node.
"""

import numpy as np
import pytest

import effector
from tests.conftest import gated_model, gated_model_jac, make_regional_data


def _old_nodes(reg):
    return reg.tree["feature_0"].nodes


@pytest.mark.parametrize("name", ["pdp", "rhale"])
def test_regional_tree_matches_find_regions(name):
    data = make_regional_data()
    finder_old = effector.space_partitioning.Best(max_depth=2)
    finder_new = effector.space_partitioning.Best(max_depth=2)

    if name == "pdp":
        reg = effector.RegionalPDP(data, gated_model, nof_instances="all")
        fx = effector.PDP(data, gated_model, nof_instances="all")
    else:
        reg = effector.RegionalRHALE(
            data, gated_model, model_jac=gated_model_jac, nof_instances="all"
        )
        fx = effector.RHALE(
            data, gated_model, model_jac=gated_model_jac, nof_instances="all"
        )

    reg.fit(0, space_partitioner=finder_old)
    fx.fit(0, centering=False)
    part = fx.find_regions(0, finder=finder_new)

    old = _old_nodes(reg)
    assert len(old) == len(part), f"{name}: region count differs"

    for node, region in zip(old, part):
        assert np.array_equal(
            node.info["active_indices"].astype(bool), region.mask
        ), f"{name}: mask differs at region {region.idx}"
        np.testing.assert_allclose(
            node.info["heterogeneity"],
            region.heterogeneity,
            atol=1e-10,
            err_msg=f"{name}: heterogeneity differs at region {region.idx}",
        )
        assert node.info.get("foc_index") == region.foc_index
        assert node.info.get("comparison") == region.comparison
        assert node.info.get("foc_split_position") == region.foc_split_position

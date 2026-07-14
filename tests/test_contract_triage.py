"""Contract tests — `effector.plot_triage` (the api shell).

The triage plane is a view over `importance` (x) and `heter_score` (y): the
scatter offsets must equal the verbs' outputs, the default threshold line is
the median heterogeneity (the explain() convention), and with `partitions` one
arrow runs from each partitioned feature's global point to each of its leaf
points (leaf coordinates = the verbs under the leaf's rule).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

import effector
from tests.conftest import gated_model, make_regional_data

matplotlib.use("Agg")

SCHEMA = {"feature_names": ["g", "gate", "flag"]}


@pytest.fixture(scope="module")
def pdp():
    data = make_regional_data(n=300)
    fx = effector.PDP(data, gated_model, nof_instances="all", schema=SCHEMA)
    fx.fit("all", centering=False)
    return fx


def _arrow_count(ax):
    return sum(
        1
        for child in ax.get_children()
        if isinstance(child, matplotlib.text.Annotation) and child.arrowprops
    )


def test_offsets_equal_verbs(pdp):
    fig, ax = effector.plot_triage(pdp, show_plot=False)
    offsets = ax.collections[0].get_offsets()
    expected = np.array([[pdp.importance(f), pdp.heter_score(f)] for f in range(3)])
    np.testing.assert_allclose(np.asarray(offsets), expected, atol=1e-12)
    plt.close(fig)


def test_threshold_line_is_median(pdp):
    fig, ax = effector.plot_triage(pdp, show_plot=False)
    hs = [pdp.heter_score(f) for f in range(3)]
    thr_lines = [ln for ln in ax.get_lines() if "threshold" in str(ln.get_label())]
    assert len(thr_lines) == 1
    assert thr_lines[0].get_ydata()[0] == pytest.approx(float(np.median(hs)))
    plt.close(fig)


def test_threshold_false_and_float(pdp):
    fig, ax = effector.plot_triage(pdp, threshold=False, show_plot=False)
    assert not [ln for ln in ax.get_lines() if "threshold" in str(ln.get_label())]
    plt.close(fig)
    fig, ax = effector.plot_triage(pdp, threshold=0.5, show_plot=False)
    line = [ln for ln in ax.get_lines() if "threshold" in str(ln.get_label())][0]
    assert line.get_ydata()[0] == 0.5
    plt.close(fig)


def test_partitions_draw_one_arrow_per_leaf(pdp):
    parts = pdp.find_regions(features=["g"])
    fig, ax = effector.plot_triage(pdp, partitions=parts, show_plot=False)
    assert _arrow_count(ax) == len(parts["g"].leaves)
    plt.close(fig)


def test_partitions_keys_by_index_too(pdp):
    part = pdp.find_regions("g")
    fig, ax = effector.plot_triage(pdp, partitions={0: part}, show_plot=False)
    assert _arrow_count(ax) == len(part.leaves)
    plt.close(fig)


def test_leaf_points_equal_masked_verbs(pdp):
    part = pdp.find_regions("g")
    fig, ax = effector.plot_triage(pdp, partitions={"g": part}, show_plot=False)
    leaf_offsets = np.concatenate(
        [np.asarray(c.get_offsets()) for c in ax.collections[1:]]
    )
    expected = np.array(
        [
            [pdp.importance("g", rule=leaf.rule), pdp.heter_score("g", rule=leaf.rule)]
            for leaf in part.leaves
        ]
    )
    np.testing.assert_allclose(
        np.sort(leaf_offsets, axis=0), np.sort(expected, axis=0), atol=1e-12
    )
    plt.close(fig)


def test_features_subset_and_names(pdp):
    fig, ax = effector.plot_triage(pdp, features=["g", "gate"], show_plot=False)
    assert np.asarray(ax.collections[0].get_offsets()).shape == (2, 2)
    plt.close(fig)


def test_unsupported_features_skipped_with_warning(monkeypatch):
    from effector import ingestion

    monkeypatch.setattr(
        effector.DerPDP, "SUPPORTED_FEATURE_TYPES", frozenset({ingestion.CONTINUOUS})
    )
    data = make_regional_data(n=300)
    fx = effector.DerPDP(
        data,
        gated_model,
        model_jac=lambda x: np.stack(
            [
                np.where((x[:, 1] > 0) & (x[:, 2] == 0), 5.0, 0.0),
                np.zeros(len(x)),
                np.zeros(len(x)),
            ],
            axis=1,
        ),
        nof_instances="all",
        schema=SCHEMA,
    )
    with pytest.warns(UserWarning, match="flag"):
        fig, ax = effector.plot_triage(fx, show_plot=False)
    assert np.asarray(ax.collections[0].get_offsets()).shape == (2, 2)
    plt.close(fig)


def test_junk_features_string_raises(pdp):
    with pytest.raises(ValueError, match="'all'"):
        effector.plot_triage(pdp, features="heterogeneous", show_plot=False)

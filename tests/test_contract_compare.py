"""Contract tests — `effector.compare` (the api shell).

`compare(*effects, feature=...)` is the cross-examination verb standing above
the engines: it queries each fitted effect's public `eval` on a shared grid
and overlays the curves. The drawn lines must equal the effects' own `eval`
(R1: a plot is a view over eval); mixing derivative-unit with level-unit
effects raises; comparisons are always centered.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

import effector
from tests.conftest import make_global, make_global_data

SCHEMA = {"feature_names": ["a", "b", "c"]}


@pytest.fixture(scope="module")
def trio():
    data = make_global_data()
    effects = [make_global(n, data, schema=SCHEMA) for n in ("pdp", "ale", "rhale")]
    yield effects
    plt.close("all")


def test_compare_returns_fig_ax(trio):
    fig, ax = effector.compare(*trio, feature="b", show_plot=False)
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert labels == ["PDP", "ALE", "RHALE"]


def test_compare_lines_match_eval(trio):
    fig, ax = effector.compare(*trio, feature=1, show_plot=False)
    for eff, line in zip(trio, ax.get_lines()):
        xdata, ydata = line.get_xdata(), line.get_ydata()
        np.testing.assert_allclose(
            ydata, eff.eval(1, xdata, centering="zero_integral"), atol=1e-8
        )


def test_compare_duplicate_methods_deduped(trio):
    pdp2 = make_global("pdp", trio[0].data, schema=SCHEMA)
    fig, ax = effector.compare(trio[0], pdp2, feature=0, show_plot=False)
    assert [ln.get_label() for ln in ax.get_lines()] == ["PDP", "PDP (2)"]


def test_compare_custom_labels(trio):
    fig, ax = effector.compare(
        *trio, feature=0, labels=["one", "two", "three"], show_plot=False
    )
    assert [ln.get_label() for ln in ax.get_lines()] == ["one", "two", "three"]
    with pytest.raises(ValueError, match="labels"):
        effector.compare(*trio, feature=0, labels=["just-one"], show_plot=False)


def test_compare_uncentered_coerced_with_warning(trio):
    with pytest.warns(UserWarning, match="zero_integral"):
        fig, ax = effector.compare(*trio, feature=0, centering=False, show_plot=False)
    ydata = ax.get_lines()[0].get_ydata()
    np.testing.assert_allclose(
        ydata,
        trio[0].eval(0, ax.get_lines()[0].get_xdata(), centering="zero_integral"),
        atol=1e-8,
    )


def test_compare_needs_two_effects(trio):
    with pytest.raises(ValueError, match="at least two"):
        effector.compare(trio[0], feature=0, show_plot=False)


def test_compare_rejects_derivative_mix(trio):
    der = make_global("derpdp", trio[0].data, schema=SCHEMA)
    with pytest.raises(ValueError, match="derivative"):
        effector.compare(trio[0], der, feature=0, show_plot=False)


def test_compare_all_derivative_ok(trio):
    der1 = make_global("derpdp", trio[0].data, schema=SCHEMA)
    der2 = make_global("derpdp", trio[0].data, schema=SCHEMA)
    fig, ax = effector.compare(der1, der2, feature=0, show_plot=False)
    assert ax.get_ylabel() == "dy/dx"


def test_compare_categorical_feature():
    rng = np.random.default_rng(3)
    data = np.stack(
        [rng.uniform(-1, 1, 300), rng.integers(0, 3, 300).astype(float)], axis=1
    )
    model = lambda x: x[:, 0] + np.where(x[:, 1] == 2.0, 1.0, 0.0)
    schema = {"feature_names": ["num", "cat"], "feature_types": ["cont", "nominal"]}
    pdp = effector.PDP(data, model, schema=schema)
    ale = effector.ALE(
        data, model, schema={**schema, "feature_types": ["cont", "ordinal"]}
    )
    fig, ax = effector.compare(pdp, ale, feature="cat", show_plot=False)
    # per-level marker series: 3 levels on the x axis
    assert all(len(ln.get_xdata()) == 3 for ln in ax.get_lines())


def test_compare_disagreeing_dims_raise(trio):
    other = make_global("pdp", make_global_data(n=50)[:, :2])
    with pytest.raises(ValueError, match="same columns"):
        effector.compare(trio[0], other, feature=0, show_plot=False)

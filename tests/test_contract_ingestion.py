"""Contract layer for R10 — the input contract (docs/design.md R10).

The load-bearing promises:
- effector is numpy-only: `from_dataframe(df)` reproduces the numpy matrix so a
  DataFrame origin gives identical numbers out (exact, atol=0);
- every class exposes the resolved `feature_metadata`;
- legacy aliases ("cont"/"cat") are normalized at the door and keep the
  categorical split path working;
- internally built objects (regional nodes, facade sub-methods) inherit the
  parent's resolved metadata instead of re-inferring from subsets.
"""

import dataclasses

import numpy as np
import pytest

import effector
from effector import ingestion
from tests.conftest import (
    GLOBAL_NAMES,
    analytic_shap_values,
    eval_mean,
    gated_model,
    linear_model,
    make_global,
    make_global_data,
    make_global_df,
    make_mixed_df,
    make_regional,
    make_regional_data,
)

# ---------------------------------------------------------------------------
# R10.1 — DataFrame / numpy parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_r10_df_numpy_parity_eval(name):
    data = make_global_data()
    xs = np.linspace(-0.8, 0.8, 7)

    m_np = make_global(name, data)
    # from_dataframe must reproduce the numpy matrix exactly -> identical output
    X_df, schema = effector.from_dataframe(make_global_df())
    # shap values must be an aligned ndarray on both paths (they bypass ingest)
    kwargs = {"shap_values": analytic_shap_values(data)} if name == "shapdp" else {}
    m_df = make_global(name, X_df, schema=schema, **kwargs)
    y_np = eval_mean(m_np, 0, xs, centering=False)
    y_df = eval_mean(m_df, 0, xs, centering=False)
    np.testing.assert_array_equal(y_np, y_df)


@pytest.mark.parametrize("name", ["regional_pdp", "regional_ale"])
def test_r10_df_numpy_parity_regional_tree(name):
    import pandas as pd

    data = make_regional_data()
    df = pd.DataFrame(data, columns=["a", "b", "c"])

    # identical resolved metadata on both paths: a float64 DataFrame column
    # would otherwise be dtype-decided continuous while the numpy heuristic
    # reads the same binary values as ordinal (both correct per R10)
    types = ["continuous", "continuous", "ordinal"]
    reg_np = make_regional(name, data, schema={"feature_types": types})
    cls = effector.RegionalPDP if name == "regional_pdp" else effector.RegionalALE
    # numpy-only: convert the DataFrame first, force the shared types, and feed
    # the plain numpy model (the encoded matrix equals `data`)
    X_df, schema = effector.from_dataframe(df)
    schema = dataclasses.replace(schema, feature_types=types)
    reg_df = cls(X_df, gated_model, schema=schema)
    part = effector.space_partitioning.Best(max_depth=2)
    reg_np.fit(0, space_partitioner=part)
    reg_df.fit(0, space_partitioner=part)

    nodes_np = [n.name for n in reg_np.tree["feature_0"].nodes]
    nodes_df = [n.name for n in reg_df.tree["feature_0"].nodes]
    # same split structure; names differ only by the column labels
    assert len(nodes_np) == len(nodes_df)
    for a, b in zip(nodes_np, nodes_df):
        assert a.replace("x_1", "b").replace("x_2", "c").replace("x_0", "a") == b


# ---------------------------------------------------------------------------
# R10.2 — metadata exposed everywhere
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_r10_feature_metadata_exposed_global(name):
    m = make_global(name, make_global_data())
    meta = m.feature_metadata
    assert isinstance(meta, ingestion.FeatureMetadata)
    assert len(meta.feature_names) == 3
    assert all(t in ingestion.VALID_FEATURE_TYPES for t in meta.feature_types)
    assert m.feature_names == list(meta.feature_names)
    assert m.feature_types == list(meta.feature_types)
    assert m.target_name == "y"


@pytest.mark.parametrize(
    "name",
    ["regional_pdp", "regional_derpdp", "regional_ale", "regional_rhale"],
)
def test_r10_feature_metadata_exposed_regional(name):
    reg = make_regional(name, make_regional_data())
    assert isinstance(reg.feature_metadata, ingestion.FeatureMetadata)
    assert len(reg.feature_types) == 3
    # binary integer-valued column -> ordinal under the numpy heuristic
    assert reg.feature_types[2] == ingestion.ORDINAL


def test_r10_feature_metadata_exposed_facade():
    fe = effector.FeatureEffect(make_global_data(), linear_model)
    assert isinstance(fe.feature_metadata, ingestion.FeatureMetadata)
    assert fe.feature_names == ["x_0", "x_1", "x_2"]


# ---------------------------------------------------------------------------
# R10.3 — aliases normalized, categorical split path intact
# ---------------------------------------------------------------------------


def test_r10_alias_stored_canonical_global():
    m = make_global(
        "pdp",
        make_global_data(),
        schema={"feature_types": ["cont", "cat", "continuous"]},
    )
    assert m.feature_types == ["continuous", "nominal", "continuous"]


def test_r10_alias_stored_canonical_regional_and_splits():
    reg = make_regional(
        "regional_pdp",
        make_regional_data(),
        schema={"feature_types": ["cont", "cont", "cat"]},
    )
    assert reg.feature_types == ["continuous", "continuous", "nominal"]
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=2))
    # the x2 split must stay categorical-style (= / ≠ in the tree display)
    names = " ".join(n.name for n in reg.tree["feature_0"].nodes)
    assert "≠" in names


def test_r10_schema_object_equals_dict():
    data = make_global_data()
    m1 = make_global("pdp", data, schema=effector.Schema(feature_names=["a", "b", "c"]))
    m2 = make_global("pdp", data, schema={"feature_names": ["a", "b", "c"]})
    assert m1.feature_names == m2.feature_names == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# R10.4 — internally built objects inherit the parent's metadata
# ---------------------------------------------------------------------------


def test_r10_regional_node_inherits_types():
    reg = make_regional("regional_pdp", make_regional_data())
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=2))
    # the ONE internally built global effect (all node eval/plot delegate to
    # its masked summaries) must NOT re-infer: parent's types verbatim
    assert reg._global_fe.feature_types == reg.feature_types
    assert reg._global_fe.cat_limit == reg.cat_limit


def test_r10_facade_submethods_inherit_types():
    fe = effector.FeatureEffect(
        make_global_data(),
        linear_model,
        schema={"feature_types": ["cont", "cat", "cont"]},
    )
    pdp = fe._get_method("pdp")
    assert pdp.feature_types == ["continuous", "nominal", "continuous"]


# ---------------------------------------------------------------------------
# R10.5 — from_dataframe fidelity: the encoded matrix drives the methods
# ---------------------------------------------------------------------------


def test_r10_mixed_df_shapdp_with_analytic_values():
    df = make_mixed_df()
    with pytest.warns(UserWarning, match="cardinality heuristic"):
        matrix, schema = effector.from_dataframe(df)
    shap_values = 0.1 * (matrix - matrix.mean(axis=0))  # any aligned array

    def model(X):  # mixed_df_model, expressed on the encoded matrix (numpy->numpy)
        return 2.0 * X[:, 0] + X[:, 1] + X[:, 2] + 0.5 * X[:, 3]

    m = effector.ShapDP(matrix, model, shap_values=shap_values, schema=schema)
    y = m.eval(0, np.linspace(-0.5, 0.5, 5), centering=False)
    assert y.shape == (5,)


# ---------------------------------------------------------------------------
# R10.6 — scaling precedence: plot kwarg > schema > None; False disables
# ---------------------------------------------------------------------------

SCALE_X = {"mean": 10.0, "std": 2.0}
SCALE_SCHEMA = {"scale_x_list": [SCALE_X, None, None]}


def _first_line_xdata(m, **plot_kwargs):
    fig, ax = m.plot(0, show_plot=False, **plot_kwargs)
    return (
        ax.lines[0].get_xdata()
        if not isinstance(ax, np.ndarray)
        else ax[0].lines[0].get_xdata()
    )


def test_r10_scale_constructor_applied_at_plot():
    data = make_global_data()
    plain = make_global("pdp", data)
    scaled = make_global("pdp", data, schema=SCALE_SCHEMA)
    x_plain = _first_line_xdata(plain, heterogeneity=False)
    x_scaled = _first_line_xdata(scaled, heterogeneity=False)
    np.testing.assert_allclose(x_scaled, x_plain * SCALE_X["std"] + SCALE_X["mean"])


def test_r10_plot_kwarg_overrides_constructor_scale():
    data = make_global_data()
    scaled = make_global("pdp", data, schema=SCALE_SCHEMA)
    override = {"mean": 0.0, "std": 5.0}
    plain = make_global("pdp", data)
    x_plain = _first_line_xdata(plain, heterogeneity=False)
    x_over = _first_line_xdata(scaled, heterogeneity=False, scale_x=override)
    np.testing.assert_allclose(x_over, x_plain * 5.0)


def test_r10_scale_false_disables():
    data = make_global_data()
    plain = make_global("pdp", data)
    scaled = make_global("pdp", data, schema=SCALE_SCHEMA)
    x_plain = _first_line_xdata(plain, heterogeneity=False)
    x_off = _first_line_xdata(scaled, heterogeneity=False, scale_x=False)
    np.testing.assert_allclose(x_off, x_plain)


def test_r10_summary_uses_stored_scale(capsys):
    reg = make_regional(
        "regional_pdp",
        make_regional_data(),
        schema={"scale_x_list": [None, None, {"mean": 100.0, "std": 1.0}]},
    )
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=1))
    reg.summary(0)
    out = capsys.readouterr().out
    # the x_2 = 0 split prints in scaled units (= 100.00) without passing
    # scale_x_list to summary
    assert "100.00" in out


# ---------------------------------------------------------------------------
# R10.7 — category_names: human-readable level labels on categorical plots
# ---------------------------------------------------------------------------

_CAT_NAMES = ["male", "female", "non-binary"]


def _cat_model(x):
    x = np.asarray(x)
    return x[:, 0] + 0.5 * x[:, 1]


def _cat_gated(x):
    # gender effect only when x1 > 0 -> RegionalPDP(gender) splits on x1
    x = np.asarray(x)
    return x[:, 0] * (x[:, 1] > 0).astype(float)


def _cat_data(n=600):
    rng = np.random.default_rng(0)
    g = rng.integers(0, 3, n).astype(float)
    g[:3] = [0.0, 1.0, 2.0]
    return np.column_stack([g, rng.uniform(-1, 1, n)])


def _xtick_texts(ax):
    axes = list(np.atleast_1d(ax).ravel()) if isinstance(ax, np.ndarray) else [ax]
    for a in axes:
        texts = [t.get_text() for t in a.get_xticklabels()]
        if any(nm in texts for nm in _CAT_NAMES):
            return texts
    return [t.get_text() for t in axes[0].get_xticklabels()]


@pytest.mark.parametrize("cls", [effector.PDP, effector.ALE, effector.ShapDP])
def test_r10_category_names_on_axis(cls):
    schema = {
        "feature_types": ["nominal", "continuous"],
        "category_names": [_CAT_NAMES, None],
    }
    m = cls(_cat_data(), _cat_model, schema=schema)
    m.fit(0, centering="zero_integral")
    _, ax = m.plot(0, centering="zero_integral", show_plot=False)
    assert _xtick_texts(ax) == _CAT_NAMES


def test_r10_category_names_default_codes_when_absent():
    m = effector.PDP(
        _cat_data(), _cat_model, schema={"feature_types": ["nominal", "continuous"]}
    )
    m.fit(0, centering="zero_integral")
    _, ax = m.plot(0, centering="zero_integral", show_plot=False)
    assert _xtick_texts(ax) == ["0", "1", "2"]


def test_r10_category_names_regional_node():
    schema = {
        "feature_types": ["nominal", "continuous"],
        "category_names": [_CAT_NAMES, None],
    }
    reg = effector.RegionalPDP(_cat_data(1500), _cat_gated, schema=schema)
    reg.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=1))
    _, ax = reg.plot(0, 1, centering="zero_integral", show_plot=False)
    assert _xtick_texts(ax) == _CAT_NAMES


def test_r10_category_names_regional_split_on_categorical():
    # regression: a node that restricts a categorical SPLIT feature to a subset
    # of its levels must not crash on re-ingest. category_names is value-keyed,
    # so `plot(continuous_feature, node)` works even though gender is now partial.
    schema = {
        "feature_types": ["nominal", "continuous"],
        "category_names": [_CAT_NAMES, None],
    }
    reg = effector.RegionalPDP(_cat_data(1500), _cat_gated, schema=schema)
    reg.fit(1, space_partitioner=effector.space_partitioning.Best(max_depth=2))
    for node_idx in range(len(reg.tree["feature_1"].nodes)):
        reg.plot(1, node_idx, show_plot=False)  # must not raise


def _ord_1based(n=1500):
    rng = np.random.default_rng(0)
    x0 = rng.integers(1, 5, n).astype(float)  # ordinal codes 1..4 (non-0-based)
    return np.column_stack([x0, rng.uniform(-1, 1, n)])


def _ord_model(z):
    return z[:, 0] * (z[:, 1] > 0) + 0.1 * z[:, 1]


def test_ale_plot_non_zero_based_ordinal_global():
    # P1 regression: ALE/RHALE .plot() built its grid from positional codes
    # 0..K-1 and fed them to eval, which rejects non-observed values -> crash
    # whenever the level codes are not 0..K-1 (here 1..4).
    a = effector.ALE(
        _ord_1based(), _ord_model, schema={"feature_types": ["ordinal", "continuous"]}
    )
    a.fit(0)
    _, ax = a.plot(0, show_plot=False)  # must not raise
    ticks = [t.get_text() for t in np.atleast_1d(ax).ravel()[0].get_xticklabels()]
    assert ticks[:4] == ["1", "2", "3", "4"]


def test_ale_plot_non_zero_based_ordinal_regional():
    # regional ALE builds a global ALE per node and calls its .plot(), so it
    # inherits the same P1 crash; the root node exercises the categorical path.
    ra = effector.RegionalALE(
        _ord_1based(), _ord_model, schema={"feature_types": ["ordinal", "continuous"]}
    )
    ra.fit(0, space_partitioner=effector.space_partitioning.Best(max_depth=2))
    ra.plot(0, 0, show_plot=False)  # must not raise

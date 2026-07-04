"""Contract layer for R10 — the input contract (docs/design.md R10).

The load-bearing promises:
- numpy in == DataFrame in: identical numbers out (exact, atol=0);
- every class exposes the resolved `feature_metadata`;
- legacy aliases ("cont"/"cat") are normalized at the door and keep the
  categorical split path working;
- internally built objects (regional nodes, facade sub-methods) inherit the
  parent's resolved metadata instead of re-inferring from subsets.
"""

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
    mixed_df_model,
)

# ---------------------------------------------------------------------------
# R10.1 — DataFrame / numpy parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", GLOBAL_NAMES)
def test_r10_df_numpy_parity_eval(name):
    data = make_global_data()
    df = make_global_df()
    xs = np.linspace(-0.8, 0.8, 7)

    m_np = make_global(name, data)
    # shap values must be an aligned ndarray on both paths (they bypass ingest)
    kwargs = {"shap_values": analytic_shap_values(data)} if name == "shapdp" else {}
    m_df = make_global(name, df, **kwargs)
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
    # the model receives a reconstructed DataFrame on the DF path (R10) —
    # an array-expecting model uses the documented escape hatch
    reg_df = cls(
        df,
        lambda x: gated_model(x.to_numpy()),
        schema={"feature_types": types},
    )
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
    node_fe = reg._create_fe_object(0, 1, None)
    # node subsets must NOT re-infer: parent's types verbatim
    assert node_fe.feature_types == reg.feature_types
    assert node_fe.cat_limit == reg.cat_limit


def test_r10_facade_submethods_inherit_types():
    fe = effector.FeatureEffect(
        make_global_data(),
        linear_model,
        schema={"feature_types": ["cont", "cat", "cont"]},
    )
    pdp = fe._get_method("pdp")
    assert pdp.feature_types == ["continuous", "nominal", "continuous"]


# ---------------------------------------------------------------------------
# R10.5 — the model-call rule, end to end
# ---------------------------------------------------------------------------


def test_r10_mixed_df_end_to_end():
    df = make_mixed_df()
    calls = []

    def recording_model(x):
        calls.append(
            (
                type(x).__name__,
                str(x["color"].dtype),
                bool(x["size"].cat.ordered),
            )
        )
        return mixed_df_model(x)

    with pytest.warns(UserWarning, match="cardinality heuristic"):
        pdp = effector.PDP(df, recording_model)
    xs = np.linspace(-0.5, 0.5, 5)
    y = pdp.eval(0, xs, centering=False)

    assert y.shape == (5,)
    assert len(calls) > 0
    for type_name, color_dtype, size_ordered in calls:
        assert type_name == "DataFrame"
        assert color_dtype == "category"
        assert size_ordered is True


def test_r10_mixed_df_shapdp_with_analytic_values():
    df = make_mixed_df()
    matrix = ingestion.ingest(df, mixed_df_model).data
    shap_values = 0.1 * (matrix - matrix.mean(axis=0))  # any aligned array
    m = effector.ShapDP(
        df,
        mixed_df_model,
        shap_values=shap_values,
        schema={"feature_types": ["continuous", "ordinal", "nominal", "ordinal"]},
    )
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

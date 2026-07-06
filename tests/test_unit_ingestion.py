"""Unit tests for effector/ingestion.py — the R10 input contract.

Layer: unit (Tier-1, tiny N). effector is numpy-only: constructors take a numpy
`data` + a numpy->numpy `model`; a DataFrame is converted first with the
`from_dataframe` convenience. The contract-level DF->numpy parity tests live in
tests/test_contract_ingestion.py.
"""

import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from effector.ingestion import (
    CONTINUOUS,
    NOMINAL,
    ORDINAL,
    Schema,
    from_dataframe,
    infer_feature_types,
    ingest,
    normalize_feature_types,
)


def _model(x):
    return np.asarray(x)[:, 0] * 1.0


def _num_data(n=30):
    rng = np.random.default_rng(21)
    return rng.uniform(-1, 1, size=(n, 3))


def _mixed_df(n=30):
    rng = np.random.default_rng(21)
    return pd.DataFrame(
        {
            "num": rng.uniform(-1, 1, n),
            "count": rng.integers(0, 3, n),
            "color": pd.Categorical(rng.choice(["r", "g", "b"], n)),
            "size": pd.Categorical(
                rng.choice(["S", "M", "L"], n),
                categories=["S", "M", "L"],
                ordered=True,
            ),
            "label": rng.choice(["yes", "no"], n).astype(object),
        }
    )


# ---------------------------------------------------------------------------
# numpy path: passthrough + rejections
# ---------------------------------------------------------------------------


def test_ingest_numpy_passthrough_identity():
    data = _num_data()
    res = ingest(data, _model)
    assert res.data is data  # no copy, no dtype cast
    assert res.model is _model  # model is never wrapped
    assert res.model_jac is None


def test_ingest_numpy_int_dtype_not_cast():
    data = np.arange(60).reshape(30, 2)  # int dtype, nunique >= cat_limit
    res = ingest(data, _model)
    assert res.data is data
    assert res.data.dtype == data.dtype


def test_ingest_rejects_1d():
    with pytest.raises(ValueError, match="2D"):
        ingest(np.arange(10.0), _model)


def test_ingest_rejects_object_ndarray():
    data = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(TypeError, match="non-numeric dtype"):
        ingest(data, _model)


def test_ingest_rejects_unknown_type():
    with pytest.raises(TypeError, match="must be a 2D numpy array"):
        ingest([[1.0, 2.0]], _model)


def test_ingest_rejects_dataframe():
    df = pd.DataFrame({"a": np.linspace(0, 1, 30)})
    with pytest.raises(TypeError, match="from_dataframe"):
        ingest(df, _model)


# ---------------------------------------------------------------------------
# numpy inference rules
# ---------------------------------------------------------------------------


def test_infer_numpy_int_valued_below_limit_is_ordinal():
    col_ord = np.tile(np.arange(3.0), 10)
    col_cont = np.linspace(0, 1, 30)
    data = np.stack([col_ord, col_cont], axis=1)
    assert infer_feature_types(data) == [ORDINAL, CONTINUOUS]


def test_infer_numpy_float_valued_is_continuous():
    # few unique values but NOT integer-valued -> continuous (standardized dummy)
    col = np.tile(np.array([-0.87, 1.15]), 15)
    data = col.reshape(-1, 1)
    assert infer_feature_types(data) == [CONTINUOUS]


def test_infer_numpy_boundary_at_cat_limit():
    # strictly-less-than cat_limit is ordinal
    col_a = np.tile(np.arange(9.0), 10)[:90]
    col_b = np.tile(np.arange(10.0), 9)[:90]
    data = np.stack([col_a, col_b], axis=1)
    assert infer_feature_types(data, cat_limit=10) == [ORDINAL, CONTINUOUS]


def test_infer_numpy_never_nominal():
    data = np.stack([np.tile(np.arange(2.0), 15), np.zeros(30) + 0.5], axis=1)
    assert NOMINAL not in infer_feature_types(data)


# ---------------------------------------------------------------------------
# from_dataframe: dtype inference table + numpy/schema extraction
# ---------------------------------------------------------------------------


def test_from_dataframe_dtype_table():
    with pytest.warns(UserWarning, match="cardinality heuristic"):
        X, schema = from_dataframe(_mixed_df())
    assert schema.feature_types == [
        CONTINUOUS,  # float
        ORDINAL,  # int, nunique < cat_limit (heuristic)
        NOMINAL,  # unordered category
        ORDINAL,  # ordered category
        NOMINAL,  # object strings
    ]
    assert schema.feature_names == ["num", "count", "color", "size", "label"]
    assert X.dtype == np.float64 and X.shape == (30, 5)


def test_from_dataframe_int_large_is_continuous_and_silent():
    df = pd.DataFrame({"a": np.arange(30)})
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # int->continuous is safe
        _, schema = from_dataframe(df)
    assert schema.feature_types == [CONTINUOUS]


def test_from_dataframe_bool_is_ordinal():
    df = pd.DataFrame({"a": np.tile([True, False], 15)})
    _, schema = from_dataframe(df)
    assert schema.feature_types == [ORDINAL]
    assert schema.category_names[0] == ["False", "True"]


def test_from_dataframe_ordered_category_keeps_declared_order():
    with pytest.warns(UserWarning):
        _, schema = from_dataframe(_mixed_df())
    # ordered 'size' -> ordinal, labels in declared (not alphabetical) order
    assert schema.feature_types[3] == ORDINAL
    assert schema.category_names[3] == ["S", "M", "L"]
    # unordered 'color' -> nominal
    assert schema.feature_types[2] == NOMINAL


def test_from_dataframe_datetime_raises():
    df = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=5)})
    with pytest.raises(ValueError, match="unsupported dtype"):
        from_dataframe(df)


def test_from_dataframe_nan_raises_naming_column():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    with pytest.raises(ValueError, match="'a'.*missing"):
        from_dataframe(df)


def test_from_dataframe_uses_column_names():
    df = pd.DataFrame({"a": np.linspace(0, 1, 30), "b": np.linspace(0, 1, 30)})
    _, schema = from_dataframe(df)
    assert schema.feature_names == ["a", "b"]


def test_from_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError, match="expects a pandas DataFrame"):
        from_dataframe(_num_data())


def test_from_dataframe_roundtrip_numpy_and_schema():
    df = pd.DataFrame(
        {
            "num": np.linspace(-1, 1, 6),
            "grade": pd.Categorical(
                ["low", "mid", "high", "low", "mid", "high"],
                categories=["low", "mid", "high"],
                ordered=True,
            ),
            "color": pd.Categorical(["r", "g", "b", "r", "g", "b"]),
        }
    )
    X, schema = from_dataframe(df)
    assert X.dtype == np.float64 and X.shape == (6, 3)
    assert schema.feature_types == [CONTINUOUS, ORDINAL, NOMINAL]
    assert schema.category_names[1] == ["low", "mid", "high"]  # ordered kept
    assert schema.category_names[2] == ["b", "g", "r"]  # unordered -> sorted
    np.testing.assert_array_equal(X[:, 1], df["grade"].cat.codes.to_numpy())
    # the (X, schema) pair drives a constructor; labels resolve by value
    res = ingest(X, _model, schema=schema)
    assert res.meta.category_names[1] == {0.0: "low", 1.0: "mid", 2.0: "high"}


# ---------------------------------------------------------------------------
# schema: define-or-infer, aliases, validation
# ---------------------------------------------------------------------------


def test_alias_cont_cat_normalized():
    assert normalize_feature_types(["cont", "cat", "ordinal"], 3) == [
        CONTINUOUS,
        NOMINAL,
        ORDINAL,
    ]


def test_invalid_type_string_raises():
    with pytest.raises(ValueError, match="invalid feature type 'categorical'"):
        normalize_feature_types(["categorical"], 1)


def test_explicit_types_override_inference():
    data = np.stack([np.tile(np.arange(3.0), 10)] * 2, axis=1)
    res = ingest(data, _model, schema={"feature_types": ["cont", "ordinal"]})
    assert res.meta.feature_types == [CONTINUOUS, ORDINAL]


def test_explicit_names_override_inference():
    res = ingest(_num_data(), _model, schema={"feature_names": ["a", "b", "renamed"]})
    assert res.meta.feature_names == ["a", "b", "renamed"]


def test_schema_dataclass_and_dict_equivalent():
    data = _num_data()
    s1 = ingest(data, _model, schema=Schema(feature_names=["a", "b", "c"]))
    s2 = ingest(data, _model, schema={"feature_names": ["a", "b", "c"]})
    assert s1.meta == s2.meta


def test_schema_unknown_key_raises():
    with pytest.raises(ValueError, match="unknown schema key"):
        ingest(_num_data(), _model, schema={"names": ["a", "b", "c"]})


def test_feature_names_wrong_length_raises():
    with pytest.raises(ValueError, match="feature_names has length 2, expected 3"):
        ingest(_num_data(), _model, schema={"feature_names": ["a", "b"]})


def test_feature_types_wrong_length_raises():
    with pytest.raises(ValueError, match="feature_types has length"):
        ingest(_num_data(), _model, schema={"feature_types": ["cont"]})


def test_scale_x_list_wrong_length_raises():
    with pytest.raises(ValueError, match="scale_x_list has length"):
        ingest(_num_data(), _model, schema={"scale_x_list": [None]})


def test_scale_x_missing_key_raises():
    scale_x_list = [{"mean": 0.0}, None, None]
    with pytest.raises(ValueError, match="missing key.*std"):
        ingest(_num_data(), _model, schema={"scale_x_list": scale_x_list})


def test_scale_y_zero_std_raises():
    with pytest.raises(ValueError, match="non-zero"):
        ingest(_num_data(), _model, schema={"scale_y": {"mean": 0.0, "std": 0}})


def test_cat_limit_bool_raises():
    with pytest.raises(TypeError, match="cat_limit"):
        ingest(_num_data(), _model, schema={"cat_limit": True})


def test_cat_limit_too_small_raises():
    with pytest.raises(ValueError, match="cat_limit"):
        ingest(_num_data(), _model, schema={"cat_limit": 1})


def test_target_name_default_and_override():
    assert ingest(_num_data(), _model).meta.target_name == "y"
    res = ingest(_num_data(), _model, schema={"target_name": "price"})
    assert res.meta.target_name == "price"


# ---------------------------------------------------------------------------
# heuristic-inference warning
# ---------------------------------------------------------------------------


def test_heuristic_warning_fires_for_numpy_int_columns():
    data = np.stack([np.tile(np.arange(3.0), 10), np.linspace(0, 1, 30)], axis=1)
    with pytest.warns(UserWarning, match="'x_0' -> ordinal"):
        ingest(data, _model)


def test_heuristic_warning_silent_for_dtype_decided():
    df = pd.DataFrame(
        {
            "num": np.linspace(0, 1, 30),
            "color": pd.Categorical(["r", "g", "b"] * 10),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        from_dataframe(df)


def test_heuristic_warning_silenced_by_explicit_types():
    data = np.stack([np.tile(np.arange(3.0), 10), np.linspace(0, 1, 30)], axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        ingest(data, _model, schema={"feature_types": ["ordinal", "cont"]})


# ---------------------------------------------------------------------------
# no-pandas guarantee
# ---------------------------------------------------------------------------


def test_numpy_path_never_imports_pandas():
    script = """
import sys

class Blocker:
    def find_module(self, name, path=None):
        return self if name == "pandas" or name.startswith("pandas.") else None
    def find_spec(self, name, path=None, target=None):
        if name == "pandas" or name.startswith("pandas."):
            raise ImportError("pandas import blocked by test")
        return None

sys.meta_path.insert(0, Blocker())

import numpy as np
import effector

data = np.random.default_rng(21).uniform(-1, 1, size=(50, 2))
pdp = effector.PDP(data, lambda x: x[:, 0])
pdp.eval(0, np.linspace(-1, 1, 5), centering=False)
print("OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


# ---------------------------------------------------------------------------
# category_names — schema field for human-readable categorical level labels
# ---------------------------------------------------------------------------


def _cat_num_data(n=30):
    """3-level categorical column 0 (codes 0/1/2) + a continuous column."""
    rng = np.random.default_rng(21)
    g = rng.integers(0, 3, n).astype(float)
    g[:3] = [0.0, 1.0, 2.0]  # guarantee all three levels are present
    return np.column_stack([g, rng.uniform(-1, 1, n)])


def test_category_names_resolved_to_value_map():
    # stored as a {feature_idx: {level_value: name}} map (value-keyed so regional
    # nodes with a level subset map correctly), not the raw positional list
    res = ingest(
        _cat_num_data(),
        _model,
        schema={
            "feature_types": ["nominal", "continuous"],
            "category_names": [["a", "b", "c"], None],
        },
    )
    assert res.meta.category_names == {0: {0.0: "a", 1.0: "b", 2.0: "c"}}


def test_category_names_wrong_length_raises():
    with pytest.raises(ValueError, match="observed levels"):
        ingest(
            _cat_num_data(),
            _model,
            schema={
                "feature_types": ["nominal", "continuous"],
                "category_names": [["a", "b"], None],
            },
        )


def test_category_names_on_continuous_raises():
    with pytest.raises(ValueError, match="not categorical"):
        ingest(
            _cat_num_data(),
            _model,
            schema={
                "feature_types": ["nominal", "continuous"],
                "category_names": [None, ["x", "y"]],
            },
        )


def test_category_names_length_must_match_dim():
    with pytest.raises(ValueError, match="expected 2"):
        ingest(
            _cat_num_data(),
            _model,
            schema={
                "feature_types": ["nominal", "continuous"],
                "category_names": [["a", "b", "c"]],
            },
        )

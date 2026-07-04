"""Input ingestion — the R10 contract (docs/design.md).

One door for `data`: a 2-D numeric numpy array or a pandas DataFrame. DataFrames
are converted to a float numpy core matrix here; everything downstream of the
constructors is numpy-only. All metadata (names, types, target name, scaling)
travels in a single `Schema`. pandas is never imported unless the caller already
passed a DataFrame (detection checks `sys.modules` only).
"""

import sys
import typing
import warnings
from dataclasses import dataclass, fields

import numpy as np

CONTINUOUS = "continuous"
ORDINAL = "ordinal"
NOMINAL = "nominal"
VALID_FEATURE_TYPES = (CONTINUOUS, ORDINAL, NOMINAL)
TYPE_ALIASES = {"cont": CONTINUOUS, "cat": NOMINAL}
DEFAULT_CAT_LIMIT = 10


@dataclass(frozen=True)
class Schema:
    """The single metadata argument of every effector constructor (R10).

    All fields are optional; whatever is not declared is inferred from the data
    (DataFrame dtypes, or numpy heuristics) or synthesized (`x_0…`, `"y"`).
    A `Schema` holds no data, so one instance can be reused across method
    constructions. Constructors also accept a plain dict with the same keys.

    Fields:
        feature_names: one name per column.
        feature_types: one of `"continuous" | "ordinal" | "nominal"` per column
            (aliases `"cont"` → continuous, `"cat"` → nominal).
        cat_limit: cardinality threshold for the int-column heuristic
            (default 10).
        target_name: name of the model output (default `"y"`).
        scale_x_list: per-feature `{"mean": .., "std": ..}` dicts (or None
            entries) to display plots in original units; plot-time `scale_x`
            overrides.
        scale_y: `{"mean": .., "std": ..}` for the output axis; plot-time
            `scale_y` overrides.
    """

    feature_names: typing.Optional[list] = None
    feature_types: typing.Optional[list] = None
    cat_limit: typing.Optional[int] = None
    target_name: typing.Optional[str] = None
    scale_x_list: typing.Optional[list] = None
    scale_y: typing.Optional[dict] = None


@dataclass(frozen=True)
class ColumnEncoding:
    """How one non-numeric DataFrame column was encoded to float codes."""

    levels: tuple  # code k -> levels[k]; ordered Categorical keeps its order
    kind: str  # "category" | "object" | "bool" (how to reconstruct)
    ordered: bool


@dataclass(frozen=True)
class FeatureMetadata:
    """Resolved input metadata, stored by every effect class as `feature_metadata`."""

    feature_names: list
    feature_types: list  # canonical three-way strings
    cat_limit: int
    categories: dict  # {col_idx: ColumnEncoding} for encoded DataFrame columns
    from_dataframe: bool
    column_dtypes: typing.Optional[dict]  # {col_name: original dtype} for DataFrames
    target_name: str
    scale_x_list: typing.Optional[list] = None
    scale_y: typing.Optional[dict] = None


@dataclass(frozen=True)
class IngestResult:
    data: np.ndarray  # the 2-D numeric core matrix
    model: typing.Callable  # wrapped iff the input was a DataFrame
    model_jac: typing.Optional[typing.Callable]
    meta: FeatureMetadata


def is_dataframe(obj) -> bool:
    """True iff `obj` is a pandas DataFrame, without ever importing pandas."""
    pd = sys.modules.get("pandas")
    return pd is not None and isinstance(obj, pd.DataFrame)


def is_categorical(ftype: str) -> bool:
    """Shared taxonomy predicate: does this feature type behave categorically?

    Accepts the legacy `"cat"` string so direct `space_partitioning.compile`
    callers keep working.
    """
    return ftype in ("cat", ORDINAL, NOMINAL)


def infer_feature_types(
    data: np.ndarray, cat_limit: int = DEFAULT_CAT_LIMIT
) -> typing.List[str]:
    """Infer three-way feature types from a numeric numpy matrix.

    Rule: a column that is integer-valued with fewer than `cat_limit` unique
    values is `"ordinal"`; everything else is `"continuous"`. `"nominal"` is
    never inferred from numpy input — it must be declared in the schema.
    """
    types = []
    for j in range(data.shape[1]):
        col = data[:, j]
        if np.issubdtype(col.dtype, np.floating):
            col = col[np.isfinite(col)]
        uniq = np.unique(col)
        integer_valued = uniq.size > 0 and bool(np.all(np.mod(uniq, 1) == 0))
        types.append(
            ORDINAL if integer_valued and uniq.size < cat_limit else CONTINUOUS
        )
    return types


def normalize_feature_types(feature_types: list, dim: int) -> typing.List[str]:
    """Map aliases to canonical type strings and validate values/length."""
    if len(feature_types) != dim:
        raise ValueError(
            f"feature_types has length {len(feature_types)}, expected {dim}"
        )
    out = []
    for j, t in enumerate(feature_types):
        if not isinstance(t, str):
            raise TypeError(
                f"feature type at position {j} must be a string, got {type(t).__name__}"
            )
        canonical = TYPE_ALIASES.get(t.lower(), t.lower())
        if canonical not in VALID_FEATURE_TYPES:
            raise ValueError(
                f"invalid feature type {t!r} at position {j}; "
                f"valid: continuous/ordinal/nominal (aliases: cont, cat)"
            )
        out.append(canonical)
    return out


def _as_schema(schema) -> Schema:
    if schema is None:
        return Schema()
    if isinstance(schema, Schema):
        return schema
    if isinstance(schema, dict):
        valid_keys = {f.name for f in fields(Schema)}
        unknown = set(schema) - valid_keys
        if unknown:
            raise ValueError(
                f"unknown schema key(s) {sorted(unknown)}; "
                f"valid keys: {sorted(valid_keys)}"
            )
        return Schema(**schema)
    raise TypeError(
        f"schema must be an effector.Schema, a dict, or None, "
        f"got {type(schema).__name__}"
    )


def _prep_cat_limit(cat_limit) -> int:
    if cat_limit is None:
        return DEFAULT_CAT_LIMIT
    if isinstance(cat_limit, bool) or not isinstance(cat_limit, int):
        raise TypeError(f"cat_limit must be an int, got {cat_limit!r}")
    if cat_limit < 2:
        raise ValueError(f"cat_limit must be >= 2, got {cat_limit}")
    return cat_limit


def _validate_scale(scale: dict, what: str):
    if scale is None:
        return
    if not isinstance(scale, dict):
        raise TypeError(f"{what} must be a dict with keys 'mean' and 'std'")
    missing = {"mean", "std"} - set(scale)
    if missing:
        raise ValueError(f"{what} is missing key(s) {sorted(missing)}")
    for key in ("mean", "std"):
        if not isinstance(scale[key], (int, float)) or isinstance(scale[key], bool):
            raise TypeError(f"{what}[{key!r}] must be a number, got {scale[key]!r}")
    if scale["std"] == 0:
        raise ValueError(f"{what}['std'] must be non-zero")


def validate_metadata(
    dim: int,
    feature_names: list,
    feature_types: list,
    cat_limit: int,
    scale_x_list: typing.Optional[list],
    scale_y: typing.Optional[dict],
    target_name: str,
    categories: typing.Optional[dict] = None,
) -> None:
    """The single validation point of the input contract (R10, R9 style)."""
    if len(feature_names) != dim:
        raise ValueError(
            f"feature_names has length {len(feature_names)}, expected {dim}"
        )
    if len(feature_types) != dim:
        raise ValueError(
            f"feature_types has length {len(feature_types)}, expected {dim}"
        )
    for j, t in enumerate(feature_types):
        if t not in VALID_FEATURE_TYPES:
            raise ValueError(
                f"invalid feature type {t!r} at position {j}; "
                f"valid: {'/'.join(VALID_FEATURE_TYPES)}"
            )
    if categories:
        for j in categories:
            if feature_types[j] == CONTINUOUS:
                raise ValueError(
                    f"feature {feature_names[j]!r} comes from a non-numeric column "
                    f"and cannot be labeled 'continuous'"
                )
    if scale_x_list is not None:
        if len(scale_x_list) != dim:
            raise ValueError(
                f"scale_x_list has length {len(scale_x_list)}, expected {dim}"
            )
        for j, scale in enumerate(scale_x_list):
            _validate_scale(scale, f"scale_x_list[{j}]")
    _validate_scale(scale_y, "scale_y")
    if not isinstance(target_name, str):
        raise TypeError(f"target_name must be a string, got {target_name!r}")


def _encode_dataframe(df, cat_limit: int):
    """Encode a DataFrame per the R10 dtype table.

    Returns (matrix, names, inferred_types, categories, column_dtypes,
    heuristic_idx) where `heuristic_idx` lists the column indices whose type
    was decided by the cardinality heuristic (int columns), not by dtype.
    """
    import pandas as pd
    from pandas.api import types as pdt

    n, dim = df.shape
    matrix = np.empty((n, dim), dtype=np.float64)
    names = [str(c) for c in df.columns]
    inferred_types = []
    categories = {}
    column_dtypes = {}
    heuristic_idx = []

    for j, col_name in enumerate(df.columns):
        col = df[col_name]
        column_dtypes[names[j]] = col.dtype
        if col.isna().any():
            raise ValueError(
                f"column {names[j]!r} contains missing values; "
                f"effector does not handle NaN — impute or drop first"
            )
        if isinstance(col.dtype, pd.CategoricalDtype):
            matrix[:, j] = col.cat.codes.to_numpy(dtype=np.float64)
            categories[j] = ColumnEncoding(
                levels=tuple(col.cat.categories.tolist()),
                kind="category",
                ordered=bool(col.dtype.ordered),
            )
            inferred_types.append(ORDINAL if col.dtype.ordered else NOMINAL)
        elif pdt.is_bool_dtype(col.dtype):
            matrix[:, j] = col.to_numpy().astype(np.float64)
            categories[j] = ColumnEncoding(
                levels=(False, True), kind="bool", ordered=True
            )
            inferred_types.append(ORDINAL)
        elif pdt.is_float_dtype(col.dtype):
            matrix[:, j] = col.to_numpy(dtype=np.float64)
            inferred_types.append(CONTINUOUS)
        elif pdt.is_integer_dtype(col.dtype):
            matrix[:, j] = col.to_numpy(dtype=np.float64)
            inferred_types.append(ORDINAL if col.nunique() < cat_limit else CONTINUOUS)
            heuristic_idx.append(j)
        elif pdt.is_object_dtype(col.dtype) or pdt.is_string_dtype(col.dtype):
            as_cat = col.astype("category")
            matrix[:, j] = as_cat.cat.codes.to_numpy(dtype=np.float64)
            categories[j] = ColumnEncoding(
                levels=tuple(as_cat.cat.categories.tolist()),
                kind="object",
                ordered=False,
            )
            inferred_types.append(NOMINAL)
        else:
            raise ValueError(
                f"unsupported dtype {col.dtype} for column {names[j]!r}; "
                f"convert it to numeric or categorical"
            )
    return matrix, names, inferred_types, categories, column_dtypes, heuristic_idx


def _make_frame_builder(columns: list, categories: dict) -> typing.Callable:
    """Build the numpy-row → DataFrame reconstructor for the model-call rule (R10)."""
    import pandas as pd

    def build(X: np.ndarray):
        X = np.asarray(X)
        cols = {}
        for j, name in enumerate(columns):
            enc = categories.get(j)
            if enc is None:
                cols[name] = X[:, j].astype(np.float64)
            else:
                codes = np.clip(np.round(X[:, j]), 0, len(enc.levels) - 1).astype(int)
                if enc.kind == "category":
                    cols[name] = pd.Categorical.from_codes(
                        codes, list(enc.levels), ordered=enc.ordered
                    )
                elif enc.kind == "bool":
                    cols[name] = codes.astype(bool)
                else:  # "object"
                    cols[name] = np.asarray(enc.levels, dtype=object)[codes]
        return pd.DataFrame(cols, columns=list(columns))

    return build


def _wrap_model(
    fn: typing.Optional[typing.Callable], frame_builder
) -> typing.Optional[typing.Callable]:
    if fn is None:
        return None

    def wrapped(X):
        return np.asarray(fn(frame_builder(X)))

    wrapped.__wrapped__ = fn
    return wrapped


def _heuristic_warning(heuristic_info: list):
    detail = ", ".join(f"{name!r} -> {t}" for name, t in heuristic_info)
    warnings.warn(
        f"Feature type(s) inferred by the cardinality heuristic: {detail}. "
        f"This guess is error-prone (a label-encoded nominal feature looks "
        f"ordinal). Declare types explicitly to silence this warning, e.g. "
        f"schema={{'feature_types': [...]}} with values "
        f"'continuous'/'ordinal'/'nominal'.",
        UserWarning,
        stacklevel=3,
    )


def ingest(
    data,
    model: typing.Callable,
    model_jac: typing.Optional[typing.Callable] = None,
    *,
    schema=None,
) -> IngestResult:
    """The one door for `data` + metadata (R10); runs before `helpers.prep_data`.

    numpy input passes through byte-identical (no dtype cast, model untouched).
    DataFrame input is encoded to a float matrix and `model`/`model_jac` are
    wrapped so they are always called with a reconstructed DataFrame.
    """
    schema = _as_schema(schema)
    cat_limit = _prep_cat_limit(schema.cat_limit)

    categories = {}
    column_dtypes = None
    heuristic_idx = []
    from_dataframe = is_dataframe(data)

    if from_dataframe:
        (
            matrix,
            df_names,
            inferred_types,
            categories,
            column_dtypes,
            heuristic_idx,
        ) = _encode_dataframe(data, cat_limit)
    elif isinstance(data, np.ndarray):
        if data.ndim != 2:
            raise ValueError(f"data must be a 2D array, got {data.ndim} dimensions")
        if data.dtype.kind not in "fiub":
            raise TypeError(
                f"data has non-numeric dtype {data.dtype}; "
                f"pass a pandas DataFrame for string/categorical columns"
            )
        matrix = data
        df_names = None
        inferred_types = None
    else:
        raise TypeError(
            f"data must be a 2D numpy array or a pandas DataFrame, "
            f"got {type(data).__name__}"
        )

    dim = matrix.shape[1]

    # define-or-infer: explicit schema field > DataFrame inference > numpy heuristic
    if schema.feature_names is not None:
        feature_names = [str(name) for name in schema.feature_names]
    elif df_names is not None:
        feature_names = df_names
    else:
        feature_names = ["x_" + str(i) for i in range(dim)]

    if schema.feature_types is not None:
        feature_types = normalize_feature_types(schema.feature_types, dim)
    elif inferred_types is not None:
        feature_types = inferred_types
        # only the risky half of the int heuristic warns: int -> continuous is
        # the expected reading, int -> ordinal may be a label-encoded nominal
        risky = [j for j in heuristic_idx if feature_types[j] == ORDINAL]
        if risky:
            _heuristic_warning([(feature_names[j], feature_types[j]) for j in risky])
    else:
        feature_types = infer_feature_types(matrix, cat_limit)
        heuristic_info = [
            (feature_names[j], t) for j, t in enumerate(feature_types) if t == ORDINAL
        ]
        if heuristic_info:
            _heuristic_warning(heuristic_info)

    target_name = schema.target_name if schema.target_name is not None else "y"

    validate_metadata(
        dim,
        feature_names,
        feature_types,
        cat_limit,
        schema.scale_x_list,
        schema.scale_y,
        target_name,
        categories,
    )

    if from_dataframe:
        frame_builder = _make_frame_builder(feature_names, categories)
        model = _wrap_model(model, frame_builder)
        model_jac = _wrap_model(model_jac, frame_builder)

    meta = FeatureMetadata(
        feature_names=feature_names,
        feature_types=feature_types,
        cat_limit=cat_limit,
        categories=categories,
        from_dataframe=from_dataframe,
        column_dtypes=column_dtypes,
        target_name=target_name,
        scale_x_list=schema.scale_x_list,
        scale_y=schema.scale_y,
    )
    return IngestResult(data=matrix, model=model, model_jac=model_jac, meta=meta)

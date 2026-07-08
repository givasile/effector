"""Input ingestion — the R10 contract (docs/design.md).

The border crossing. `data` must be a 2-D numeric numpy array and `model`
(and `model_jac`) must be numpy-in / numpy-out callables — effector is
numpy-only. `ingest` validates that, then auto-infers whatever metadata the
caller did not declare in the `Schema` (names `x_0…`, the numpy type
heuristic, target `"y"`).

A pandas DataFrame is *not* accepted as `data`: it is hard-rejected with a
pointer to `from_dataframe`, the opt-in convenience that reads a DataFrame's
names/dtypes/levels into `(X, Schema)`. `from_dataframe` never touches the
model — framework/DataFrame conversion is the user's wrapper's job. pandas is
never imported unless the caller reaches for that helper (detection checks
`sys.modules` only).
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

    All fields are optional; whatever is not declared is inferred from the numpy
    data (type heuristic) or synthesized (`x_0…`, `"y"`). `effector.from_dataframe`
    populates one from a DataFrame's names/dtypes/levels. A `Schema` holds no
    data, so one instance can be reused across method constructions. Constructors
    also accept a plain dict with the same keys.

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
        category_names: per-feature list of human-readable level names for a
            categorical (ordinal/nominal) feature — one name per observed level
            in ascending order — shown on the plot axis instead of the numeric
            codes. `None` entries (and non-categorical features) keep the codes.
    """

    feature_names: typing.Optional[list] = None
    feature_types: typing.Optional[list] = None
    cat_limit: typing.Optional[int] = None
    target_name: typing.Optional[str] = None
    scale_x_list: typing.Optional[list] = None
    scale_y: typing.Optional[dict] = None
    category_names: typing.Optional[list] = None


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
    target_name: str
    scale_x_list: typing.Optional[list] = None
    scale_y: typing.Optional[dict] = None
    category_names: typing.Optional[dict] = None  # {feature_idx: {level_value: name}}


@dataclass(frozen=True)
class IngestResult:
    data: np.ndarray  # the 2-D numeric core matrix (the input array, unchanged)
    model: typing.Callable  # the user's model, passed through untouched
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
    category_names: typing.Optional[list] = None,
    level_counts: typing.Optional[dict] = None,
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
    if scale_x_list is not None:
        if len(scale_x_list) != dim:
            raise ValueError(
                f"scale_x_list has length {len(scale_x_list)}, expected {dim}"
            )
        for j, scale in enumerate(scale_x_list):
            _validate_scale(scale, f"scale_x_list[{j}]")
    if category_names is not None:
        if len(category_names) != dim:
            raise ValueError(
                f"category_names has length {len(category_names)}, expected {dim}"
            )
        for j, names in enumerate(category_names):
            if names is None:
                continue
            if not is_categorical(feature_types[j]):
                raise ValueError(
                    f"category_names[{j}] is set but feature {feature_names[j]!r} "
                    f"is {feature_types[j]!r}, not categorical (ordinal/nominal)"
                )
            if level_counts is not None and len(names) != level_counts.get(j):
                raise ValueError(
                    f"category_names[{j}] has {len(names)} names but feature "
                    f"{feature_names[j]!r} has {level_counts.get(j)} observed levels"
                )
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
    """The border crossing for `data` + metadata (R10); runs before
    `helpers.prep_data`.

    `data` must be a 2-D numeric numpy array; `model`/`model_jac` pass through
    untouched (effector is numpy-only, so they must already be numpy-in /
    numpy-out). A pandas DataFrame is rejected with a pointer to
    `from_dataframe`. Whatever the `Schema` does not declare is inferred here.
    """
    schema = _as_schema(schema)
    cat_limit = _prep_cat_limit(schema.cat_limit)

    if is_dataframe(data):
        raise TypeError(
            "effector is numpy-only: `data` must be a 2-D numeric numpy array, "
            "not a pandas DataFrame. Convert it first:\n"
            "    X, schema = effector.from_dataframe(df)\n"
            "and pass a numpy->numpy `model` (wrap a DataFrame/torch/tf model "
            "yourself)."
        )
    if not isinstance(data, np.ndarray):
        raise TypeError(f"data must be a 2D numpy array, got {type(data).__name__}")
    if data.ndim != 2:
        raise ValueError(f"data must be a 2D array, got {data.ndim} dimensions")
    if data.dtype.kind not in "fiub":
        raise TypeError(
            f"data has non-numeric dtype {data.dtype}; encode it to a numeric "
            f"matrix first (see effector.from_dataframe for DataFrame columns)"
        )
    if not callable(model):
        raise TypeError(
            f"`model` must be a numpy->numpy callable, got "
            f"{type(model).__name__}. Wrap your model first — see "
            f"effector.adapters (from_sklearn / classifier_proba / from_torch) "
            f"or write the wrapper yourself, then verify it with "
            f"effector.adapters.check(model, X)."
        )
    if model_jac is not None and not callable(model_jac):
        raise TypeError(
            f"`model_jac` must be a numpy->numpy callable, got "
            f"{type(model_jac).__name__}."
        )
    matrix = data
    dim = matrix.shape[1]

    # define-or-infer: explicit schema field > numpy heuristic > synthesized
    if schema.feature_names is not None:
        feature_names = [str(name) for name in schema.feature_names]
    else:
        feature_names = ["x_" + str(i) for i in range(dim)]

    if schema.feature_types is not None:
        feature_types = normalize_feature_types(schema.feature_types, dim)
    else:
        feature_types = infer_feature_types(matrix, cat_limit)
        heuristic_info = [
            (feature_names[j], t) for j, t in enumerate(feature_types) if t == ORDINAL
        ]
        if heuristic_info:
            _heuristic_warning(heuristic_info)

    target_name = schema.target_name if schema.target_name is not None else "y"

    # observed level counts for categorical features (validates category_names)
    level_counts = {
        j: int(np.unique(matrix[:, j]).size)
        for j, t in enumerate(feature_types)
        if is_categorical(t)
    }

    validate_metadata(
        dim,
        feature_names,
        feature_types,
        cat_limit,
        schema.scale_x_list,
        schema.scale_y,
        target_name,
        schema.category_names,
        level_counts,
    )

    # resolve category_names (per-feature name list, one per ascending observed
    # level) to a {level_value: name} map, so it maps by value and survives to
    # regional nodes, where a split feature may show only a subset of its levels
    category_names_map = None
    if schema.category_names is not None:
        category_names_map = {}
        for j, names in enumerate(schema.category_names):
            if names is None:
                continue
            levs = np.unique(matrix[:, j])
            category_names_map[j] = {float(lv): str(n) for lv, n in zip(levs, names)}

    meta = FeatureMetadata(
        feature_names=feature_names,
        feature_types=feature_types,
        cat_limit=cat_limit,
        target_name=target_name,
        scale_x_list=schema.scale_x_list,
        scale_y=schema.scale_y,
        category_names=category_names_map,
    )
    return IngestResult(data=matrix, model=model, model_jac=model_jac, meta=meta)


def from_dataframe(df, *, cat_limit: int = DEFAULT_CAT_LIMIT):
    """Extract a numpy matrix and a populated `Schema` from a pandas DataFrame.

    A pure convenience for the common "I started from a DataFrame" case. It
    reads column names, maps dtypes to feature types, and pulls categorical
    level labels, returning `(X, schema)` so you can call any effector
    constructor as `Method(X, model, schema=schema)`.

    It does **not** touch your model: effector is numpy-only, so `model` must be
    a numpy->numpy callable. If your model consumes a DataFrame (e.g. an sklearn
    Pipeline), wrap it yourself into a numpy->numpy function.

    The returned schema is a *proposal you should inspect* — in particular the
    int-column type guess (ordinal vs continuous vs a label-encoded nominal) is
    the one thing no extractor can know for sure. Override any field you don't
    like before passing it on.

    Args:
        df: a pandas DataFrame with numeric / bool / categorical / string
            columns (no missing values).
        cat_limit: cardinality threshold for the int-column ordinal heuristic
            (default 10), recorded on the returned schema.

    Returns:
        `(X, schema)` where `X` is a `(N, D)` float64 numpy array and `schema`
        is an `effector.Schema` with `feature_names`, `feature_types`,
        `cat_limit`, and `category_names` populated.
    """
    if not is_dataframe(df):
        raise TypeError(
            f"from_dataframe expects a pandas DataFrame, got {type(df).__name__}"
        )
    cat_limit = _prep_cat_limit(cat_limit)
    matrix, names, inferred_types, categories, _dtypes, heuristic_idx = (
        _encode_dataframe(df, cat_limit)
    )

    # human-readable level names, one per *observed* level in ascending code
    # order — exactly the shape Schema.category_names expects, so an unused
    # declared category never trips the length check downstream
    category_names = [None] * len(names)
    for j, enc in categories.items():
        observed = np.unique(matrix[:, j])
        category_names[j] = [str(enc.levels[int(c)]) for c in observed]
    if not any(n is not None for n in category_names):
        category_names = None

    # same guidance the numpy door gives: only int -> ordinal is a risky guess
    risky = [j for j in heuristic_idx if inferred_types[j] == ORDINAL]
    if risky:
        _heuristic_warning([(names[j], inferred_types[j]) for j in risky])

    schema = Schema(
        feature_names=names,
        feature_types=inferred_types,
        cat_limit=cat_limit,
        category_names=category_names,
    )
    return matrix, schema

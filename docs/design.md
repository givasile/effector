# Design contract (the "constitution")

These are the core rules every `effector` class follows after the 2026-07
homogenization refactor (PLAN Part III §1, LOGBOOK #3–#6, #13). New features
must follow them too; the contract-layer tests (`tests/test_contract_*.py`)
enforce most of them mechanically.

## R1 — Lifecycle

`fit` does the hard work once and stores everything under
`feature_effect["feature_{i}"]`. `eval(feature, xs, centering=<class default>)`
returns the **mean effect only — one return type, always** — and never
recomputes unless `requires_refit`. `plot` is a thin wrapper over `eval` /
stored state plus one `vis.*` call: the plot layer draws, it does not compute.

## R2 — Heterogeneity semantics

Heterogeneity does **not** go through `eval`; it has its own surface, an
aggregation ladder with one consumer per level:

- **`payload(feature)` → dict** — the method's honest raw object (ICE table,
  per-bin variances, shap cloud) from stored state.
- **`eval_heter(feature, xs)` → `(T,)`** — the heterogeneity curve h(xs).
  h is the *variance* of the method's own per-instance effect object
  (PDP: centered ICE levels; DerPDP: d-ICE slopes; (RH)ALE: per-bin slope
  variance as a step function; ShapDP: the residual spline). Variance
  internally, std only at the plot layer. **No centering kwarg** — h is
  invariant to centering, and the signature enforces it. Every plotted
  band/error-bar equals `eval_heter` (R1 extended to heterogeneity).
  Regional twin: `eval_heter(feature, node_idx, xs)`.
- **`heter_score(feature)` → float ≥ 0** — the one method-agnostic scalar,
  consumed by regional splitting and the interaction submodule.

## R3 — Centering vocabulary

`{False, "zero_integral" (=True), "zero_start"}`, normalized once by
`helpers.prep_centering`. Each class declares its default **once** as the
`DEFAULT_CENTERING` class attribute; `fit/eval/plot` signatures use
`centering=None` → class default instead of per-method literals.

## R4 — State schema

`feature_effect["feature_{i}"]` always contains `norm_const: float | None`
(`None` means "not centered" — never a sentinel value) plus the
method-specific payload; `fit_args["feature_{i}"]` records the kwargs needed
to detect refit.

## R5 — One method registry

`effector.method_registry` holds the single
`{canonical_name: (cls, needs_jac, uses_data_effect, display_name)}` table
plus aliases. `FeatureEffect`, `RegionalEffectBase._create_fe_object`, and
plot titles all read it; per-method if/elif chains are a bug.

## R6 — String-argument registries

Exactly one alias table per concept, resolved by one `return_default`-style
function, and validation always goes through the resolver (binning:
`"fixed" | "greedy" | "dp"`; partitioner: `"best" | "best_level_wise"`).

## R7 — Plot contract

Every `vis.*` function and every public `.plot` returns `(fig, ax)` when
`show_plot=False` and `None` otherwise — uniformly, global and regional.

## R8 — Constructor contract

Canonical parameter order `data, model, model_jac=None, *, data_effect,
nof_instances, axis_limits, schema, random_state, ...` — everything after
`model_jac` keyword-only, so positional shuffles can't bite. All input
metadata (names, types, target name, scaling) lives in the single `schema`
argument (R10); metadata never appears as separate constructor kwargs.

Reproducibility is contractual: every constructor takes `random_state`
(default `21`; `None` opts into fresh randomness), two identical
constructions give identical `eval`/`fit`/`plot` output, and no effect-class
code touches the global `np.random` state — every sampling site creates its
own `np.random.default_rng(random_state)`. (The one exception is
`datasets.IndependentUniform`, which keeps its legacy seeded global draw so
the executed notebooks stay a valid regression oracle.)

## R9 — Errors & messages

`ValueError`/`TypeError` (not bare `assert`) for user input; `warnings.warn`
or `logging` (not `print`) inside heterogeneity functions.

## R10 — Input contract

**Accepted `data` types.** A 2-D numeric numpy array, or a pandas DataFrame.
Anything else → `TypeError`. pandas is never a hard dependency: detection
checks `sys.modules` only, and the numpy path never imports pandas (proven by
a subprocess test). DataFrames are converted to a float numpy core matrix at
the door (`effector.ingestion.ingest`, called before `helpers.prep_data` in
every constructor); everything downstream is numpy-only.

**One metadata argument.** All input metadata travels in `schema=` — an
`effector.Schema` (frozen dataclass) or a plain dict with the same keys:
`feature_names`, `feature_types`, `cat_limit` (default 10), `target_name`,
`scale_x_list`, `scale_y`, `category_names` — every field optional. Unknown dict keys →
`ValueError` listing the valid ones. A schema is reusable across method
constructions.

**Define-or-infer.** Precedence per field: explicit schema field >
DataFrame inference > numpy heuristic > synthesized default (`x_0…`, `"y"`).

**Feature-type taxonomy** — three-way, `"continuous" | "ordinal" | "nominal"`,
with door-normalized aliases `"cont"` → continuous, `"cat"` → nominal.
Inference:

| source | rule |
|---|---|
| DataFrame float column | continuous |
| DataFrame int column | ordinal if `nunique < cat_limit`, else continuous |
| DataFrame bool column | ordinal (codes 0/1) |
| DataFrame `Categorical(ordered=True)` | ordinal, declared category order kept |
| DataFrame unordered category / object / string | nominal (codes via `astype("category")`) |
| DataFrame datetime / other | `ValueError` |
| numpy column, integer-valued, `nunique < cat_limit` | ordinal |
| numpy column, otherwise | continuous — **nominal is never inferred from numpy** |

Types decided by the *cardinality heuristic* (the two int rules) and not
declared in the schema trigger one `UserWarning` naming the columns and the
one-line `schema={"feature_types": [...]}` fix; dtype-decided types are
silent. NaN anywhere in a DataFrame → `ValueError` naming the column.

**Model-call rule.** If `data` was a DataFrame, `model` (and `model_jac`) are
always called with a reconstructed DataFrame: original column names/order,
encoded columns decoded to their original values/dtypes, numeric columns
float64, codes rounded and clipped to the nearest level. Escape hatch for
array-expecting models: `lambda X: f(X.to_numpy())`. Precomputed
`data_effect` / `shap_values` align with the *encoded* matrix and pass
through untouched.

**Scaling precedence.** `scale_x_list`/`scale_y` in the schema are
construction-time defaults; a plot-time `scale_x`/`scale_y` dict overrides,
`False` at plot time explicitly disables an inherited scale.

**One validation point.** `effector.ingestion.validate_metadata` (R9 style)
checks name/type list lengths against `dim`, canonical type values, scale
dict shapes (`{"mean","std"}`, `std != 0`), `cat_limit` sanity, and rejects a
`continuous` label on an encoded (string-source) column.

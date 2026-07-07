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
  Masked/regional form: `eval_heter(feature, xs, mask)` (the mask of a
  `Partition` region — R11/R12).
- **`heter_score(feature)` → float ≥ 0** — the one method-agnostic scalar,
  consumed by `find_regions` and the interaction submodule.

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
`{canonical_name: (cls, needs_jac, display_name, ...capabilities)}` table
plus aliases. `FeatureEffect` and plot titles all read it;
per-method if/elif chains are a bug.

## R6 — String-argument registries

Exactly one alias table per concept, resolved by one `return_default`-style
function, and validation always goes through the resolver (binning:
`"fixed" | "greedy" | "dp"`; partitioner: `"best" | "best_level_wise"`).

## R7 — Plot contract

Every `vis.*` function and every public `.plot` returns `(fig, ax)` when
`show_plot=False` and `None` otherwise — uniformly, for global effects and
`Partition.plot`.

## R8 — Constructor contract

Canonical parameter order `data, model, model_jac=None, *, data_effect,
nof_instances, axis_limits, schema, random_state, ...` — everything after
`model_jac` keyword-only, so positional shuffles can't bite. `data` is a
numpy array and `model`/`model_jac` are numpy→numpy callables (R10). All input
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

**Accepted `data` type.** A 2-D numeric numpy array. Anything else →
`TypeError`; a pandas DataFrame is rejected with a pointer to
`effector.from_dataframe`. effector is numpy-only — `data`, `model`, and
`model_jac` all live in numpy, so the model is called exactly as given and is
never wrapped. pandas is never a dependency of the compute path (the numpy door
never imports it, proven by a subprocess test).

**Model contract.** `model` is `Callable[[np.ndarray[N, D]], np.ndarray[N]]`
and `model_jac` (optional) is `Callable[[np.ndarray[N, D]], np.ndarray[N, D]]`.
A model trained on a DataFrame, a torch/tf tensor, or an sklearn `Pipeline` is
the user's to wrap into a numpy→numpy callable — dtype, device, batching, and
any DataFrame reconstruction included. See the *"effector is purely numpy
based"* quickstart guide.

**One metadata argument.** All input metadata travels in `schema=` — an
`effector.Schema` (frozen dataclass) or a plain dict with the same keys:
`feature_names`, `feature_types`, `cat_limit` (default 10), `target_name`,
`scale_x_list`, `scale_y`, `category_names` — every field optional. Unknown dict keys →
`ValueError` listing the valid ones. A schema is reusable across method
constructions.

**Define-or-infer.** Precedence per field: explicit schema field > numpy
heuristic > synthesized default (`x_0…`, `"y"`).

**Feature-type taxonomy** — three-way, `"continuous" | "ordinal" | "nominal"`,
with door-normalized aliases `"cont"` → continuous, `"cat"` → nominal. From a
numpy column:

| source | rule |
|---|---|
| numpy column, integer-valued, `nunique < cat_limit` | ordinal |
| numpy column, otherwise | continuous — **nominal is never inferred from numpy** |

Types decided by the *cardinality heuristic* (the int rule) and not declared in
the schema trigger one `UserWarning` naming the columns and the one-line
`schema={"feature_types": [...]}` fix.

**`from_dataframe` convenience.** `X, schema = effector.from_dataframe(df)`
reads a DataFrame's column names, dtypes, and category levels into `(X, Schema)`
so you can call any constructor as `Method(X, model, schema=schema)`. It
converts *data* only — it never touches the model. dtype → type mapping:

| DataFrame column | inferred type |
|---|---|
| float | continuous |
| int | ordinal if `nunique < cat_limit`, else continuous |
| bool | ordinal (codes 0/1) |
| `Categorical(ordered=True)` | ordinal, declared category order kept |
| unordered category / object / string | nominal (codes via `astype("category")`) |
| datetime / other | `ValueError` |

NaN anywhere → `ValueError` naming the column. The returned schema is a
*proposal to inspect*: the int-column guess (ordinal vs continuous vs a
label-encoded nominal) is the one thing no extractor can know for sure.

**Scaling precedence.** `scale_x_list`/`scale_y` in the schema are
construction-time defaults; a plot-time `scale_x`/`scale_y` dict overrides,
`False` at plot time explicitly disables an inherited scale.

**One validation point.** `effector.ingestion.validate_metadata` (R9 style)
checks name/type list lengths against `dim`, canonical type values, scale
dict shapes (`{"mean","std"}`, `std != 0`), `cat_limit` sanity, and
`category_names` lengths against the observed levels.

## R11 — Regional ≡ masked global

A regional effect is **not** a re-instantiated global object on a data
subset; it is the one global object's own summary restricted by a boolean
mask. Every global `eval`/`eval_heter`/`heter_score`/`plot` accepts
`mask=` (boolean, shape `(N,)`). Regional questions are asked with
`find_regions(feature) -> Partition` (R12): the returned `Partition`'s
`eval`/`eval_heter`/`plot(idx)` call back into the one fitted global object
with that region's mask (plus `feature_label=` for the region title). One
source of truth: the heterogeneity shown by `Partition.show()` **is**
`heter_score(feature, mask)`, and every plotted region band derives from
`eval_heter(feature, xs, mask)`.

**The invariant.** `axis_limits` is the immutable global frame, fixed at
construction. A mask never mutates stored state — no axis_limits, no bins,
no payloads. Anything region-shaped is transient, derived per call via
`_effective_limits(feature, mask)` = `[min, max]` of the masked column
(categorical analog: `_level_weights`). Its only consumers: masked
centering constants, the masked plot x-window, and degeneracy guards
(empty mask / collapsed interval → `ValueError`).

**Model-free.** Masked surfaces extend the single-model-touch constitution
to eval/plot: everything is re-summarized from the stored per-instance
local effects, zero model calls (contract-tested). The one allowed
exception: PDP `eval(mask=)` at points off the cached grid recomputes ICE
on `data[mask]` exactly — symmetric with global PDP eval.

**Frame semantics.** Structure frozen on the global frame stays frozen:
ALE keeps its bin edges, PDP its grid. Bins/levels left empty by the mask
are interpolated (`fill_nans`) and edge bins extend flat — a region never
re-touches the model to re-support the frame. RHALE/ShapDP store
per-instance local effects, so masked calls re-run binning; the
`binning_scope` fit kwarg (`"global"` default | `"effective"`) selects the
range handed to `find_limits`, is recorded in `fit_args`, and is replayed
on every masked call — split search and display always share it.

## R12 — Regions are values, not state

`find_regions(feature) -> Partition` is a **query**, not a mutation: it stores
nothing on the effect object, which holds only the canonical global state (a
fit). *Store what is canonical (a fit), return what is exploratory (a
partition).* A `Partition` depends on the search config, so there is no single
canonical one to store; two `find_regions` calls return equal-but-distinct
values, and the effect gains no public attribute. The `Partition` is a value
object — an ordered list of `Region`s (each a boolean mask + heterogeneity +
split metadata) with `show`/`eval`/`eval_heter`/`plot`/`to_dict`; it binds a
reference to its producing effect only for the `eval`/`plot` sugar, and
`to_dict` is the serialization boundary (the effect is never serialized).

**Finder seam.** A region finder consumes only `(score_fn: mask -> float, data,
metadata, its own config)` and returns a `Partition`. `heter_score(feature,
mask)` is the score; the finder owns the min-points / degeneracy guard (the
`BIG_M` vocabulary lives in the finder, never in the effect). New finders (ICE
clustering, subgroup discovery, a user `groupby`) plug in with zero changes
elsewhere, so `Partition` must not structurally assume a tree — hierarchy is
optional display metadata (`parent_idx`).

**Invisible memos are not state.** Performance caches (the masked-summary memo)
are allowed inside the effect because they are semantically transparent — keyed
by fit epoch, so a refit invalidates them, and they never change an answer, only
its latency. A cache is not API surface; a stored partition would be.

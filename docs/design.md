# Design contract (the "constitution")

These are the core rules every `effector` class follows after the 2026-07
homogenization refactor (PLAN Part III §1, LOGBOOK #3–#6, #13). New features
must follow them too; the contract-layer tests (`tests/test_contract_*.py`)
enforce most of them mechanically.

## R1 — Lifecycle

Construction ingests and nothing else. `fit(features, **config)` *declares the
method configuration* (binning, order, scope, default centering — the kwargs
`eval`/`plot` deliberately do not accept) and eagerly warms the two caches
(R14); nothing `fit` does is unavailable lazily — every query silently ensures
what it needs through the same gates, and fit-then-query equals
never-fit-just-query byte for byte. `eval(feature, xs, centering=<class
default>)` returns the **mean effect only — one return type, always**. `plot`
is a thin wrapper over the same summaries plus one `vis.*` call: the plot
layer draws, it does not compute.

## R2 — Heterogeneity semantics

Heterogeneity does **not** go through `eval`; it has its own surface, an
aggregation ladder with one consumer per level:

- **`payload(feature)` → dict** — the method's honest raw object (ICE table,
  per-bin variances, shap cloud): the all-ones summary (R14).
- **`eval_heter(feature, xs)` → `(T,)`** — the heterogeneity curve h(xs).
  h is the *variance* of the method's own per-instance effect object
  (PDP: centered ICE levels; DerPDP: d-ICE slopes; (RH)ALE: per-bin slope
  variance as a step function; ShapDP: the interpolated per-bin φ variance).
  Variance
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

An effect object holds exactly two caches and one config (R14): the
local-effects store `_local[feature]` (frame-carrying, model-derived), the
summaries memo `_summaries` (payloads *and* centering constants, keyed by
`(feature, epoch, mask_key[, mode])`), and `fit_args["feature_{i}"]` (the
declared config). There is no separate "fitted state": the fitted payload *is*
the all-ones entry of the memo, and `payload(feature)` returns it. Centering
constants are summaries — always derived from the local effects on demand,
never stored as fitted state and never a refit trigger.

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
`mask=` (boolean, shape `(N,)`) and, as sugar over it, `rule=` — an
`effector.Rule` or a string like `"temp < 6.5 and workingday == 0"`,
parsed with the effect's metadata and applied to the effect's data
(`Rule.contains` is the single rule→mask site; mutually exclusive with
`mask=`). Regional questions are asked with
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
object — an ordered list of `Region`s with
`show`/`show_axes`/`eval`/`eval_heter`/`plot`/`to_dict`; it binds a
reference to its producing effect only for the `eval`/`plot` sugar, and
`to_dict` is the serialization boundary (the effect is never serialized).

**Rule-primary regions.** A `Region`'s identity is its `Rule`
(`effector.rules`): a normalized conjunction of per-feature conditions —
half-open intervals matching the split mask semantics (`x < t` / `x >= t`)
and explicit level sets (a `!=` split materializes the complement over the
observed levels). Membership (`rule.contains`), display (`rule.format`),
and serialization (`rule.to_dict`) derive from that one object, so they
cannot drift apart. The boolean mask is a derived cache stamped against one
dataset; `to_dict` serializes rules + scalar stats (O(regions), no masks),
and `bind(effect)` recomputes every mask from its rule and verifies it —
against the finder's mask on the `find_regions` path (an exactness
tripwire) or the stored `nof_instances` on the `from_dict` path. The
constructor enforces the partition invariant: leaves pairwise disjoint and
jointly covering the root. `Partition.from_rules` builds a user-authored
partition through the same validation — manual and automated regional
analysis converge on one value type.

**Finder seam.** A region finder consumes only `(score_fn: mask -> float, data,
metadata, its own config)` and returns a `Partition`. `heter_score(feature,
mask)` is the score; the finder owns the min-points / degeneracy guard (the
`BIG_M` vocabulary lives in the finder, never in the effect). New finders (ICE
clustering, subgroup discovery, a user `groupby`) plug in with zero changes
elsewhere, so `Partition` must not structurally assume a tree — hierarchy is
optional display metadata (`parent_idx`).

**Proposer seam.** Inside the built-in finders, candidate enumeration is its
own protocol (`effector.proposers`): a proposer maps a conditioning feature to
`CandidateSplit`s — ordered tuples of disjoint, jointly-covering `Condition`s
on that feature, so a split is *parent rule → child conditions* (child i =
`parent_rule.refine(conditions[i])`, mask = parent mask ∧ condition mask).
Candidates are parent-independent (a level-wise finder applies one candidate
to every node of a level) and k-way by construction. The defaults reproduce
the classic search: binary thresholds on an interior grid for continuous
features, one-vs-rest over observed levels for categorical ones (the `!=`
complement is materialized here, as an explicit level set). The richer named
proposers — `subsets` / `ordered` (similarity-seriated for nominal levels) /
`multiway` for categorical conditioning features, `quantiles` (k-way marginal
cuts) for continuous ones — are selected per feature type from the finder
constructor (`categorical_proposer=` / `continuous_proposer=`, a name or a
proposer instance); the finder's `proposer_factory` attribute remains the raw
seam, so custom proposers plug in without touching the search or the
construction — finders build rule-primary `Region`s directly; there is no
tree intermediate.

**Invisible memos are not state.** Performance caches (the summaries memo,
R14) are allowed inside the effect because they are semantically transparent —
keyed by the feature's epoch, so a frame or config change makes stale entries
unreachable, and they never change an answer, only its latency. A cache is not
API surface; a stored partition would be.

## R13 — Importance

`importance(feature, mask=None) -> float >= 0` is the method-agnostic scalar
measuring how much the **mean effect** of a feature varies — the μ-twin of
`heter_score` (which measures per-instance spread), evaluated the same way it is:
continuous features over the **(masked) data values**, discrete features
frequency-weighted over levels. It is centering-invariant (no `centering` kwarg —
the signature enforces it, and the std of the mean effect is invariant to the
additive centering constant) and model-free (re-summarized from the cached local
effects; the masked variant re-summarizes the subset through the same memo).

Like `heter_score`, it is a **std-type quantity in output units** (R2), which is
what makes it comparable across methods and features. The default — the standard
deviation of the mean effect — is therefore the *only* definition for PDP, ALE,
RHALE and **ShapDP**: for a linear model all four recover `|a_j| * std(x_j)`.
**d-PDP is the one override**: its mean effect is already a derivative, whose
dispersion is ~0 for a locally-linear model (a useless ranking), so it uses the
mean `|derivative|` over the (masked) data values, bridged into output units by
the feature's dispersion — which recovers the same `|a_j| * std(x_j)`.

*(Historical note: ShapDP once overrode this with the canonical `mean(|phi_s|)`.
That was removed — it is an L1 functional, off by the distribution-dependent
`E|x|/std(x)` factor, and φ absorbs half of each pairwise interaction, which the
other methods' importances exclude. Both biases made ShapDP the triage-plane
outlier. See `docs/method_semantics.md`.)*

`importances(mask=None) -> (D,)` is the per-feature vector, warning once (R9) for
feature types a method cannot explain. effector never sees `y`, so
loss/permutation importance is out of scope by construction — importance here is
a property of the fitted effect, not of a held-out error.

## R14 — Two-block lifecycle

The engine has exactly two caches, owned entirely by `GlobalEffectBase`.

**Cache (a) — local effects** (the only model-touching block). One entry per
feature, `{"frame": tuple, ...instance-aligned arrays}`, plus shared raw
material computed once per object (`_jac`, `_shap`, `_y_pred`). Every array is
instance-aligned, so a boolean `(N,)` mask slices it in one expression. The
*frame* is the discretization the method's local effect is defined on:

| method | local effect | frame |
|---|---|---|
| PDP / DerPDP | (d-)ICE columns, one per x-position | `()` — the position store only grows |
| ALE (continuous) | per-instance bin secants | `("fixed", nof_bins, min_points)` |
| (RH)ALE (categorical) | adjacent-level differences | `("order", (levels…))` |
| RHALE (continuous) | jacobian column | `()` (view of the shared table) |
| ShapDP | shap column | `()` (view of the shared table) |

**The one retrigger rule.** Recompute a local-effects entry iff it is absent or
its stored frame differs from the frame derived from the current config
(tuple equality, checked on every access). Same frame → the cache only grows
(PDP positions are appended, missing-only, no invalidation — growth is
additive). Frame change → replace the entry and bump the feature's **epoch**.

**Cache (b) — summaries** (pure numpy, LRU-bounded). Entries are payloads,
keyed `(feature, epoch, mask_key)`, and centering constants, keyed
`(feature, epoch, mask_key, mode)`. Payloads come in two archetypes, both
plain-numpy dicts (serializable, no callables): *binned* —
`limits`/`bin_effect`/`bin_variance` (+ `levels` on a discrete axis) for
ALE/RHALE/ShapDP, whose readers accumulate or interpolate between bins — and
*gridded* — `grid`/`mean`/`heter` for (d-)PDP, read by exact lookup at grid
positions. `mask_key` normalizes `None` and all-ones
to a single key (rule M1) — that equivalence lives in exactly one function.
The epoch bumps on frame replacement **or** config change (a refit with new
binning/scope), so stale summaries become *unreachable*: staleness is handled
by key structure, never by deletion logic.

**Model-touch inventory.** The model is called in exactly three situations:
(i) filling cache (a), once per frame; (ii) (d-)PDP evaluation at a
never-before-seen position — the missing columns only, cached forever (the
masked off-grid variant instead recomputes transiently on `data[mask]` and
caches nothing — partial-N columns cannot enter an instance-aligned cache);
(iii) `_y_pred`, once. Everything else — masked/regional surfaces,
`heter_score`, `importance`, centering constants, repeated evals and plots —
is zero model calls, pinned by counting-model contract tests.

**Subclass contract.** A method implements a frame declaration
(`_frame_from_config`) and three pure kernels, each split into a continuous
and a categorical variant — `_compute_local_cont|_cat` (the only kernels that
may touch the model), `_summarize_cont|_cat` (numpy in, payload dict out),
`_eval_payload_cont|_cat` (payload + xs in, numbers out) — and contains **no
cache, retrigger, or mask logic, ever**. The base owns the dispatch: each
gate-facing name (`_compute_local`, `_summarize`, `_eval_payload`,
`_compute_norm_const`) forks on `_is_cat(feature)` exactly once, so a kernel
body never inspects the feature type. A type-agnostic kernel is declared with
a class-level alias (`_compute_local_cat = _compute_local_cont`), never a fork
in the body. The `_cat` defaults raise — unreachable when
`SUPPORTED_FEATURE_TYPES` excludes ordinal/nominal, so continuous-only
methods (DerPDP) skip them entirely. Methods that don't fit the mold override
a named hook (`_eval_mean`, `_importance`, the norm-const shape) — they
never bypass the gates. `tests/toy_method.py` is the reference implementation
and the contract suite's guinea pig.

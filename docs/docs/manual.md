# The package manual

This page is the long-form reference for effector's architecture and API: what
each object is, what it stores, and the mental model behind every verb. It is
the deep companion to [the mental model](./mental_model.md) (the short
version) and is governed by the same written constitution —
[the design contract](./design.md), rules **R1–R14** — most of which is
mechanically enforced by the contract-test layer (`tests/test_contract_*.py`).
When this page cites "R7", that is a pointer into that document.

## 1. What effector is, in one page

`effector` is a Python library for **feature-effect explanations** of
black-box models: global effects (PDP, d-PDP, ALE, RHALE, SHAP-DP), their
**heterogeneity**, per-feature **importance**, and **regional effects**
(heterogeneity-reducing subregions à la REPID/GADGET), plus a one-click
`explain` report on top.

**The four sentences that summarize the design:**

1. **effector is numpy-only** (R10). `data` is a 2-D numeric numpy array;
   `model` and `model_jac` are numpy→numpy callables, called exactly as given,
   never wrapped. All metadata travels in one `schema=` argument. DataFrames
   enter only through the explicit converter `effector.from_dataframe`, and
   common model objects through `effector.adapters` — both return plain
   objects that *you* pass to the constructor.
2. **One stateful engine per method, two caches** (R14). An effect object
   holds the data, the model handle, the model-derived *local effects*
   (computed once per feature — the only model touch), and a memo of cheap
   numpy *summaries*. Every public verb is a query over those caches.
3. **A regional effect is not a new object — it is the one global object
   restricted by a mask** (R11). Every `eval`/`eval_heter`/`heter_score`/
   `importance`/`plot` accepts `mask=` (or `rule=` sugar); regional search is
   the query `find_regions(feature) -> Partition`, a value (R12).
4. **Queries return values, state stays in the engine** (R12). `importance`
   returns a float, `find_regions` a `Partition`, `explain` a `Report`. The
   engine never remembers your analysis; your notebook variables are the
   session.

### Package layout

```
effector/
  ingestion.py            # R10 border crossing: ingest(), Schema, from_dataframe
  adapters.py             # model side of the border: from_sklearn, from_torch,
                          #   classifier_proba, check
  global_effect.py        # GlobalEffectBase (the engine: caches, gates, dispatch)
  global_effect_pdp.py    # PDP, DerPDP        (via PDPBase)
  global_effect_ale.py    # ALE, RHALE         (via ALEBase)
  global_effect_shap.py   # ShapDP
  feature_effect.py       # FeatureEffect      (comparison facade, not a subclass)
  method_registry.py      # R5: the single per-method capability table
  rules.py                # Rule: normalized conjunctions of feature conditions
  partition.py            # Partition / Region value objects
  proposers.py            # candidate-split enumeration (the proposer seam)
  space_partitioning.py   # Best / BestLevelWise region finders
  axis_partitioning.py    # x-axis binning: Fixed / DynamicProgramming /
                          #   Agglomerative / Quantile
  report.py               # explain() -> Report / FeatureReport
  visualization.py        # all plotting + compare + plot_triage; draw-only
  ordering.py             # similarity seriation for nominal ALE
  helpers.py, utils.py    # prep_* helpers; numeric kernels
  theme.py                # set_theme("light"|"dark"|"paper"|"default")
  models.py, datasets.py, benchmarks.py   # synthetic models, data, ground truths
```

`visualization` is a set of pure draw-only functions — *the plot layer draws,
it does not compute* (R1).

**Public API** (`effector.`): the five effect classes `PDP`, `DerPDP`, `ALE`,
`RHALE`, `ShapDP`; the facade `FeatureEffect`; the values `Partition`,
`Region`, `Report`, `Rule`; the functions `explain`, `from_dataframe`,
`compare`, `plot_triage`, `set_theme`; the config `Schema`; and the modules
`adapters`, `axis_partitioning`, `space_partitioning`, `proposers`, `rules`,
`ingestion`, `models`, `datasets`, `benchmarks`, `theme`.

## 2. The input contract (R10): the numpy door

All constructors share one entry point, `ingestion.ingest(data, model,
model_jac, schema=)` — "the border crossing". Its job is validation and
metadata resolution, nothing else:

- `data` must be a **2-D numeric numpy array** `(N, D)`. Anything else raises
  `TypeError`. A pandas DataFrame is *hard-rejected* with a message pointing
  to `effector.from_dataframe`. pandas is never imported by the compute path.
- `model: (N, D) -> (N,)` and optional `model_jac: (N, D) -> (N, D)` **pass
  through untouched**. Wrapping a DataFrame/torch/sklearn model into a
  numpy→numpy callable is the user's job — with `effector.adapters` doing the
  common cases.

### `Schema` — the one metadata argument

Every constructor takes `schema=`, either an `effector.Schema` (frozen
dataclass, holds no data, reusable across constructions) or a plain dict with
the same keys. Every field is optional; unknown dict keys raise `ValueError`
listing the valid ones.

| field | meaning |
|---|---|
| `feature_names` | one name per column (default synthesized `x_0, x_1, …`) |
| `feature_types` | per column: `"continuous"` / `"ordinal"` / `"nominal"` (aliases `"cont"`, `"cat"`) |
| `cat_limit` | cardinality threshold for the int-column heuristic (default 10) |
| `target_name` | name of the model output (default `"y"`) |
| `scale_x_list` / `scale_y` | `{"mean", "std"}` dicts to display plots in original units |
| `category_names` | per-feature level names shown on categorical axes instead of codes |

**Define-or-infer.** Per field the precedence is: explicit schema field >
numpy heuristic > synthesized default. The type heuristic is deliberately
conservative: an integer-valued column with fewer than `cat_limit` unique
values is `ordinal`; everything else is `continuous`; **nominal is never
inferred from a numpy matrix** (numbers carry no "unordered" evidence).
Columns whose type was *guessed* by the cardinality rule trigger one
`UserWarning` naming them and the one-line `schema={"feature_types": [...]}`
fix.

### `from_dataframe` — the explicit pandas ramp

```python
X, schema = effector.from_dataframe(df)     # data conversion only; never touches the model
pdp = effector.PDP(X, model, schema=schema)
```

It reads names, dtypes, and category levels into `(X, Schema)`: float →
continuous; int → ordinal-or-continuous by `cat_limit`; bool → ordinal 0/1;
ordered `Categorical` → ordinal with the declared order; unordered
category/object/string → nominal via category codes; datetime → `ValueError`;
any NaN → `ValueError` naming the column. The returned schema is a *proposal
to inspect* — the int-column guess is the one thing no extractor can know for
sure.

The resolved metadata is stored on every object as `obj.feature_metadata`
with flat mirrors `feature_names`, `feature_types`, `cat_limit`,
`target_name`, `scale_x_list`, `scale_y`. Every public verb accepts a feature
**index or name** — names resolve through this metadata.

### `adapters` — the model side of the border

```python
model = effector.adapters.from_sklearn(est)                 # estimator/Pipeline -> callable
model = effector.adapters.classifier_proba(clf, class_=1)   # classifiers: P(class=k)
model, model_jac = effector.adapters.from_torch(net)        # torch: callable + autograd jacobian
effector.adapters.check(model, X)                           # the handshake: probe on 2 rows
```

Adapters only *return* plain callables — you make the final pass into the
constructor. Nothing is auto-detected, so nothing can be silently wrong.

## 3. The engine: the two-block lifecycle (R14)

Every effect class is `GlobalEffectBase` + method-specific kernels. The engine
owns exactly **two caches** and one config:

```
construct ──────────► queries: eval / eval_heter / heter_score / importance /
   │                           payload / plot / find_regions / …
   │  ingest (R10), prep_data (axis_limits filter + nof_instances subsample)
   │
   ├─ cache (a): local effects   ← THE ONLY MODEL-TOUCHING BLOCK
   │     one entry per feature: {"frame": tuple, ...instance-aligned arrays}
   │     + shared raw material (_jac, _shap, _y_pred), once per object
   │
   └─ cache (b): summaries memo  ← pure numpy, LRU-bounded
         payloads keyed (feature, epoch, mask_key)
         centering constants keyed (feature, epoch, mask_key, mode)
```

**`fit` declares, gates ensure** (R1). `fit(features, **config)` *declares*
the method configuration (binning, order, scope, default centering — kwargs
`eval`/`plot` deliberately do not accept) and eagerly warms the caches.
Nothing `fit` does is unavailable lazily: every query silently ensures what it
needs through the same gates, and fit-then-query equals never-fit-just-query
byte for byte.

**The single-model-touch constitution.** The model is called in exactly three
situations: (i) filling cache (a), once per frame; (ii) (d-)PDP evaluation at
a never-before-seen position — the missing ICE columns only, cached forever;
(iii) `_y_pred`, once. Everything else — masked/regional surfaces,
`heter_score`, `importance`, centering constants, the whole `find_regions`
split search, repeated evals and plots — is **zero model calls**, pinned by
counting-model contract tests.

**What a "local effect" is, per method** — the per-instance object in cache (a),
and the *frame* (discretization) it is defined on:

| method | per-instance local effect | frame |
|---|---|---|
| PDP / DerPDP | (d-)ICE columns, one per x-position | `()` — the position store only grows |
| ALE (continuous) | secants of `f` across each instance's bin | `("fixed", nof_bins, min_points)` |
| (RH)ALE (categorical) | adjacent-level differences | `("order", (levels…))` |
| RHALE (continuous) | jacobian column `df/dx_s` | `()` (view of the shared table) |
| ShapDP | the SHAP value `phi_s` per instance | `()` (view of the shared table) |

**The one retrigger rule.** A local-effects entry is recomputed iff it is
absent or its stored frame differs from the frame derived from the current
config. Same frame → the cache only grows. Frame change (or a refit with new
config) → replace the entry and bump the feature's **epoch**; summaries are
keyed by epoch, so stale entries become *unreachable* — staleness is handled
by key structure, never by deletion logic.

**Immutability of the frame.** `axis_limits` (a `(2, D)` array, given or
inferred) is fixed at construction and never mutated afterwards — by masks,
refits, or anything else. Anything region-shaped is transient and derived per
call (§5).

**Reproducibility** (R8) is contractual: every constructor takes
`random_state` (default `21`, `None` opts into fresh randomness); two
identical constructions give identical output; no effect-class code touches
global `np.random`.

## 4. The global API surface

### Constructor (R8)

Canonical parameter order, everything after `model_jac` keyword-only:

```python
Method(data, model, model_jac=None, *,
       data_effect=None,          # (RH)ALE: precomputed jacobian, skips the model touch
       nof_instances=10_000,      # int subsample or "all"   (ShapDP default: 1_000)
       axis_limits=None,          # (2, D) or None -> inferred from data
       schema=None,               # Schema | dict (section 2)
       random_state=21)
```

Per-class deviations: `PDP`/`DerPDP` take no `data_effect`; `ShapDP`
additionally takes `shap_values=` (inject precomputed values),
`backend="shap"|"shapiq"`, `budget=512`, and explainer kwargs. Injection
kwargs (`data_effect`, `shap_values`) all mean the same thing: *the model
touch already happened elsewhere; never call the model.*

### `fit(features="all", *, centering=<class default>, **method_kwargs)`

Declares the configuration and warms the caches (R1/R14). Method-specific
kwargs:

| method | fit kwargs beyond `centering` |
|---|---|
| PDP / DerPDP | `use_vectorized=True` |
| ALE | `binning_method="fixed"`, `order=None` (categorical level order) |
| RHALE | `binning_method="dp"`, `order=None`, `binning_scope="global"` |
| ShapDP | `binning_method="dp"`, `binning_scope="global"` |

`binning_method` accepts a string (`"fixed" | "dp" | "greedy"` — one alias
table, one resolver, R6) or a configured instance from
`effector.axis_partitioning` (`Fixed`, `DynamicProgramming`, `Agglomerative`,
`Quantile`).

### `eval(feature, xs, centering=None, mask=None, rule=None) -> (T,)`

**One return type, always** (R1): the mean effect at `xs`, a single `(T,)`
array. Heterogeneity deliberately does *not* travel through `eval`.

**Centering vocabulary** (R3): `{False, "zero_integral" (=True),
"zero_start"}`. `centering=None` means "use the class default", declared once
per class as `DEFAULT_CENTERING` (PDP, ALE, RHALE, ShapDP: `"zero_integral"`;
DerPDP: `False`).

### The heterogeneity ladder (R2)

Heterogeneity has its own surface — an aggregation ladder with one consumer
per level:

```python
payload(feature)                            -> dict    # the raw fitted object
eval_heter(feature, xs, mask=None)          -> (T,)    # the heterogeneity curve h(xs)
heter_score(feature, mask=None, rule=None)  -> float   # the one method-agnostic scalar >= 0
```

`eval_heter` returns the **variance** of the method's own per-instance effect
object (PDP: centered ICE levels; DerPDP: d-ICE slopes; (RH)ALE: per-bin slope
variance as a step function; ShapDP: the interpolated per-bin φ variance).
Variance internally, std only at the plot layer. There is **no centering
kwarg** — h is invariant to centering and the signature enforces it. Every
plotted band/error bar equals `eval_heter` output.

`heter_score` is the mean of `eval_heter` over a uniform grid on the feature's
interval (frequency-weighted over levels for categorical features). It is the
single scalar the regional split search consumes — the seam that decouples
regional search from the effect method entirely.

### `importance` (R13): the μ-twin

```python
importance(feature, mask=None, rule=None)  -> float >= 0
importances(mask=None, rule=None)          -> (D,)   # NaN + one warning for unsupported types
```

`importance` measures how much the **mean effect** varies — the μ-twin of
`heter_score` (which measures per-instance spread), evaluated the same way it
is. Model-free, centering-invariant (no `centering` kwarg by design). Default:
the std of the mean effect; `ShapDP` overrides with the canonical
`mean(|phi|)`; `DerPDP` with `mean(|derivative|)`. effector never sees `y`, so
loss/permutation importance is out of scope by construction — importance here
is a property of the fitted effect.

**Importance and heterogeneity are orthogonal.** A feature can be highly
heterogeneous yet have low importance if its mean effect cancels out — that is
exactly the feature a global average hides and a regional analysis reveals
(the top-right corner of `plot_triage`).

### `plot(feature, ...)` (R7)

A thin wrapper over the same summaries plus one `vis.*` call. Uniform rule:
every `.plot` (and `Partition.plot`) returns `(fig, ax)` when
`show_plot=False` and `None` otherwise. Common kwargs: `heterogeneity`
(`False | True | "std" | "std_err" | "ice" | "shap_values"`; defaults vary:
PDP `"ice"`, (RH)ALE/ShapDP `True`), `centering`, `scale_x`/`scale_y`
(plot-time override of schema scaling; `False` explicitly disables an
inherited scale), `y_limits`, `dy_limits` (derivative units),
`show_avg_output`, and `mask=`/`rule=`/`feature_label=` (§5). Categorical
features render as bars at observed levels with whiskers; `category_names`
from the schema label the ticks.

### Feature-type capability matrix

Declared per class, mirrored into the R5 registry, and enforced with a clear
`ValueError` when a method is asked for an unsupported type:

| method | continuous | ordinal | nominal | categorical strategy |
|---|---|---|---|---|
| PDP | yes | yes | yes | ICE evaluated at observed levels |
| DerPDP | yes | — | — | — |
| ALE | yes | yes | yes | adjacent-level differences |
| RHALE | yes | yes | — | level diffs, grouped |
| ShapDP | yes | yes | yes | per-level SHAP statistics |

Shared rule: discrete features are evaluated **only at observed levels** —
`eval`/`eval_heter` elsewhere raises `ValueError`; the model is never queried
at non-existing category values. For nominal ALE the level order matters;
`order=` accepts an explicit list, `"similarity"` (MDS seriation from
`effector.ordering`), or the default deterministic encoded order.

## 5. Masked evaluation (R11): regional ≡ masked global

Every `eval`, `eval_heter`, `heter_score`, `importance`, and `plot` accepts
`mask=` — a boolean `(N,)` array over the object's (post-subsampling)
instances — and, as sugar over it, `rule=`:

```python
mask = X[:, 3] == 0                              # e.g. "non-working days"
pdp.eval(0, xs, mask=mask)                       # the effect *within* the subregion
pdp.plot("hr", rule="workingday == 0")           # same thing, by rule string
pdp.heter_score("hr", rule=parts["hr"][2].rule)  # or a Region's Rule object
```

A `rule` is an `effector.Rule` (a normalized conjunction of per-feature
conditions) or a string like `"temp < 6.5 and workingday == 0"`, parsed with
the effect's metadata; `Rule.contains(data)` is the single rule→mask site.
`rule=` and `mask=` are mutually exclusive.

Semantics — and the invariant behind them:

- **Nothing is stored.** A mask never mutates state — not `axis_limits`, not
  bins, not payloads. Anything region-shaped is transient, derived per call
  via `_effective_limits(feature, mask)` = `[min, max]` of the masked column
  (categorical analog: within-mask level frequencies). Its only consumers:
  masked centering constants, the masked plot x-window, and degeneracy guards
  (empty mask / collapsed interval → `ValueError`).
- **Model-free.** A masked call re-summarizes the *stored per-instance local
  effects* through the summaries memo — zero model calls (contract-tested).
  The one allowed exception: PDP `eval(mask=)` at points off the cached grid
  recomputes ICE on `data[mask]` transiently, symmetric with global PDP eval.
- **Frame semantics.** Structure frozen on the global frame stays frozen. PDP
  keeps its grid and re-averages the masked ICE *columns*; ALE keeps its bin
  edges (secants are edge-bound — re-binning would require model calls) and
  re-averages masked per-bin stats, interpolating bins the mask left empty
  with flat extension at the edges. RHALE/ShapDP store genuinely per-instance
  local effects, so a masked call **re-runs binning** on the masked subset.
- **`binning_scope`** (RHALE/ShapDP fit kwarg): the x-range handed to the
  binner on masked re-binning — `"global"` (default; the full `axis_limits`
  interval) or `"effective"` (the masked column's own `[min, max]`, packing
  the bin budget into the region). Recorded in `fit_args` and replayed on
  every masked call, so the split search's `heter_score(mask)` and every
  masked eval/plot always share the same scope.

## 6. Regional analysis: `find_regions` → `Partition` (R12)

Regional questions are a **query** on the fitted global effect — nothing is
stored on the effect; a `Partition` is a value:

```python
part  = pdp.find_regions("hr")                          # one feature -> Partition
parts = pdp.find_regions(features="heterogeneous")      # several -> {name: Partition}
part.show()                                             # the partition tree + level stats
```

Exactly one of `feature`/`features` is given. The plural form accepts a list
of indices/names, `"all"` (every feature this method supports), or
`"heterogeneous"` (supported features whose `heter_score` is at or above the
median — the same threshold convention `effector.explain` uses), and returns
the `{feature_name: Partition}` dict that
`plot_triage(effect, partitions=...)` consumes directly.

The search is model-free: every candidate's score is
`heter_score(feature, mask)`, re-summarized from the cached local effects.
There are no method fit kwargs here — the binning/scope are exactly those the
feature was fitted with, replayed.

### Finders

`finder=` names the search strategy: `"best"` (node-wise recursive, default),
`"best_level_wise"` (one split per level), or a configured instance:

```python
finder = effector.space_partitioning.Best(
    min_heterogeneity_decrease_pcg=0.1,   # a split must cut weighted heter by >= 10%
    heter_small_enough=0.001,             # stop early when a node is homogeneous
    max_depth=2, min_samples_leaf=10,
    numerical_features_grid_size=20,      # candidate thresholds per numeric feature
)
part = pdp.find_regions("hr", finder=finder,
                        candidate_conditioning_features="all")
```

A finder consumes only `(score_fn: mask -> float, data, metadata, its own
config)` and returns a `Partition` — new finders (ICE clustering, subgroup
discovery, a user `groupby`) plug in with zero changes elsewhere.

Inside the built-in finders, candidate enumeration is its own protocol
(`effector.proposers`): a proposer maps a conditioning feature to candidate
splits — ordered tuples of disjoint, jointly-covering conditions. The defaults
reproduce the classic search (binary thresholds on an interior grid for
continuous features; one-vs-rest over observed levels for categorical ones).
Richer named proposers are selected per feature type from the finder
constructor: `categorical_proposer=` `"subsets"` / `"ordered"`
(similarity-seriated) / `"multiway"`, and `continuous_proposer=` `"quantiles"`
(k-way marginal cuts) — or any proposer instance.

### `Partition` and `Region`

A `Partition` is an ordered container of `Region`s; each region's identity is
its `Rule`:

```python
len(part)                       # number of regions (root + descendants)
part.leaves                     # the leaf regions
for r in part:
    r.idx, r.level, r.rule, r.heterogeneity, r.nof_instances, r.weight

part.mask(2)                    # boolean (N,) mask of region 2 (a copy)
part.label(2)                   # the region's condition, human-readable
part.eval(2, xs)                # mean effect within region 2
part.eval_heter(2, xs)          # heterogeneity within region 2
part.plot(2, heterogeneity="ice")
part.show()                     # print the tree;  part.show_axes(): per-feature intervals
d = part.to_dict()              # fully serializable: rules + stats, no masks, no model
```

One source of truth: the heterogeneity a `Partition` reports **is**
`heter_score(feature, mask)` on each region's mask, and every plotted region
band derives from `eval_heter(feature, xs, mask)` — global and regional
cannot disagree.

A partition restored with `Partition.from_dict(d)` is **unbound**: rules,
labels, and stats work; `eval`/`plot` need `bind(effect)`, which recomputes
every mask from its rule and verifies it. `Partition.from_rules([...],
effect=..., feature=...)` builds a user-authored partition through the same
validation — manual and automated regional analysis converge on one value
type.

## 7. The one-click report: `explain` → `Report`

`explain` is the workbench walked with defaults and nobody intervening — one
model touch, then everything model-free:

```python
report = effector.explain(
    X, model,                    # + model_jac= for method="rhale"/"derpdp"
    schema=schema,
    method="pdp",                # "pdp" | "ale" | "rhale" | "shapdp" | "derpdp"
    top_k=5,
    heter_threshold=None,        # default: median heter_score across ranked features
    finder="best",
    nof_instances=10_000,
)
report.show()                    # importance-ranked table + per-feature partition trees
report.plot_importance()         # horizontal bar chart
report.to_html("report.html")    # ONE self-contained page (figures inlined as base64)
```

It fits the chosen method, ranks features by `importance`, and for the
top-`top_k`: computes the mean-effect + heterogeneity curves and, when the
feature clears `heter_threshold`, runs `find_regions`. Growing `top_k` costs
no extra model calls.

The `Report` is a value and round-trips without the estimator:

```python
d = report.to_dict()
report2 = effector.Report.from_dict(d)   # unbound: text + importance chart work;
report2.show()                           # live re-plot raises
```

Each `report.features[i]` is a `FeatureReport` with `feature, name,
importance, heter_score, xs, y, h, partition`.

## 8. Comparison and triage

### `plot_triage` — the to-do list

```python
effector.plot_triage(pdp)                        # importance (x) vs heterogeneity (y)
effector.plot_triage(pdp, partitions=parts)      # + arrows: each global point -> its leaves
```

Bottom-left: ignore. Bottom-right: important and fully described by its mean
effect. **Top-right: important *and* heterogeneous — where `find_regions`
should look.** With `partitions` (exactly what
`find_regions(features=...)` returns) the plot becomes the before/after
story: leaves of a good partition move right (more decisive) and down (spread
explained).

### `compare` — cross-examination of fitted engines

```python
effector.compare(pdp, rhale, shapdp, feature="hr", centering=True)
```

Overlays the mean effect of engines *you already hold* — different methods, or
different models over the same columns. It computes nothing itself and stores
nothing. Derivative-unit effects (`DerPDP`) cannot be mixed with output-unit
ones; `centering=False` is coerced to `"zero_integral"` with a warning (a
comparison is only meaningful centered).

### `FeatureEffect` — the single-model shortcut

Not a subclass — a facade that ingests once and lazily builds per-method
engines (through the R5 registry) sharing the *same* data, subsample, axis
limits, and schema, so the only difference between curves is the method:

```python
fe = effector.FeatureEffect(X, model, model_jac=None, schema=schema)
curves = fe.eval(0, xs, methods=["PDP", "ALE", "RHALE"])   # {"PDP": (T,), ...}
fe.plot(0, methods=["PDP", "ALE", "RHALE", "ShapDP"],
        method_kwargs={"ShapDP": {"nof_instances": 300}})
```

`ShapDP` is opt-in (slower; needs the `shap` package). Methods that don't
support the feature's type are dropped via the capability matrix. `DerPDP` is
excluded from the pool (derivative units).

## 9. Supporting modules

- **`method_registry`** (R5): the single `{canonical_name: MethodSpec(cls,
  needs_jac, display_name, ...capabilities)}` table plus aliases (`"shap"` →
  `shapdp`). `FeatureEffect`, `explain`, and plot titles all read it; a
  per-method `if/elif` chain anywhere else is, by contract, a bug.
- **`axis_partitioning`**: x-axis binning for (RH)ALE/ShapDP. `Fixed`
  (equal-width), `DynamicProgramming` (optimal variance-based),
  `Agglomerative` (greedy merge), `Quantile`. All share a `Constraints`
  object and report a typed `NoBinningReason` when a feature can't be binned.
- **`rules`**: the `Rule` algebra — normalized conjunctions of per-feature
  conditions (half-open intervals `x < t` / `x >= t`, explicit level sets).
  Membership (`contains`), display (`format`), and serialization (`to_dict`)
  derive from the one object, so they cannot drift apart.
- **`visualization`**: pure draw-only functions; every function obeys the R7
  return rule. `effector.set_theme("light" | "dark" | "paper" | "default")`
  switches the house matplotlib theme globally (palette colors apply
  per-artist by default; chrome rcParams only on explicit opt-in).
- **`models` / `datasets` / `benchmarks`**: synthetic models with analytic
  jacobians, data generators + real datasets (`BikeSharing`), and paired
  (model, distribution) objects with **closed-form ground-truth effects** used
  by the test suite and executed notebooks as regression oracles.
- **Error style** (R9): `ValueError`/`TypeError` for user input (never bare
  `assert`); `warnings.warn` or `logging` (never `print`) in internals.

## 10. Worked examples

### 10.1 End-to-end, numpy-native

```python
import numpy as np
import effector

X = np.random.uniform(-1, 1, (1_000, 3))
model = lambda x: x[:, 0] ** 2 + x[:, 1] * (x[:, 2] > 0)      # (N,3) -> (N,)

pdp = effector.PDP(X, model, nof_instances="all")
pdp.fit(features=0, centering="zero_start")
xs = np.linspace(-1, 1, 100)
y  = pdp.eval(0, xs)                    # (100,) mean effect, class-default centering
h  = pdp.eval_heter(0, xs)              # (100,) variance of centered ICE at xs
H  = pdp.heter_score(0)                 # one scalar
I  = pdp.importance(0)                  # its μ-twin
pdp.plot(0, heterogeneity="ice")        # shows; -> None
fig, ax = pdp.plot(0, show_plot=False)  # returns the figure instead (R7)
```

### 10.2 From a DataFrame + an sklearn model

```python
X, schema = effector.from_dataframe(df)          # names, dtypes, levels -> Schema
model = effector.adapters.from_sklearn(est)      # estimator -> numpy callable
effector.adapters.check(model, X)                # the handshake

ale = effector.ALE(X, model, schema=schema)
ale.fit(features="all", binning_method="fixed")
ale.plot("season")                               # nominal feature -> level bars, named ticks
```

### 10.3 A torch model

```python
model, model_jac = effector.adapters.from_torch(net)    # callable + autograd jacobian
rhale = effector.RHALE(X, model, model_jac, schema=schema)
```

(or hand-write the numpy→numpy wrappers — the contract is yours to meet
either way.)

### 10.4 Regional analysis, and the same thing by hand

```python
pdp.fit("hr")
part = pdp.find_regions("hr")
part.show()                              # tree: idx, rule, heter = heter_score(mask)
pdp.plot("hr", rule=part.leaves[0].rule)

# what that call *is*, spelled out (R11):
pdp.plot("hr", mask=part.mask(part.leaves[0].idx))
# ...and an ad-hoc subregion needs no search at all:
pdp.plot("hr", rule="workingday == 0")
```

### 10.5 Rank, drill, triage — the workbench loop

```python
pdp = effector.PDP(X, model, schema=schema)
pdp.fit("all")
effector.plot_triage(pdp)                             # who matters, who is hiding something
parts = pdp.find_regions(features="heterogeneous")    # explain the spread
effector.plot_triage(pdp, partitions=parts)           # the analysis, with receipts
```

### 10.6 Injecting precomputed quantities (skip the model touch)

```python
shapdp = effector.ShapDP(X, model, shap_values=phi)         # phi: (N, D), computed elsewhere
rhale  = effector.RHALE(X, model, data_effect=jacobian)     # jacobian: (N, D)
# both objects will never call `model` for fitting/eval/plot
```

## 11. Migrating from the `Regional*` classes

The `RegionalPDP` / `RegionalALE` / `RegionalRHALE` / `RegionalShapDP` /
`RegionalDerPDP` classes were **removed**. Regional questions are now a query
on the global effect:

| Old (removed) | New |
|---|---|
| `RegionalPDP(...).fit(space_partitioner=P)` | `PDP(...).fit(); .find_regions(f, finder=P)` |
| `reg.summary(features=f)` | `partition.show()` |
| `reg.plot(feature=f, node_idx=i)` | `partition.plot(i)` or `pdp.plot(f, rule=partition[i].rule)` |
| `reg.eval(f, node_idx=i, xs)` | `partition.eval(i, xs)` or `pdp.eval(f, xs, mask=partition.mask(i))` |
| `reg.tree["feature_f"].nodes` + `node.info[...]` | iterate `partition`; `region.rule/.heterogeneity/.weight` |
| — (no importance) | `effect.importance(f)`, `effect.importances()` |
| — (no report) | `effector.explain(...) -> Report`, `report.to_html()` |

## 12. Contract quick reference (R1–R14)

| # | rule, in one line |
|:--|:--|
| R1 | `fit` declares config + warms caches; `eval` returns the mean effect only, one type; `plot` draws, never computes |
| R2 | heterogeneity ladder `payload` → `eval_heter` (variance, no centering kwarg) → `heter_score`; all accept `mask` |
| R3 | centering vocabulary `{False, "zero_integral"=True, "zero_start"}`; class default declared once |
| R4 | state = two caches + declared config: local effects (frame-carrying) + summaries memo (epoch-keyed) |
| R5 | one method registry; per-method if/elif chains are a bug |
| R6 | one alias table + one resolver per string argument (binning, finder) |
| R7 | every plot returns `(fig, ax)` iff `show_plot=False`, else `None` |
| R8 | canonical constructor order, keyword-only after `model_jac`; `random_state` contractual |
| R9 | `ValueError`/`TypeError` for user input; `warnings`/`logging`, never `print` |
| R10 | numpy-only door: 2-D numpy `data`, numpy→numpy model, one `schema=`, `from_dataframe`/`adapters` ramps |
| R11 | regional ≡ masked global: transient masks (and `rule=` sugar) over one immutable global frame, model-free |
| R12 | regions are values, not state: `find_regions -> Partition`, rule-primary, serializable; memos are invisible |
| R13 | `importance` = dispersion of the mean effect, the μ-twin of `heter_score`; effector never sees `y` |
| R14 | two-block lifecycle: frame-gated local effects + epoch-keyed summaries memo; three model-touch situations |

**Where to look next:** [the design contract](./design.md) (normative),
[method semantics](./method_semantics.md) (per-method exactness formulas by
feature type, including masked semantics), and `tests/test_contract_*.py`
(the enforcement).

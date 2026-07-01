# Homogenization Plan — stabilizing the core before v0.3 features

Goal: agree on a small set of **core rules** for the method API (`fit` / `eval` / `plot` and
the state they share), then refactor each submodule to follow them. The package's power is
that any global method's `.eval()` can be reused to compute heterogeneity → regional effects
→ (future) importance/comparison features. Every deviation from the common contract is a
place where the next feature has to special-case a method.

Everything below comes from a full read of `effector/` (~7.2k LOC). Sections: (0) mental
model as-is, (1) proposed core rules, (2) per-submodule changes, (3) bugs found (mostly
caused by non-DRY spots — evidence the homogenization pays off), (4) suggested order.

---

## 0. The mental model today

Two parallel families, one facade:

- **Global**: `GlobalEffectBase` (`global_effect.py`) holds shared preprocessing
  (axis-limit filtering → subsampling → names) and the lazy-refit machinery
  (`is_fitted`, `fit_args`, `requires_refit`). Subclasses: `ALE`/`RHALE` (bin-based,
  via `compute_ale_params`), `PDP`/`DerPDP` (ICE-matrix based), `ShapDP`
  (shap values + piecewise-linear interpolation).
- **Regional**: `RegionalEffectBase` (`regional_effect.py`) repeats the same
  preprocessing, builds a per-feature heterogeneity callable **from the corresponding
  global method's `.eval(heterogeneity=True)`**, hands it to a space partitioner
  (`space_partitioning.Best`), stores a `Tree`, and serves `eval/plot` by instantiating
  a fresh global-method object on each node's subset (`_create_fe_object`).
- **Facade**: `FeatureEffect` (`feature_effect.py`) — new — shares data once and overlays
  the `eval` of several global methods.
- **Support**: `axis_partitioning` (bin-limit strategies: Fixed/Greedy/DP),
  `space_partitioning` (+`tree`), `visualization` (matplotlib), `helpers`/`utils`
  (prep + ALE math), `models`/`datasets` (synthetic + BikeSharing).

The design is right. The drift is in the details: default values, state schemas,
string registries, return contracts, and duplicated loops.

---

## 1. Proposed core rules (the "constitution")

Agree on these first; every submodule change below is an application of one of them.

- **R1 — Lifecycle**: `fit(features="all", centering=<class default>, **method_kwargs)`
  computes and stores everything under `feature_effect["feature_{i}"]`;
  `eval(feature, xs, heterogeneity=False, centering=<class default>)` never recomputes
  unless `requires_refit`; `plot` is always a thin wrapper over `eval` + one `vis.*` call.
- **R2 — Heterogeneity semantics**: pick **one** quantity that `eval(..., heterogeneity=True)`
  returns for all methods — recommend **variance** internally, converted to **std only at
  the plot layer** — and name fields accordingly (today: ALE returns bin *variance*, PDP
  returns `np.var`, ShapDP returns a spline fit to variance but names it `spline_std`,
  and every docstring says "std").
- **R3 — Centering vocabulary**: `{False, "zero_integral" (=True), "zero_start"}` via
  `helpers.prep_centering` (already true). Each class declares its default **once** as a
  class attribute (e.g. `DEFAULT_CENTERING = "zero_integral"`), and `fit/eval/plot`
  signatures use that attribute instead of hardcoding different literals per method.
- **R4 — State schema**: `feature_effect["feature_{i}"]` always contains
  `norm_const: float | None` (use `None`, not the `1e8` `EMPTY_SYMBOL` or `np.nan` —
  today all three conventions coexist) plus a method-specific payload;
  `fit_args["feature_{i}"]` records the kwargs needed to detect refit.
- **R5 — One method registry**: a single `{canonical_name: (cls, needs_jac, uses_data_effect,
  display_name)}` table (plus aliases), used by `FeatureEffect._REGISTRY`,
  `RegionalEffectBase._create_fe_object`, and plot titles. Today the same knowledge lives
  in ≥3 places as if/elif chains and string comparisons.
- **R6 — String-argument registries**: exactly one alias table per concept, resolved by one
  `return_default`-style function, and *asserts always match the resolver*
  (binning: `"fixed" | "greedy" | "dp"`; partitioner: `"best" | "best_level_wise"`).
- **R7 — Plot contract**: every `vis.*` function and every public `.plot` returns
  `(fig, ax)` when `show_plot=False` and `None` otherwise — uniformly.
- **R8 — Constructor contract**: canonical parameter order
  `data, model, model_jac=None, *, data_effect, nof_instances, axis_limits,
  feature_types, cat_limit, feature_names, target_name, ...` — everything after
  `model_jac` keyword-only, so the current per-class positional shuffles can't bite.
- **R9 — Errors & messages**: `ValueError`/`TypeError` (not bare `assert`) for user input;
  `warnings.warn` or `logging` (not `print`) inside heterogeneity functions.

---

## 2. Per-submodule proposals

### 2.1 `helpers.py` / `utils.py` (foundations — do first)

- Split the constant collision: `BIG_M`, `EPS`, and `EMPTY_SYMBOL` are semantically
  different but `BIG_M == EMPTY_SYMBOL == 1e8`. Replace `EMPTY_SYMBOL` with `None`
  everywhere `norm_const` is stored (R4); keep `BIG_M` only as the partitioning penalty.
- Delete dead code: `prep_dale_fit_params`, `prep_ale_fit_params` (no callers).
- `prep_features`: validate feature indices are in range and raise `ValueError` (R9);
  currently accepts anything.
- Fix truncated docstring `indices_within_limits` ("insi").
- `utils.compute_ale_params`: doctest examples reference undefined names
  (`bin_values`, `bin_limits` vs the defined `df_dxs`, `limits`) and show a
  `bin_estimator_variance` key that the function no longer returns — fix examples to match.
- Unify numerical differentiation: `utils.compute_jacobian_numerically` uses forward
  difference with `eps=1e-8`, while `ice_*` in `global_effect_pdp.py` inlines central
  difference with `1e-6`. One helper, one scheme (central, one eps), used by both.

### 2.2 `global_effect.py` (base class — the heart of the change)

- **Hoist the fit loop** (R1). `ALEBase._fit_loop` already does the right thing:
  `prep_features` → `prep_centering` → per-feature `_fit_feature` → `norm_const` →
  `is_fitted` → `fit_args`. Move it to `GlobalEffectBase`; PDP and ShapDP currently
  re-implement the identical loop inline (`global_effect_pdp.py:100-110`,
  `global_effect_shap.py:399-417`).
- **Make `_eval_unnorm` the abstract kernel** and `eval` a *concrete base method*.
  All three `eval`s repeat the same skeleton: `prep_centering` → `requires_refit`→`fit` →
  assert limits → evaluate → subtract `norm_const` → return `y` or `(y, het)`.
  Only the middle "evaluate" differs (ALE: accumulated bin effects; PDP: ICE matrix mean/var;
  ShapDP: spline). With `_eval_unnorm(feature, xs, heterogeneity)` abstract, `eval`
  is written once, and everything downstream (regional heterogeneity functions,
  `FeatureEffect`, future importance measures) inherits identical behavior.
- **Generalize `_compute_norm_const`** (currently ALE-only): it only needs `_eval_unnorm`,
  so it moves to the base unchanged — `zero_integral` = mean over a linspace,
  `zero_start` = value at the left limit. PDP's inline norm-const logic
  (`_fit_feature`) and ShapDP's (`_fit_feature` lines 250-261) collapse into it.
- **Fix `requires_refit`**: the `norm_const is None` check is dead today because
  sentinels (`1e8`/`np.nan`) are stored instead of `None`; after R4 it becomes live again.
  Also record `points_for_centering` in `fit_args` consistently (ALE stores only
  `centering`; PDP/Shap store both).
- Remove dead state `self.avg_output` (set to `None` in `__init__`, never written,
  always passed as `None` into `prep_avg_output`). Either compute it lazily in a
  `avg_output` property or delete.
- Class-attribute identity (R5): replace the `method_name: str` constructor threading
  (`ALEBase` even sets `self.method_name` twice, unlowered then lowered) with class
  attributes `name = "ale"`, `display_name = "Accumulated Local Effects (ALE)"`.
  Plot titles then stop being `if self.method_name == "pdp"` string comparisons.

### 2.3 `global_effect_ale.py` / `global_effect_pdp.py` / `global_effect_shap.py`

- After 2.2, each file keeps only: constructor, `_fit_feature` (payload computation),
  `_eval_unnorm`, and a thin `plot`. Concretely removable duplication:
  - ALE/RHALE `_fit_feature` share the find-limits → assert → `compute_ale_params`
    sequence; keep one implementation parameterized by how `data_effect` is obtained.
  - `RHALE.compile()` and `RegionalRHALE.compile()` are identical — move to a shared
    helper (e.g. `utils.prep_data_effect(data, model, model_jac)`).
  - The identical "Impossible to compute bins…" assertion appears 3× (ALE, RHALE, ShapDP)
    — one raise-helper next to `compute_ale_params`.
- **Align `eval`/`fit` defaults** (R3) — today:

  | method | fit centering | eval centering | eval heterogeneity |
  |---|---|---|---|
  | ALE/RHALE | `True` | `True` | `False` |
  | PDP/DerPDP | `False` | `False` | `False` |
  | ShapDP | `True` | `True` | **`True`** |

  Decide per-class defaults deliberately (ALE-family must center to be meaningful; that's
  fine) but expose them as the single class attribute, and make ShapDP's
  `heterogeneity=True` default `False` like everyone else.
- **PDP `eval(return_all=True)`**: it silently ignores `heterogeneity` and changes the
  return type. Since regional PDP relies on it, keep the capability but consider renaming
  the concept into the contract (e.g. a separate documented method `eval_all(feature, xs)`
  or keep kwarg but document in base). Decide once — this is the only eval whose signature
  deviates.
- **ShapDP specifics**:
  - `plot` line 517: `if heterogeneity == "std" or True` — always truthy; fix to
    `== "std"`. Also `plot` is the only one *not* calling `prep_centering`.
  - Rename `spline_std` → `spline_var` (it is fit on `bin_variance`) and take the sqrt at
    the vis layer (R2).
  - The giant duplicated "Code behind the scene" docstring block appears twice in this
    file *and* twice in `regional_effect_shap.py` — factor the explainer-construction into
    a module-level function `_compute_shap_values(model, data, backend, budget,
    explainer_kwargs, explanation_kwargs)` and let docs reference it once.
- **PDP naming**: `ice_non_vectorized`/`ice_vectorized` docstring examples call functions
  with parameters that don't exist (`heterogeneity=`, `model_returns_jac=`) — stale;
  fix when touching. The `use_vectorized` kwarg threads through fit/eval/plot on PDP only —
  acceptable method-specific kwarg under R1, but it must be recorded in `fit_args`
  (it is) and forwarded consistently by regional (it is, via `kwargs_fitting`).
- **Constructor order** (R8): `PDPBase(..., axis_limits, nof_instances, ...)` vs
  `ALEBase(..., nof_instances, axis_limits, ...)` vs `RHALE(..., nof_instances,
  axis_limits, data_effect, ...)`. Make keyword-only; the regional files currently call
  these constructors positionally and only survive because each call site memorized its
  own class's order.

### 2.4 `visualization.py`

- Extract the shared frame: every function repeats fig/ax creation, `trans_affine`
  x-scaling, avg-output hline, xlabel from `feature_names` fallback, ylabel from
  `target_name`, legend, `y_limits`, `show_plot`-return. One `_finalize_ax(...)` +
  one `_scale_xy(...)` kills ~40% of the file and guarantees R7.
- Fix the 0-based/1-based clash: vis fallback labels are `x_%d % (feature+1)` while
  `helpers.get_feature_names` produces `x_0, x_1, ...`. Pick 0-based (matches API indices).
- `plot_pdp_ice`: `std_err = np.sqrt(np.var(...))` *is* the std, not the standard error —
  fix or drop the option; `std` computed twice; `nof_ice` clamping half-done at the top and
  redone in the `"ice"` branch.
- `is_derivative` is accepted but never passed by `PDPBase._plot` — so DerPDP with
  `scale_y` gets affine-transformed like a level plot (wrong units). Wire it from the
  method's class attribute (R5 registry knows the method) or drop the param.
- ALE plot grid: `ale_plot` hardcodes `np.linspace(..., 1000)` while PDP/Shap take
  `nof_points` — expose the same knob everywhere (ALE.plot currently has no `nof_points`).
- Standardize heterogeneity-option names across plots (`"std"`, `"ice"`,
  `"shap_values"`): keep per-method extras, but `True` must mean the same thing (`"std"`)
  everywhere — already enforced by `prep_confidence_interval`, just document it in one place.

### 2.5 `regional_effect.py` + `regional_effect_{ale,pdp,shap}.py`

- **Template-method `fit`** (mirror of 2.2): every regional `fit` repeats
  resolve-partitioner → assert min-points → `prep_features` → `tqdm` loop
  { method-specific global precompute → `_create_heterogeneity_function` → `_fit_feature` }
  → store kwargs. Hoist the skeleton into `RegionalEffectBase.fit` with two abstract
  hooks: `_precompute_global(feature, **kwargs)` and
  `_create_heterogeneity_function(feature, ...)`.
- **Kill the `locals()` idioms** — they already caused real bugs:
  - `regional_effect_ale.py:236-243` and `:442-449` slice
    `list(all_arguments.keys())[:3]` with a comment saying "first 8 arguments", and filter
    `kwargs_fitting` on the misspelled key `"binnning_method"` → **`kwargs_fitting` is
    always empty**, so `eval`/`plot` refit regional (RH)ALE with *default* binning even
    when the user chose otherwise.
  - Every `plot` builds `kwargs = locals(); kwargs.pop("self")` and `_plot(kwargs)`
    pops/renames keys — replace with explicit named parameters passed explicitly.
  Replace both with explicit dicts: `self.kwargs_subregion_detection = {...}`,
  `self.kwargs_fitting = {...}` written out per method.
- **Partitioner registry** (R6): `RegionalEffectBase._fit_feature` asserts
  `space_partitioner in ["best", "cart"]` but `space_partitioning.return_default`
  resolves `["best", "best_level_wise"]` — `"cart"` passes the assert then crashes,
  `"best_level_wise"` is valid but fails the assert. Also each regional `fit` *and*
  `_fit_feature` both do the string→object conversion (twice per call, with a
  `copy.deepcopy` in one path only). Resolve once, at the top of base `fit`.
- **`_create_fe_object` → registry** (R5): replace the five-way if/elif on
  `self.method_name` with the shared method registry; it also removes the smell of the
  base class reading `self.global_shap_values`, which only exists on `RegionalShapDP`.
- **Unify signatures**:
  - `features` param: required-positional in `RegionalALE.fit`/`RegionalShapDP.fit`,
    defaulted `"all"` in the other three — default everywhere.
  - `plot` signatures drift (`RegionalPDP.plot` annotates `heterogeneity: bool = "ice"`;
    only `RegionalShapDP.plot` returns the figure) — one shape, one return rule (R7).
  - Heterogeneity-function failure handling: (RH)ALE/Shap `print(...)`, PDP silent —
    use `warnings.warn` (R9).
- `eval` (base) deep-copies `kwargs_fitting` then refits a fresh global object per call —
  fine for correctness, but note: once `kwargs_fitting` is fixed (bug above), verify
  `centering` interplay for DerPDP (docstring itself warns `centering=True` is wrong for
  d-PDP, yet `eval`'s default is `True` for every method — tie the default to the class,
  R3).
- Preprocessing duplication: the whole axis-limits→subsample→names block in
  `RegionalEffectBase.__init__` is copy-pasted from `GlobalEffectBase.__init__` (plus
  feature types). Extract `helpers.prep_data(data, axis_limits, nof_instances,
  data_effect)` used by both (and by `FeatureEffect`, which is a third copy).

### 2.6 `axis_partitioning.py`

- **Fix the string registry** (R6): `return_default` accepts `"dp"`, but `RHALE.fit`
  asserts `["greedy", "dynamic", "fixed"]` — so the documented `"dp"` fails RHALE's
  assert, and the allowed `"dynamic"` crashes inside `return_default`. **DP is currently
  unreachable via string.** One alias table (`"dp"`), asserts derived from it.
- Naming leftovers: docstrings and error messages still say `effector.binning_methods.*`
  (the module's old name) — update to `axis_partitioning`.
- `Base.find` vs subclasses' `find_limits`: the base defines `find` (never overridden or
  called) — rename base to `find_limits` and make it the abstract method.
- `Base.__init__` docstring describes a completely different signature (feature/data/
  data_effect/axis_limits) — rewrite.
- `Fixed.find_limits` computes `self.limits = False` when `_none_valid_binning()` and then
  **unconditionally overwrites it** with a linspace two lines later (only the
  `min_points is not None` branch can restore `False`) — restructure into early returns
  like Greedy/DP.
- Type hygiene: `min_points_per_bin: int = 2.0` (DP) and `=0.0` (Fixed) are floats.
- Dead code: the commented `_is_categorical` blocks (3×) and unused `_cat_limit` locals —
  either implement categorical handling (it's a v0.3 idea anyway) or delete the comments.

### 2.7 `space_partitioning.py` + `tree.py`

- `Best` internally names itself `"cart"` (`super().__init__("Cart")`) while the public
  registry calls it `"best"` — pick one name (R6; also fixes the 2.5 assert mess).
- `Best` and `BestLevelWise` duplicate: the entire constructor + its long docstring, and
  ~80% of `_single_node_split` / `single_level_splits` (candidate-position generation,
  exhaustive scan, weighted-heterogeneity matrix, argmin unravel). Hoist a shared
  `_evaluate_splits(active_indices_list) -> split_dict` into `Base`; the two classes then
  differ only in recursion strategy (node-wise vs level-wise).
- `BestLevelWise._splits_to_tree` ends with `self.important_splits = None; self.splits =
  None` labeled "hack to check if … used after" — remove once tests pass.
- `compile()` accepts `candidate_conditioning_features="all"` handling inline; move that
  normalization to `helpers` alongside `prep_features`.
- `tree.py` is in good shape; minor: `get_node_by_idx` is a linear scan (fine at this
  size), `show_*` methods `print` directly — acceptable for a summary API, but consider
  returning strings so `summary()` output is testable.

### 2.8 `feature_effect.py` (facade)

- Reuse the shared registry (R5) instead of private `_REGISTRY`/`_ALIASES`/`_DISPLAY`.
- Reuse `helpers.prep_data` (2.5) instead of the third copy of the preprocessing block.
- Once base `eval` is uniform (2.2), the facade needs no per-method knowledge beyond the
  registry — it becomes ~80 lines. Its `plot`-only API can then grow `eval` for free
  (return a `{method: y}` dict), which fits the "use .eval() of any method to scale" goal.

### 2.9 Dead / vestigial modules

- `interaction.py` (293 lines): fully commented out and imports functions that no longer
  exist (`pdp_1d_vectorized`, ...). Delete from the package (git history keeps it); it
  ships dead weight and its ideas belong in V03_IDEAS.md.
- `utils_integrate.py`: only `mean_1d_linspace` is used (by `_compute_norm_const`).
  Move it into `utils.py` and delete the module (quad/expectation helpers have no callers).
- `tree.py` bottom: 50 lines of commented-out `DataTransformer` — delete.

### 2.10 `models.py` / `datasets.py` (low priority)

- `datasets.RealDatasetBase.split` shuffles with no seed → irreproducible train/test
  splits between runs; take a `seed` param like `generate_data` does.
- Typos/annotations: `standarize` → `standardize`, `np.array` → `np.ndarray` in
  annotations, `IndependentUniform.generate_data` does a pointless extra `shuffle`.
- `BikeSharing.postprocess` hardcodes magic un-normalization constants (8/47, 100, 67) —
  add a comment with the source (hour, humidity, windspeed scalings) or named constants.

---

## 3. Bugs found during the read (fix as part of the relevant submodule pass)

| # | Where | Bug | Submodule pass |
|---|---|---|---|
| B1 | `regional_effect_ale.py:242,448` | `"binnning_method"` typo (and `[:3]` locals slice) → `kwargs_fitting` always empty → regional (RH)ALE eval/plot ignore user's binning method | 2.5 |
| B2 | `global_effect_ale.py:569` + `axis_partitioning.py:432` | `"dp"` fails RHALE assert; `"dynamic"` passes assert then crashes in `return_default` → DP binning unreachable by string | 2.6 |
| B3 | `regional_effect.py:117` | assert allows `"cart"` (crashes later), rejects valid `"best_level_wise"` | 2.5 |
| B4 | `global_effect_shap.py:517` | `heterogeneity == "std" or True` always truthy | 2.3 |
| B5 | `visualization.py` (`plot_pdp_ice`) | `is_derivative` never wired → DerPDP + `scale_y` scales values as levels (adds mean); `std_err` is actually std | 2.4 |
| B6 | `axis_partitioning.py` (`Fixed.find_limits`) | `_none_valid_binning` result overwritten; can return limits when binning was deemed impossible | 2.6 |
| B7 | `global_effect.py` (`requires_refit`) | `norm_const is None` branch dead (sentinels stored instead of `None`) | 2.2 |
| B8 | naming/semantics | ShapDP `spline_std` holds variance; PDP eval returns variance while docstrings promise std | 2.2/2.3 |
| B9 | `global_effect_pdp.py:694` | vectorized numerical d-ICE marked `TODO: needs test, something is wrong` — decide: test it or route to non-vectorized | 2.3 |

---

## 4. Suggested application order

Each step is a standalone PR with tests green (`make test`); later steps depend on earlier ones.

1. **helpers/utils foundation** (2.1) — `prep_data`, `None` for `norm_const`, delete dead
   helpers, alias tables for binning/partitioner strings (fixes B2 groundwork).
2. **`global_effect.py` base** (2.2) — `_eval_unnorm` kernel, base `eval`, base `_fit_loop`,
   base `_compute_norm_const`, class-attr identity + defaults (fixes B7).
3. **Global methods** (2.3) — slim ALE/PDP/Shap down to kernels; align defaults
   (fixes B2, B4, B8, B9-decision).
4. **`visualization.py`** (2.4) — `_finalize_ax`, uniform returns, wire `is_derivative`
   (fixes B5).
5. **Regional family** (2.5) — template `fit`, explicit kwargs dicts, registry-based
   `_create_fe_object`, uniform plots (fixes B1, B3).
6. **Partitioning** (2.6, 2.7) — registries, `Fixed` control flow (fixes B6),
   Best/BestLevelWise dedup.
7. **Facade + cleanup** (2.8, 2.9, 2.10) — registry reuse, delete `interaction.py`,
   fold `utils_integrate`, datasets seed.

After step 7, the "core rules" in §1 should be copied into `CONTRIBUTING.md` (or a
`docs/design.md`) as the contract new features must follow.

---

## 5. Effort & payoff estimate

### Effort (applying submodule-by-submodule, incl. review + running the suite)

| step | scope | est. |
|---|---|---|
| 1 | helpers/utils foundation | ~1 h |
| 2 | `global_effect.py` base (`_eval_unnorm`, base `eval`/`_fit_loop`) | ~1.5 h (most delicate) |
| 3 | slim ALE/PDP/Shap, align defaults | ~1.5 h |
| 4 | visualization | ~1 h |
| 5 | regional family | ~1.5 h |
| 6 | axis + space partitioning | ~1 h |
| 7 | facade + dead-code cleanup | ~0.5 h |
| **total** | | **~8 h ≈ 1 focused working day** |

Combined with the test-suite work (see `TESTING_PLAN.md` §7, ~1 day), the realistic
total is **~2 focused working days**: tests first (day 1), refactor (day 2). The 5-min
test budget keeps each refactor step's feedback loop under a minute for the gate.

### Payoff — code reduction (package today: **7,157 LOC**)

| source | est. reduction |
|---|---|
| delete `interaction.py` (fully commented out) | −293 |
| fold `utils_integrate.py` into `utils` (keep ~15 live lines) | −85 |
| commented-out blocks (`tree.py` DataTransformer, `axis_partitioning` `_is_categorical` ×3) | −105 |
| dead helpers (`prep_dale_fit_params`, `prep_ale_fit_params`) | −25 |
| SHAP "code behind the scene" docstring ×4 → 1 | −115 |
| base-class hoisting (3 `eval` skeletons → 1, 2 fit loops, 2 norm-const impls) | −120 |
| regional template `fit` + registry `_create_fe_object` + explicit kwargs | −100 |
| `visualization` `_finalize_ax`/`_scale_xy` extraction | −90 |
| `Best`/`BestLevelWise` ctor + split-evaluation dedup | −110 |
| new shared code (registry, `prep_data`, raise-helpers) | **+100** |
| **net** | **≈ −950 LOC (−13%), 7.2k → ~6.2k** |

The deeper payoff is structural: after step 2, `eval` exists **once**; every future
feature (importance measures, method comparison, exports) is written once against the
base instead of 5×, and the bug classes found in §3 (`locals()` capture, drifting string
registries, per-method default drift) become impossible rather than merely fixed.

---

## 6. After testing + homogenization — strategic priorities (2026-07 strategic review)

The composable core (regional = partitioner + any global `.eval`) is the package's real
asset; these are the gaps that most threaten it, in priority order. They feed Phase 2 of
`ROADMAP.md` / `V03_IDEAS.md` and should come **before** further feature ideas.

1. **Reproducibility end-to-end** (small, high trust-impact). `nof_instances`
   subsampling uses global `np.random` with no seed; dataset splits are unseeded — two
   runs give two different explanations. Add `random_state`/`seed` to every constructor
   and `datasets.*`, thread it through `prep_nof_instances`, and add a contract test:
   two identical constructions → identical `eval` output. For an XAI package this is a
   core value, not a nice-to-have.
2. **Keep the contract enforced, not conventional.** The §1 rules go into
   `CONTRIBUTING.md` / `docs/design.md`, and the contract layer of `TESTING_PLAN.md`
   (§3.1) becomes the merge gate for any new method class: a method is "done" when it
   passes the parametrized contract suite. This is what prevents the §3 bug classes from
   re-accumulating.
3. **Heterogeneity semantics as the brand.** After R2 lands (variance internally, std at
   the plot layer), document the exact heterogeneity definition per method on one docs
   page and fill the `TODO` math placeholders in the ALE/RHALE docstrings — the flagship
   methods currently ship with empty definitions.
4. **Real-world tabular reach** (the adoption gate, sequenced after 1–3): categorical
   features as *feature of interest* (today categoricals only work as conditioning
   features), pandas/DataFrame ingestion (names/types inferred), and a classification
   story (`predict_proba` guidance or a thin wrapper). These are the first things a
   practitioner with a real dataset hits; heterogeneity-on-categoricals is also open
   research space the package is positioned to own.
5. **Regional UX: stop leaking internals.** Replace magic `node_idx` ints with
   addressable regions (`for region in reg.regions(feature): region.plot()`, or
   node names/conditions), and make `summary()` return a structured object with printing
   optional. Enabled cheaply by the R5 registry + tree refactor.
6. **Performance posture: decide, then state it.** Regional `eval`/`plot` refits a fresh
   global object per call; `copy.deepcopy(data)` sits inside the ICE kernels; there is no
   node-level caching. Decide the supported scale (e.g. "N ≤ 100k interactive"), add one
   benchmark script, and cache per-node fitted objects — before someone benchmarks
   effector against `shap` at 1M rows.
7. **2D / interaction effects.** The deleted `interaction.py` ideas (H-index, 2D
   PDP/ALE) return as *new* code written once against the unified base — only after
   1–3 are done.

### 6.4a Categorical-FOI spec (detail for item 4)

Decision (2026-07-02): categorical features as feature-of-interest do **not** break the
method symmetry — it survives at the level that matters: every supported method reduces
to *per-bin effect + per-bin variance with bins = levels*, so `eval(feature, levels,
heterogeneity=True)` keeps the one contract and the regional machinery (which only
consumes the heterogeneity callable) works unchanged.

Capability matrix (becomes two R5-registry fields: `supports_categorical_foi: bool` +
the cat strategy — code, not convention):

| method | categorical kernel | verdict |
|---|---|---|
| PDP | ICE evaluated at levels; h(k) = Var over instances per level | exact |
| ALE | local differences between *adjacent levels* (never bin edges — invalid category values) | exact for ordinal; induced order for nominal |
| RHALE | level differences (= discrete derivative) + Greedy/DP merging over the level order = **adaptive level grouping** | principled analog; keeps RHALE's identity (precedent: RHALE already falls back to numerical diff without `model_jac`) |
| ShapDP | per-level mean/variance of shap values (coalitions need no order/distance); step lookup instead of `interp1d` | exact — cleanest of all |
| DerPDP | — rejected with a clear error | already the asymmetric method (derivative units, excluded from `FeatureEffect`); its cat analog is derivable from PDP-cat bars |

Design points:

- **Nominal vs ordinal is the user's call**, surfaced as `order=`:
  `order=[...]` (declared, e.g. education) reduces to the ordinal case;
  `order="similarity"` (Molnar/iml: summed KS distance of other features across levels →
  seriate to 1D — adjacent differences meaningful, curve shape order-dependent, say so in
  docs); `order="effect"` (sort by PDP value — display only).
- **Centering maps cleanly**: `zero_integral` → frequency-weighted mean over levels = 0;
  `zero_start` → reference level = 0 (the natural categorical centering, as in dummy
  coding).
- **Regional-on-cat is the payoff**: heterogeneity per level → frequency-weighted scalar
  → the partitioner works as-is; flip `search_partitions_when_categorical` once defined.
  ("For which subgroups is the *weekday* effect stable?" — no competing package has it.)
- **Extras that fall out free**: automatic level grouping = existing Greedy merging over
  the level order; high cardinality = rare-level pooling into `"other"` via `cat_limit`.
- **Plots**: bars with heterogeneity whiskers; `heterogeneity="ice"` → jittered per-level
  dots; shap scatter already works per level.

Effort estimate (assumes homogenization + contract suite are done; tests are cheap —
small N, no SHAP beyond one tiny case):

| phase | scope | est. |
|---|---|---|
| 0 | `feature_types` in global classes + registry fields + DerPDP clear error | ~1 h |
| 1 | PDP-cat: eval at levels, centering, bar plot + tests (new closed-form cat model in `effector.models`) | ~2.5 h |
| 2 | heterogeneity per level + regional-on-cat (flip the flag, contract + GT tests) | ~2 h |
| 3 | ALE-cat ordinal: level-pair local-effect kernel, reuse `compute_ale_params` | ~3 h |
| 4 | RHALE-cat: transition diffs as `data_effect` + Greedy/DP level grouping | ~2 h |
| 5 | ShapDP-cat: level bins + step lookup | ~1.5 h |
| 6 | nominal `order=` ("similarity" seriation is most of the code) + docs honesty pass | ~3 h |
| **total** | | **~15 h ≈ 2 focused days** (phases 0–5; +½ day for nominal ordering) |

Sequencing: 0 → 1 → 2 (differentiator) → 3 → 4 → 5 → 6. Each phase lands with its own
closed-form ground-truth test per `TESTING_PLAN.md` §8 ("new ground truths over new
smoke tests").

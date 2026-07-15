# Changelog

## Unreleased

### Fixed

- **Display units honor the schema's `scale_y`**: every surface labeled "(target units)" — `effector.plot_triage`, `CALM.plot_triage`, and the report's importance/heterogeneity bars, triage plane, ranked tables, ledger heter columns, and `heter_threshold` chip — now bridges the model-unit scalars by `scale_y["std"]`, so a model trained on a standardized target reads in the target's own scale (matching the curve plots, which already rescaled). Unbound reports render the same as bound ones: `Report` stamps `scale_y`/`scale_x_list`, and the unbound curve fallback applies them. The engine verbs (`importance`, `heter_score`) and all stamped/serialized payloads stay in model-output units, so `find_regions` thresholds and old dicts are unaffected.

# [0.5.0] - 2026-07-14

### Breaking

- **`Regional*` classes removed** (`RegionalPDP`, `RegionalDerPDP`, `RegionalALE`, `RegionalRHALE`, `RegionalShapDP`) with no backward-compatibility shim. Regional questions are now asked on the **global** effect object via `find_regions(feature) -> Partition` (design contract R12: partitions are values, not stored state). Migration: `RegionalPDP(X, f).fit(0); r.summary(0); r.plot(0, i)` → `pdp = PDP(X, f); pdp.fit(0); part = pdp.find_regions(0); part.show(); part.plot(i)`. The old `space_partitioner=` fit kwarg becomes the `finder=` kwarg of `find_regions`; `r.eval(0, i, xs, heterogeneity=True)` (tuple) splits into `part.eval(i, xs)` + `part.eval_heter(i, xs)`.
- **Two-block lifecycle (R14)**: `fit` computes the frame-gated local effects — the *only* model touch — and every summary (centering included) is a memoized, model-free reduction on top. The `points_for_centering` fit kwarg and the `requires_refit` machinery are gone; changing `centering` at `eval`/`plot` time never recomputes local effects behind your back.
- **Scalar units contract**: `heter_score` and `importance` now return values in **output units** (RMS over the data), directly comparable to the target's scale and to each other. Slope-scale methods (RHALE, DerPDP) are bridged by the feature's std; nominal features use order-free all-pairs level differences. `ShapDP` no longer overrides `importance` with `mean(|φ|)` — it inherits the shared default. Any thresholds tuned against the old variance-flavored numbers must be recalibrated.
- `Report.to_html(path)` writes the file and returns `None` (it used to return the multi-megabyte HTML string, echoing it in interactive shells).

### Added

- `GlobalEffectBase.find_regions(feature, *, finder="best", candidate_conditioning_features="all") -> Partition` on every global class — a model-free heterogeneity split search (every candidate scored by `heter_score(feature, mask)`, re-summarized from the cached local effects). Returns a `Partition` value object (`effector.Partition` / `effector.Region`) with `show`/`eval`/`eval_heter`/`plot`/`to_dict`; nothing is stored on the effect.
- A **finder protocol** on `space_partitioning` (`Base.find_regions`): a finder consumes only `(score_fn: mask->float, data, metadata, its own config)` and returns a `Partition`, so new finders plug in with no changes elsewhere. The min-points / degeneracy (`BIG_M`) guard lives in the finder, never in the effect.
- **Named split proposers** on the built-in finders: `categorical_proposer=` (`"one_vs_rest"` default | `"subsets"` | `"ordered"` — natural order for ordinal levels, similarity seriation for nominal, or an explicit order | `"multiway"`) and `continuous_proposer=` (`"threshold"` default | `"quantiles"` — one k-way candidate per child count at the marginal quantiles), both also accepting a proposer instance (`effector.proposers`). Defaults unchanged; the finder's `proposer_factory` attribute remains the raw extension seam.
- An invisible bounded-LRU **masked-summary memo** on the effect, keyed by `(feature, fit_epoch, mask)`, so repeated masked calls (the split search re-proposing candidate masks, a plot after a search) skip re-summarization. It is semantically transparent — a refit bumps the epoch and invalidates it; it never changes an answer, only its latency.
- **`effector.explain(data, model, ...) -> Report`** — a one-call pipeline: fit once, rank features by `importance` (R13), draw the mean-effect + heterogeneity curves for the top-k, and `find_regions` (R12) on the heterogeneous ones. Returns a serializable `Report` value (`effector.Report`) with `show()` (importance-ranked table + region trees), `plot_importance()`, `to_html()` (a self-contained page — every figure inlined as a base64 PNG, no external assets), and `to_dict()`/`from_dict()` (round-trips without an effect). All model calls happen through the single `fit`; the rest is model-free, so the cost does not grow with `top_k`.
- **`importance(feature, mask=None) -> float`** and **`importances(mask=None) -> (D,)`** on every global class (design contract R13) — the μ-twin of `heter_score`: the dispersion of the *mean* effect, evaluated the same way (continuous → std over the (masked) data values; discrete → frequency-weighted std over levels). PDP, ALE, RHALE and ShapDP all use that one default; DerPDP is the sole override (bridged `mean(|derivative|)`, since its mean effect is already a derivative). Model-free, centering-invariant (no `centering` kwarg), masked variant free through the memo. `importances` returns `NaN` (with one `UserWarning`) for feature types a method cannot explain. effector never sees `y`, so loss/permutation importance is out of scope by construction.
- **`effector.adapters`** — model wrappers that make the numpy-only final pass explicit: `from_sklearn(estimator)`, `classifier_proba(predict_proba, class_)`, `from_torch(module)`, and `check(model, X)` (the two-row handshake probe). An adapter *returns* a plain callable; it never installs one behind your back.
- **Feature names as strings everywhere**: every verb that takes a feature (`plot`, `eval`, `importance`, `heter_score`, `find_regions`, ...) accepts the schema name (`"hr"`) as well as the index.
- **Plural `find_regions(features=...)`**: a list, `"all"`, or `"heterogeneous"` (above-median `heter_score`) returns `{name: Partition}` — the input `select_regions` and `plot_triage` eat.
- **`select_regions(partitions=None, ..., min_r2_gain=0.01) -> CalmSequence`** — greedy explained-variance selection across features: starting from the GAM (every feature global), each round applies the split with the largest *sequential* R² gain and stops when no remaining split adds `min_r2_gain` of `Var(f̂)`. Returns the chain `[GAM, calm1, ...]` of **CALM** snapshots (`effector.CALM` / `effector.CalmSequence` — serializable value objects with stamped per-feature `importances`/`heter_scores`, `plot_triage`, `bind(effect)`); rejected splits land in `.skipped` with a reason (`redundant` / `below_threshold`). `CalmSequence.show()` prints the same EXPLAINED VARIANCE / REJECTED SPLITS tables as the report. Derivative-scale methods (`DerPDP`) raise: summing their curves does not approximate `f̂`.
- **Explained variance in the report**: `explain` opens with the headline — the share of the model's variance the GAM (global curves only) and the final CALM (with the accepted splits) reproduce — and the report carries the full decision sequence as a ledger (table + bar): sequential ΔR² per accepted split, a counterfactual `solo` column per rejected one. Features whose split was rejected keep their global read; the regional plots are demoted with the reason.
- **`effector.rules`** — the predicate algebra under regions: `Interval`, `LevelSet`, `Rule` (a normalized conjunction of per-feature conditions) with `contains`/`intersect`/`refine`/`format` and a string parser that resolves schema level names (`rule="workingday == no"`). `Partition`/`Region` are rule-primary (v2 serialization: rules + stamped stats, never masks); masked verbs accept `rule=` sugar wherever `mask=` is accepted.
- **`effector.compare(effect_a, effect_b, feature=...)`** — overlay fitted engines on one feature's axis — and **`effector.plot_triage(effect, partitions=None)`** — the importance × heterogeneity plane, with before/after arrows when partitions are passed.
- **`grid(feature)`** and **`.explain(...)`** on every engine: the model-free evaluation grid, and the one-liner without leaving a constructed session.
- **Categorical capability matrix completed**: `DerPDP` handles ordinal/nominal features via ICE level differences (transition bars, no jacobian on discrete axes); `RHALE` on a nominal feature falls back to ALE instead of raising — no method/type holes remain.
- **Dataset loaders** `effector.datasets.MedicalCosts`, `AirfoilSelfNoise`, `AdultIncome` (+ the corresponding real-example notebooks).

### Changed

- **Plot redesign across every figure**: titles carry the identity (feature · method · global/regional), rules read `where (workingday = no) and (temp < 6.86)` and resolve schema level names, categorical plots draw at the labeled levels with per-level counts, one tick/legend discipline, semantic theme tokens, ALE panel redesign, triage labels repel. `report.to_html()` reads like the pipeline: triage overview first, per-feature global + regional, before/after arrows, shared y across the page and shared x per feature.
- **`Report.show()` prints tables** — DATA & MODEL (in `scale_y` units, matching the figures), the EXPLAINED VARIANCE ledger with the `solo` column, REJECTED SPLITS with reasons, and the ranked FEATURES table (`#regions` = accepted leaves) — followed by the accepted partition trees. Unicode box-drawing by default, `show(ascii=True)` for terminals that mangle it. R² and every ΔR² print as `%` — one unit everywhere (read `+18.3%` as an absolute move on the 0–100% R² scale).
- **ShapDP payloads are pure numpy** (binned means replace `scipy.interpolate` splines on the eval path) and the global-effect base dispatches continuous/categorical kernels in one place — no output change, one seam for new methods.
- internal (no output change): tightened the compute/draw boundary (R1). `visualization.plot_pdp_ice` no longer computes the mean line or std/std-err band from the raw ICE table — the PDP method hands it the pre-computed curve + band (the band is now definitionally `sqrt(eval_heter)`, the single source of heterogeneity). PDP's per-instance `norm_const` reduction moved from a `np.ndim` branch in the base `eval` to an overridable `_mean_norm_const` hook. Centering-grid resolution now flows from the single `helpers.NOF_INTERNAL_POINTS` knob instead of a hard-coded `30`.

### Notes

- `Partition.plot(idx)` inherits each global class's own `plot` defaults (e.g. PDP still defaults `heterogeneity="ice"`), so there is no visual change versus the old per-method regional plot wrappers.

### Fixed

- global effects: a centering-triggered auto-refit (e.g. `fit(centering=False, order=[...])` then `eval`/`plot` with centering) no longer discards the method-specific `fit` kwargs — `order`/`binning_method` (and `use_vectorized`) are replayed from the original `fit`, overriding only `centering`, instead of silently falling back to the defaults
- the split search no longer overflows on a high-cardinality categorical conditioning feature (the subset enumeration is capped)
- the numerical jacobian tolerates models that return `(N, 1)` column vectors instead of `(N,)`
- `Report.to_html` renders a note instead of crashing when an accepted rule pins the feature of interest to a single value inside a leaf (no axis to draw a curve over)
- `Report.show`/`to_html`: the DATA & MODEL header now speaks the figures' `scale_y` units (it printed raw model-output units); the `#regions` column counts a partition's **leaves** (it counted tree nodes, disagreeing with the regional section); the HTML rejected-split notes say `%` like every other surface (two `-pt` leftovers)

# [0.4.0] - 2026-07-06

### Breaking

- one `schema=` argument replaces the metadata kwargs on every public constructor: `feature_names=`, `target_name=` (all classes) and `feature_types=`, `cat_limit=` (regional classes) are removed; pass `schema={"feature_names": ..., "feature_types": ..., "cat_limit": ..., "target_name": ..., "scale_x_list": ..., "scale_y": ...}` (or an `effector.Schema`) instead — every field optional, explicit fields win over inference
- SHAP configuration moved to the constructor: `budget`, `shap_explainer_kwargs`, `shap_explanation_kwargs` are no longer `fit` kwargs on `ShapDP`/`RegionalShapDP` (and `RegionalShapDP` gains `shap_values=` for parity); `fit` keeps only the analysis args
- all `fit` signatures are keyword-only after `features`; `points_for_mean_heterogeneity` removed from regional `fit` (internal, shares one grid constant with `points_for_centering`)
- `DerPDP.plot`/`RegionalDerPDP.plot`: `dy_limits` renamed to `y_limits` (its only axis); ALE/RHALE plots dropped `nof_points` (the curve is drawn exactly at the bin limits)
- `utils.get_feature_types` deprecated (delegates to `effector.ingestion.infer_feature_types`) and its vocabulary changed to the three-way taxonomy

### Added

- **effector is numpy-only** (R10): `data` must be a 2-D numeric numpy array and `model`/`model_jac` are `numpy → numpy` callables, called exactly as given (never wrapped) — a DataFrame passed to a constructor is rejected with a pointer to `from_dataframe`. `effector.from_dataframe(df) -> (X, schema)` is the opt-in convenience that reads a DataFrame's names/dtypes/category labels into a numpy matrix + populated `Schema` (it never touches the model); pandas stays an optional dependency, never imported on the numpy path
- three-way feature taxonomy `continuous`/`ordinal`/`nominal` (aliases `cont`/`cat`), stored on every class as `feature_types` + `feature_metadata`; heuristic type inferences (low-cardinality int) emit a `UserWarning` nudging an explicit declaration
- **categorical features as feature of interest** (docs/method_semantics.md is the exactness contract): PDP/ICE evaluate only at the observed levels (bars + jittered ICE dots); ALE accumulates adjacent-level differences (exact for ordinal; nominal defaults to the encoded order with a documented caveat, or `order=[...]` / `order="similarity"` KS-seriation on `fit`); RHALE does discrete derivatives + Greedy/DP adaptive level grouping (ordinal only); ShapDP does per-level mean/variance with a step lookup; DerPDP and RHALE-on-nominal raise clear errors; centering and `heter_score` become frequency-weighted over levels
- **regional effects on categorical features**: `search_partitions_when_categorical` now defaults to `True`, the heterogeneity of a categorical feature of interest is the frequency-weighted per-level variance, and every regional method finds subgroups for per-level effects (see `notebooks/synthetic-examples/08_categorical_features.ipynb`)
- `scale_x_list`/`scale_y` accepted at construction (schema) as plot defaults; plot-time dicts override, `False` disables
- `models.ConditionalCategorical` closed-form ground-truth model; `effector.ordering.similarity_order` (scipy-only Molnar/iml seriation)
- input contract spec: `docs/design.md` R10 (accepted data types, `Schema` metadata argument, three-way feature taxonomy, define-or-infer, model-call rule, scaling precedence) and `docs/method_semantics.md` (the exact `eval`/`eval_heter`/`heter_score`/`plot` formulas per method and feature type)
- `category_names` schema field: per-feature human-readable level labels shown on categorical plot axes instead of the numeric codes; resolved to a value-keyed map at ingest, so regional nodes that restrict a categorical feature to a subset of its levels still label correctly
- `FeatureEffect` facade on a categorical feature of interest drops the methods its type doesn't support (e.g. `RHALE` on a nominal) with a `UserWarning`, overlays the rest at the observed levels, and raises only when nothing is left

### Changed

- unified defaults: `nof_instances` is 10,000 everywhere except SHAP-based classes (1,000) — `RegionalALE`/`RegionalRHALE` were 100,000 and `FeatureEffect` 1,000; `nof_ice`/`nof_shap_values` default to 100; plot grids default to 100 points; one `heterogeneity` vocabulary on all plots (`False | "std" | method-native`, `True` ≡ `"std"`)
- type inference now runs on the full data (before `nof_instances` subsampling); regional node objects and facade sub-methods inherit the parent's resolved metadata instead of re-inferring from subsets
- `space_partitioning.compile`: `categorical_limit` renamed to `cat_limit`
- `PDP`/`RegionalPDP` default centering is now `zero_integral` (was `False`), so `eval(centering=None)` and the global/regional plots center consistently with `ALE`/`ShapDP`
- the method / feature-type capability matrix (e.g. `RHALE` and `DerPDP` on nominal) is enforced at regional `fit` time, not only later at `plot`, so `fit`/`summary`/`plot` stay consistent for an unsupported feature of interest

### Fixed

- `ALE`/`RHALE` `.plot()` crashed on a categorical feature whose level codes are not `0..K-1` (e.g. ordinal hours `1..24`): the plot grid was built from positional codes `0..K-1` and rejected by `eval`; it now draws at the observed level values
- `tree` display no longer crashes on per-feature `None` entries in `scale_x_list`
- `helpers.indices_within_limits` raises `ValueError` instead of a bare `assert` when `axis_limits` exclude every point
- removed the phantom `avg_output` constructor docstring on `ShapDP` and the stale `ice_non_vectorized` docstring example
- `helpers.prep_data`: the subsample `indices` (stored as `self.indices` on every effect object) are now original-relative — they index `data` as passed in even when the `axis_limits` filter dropped rows (`data_in[indices] == data_out`), giving a stable handle back to the user's rows

# [0.3.0] - 2026-07-04

### Added

- `random_state` on every public effect-class constructor (`PDP`, `DerPDP`, `ALE`, `RHALE`, `ShapDP`, the 5 regional variants, and `FeatureEffect`); default `21`, so two identical constructions give identical `eval`/`fit`/`plot` output out of the box, `None` opts into fresh randomness. The seed also drives plot-time ICE/SHAP-scatter subsampling and is passed to the shap/shapiq explainer (`seed=`/`random_state=`) unless overridden via `shap_explainer_kwargs`. No effector code touches the global `np.random` state anymore.

# [0.2.1] - 2026-07-01

### Changed

- modernized GitHub Actions CI: replaced `black`/`flake8`/`isort` with `ruff`, added a Python 3.10-3.13 test matrix, gated PyPI publishing behind a main-ancestry check + test run + wheel smoke test, added a docs build-check on PRs, pinned workflow permissions, added concurrency cancellation, and added `dependabot.yml`
- PyPI releases now auto-create a matching GitHub Release with notes from this changelog

### Fixed

- fixed a latent `NameError` in `RegionalEffectBase._fit_feature`'s string-based `space_partitioner` fallback (`effector/regional_effect.py`)
- fixed `prep_dale_fit_params` validating `max_nof_bins` twice instead of `min_points_per_bin` (`effector/helpers.py`)

# [0.2.0] - 2025-07-21

### Changed

- updated the guide that measures runtimes of regional methods
- moved `shap`/`shapiq` to an optional `effector[shap]` extra, so the core install stays lightweight (`numpy`, `scipy`, `matplotlib`, `tqdm`); `ShapDP`/`RegionalShapDP` now raise a clear `ImportError` pointing to `pip install effector[shap]` if used without it
- moved `ucimlrepo` to the `tutorials` extra

### Removed

- dropped the unused `overrides` dependency


# [0.1.12] - 2025-07-13

### Fixed

- fixed bug in `project.toml`, added dependency to `overrides`

## [0.1.11] - 2025-07-10

### Added

- added codecov to the project

### Fixed

- fixed all notebooks to work with the latest version of Effector
- fixed bug in `RegionalPDP` plots, where `centering` was not working properly, it was set to `False` and the user defined argument was not being passed.

## [0.1.10] - 2025-06-18

### Fixed

- bug in `project.toml`

## [0.1.9] - 2025-06-18

### Fixed

- bug in github actions

## [0.1.8] - 2025-06-18

### Added

- NO2 real example

### Fix 

- fixed documentation errors, fixed node indexes

### Changed

- added dependecies to `requirement-dev.txt`

## [0.1.7] - 2025-05-08

### Added

- added organizations the that support the project

## [0.1.6] - 2025-04-14

### Fix

- fixed bug in `project.toml`, added dependency to `shapiq`

### Changed

- REAME.md and index.md: added refereces to effector

## [0.1.5] - 2025-03-24

### Added 

- added `ucimlrepo` to the dependencies
- `space_partitioning` module with `Best` and `BestDepthWise` classes for space partitioning.
- `space_partitioning.Best` (default) is a node-wise partitioning, i.e., it splits each node based on the split that maximizes the heterogeneity drop.
- `space_partitioning.BestDepthWise` is a depth-wise partitioning, i.e., all nodes of a certain level are split based on the same condition.

### Fixed

- fixed bug in `space_partitioning.Best()`; partitioning now checks that the absolute heterogeneity should be over a threshold to be considered a valuable split
- fixed bug in RegionalPDP plots, where `centering` was not working properly, it was set to `False` and the user defined argument was not being passed.

### Changed

- default plot titles now display full method name, e.g., `Accumulated Local Effects` instead of `ALE`.
- added support for shapiq backend in `shap_dp` and `shap_regional_dp` (added as alternative to `shap`)
- set the default value of `heter_small_enough` to 0.001 (from 0.00)
- set the default value of `centering` to `True` for `RegionalPDP` plots (from `False`)
- set the default value of `centering` to `False` for `RegionalDerPDP` plots (from `True`)


## [0.1.4] - 2025-02-26

### Changed

- default plot titles now display full method name, e.g., `Accumulated Local Effects` instead of `ALE`.
- update all notebooks with new names

## [0.1.3] - 2025-02-25

### Changed

- shap_dp (both global and regional) can now take custom arguments for the SHAP explainer

## [0.1.2] - 2025-02-22

### Changed

- all plots return a `fig, ax` tuple, if the user wants to modify the plot further.
- default plot titles now display full method name, e.g., `Accumulated Local Effects` instead of `ALE`.
- changed README.md to reflect the new changes.

### Added 

- license
- documentation for space partitioning methods

## [0.1.1] - 2025-02-17

### Changed

- Updated GitHub Actions workflows:

  - Modified `pulish_to_pypi.yml` to be triggered only on major or minor version changes.
  - Modified `publish_documenation.yml` to be triggered only on major or minor version changes.

- add changelog.md to the documentation

## [0.1.0] - 2025-02-17

### Added

- Initialized changelog file and added basic versioning structure.

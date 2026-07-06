# Changelog

# [Unreleased]

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

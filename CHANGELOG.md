# Changelog

# [Unreleased]

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

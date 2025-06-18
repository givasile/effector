# Changelog

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
>>>>>>> main

### Changed

- all plots return a `fig, ax` tuple, if the user wants to modify the plot further.
<<<<<<< HEAD
<<<<<<< HEAD
=======
- default plot titles now display full method name, e.g., `Accumulated Local Effects` instead of `ALE`.
>>>>>>> main
=======
>>>>>>> main
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

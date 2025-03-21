# Changelog

# Fix

- fixed bug in `space_partitioning.Best()`; partitioning now checks that the absolute heterogeneity should be over a threshold to be considered a valuable split
- fixed bug in RegionalPDP plots, where `centering` was not working properly, it was set to `False` and the user defined argument was not being passed.

# Changed

- set the default value of `heter_small_enough` to 0.001 (from 0.00)


<<<<<<< HEAD
<<<<<<< HEAD
## [Unreleased]

### Changed

- moved to shapiq library instead of shap
- default plot titles now display full method name, e.g., `Accumulated Local Effects` instead of `ALE`.

=======
>>>>>>> main
## [0.1.2] - 2025-02-17
=======
## [0.1.5] - 2025-02-267

### Changed

- added `ucimlrepo` to the dependencies

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

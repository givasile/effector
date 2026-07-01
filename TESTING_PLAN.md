# Testing Plan — a full safety net before the homogenization refactor

Goal: before touching the codebase per `HOMOGENIZATION_PLAN.md`, make the test suite
strong enough that any behavioral break in `fit / eval / plot` across all 11 method
classes is caught. Constraint: **total runtime ≤ 5 min** (no real-world models/datasets
in the gate).

Measured baseline (2026-07, local, uv + pytest):

| set | tests | time |
|---|---|---|
| `make test` (`-m "not slow"`) | 18 | **~28 s** |
| `-m "slow"` (linear, gam, regional) | 3 | **~117 s** |
| notebook execution | 0 collected (see P2) | — |
| **total today** | 21 | **~2.5 min** |

So there is ~2.5 min of headroom for new tests — plenty, if SHAP usage is kept tiny.

---

## 1. What exists today (inventory)

| file | kind | covers | verdict |
|---|---|---|---|
| `test_functional.py::TestExample2` | functional | RHALE vs closed-form ALE of a linear model, Fixed + DP binning | good, keep |
| `test_functional.py::TestBinEstimation` | unit | Greedy/DP bin limits vs known piecewise-linear model, `min_points` edge cases incl. `limits is False` | good, keep (move to axis-partitioning tests) |
| `test_functional_linear.py` (slow) | functional | all 5 global methods + both SHAP backends, `eval` on linear model | **broken: `np.allclose` without `assert` — asserts nothing** |
| `test_functional_gam.py` (slow) | functional | same, on a GAM | **broken: same no-op asserts** |
| `test_regional_methods.py` (slow) | functional | all 5 regional methods + shapiq, `eval` on node 3 | **broken: same no-op asserts**; also hardcodes `node_idx=3` with no check the tree actually has 4 nodes |
| `test_plots.py` | smoke | `.plot(show_plot=False)` for 5 global methods, with/without scale_x/scale_y | crash-only; never inspects the figure; 15 s (ShapDP with N=1000 dominates) |
| `test_feature_effect.py` | contract | facade: fig/ax return, caching, aliases, warnings | good model to imitate |
| `test_space_partitioning.py` | unit | `Best` + `BestLevelWise` on a Gini-impurity toy; heterogeneity decreases per level | good; doesn't check *where* it splits |
| `test_tree.py` | smoke | `show_full_tree` / `show_level_stats` print without crashing | no value asserts |
| `test_unit.py` | unit | `ice_vectorized` (values, jac, finite-diff) | good; non-vectorized path untested |
| `test_models.py`, `test_datasets.py` | smoke | shapes only; **BikeSharing downloads from UCI (network!, ~7 s)** | trim/mark |
| `notebook_execution.py` | e2e | executes synthetic notebooks | **never collected** (filename doesn't match `test_*.py`, paths relative to `tests/`) |

Ground truths already derived in `notebooks/synthetic-examples/` and asserted inside
notebook cells (`np.testing.assert_allclose`, atol 1e-1/2e-1/1e-2):

- **05_…_global**: `ConditionalInteraction` — closed-form centered PDP, ALE, RHALE per feature.
- **05_…_heter**: same model — closed-form PDP heterogeneity + ALE `bin_variance`.
- **06_…_global**: `GeneralInteraction` — PDP/ALE/RHALE.
- **07_…_global**: `ConditionalInteraction4Regions` — PDP/ALE/RHALE (4 features).
- **05_…_regional**: WIP (3 cells, no asserts yet) — but the regional ground truth is known:
  splitting on $x_2 = 0$ makes the $x_1$ effect $\pm x_1^2$ with ~zero heterogeneity per region.

**These notebook asserts are the best functional tests in the repo and they never run in CI.**

## 2. Problems to fix first (P-items)

- **P1 — no-op assertions.** Replace every bare `np.allclose(...)` with
  `np.testing.assert_allclose(...)` in `test_functional_linear.py`,
  `test_functional_gam.py`, `test_regional_methods.py`. Then actually run them: they may
  fail today and reveal the real tolerances. This is step zero — the three broadest tests
  currently prove nothing.
- **P2 — notebook tests uncollected.** Rename to `test_notebooks.py`, resolve paths from
  `__file__`, mark the whole module `@pytest.mark.slow`. Keep out of the 5-min gate
  (each notebook takes ~½–2 min); the correctness content gets *ported* into pytest (§3.3)
  so the gate doesn't need them.
- **P3 — network in the gate.** `test_datasets.py` fetches UCI BikeSharing (~7 s, flaky
  offline). Mark `slow` (or skip when offline); keep `IndependentUniform` in the gate.
- **P4 — global seeding.** `test_functional.py` seeds at import; several tests rely on
  cross-test seed state. Seed inside each test (or an autouse `rng` fixture).
- **P5 — stale doctests.** `utils.compute_ale_params` and `ice_*` docstring examples
  reference undefined names / old signatures. Either fix and enable
  `--doctest-modules` for `utils.py`, or fix and leave unrun — but don't leave wrong
  examples that look like tests.

## 3. New tests to add

Organize as three layers (flat files with prefixes is fine; directories optional):

```
tests/
  conftest.py                        # shared fixtures (NEW)
  test_unit_*.py                     # one module in isolation
  test_contract_*.py                 # API-shape rules, parametrized over ALL methods (NEW layer)
  test_functional_*.py               # synthetic ground truths
  test_notebooks.py                  # slow, execution-only
```

`conftest.py` fixtures — kills the copy-pasted linear/GAM models in 6 files:
- `rng` (seeded `np.random.default_rng`),
- datasets: `uniform_2d`, `uniform_3d` (via `effector.datasets.IndependentUniform`),
- models: `linear` (f, jac), `gam` (f, jac), and the four `effector.models.*` classes,
- registries: `GLOBAL_METHODS = [ALE, RHALE, PDP, DerPDP, ShapDP]`,
  `REGIONAL_METHODS = [...]` with per-class constructor kwargs — the test-side mirror of
  the method registry (R5); parametrize with `pytest.param(..., id="ale")`.

### 3.1 Contract layer (NEW — this is what protects the refactor)

`test_contract_global.py` — parametrized over all 5 global classes, tiny data
(N=200, D=3, linear model; ShapDP with N=50, `budget=128`). For each method assert:

- **C1 return shapes/types**: `eval(0, xs)` → `(T,)` ndarray; `eval(0, xs,
  heterogeneity=True)` → tuple of two `(T,)` arrays; heterogeneity values ≥ 0.
- **C2 centering semantics**: `zero_integral` → `mean(eval(xs~linspace)) ≈ 0`;
  `zero_start` → `eval(xs[0]) ≈ 0`; `centering=True` ≡ `"zero_integral"` (identical output);
  `centering=False` differs from centered by a constant shift.
- **C3 lazy fit + refit**: `eval` works without explicit `fit`; after
  `fit(features=0, centering="zero_start")`, calling `eval(0, xs,
  centering="zero_integral")` returns correctly re-centered values (exercises
  `requires_refit`, catches B7); calling `eval` twice gives identical results.
- **C4 `fit(features=...)` variants**: int, list, `"all"` equivalent for the shared feature.
- **C5 plot contract**: `plot(0, show_plot=False)` returns `(fig, axes)`;
  `show_plot=True` path smoke-tested once with `matplotlib.use("Agg")` + `plt.close("all")`
  (autouse fixture). *(Today ShapDP/ALE differ — write as the target contract, xfail
  per-method where it fails, see §4.)*
- **C6 constructor equivalence**: kwargs vs positional produce same eval (guards the R8
  keyword-only migration); `nof_instances="all"` vs int subsampling shapes; manual
  `axis_limits` filters points outside.
- **C7 heterogeneity=True default-agreement**: return arity is the same for every method
  (catches ShapDP's deviant `heterogeneity=True` default when it changes).

`test_contract_regional.py` — parametrized over the 5 regional classes, fixed
tiny dataset with one obvious split (the `5*x1*1{x2>0, x3==0}` model already in
`test_regional_methods.py`, N=500):

- **RC1**: `fit(0)` then `summary(features=0)` runs; the fitted `tree["feature_0"]`
  exists and node 0 is the root with `weight == 1.0`.
- **RC2**: `eval(0, node_idx, xs)` works for every node index in the tree and returns
  the C1 shapes; invalid `node_idx` raises.
- **RC3**: `plot(feature=0, node_idx=k, show_plot=False)`-equivalent runs for every
  method (today regional plots always show; write against target contract R7, xfail).
- **RC4 — fit-kwargs propagation** (catches **B1**): fit `RegionalRHALE` with
  `binning_method=Fixed(nof_bins=7)`; assert the fe-object used by `eval` was fitted
  with 7 bins (e.g. via the returned curve's number of linear segments, or after the
  refactor via `kwargs_fitting`). Written as `xfail(strict=True)` today.
- **RC5 — string args** (catches **B3**): `space_partitioner="best"` works;
  `"best_level_wise"` works (xfail today); junk raises `ValueError`.

`test_contract_registries.py`:

- binning strings per method: ALE accepts only `"fixed"`; RHALE accepts `"fixed"`,
  `"greedy"`, `"dp"` (**xfail today — B2**); unknown strings raise; instances pass through.
- `axis_partitioning.return_default` / `space_partitioning.return_default` round-trips.
- centering strings: `prep_centering` accepts `{False, True, "zero_integral",
  "zero_start"}`, rejects `"zero_mean"`.
- `helpers.prep_features` rejects out-of-range indices (spec — today it doesn't).

### 3.2 Unit layer (protects HOMOGENIZATION step 1 & 6)

`test_unit_utils.py` — the numerical kernels the whole package sits on (all µs-fast):
- `compute_bin_effect`, `compute_bin_variance` (incl. NaN for empty/single-point bins),
- `fill_nans` (interior/edge interpolation; all-NaN raises `AllBinsHaveAtMostOnePointError`),
- `apply_bin_value`, `compute_accumulated_effect` (the docstring examples, as real asserts,
  incl. `square=True`),
- `compute_ale_params` end-to-end on a hand-computable 6-point example,
- `compute_local_effects` on a linear model (exact),
- `compute_jacobian_numerically` vs analytic jac (linear + quadratic),
- `get_feature_types` cat/cont boundary at `cat_limit`.

`test_unit_helpers.py` — `prep_features`, `prep_centering`, `prep_confidence_interval`,
`prep_nof_instances` (int < N, int > N, `"all"`), `axis_limits_from_data`,
`indices_within_limits`, `camel_to_snake`.

`test_unit_axis_partitioning.py` — move `TestBinEstimation` here; add:
- `Fixed`: limits are the exact linspace; `min_points` violation → `False`
  (**xfail today — B6**); single-unique-value data → `False`.
- Greedy/DP on constant-effect data → merges to few bins (behavioral pin).

`test_unit_space_partitioning.py` — keep existing; add:
- *split-position correctness*: heterogeneity function with known optimum → assert the
  root split is on feature 1 near 3.0 (the existing toy already makes this checkable),
- categorical conditioning feature → `==`/`!=` split,
- no split accepted when `min_heterogeneity_decrease_pcg` is huge → tree has only root,
- `Best` vs `BestLevelWise` find the same first split on a clean toy.

`test_unit_tree.py` — replace prints with asserts: node `weight`/`nof_instances`
computed from `active_indices`; `create_node_name`/`set_display_name` exact strings
(with and without `scale_x_list`); `get_level_stats` value; missing-key `KeyError`.

`test_unit_pdp_kernels.py` — extend `test_unit.py`: `ice_non_vectorized` ≡
`ice_vectorized` on random data (values and d-ICE, jac and finite-diff paths) — settles
the `TODO: needs test, something is wrong` (B9) one way or the other.

### 3.3 Functional layer (port the notebook ground truths)

One file per synthetic model, using `effector.models.*` + `IndependentUniform`
(N=1_000, `Fixed(nof_bins=31)` where the notebooks do — each runs in ~1 s):

- `test_functional_conditional_interaction.py` (from **05_global + 05_heter**):
  - centered PDP/ALE/RHALE vs closed-form for features 0,1,2 (atol 1e-1, masking the
    ALE jump-bin exactly as the notebook does),
  - PDP heterogeneity vs closed form; ALE `bin_variance` vs closed form,
  - **regional** (from the WIP 05_regional): `RegionalPDP/ALE/RHALE.fit(0)` → root split
    is on feature 1 at ≈0.0; per-region effect ≈ ±x² (centered) with heterogeneity ≈ 0.
    This single test is the strongest guard for the whole regional refactor (steps 2.5–2.7).
- `test_functional_general_interaction.py` (from **06**): PDP/ALE/RHALE vs closed form.
- `test_functional_4_regions.py` (from **07**): PDP/ALE/RHALE, 4 features; also assert
  Regional finds *two* split levels (x2 then x3) — 4 leaf regions.
- Fix + keep `test_functional_linear.py` / `test_functional_gam.py` (P1). Split each
  into parametrized per-method tests so a failure names the method. Move the non-SHAP
  cases out of `slow` (they're fast); keep `shap`/`shapiq` cases slow **and** add one
  tiny fast SHAP case to the gate: N=50, D=2, `budget=128` (a few seconds) so the gate
  is not SHAP-blind.
- `test_regional_methods.py`: fix P1; derive `node_idx` from the fitted tree instead of
  hardcoding 3; reduce ShapDP N (100 → 50) to cut its 36 s.

### 3.4 Plot-content layer (protects steps 2.3–2.4)

`test_contract_plots.py` — beyond crash-testing, inspect the returned `(fig, ax)`:
- the mean-effect `Line2D` y-data ≈ `method.eval(feature, xdata, centering=...)`
  (ties plot to eval — R1),
- with `scale_x`/`scale_y`: line data equals affine-transformed eval output — for a
  *derivative* plot (DerPDP) the y-data must be scaled by `std` only, not shifted
  (**xfail today — B5**),
- `y_limits` respected; number of ICE lines == `nof_ice`; legend labels stable,
- trim runtime: current 15 s → N=200 everywhere, ShapDP via precomputed `shap_values`
  passed to the constructor (0 s SHAP), `plt.close("all")` autouse.

### 3.5 Facade & data

- `test_feature_effect.py`: keep; add one numeric check — each overlaid curve equals the
  corresponding method's own centered `eval` on the same grid.
- `test_models.py`: add jacobian-vs-finite-difference check per model (cheap, exact-ish).
- `test_datasets.py`: split BikeSharing into `slow`; add seeded-split reproducibility
  test once datasets get a `seed` (HOMOGENIZATION 2.10).

## 4. Policy for the known bugs (B1–B9)

Write **specification tests** for the behavior the refactor will establish and mark them
`@pytest.mark.xfail(strict=True, reason="B<n>: ...")`. They fail today (documenting the
bug), and `strict=True` forces removing the marker the moment the fix lands — the test
then becomes the permanent regression guard. Covered: B1 (RC4), B2/B3 (registries),
B5 (plot-content), B6 (Fixed), B7 (C3). B4/B8 get direct unit tests written against the
*fixed* semantics (xfail likewise). Everything not xfail-marked must be green on the
current codebase **before** refactor step 1 starts (this whole plan = "PR 0").

## 5. Runtime budget (target ≤ 5 min for `make test-all`; gate ≤ ~1.5 min)

| group | est. time | in gate? |
|---|---|---|
| unit layer (§3.2) | < 5 s | yes |
| contract layer (§3.1), non-SHAP | ~10 s | yes |
| contract ShapDP cases (N=50, budget 128) | ~10 s | yes |
| functional 05/06/07 ports incl. regional (§3.3) | ~20–30 s | yes |
| plots (§3.4, trimmed) | ~5 s | yes |
| facade + models + IndependentUniform | ~5 s | yes |
| **gate total** | **~60–75 s** | |
| linear/gam SHAP-heavy cases (fixed asserts, trimmed N) | ~60 s | slow |
| regional all-methods incl. shap/shapiq (trimmed) | ~30 s | slow |
| BikeSharing (network) | ~10 s | slow |
| **`test-all` total (excl. notebooks)** | **~3 min** | |
| notebook execution (`test_notebooks.py`) | ~5–10 min | slow, manual/docs-CI only |

Speed rules to keep it there: SHAP only with N ≤ 100 and explicit `budget`; prefer
passing precomputed `shap_values` where SHAP itself isn't under test; `nof_instances`
small everywhere; matplotlib `Agg` + autouse `close("all")`; no network in the gate.

## 6. Order of work

1. **P1–P5** (fix silent asserts, notebook collection, network, seeding, doctests) — and
   see what actually fails; adjust tolerances knowingly.
2. `conftest.py` + contract layer (§3.1) — green except the xfail(B*) specs.
3. Unit layer (§3.2).
4. Functional ports of notebooks 05/06/07 incl. the regional ground truth (§3.3).
5. Plot-content layer (§3.4) + trim `test_plots.py`/slow tests to budget.
6. Freeze: run `make test-all` twice (seed stability), record times in this file,
   then start `HOMOGENIZATION_PLAN.md` step 1.

Mapping safety → refactor steps: step 1 (helpers/utils) ← §3.2; step 2–3 (base + global
methods) ← §3.1 C1–C7 + §3.3; step 4 (visualization) ← §3.4; step 5 (regional) ← RC1–RC5 +
regional ground truths; step 6 (partitioning) ← §3.2 partitioning tests; step 7 (facade,
cleanup) ← §3.5.

---

## 7. Effort & coverage estimate

### Effort (writing + review + tuning tolerances)

| step (§6) | scope | est. |
|---|---|---|
| 1 | P1–P5 fixes, see what really fails | ~1 h |
| 2 | `conftest.py` + contract layer | ~2 h |
| 3 | unit layer | ~1.5 h |
| 4 | functional ports of notebooks 05/06/07 + regional GT | ~1.5 h |
| 5 | plot-content layer + runtime trimming | ~1 h |
| 6 | freeze: double `test-all` run, record times | ~0.5 h |
| **total** | | **~7.5 h ≈ 1 focused working day** |

### Coverage — measured baseline (gate, `-m "not slow"`, 2026-07-01) vs target

Total today: **77%** (1941 stmts, 451 missed). Per module, worst first:

| module | today | after plan | how |
|---|---|---|---|
| `regional_effect.py` | **18%** | ≥ 85% | RC1–RC5 + regional ground truths (today regional code is *only* executed by slow tests whose asserts are no-ops) |
| `regional_effect_ale.py` | **22%** | ≥ 85% | same |
| `regional_effect_shap.py` | **25%** | ≥ 80% | same (tiny-N SHAP) |
| `regional_effect_pdp.py` | **27%** | ≥ 85% | same |
| `utils_integrate.py` | 28% | n/a | module deleted in refactor step 1 (live part folds into `utils`, covered by §3.2) |
| `helpers.py` | 76% | ≥ 95% | `test_unit_helpers.py` (+ dead-code deletion) |
| `global_effect_shap.py` | 79% | ≥ 90% | fast tiny-SHAP case + C1–C7 |
| `axis_partitioning.py` | 86% | ≥ 92% | Fixed/`return_default` edge cases |
| everything else | 89–99% | ≥ 92% | contract + unit layers |
| **TOTAL (gate)** | **77%** | **≥ 90%** | |

Caveat that the % understates the real gap: today's line coverage is mostly *execution*,
not *verification* — `test_plots.py` only checks "doesn't crash", and the three slow
functional tests assert nothing (P1). After this plan, every covered line sits behind an
actual assertion (shapes, closed-form values, or figure content), which is the property
the refactor needs. Re-measure and update this table at the freeze (§6 step 6).

---

## 8. After the freeze — the suite as permanent enforcement

This suite is not a one-off scaffold for the refactor; it becomes the mechanism that
keeps the core rules true (see `HOMOGENIZATION_PLAN.md` §6, item 2):

- **Contract layer = merge gate for new methods.** Any new effect class (global or
  regional) is added to the `conftest.py` registry and must pass the parametrized
  contract suite (§3.1) unchanged. A method that needs a special case in the contract
  tests is a design smell to resolve *before* merging, not a test to fork.
- **Reproducibility contract** (strategic priority 1): once `random_state`/`seed` land,
  add C8 — two identically-constructed objects produce byte-identical `eval` output —
  and a seeded-split test for `datasets.*`. These stay in the gate forever.
- **New ground truths over new smoke tests.** Future features (categorical FOI,
  2D effects, classification) get the same treatment as §3.3: a closed-form synthetic
  model in `effector.models` + derivation in a notebook + ported asserts in pytest.
  Crash-only tests are not accepted as the sole coverage for new functionality.
- **Budget is law.** The gate stays ≤ ~90 s and `test-all` ≤ 5 min; anything above goes
  behind `slow`. Re-record the timing table (§0) and the coverage table (§7) whenever a
  layer is added, so drift is visible in the diff.

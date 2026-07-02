# effector — road to v1.0: the master plan

One document for the whole effort (merged 2026-07-02 from the five standalone
plans). The working order is the reading order:

- **Part I — Roadmap & workflow**: git/release policy, tooling, the three phases.
- **Part II — Testing plan**: the safety net built *before* touching the code.
- **Part III — Homogenization plan**: core rules + per-submodule refactor (+ §6
  strategic priorities and the categorical-FOI spec).
- **Part IV — Features plan**: importance, interactions, x-by-design, GADGET,
  and the specced-upgrades bundle.
- **Part V — Ideas log**: running notes from the notebook walkthrough (append here).

Cross-references use "Part N §x". Docs-site work is tracked separately in
`docs/DOCS_PIPELINE_PLAN.md`.

---

# Part I — Roadmap & workflow

*(was `ROADMAP.md` — “Road to v3”)*

Work plan for `effector` v3. Active branch: `road-to-v3`.

> **Note on ordering:** Phase 1 depends on Phase 2. The reformatting the existing
> code needs depends on where we're heading, so the new features must be known
> before (or while) auditing. Practical working order: pin down the Phase 2
> features first, then run the Phase 1 audit through that lens.

---

## Git & Release Workflow

Lightweight **GitHub Flow** — simple, PR-based, ship small and often.

**Branching model**
1. `main` is always releasable and **protected** (CI must be green to merge).
2. Every change starts on a **short-lived branch off `main`**, named by intent:
   - `feat/…` new feature · `fix/…` bug fix · `chore/…` tooling/deps ·
     `docs/…` documentation · `refactor/…` internal change
3. Open a **PR** → CI runs (pytest + coverage) → **squash-merge** → delete the branch.
4. No long-lived integration branch, no GitFlow. Merge to `main` in small PRs.

**Versioning (SemVer + release-early)** — current version is `0.2.0`.
- Bug fixes / docs / cleanups → **patch**: `0.2.1`, `0.2.2`, …
- New backward-compatible features → **minor**: `0.3.0`, `0.4.0`, …
- The milestone concluding this effort (stable, settled public API) → **`1.0.0`**
  (this is what "road to v3" maps to; `0.x → 1.0.0` signals API stability).
  Optional pre-releases: `1.0.0rc1` (PEP 440).
- Each release = a PR merged to `main` + a `vX.Y.Z` tag → auto-publishes to PyPI
  (`publish_to_pypi.yml`, triggered on `v*.*` tags).
- Ship incrementally; **never** accumulate everything into one big "v3" drop.

Decided: **Model A** — the PR branch is the disposable staging area; no permanent
`develop`/`staging` branch. "Test before it's real" is enforced by two gates below.

**Action items to formalize this**
- [x] Add `CONTRIBUTING.md` documenting the flow (PR #7) — also wired into the docs site
      (nav + committed copy + release-time refresh in `publish_documentation.yml`).
- [x] **Gate 1 — protect `main`:** branch protection ON — requires 1 approving review
      + `Run tests` green; force-push/delete blocked; admin bypass kept for solo velocity.
- [ ] **Gate 2 — protect releases:** make `publish_to_pypi.yml` run the test suite
      as a `needs:` job before build/publish (a `vX.Y.Z` tag can't ship if tests fail).
- [ ] **Test matrix:** run CI across Python `3.10 / 3.11 / 3.12 / 3.13` (currently 3.10 only).
- [ ] **⚠️ TEMPORARY — restore full test coverage:** the merge gate currently runs only
      fast tests (`-m "not slow"`); the 3 slow core-functional tests (`test_regional`,
      `test_gam`, `test_linear`) are marked `slow` and skipped on PRs (PR #10, for speed).
      Restore by running the full suite (incl. slow) on push to `main` and/or nightly.
- [ ] Clean up the ~20 stale remote branches once their work is confirmed merged.
- [ ] This roadmap + git policy lands via its own PR (branch `road-to-v3`).

---

## Tooling modernization

- [x] **Migrate to uv** (PR #12) — deps consolidated into `pyproject.toml`
      `[dependency-groups]` (`test`/`docs`/`dev`); `requirements-*.txt` deleted;
      `uv.lock` committed; Makefile + CI reworked around uv. CI now ~42s.
- [x] **Doc deps pinned** — now covered by `uv.lock`, so the unpinned-install class
      of breakage (the mkdocstrings typo) can't recur.
- [x] CI actions bumped (`checkout@v4`) and duplicate pytest run dropped (PR #12).
- [ ] **Gate 2 — tests before PyPI publish** (still the old pip-based `publish_to_pypi.yml`).
- [ ] **Test matrix 3.10–3.13** — ⚠️ note: `numba`/`llvmlite` (via `shap`) needs versions
      with 3.11–3.13 wheels; docs/tests currently pinned to 3.10 (PR #13).
- [ ] Add `concurrency:` groups + least-privilege `permissions:` to workflows.
- [ ] **ruff** to replace `black` + `flake8` + `isort`.
- [ ] Add `py.typed`; clean up stale `[tool.setuptools]` block in `pyproject.toml`.

---

## Phase 1 — Audit existing code

Go over the current codebase and decide, per module/class, what to **keep**,
**update**, **refactor**, or **drop**.

- [ ] (to be filled in once Phase 2 features are known)

## Phase 2 — Add new features

Features to add (to be specified):

- [ ] _TBD — feature list goes here_

## Phase 3 — Test everything and ship

- [ ] Full test pass (`make test`)
- [ ] Fix regressions
- [ ] Release / ship v3


---

# Part II — Testing plan

Goal: before touching the codebase per Part III, make the test suite
strong enough that any behavioral break in `fit / eval / plot` across all 11 method
classes is caught.

The suite is a **hierarchy of promises** (review 2026-07-02): the **contract** layer
tests the *signature* — the promise every method makes; the **unit** layer tests the
*actual numbers* of each small piece in isolation; the **functional** layer tests that
the *pieces work well together* (closed-form ground truths, end-to-end).

Runtime constraint — **two tiers** (details in §5):

- **Sanity tier, ≤ 3 min** (`make test`, every PR): all three layers, trimmed —
  small N, tiny SHAP budgets, no network, no notebooks. A fast "everything is
  probably good" estimate that is never blind to a whole layer.
- **Exhaustive tier, ≤ 10 min** (`make test-all`, pre-merge to `main` / nightly /
  pre-release): the same three layers at full strength — full-N SHAP backends,
  network datasets, notebook execution — "all three layers are perfect".

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

*(Note 2026-07-02, LOGBOOK #4: the target API changed — `eval` returns the mean only
(one return type); heterogeneity moves to a method-specific payload accessor + the
agnostic scalar H. The C-items below are drafted against the OLD eval surface and
get rewritten at constitution-writing time: C1 simplifies, C7 disappears, new
C-items cover the payload accessor, H, and h's centering-invariance.)*

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

*(Review 2026-07-02: this layer is written **first** — it is the stable anchor of the
whole effort (§6). The test list below is a proposal; the final selection is agreed
test-by-test when we write them.)*

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

## 5. Runtime budget — two tiers: sanity ≤ 3 min, exhaustive ≤ 10 min

**Tier 1 — sanity** (`make test`, `-m "not slow"`, runs on every PR): a fast
estimate that everything is good. Covers all three layers, trimmed.

| group | est. time |
|---|---|
| unit layer (§3.2) | < 5 s |
| contract layer (§3.1), non-SHAP | ~10 s |
| contract ShapDP cases (N=50, budget 128) | ~10 s |
| functional 05/06/07 ports incl. regional (§3.3) | ~20–30 s |
| plots (§3.4, trimmed) | ~5 s |
| facade + models + IndependentUniform | ~5 s |
| linear/gam SHAP-heavy cases (fixed asserts, trimmed N) | ~60 s |
| regional all-methods incl. shap/shapiq (trimmed) | ~30 s |
| **tier-1 total** | **~2.5–3 min** |

**Tier 2 — exhaustive** (`make test-all`, pre-merge to `main` / nightly /
pre-release): everything in tier 1 plus the pieces that make the three layers
airtight rather than probable.

| group | est. time |
|---|---|
| tier 1 | ~3 min |
| BikeSharing (network) | ~10 s |
| notebook execution (`test_notebooks.py`, synthetic ground-truth notebooks) | ~5–6 min |
| **tier-2 total** | **≤ 10 min** |

Speed rules to keep both budgets: SHAP only with N ≤ 100 and explicit `budget`; prefer
passing precomputed `shap_values` where SHAP itself isn't under test; `nof_instances`
small everywhere; matplotlib `Agg` + autouse `close("all")`; no network in tier 1;
if tier 2 outgrows 10 min, trim which notebooks execute (the ground truths are ported
to pytest anyway — §3.3), don't relax the budget.

## 6. Order of work (revised 2026-07-02 — interleaved with Part III)

The contract layer is Part III §1 compiled into executable form, so the working
order interleaves the two parts: **anchor behavior first, then agree the rules,
then encode them as tests, then refactor until the encoded rules hold.**

1. **P1–P5** (fix silent asserts, notebook collection, network, seeding, doctests) — and
   see what actually fails; adjust tolerances knowingly.
2. **Functional anchor (§3.3)**: port notebooks 05/06/07 incl. the regional ground
   truth; repair + parametrize the linear/GAM tests. These test *what* is computed,
   never *how* — they must stay green through the entire effort.
3. **Agree the constitution (Part III §1) as text** — a design-review step, ~zero code.
4. **Contract layer (§3.1)** + `conftest.py` registry: the constitution as executable
   spec — green where a rule already holds today, `xfail(strict=True)` where the
   refactor must make it true (the same mechanism §4 uses for bugs B1–B9).
5. **Unit layer (§3.2)** and **plot-content layer (§3.4)**; trim to the tier budgets (§5).
6. **Freeze**: run `make test-all` twice (seed stability), record times in this file.
   Then start the Part III implementation (its steps 1–7). Definition of done for the
   whole refactor: **functional layer still green + zero xfail markers left**.

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
keeps the core rules true (see Part III §6, item 2):

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
- **Budget is law.** The sanity tier stays ≤ 3 min and the exhaustive tier ≤ 10 min;
  anything above goes behind `slow` (tier 2) or gets trimmed. Re-record the timing
  table (§0) and the coverage table (§7) whenever a layer is added, so drift is
  visible in the diff.


---

# Part III — Homogenization plan

*(was `HOMOGENIZATION_PLAN.md` — “Homogenization Plan — stabilizing the core before v0.3 features”)*

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

- **R1 — Lifecycle** *(revised 2026-07-02, LOGBOOK #3/#4)*: `fit` does the hard work
  once and stores everything under `feature_effect["feature_{i}"]`; `eval(feature, xs,
  centering=<class default>)` returns the **mean effect only — one return type,
  always** — and never recomputes unless `requires_refit`; `plot` is always a thin
  wrapper over `eval`/stored state + one `vis.*` call, zero own computation. Exact
  signatures are decided at constitution-writing time, not here.
- **R2 — Heterogeneity semantics** *(revised 2026-07-02, LOGBOOK #4 — heterogeneity
  does NOT go through `eval`)*:
  - **h, method-specific**: the variance of the method's own per-instance effect
    object (PDP: ICE levels; DerPDP: d-ICE slopes; ALE/RHALE: local slopes per bin;
    ShapDP: shap values). Convention: **variance** internally, std only at the plot
    layer; invariant to centering (contract-testable). Units differ per method —
    accepted, since nothing method-agnostic consumes h directly.
  - **payload accessor** (name/shape TBD): every method exposes its honest
    method-specific object (ICE table, per-bin variances, shap cloud) from stored
    state. PDP's old `eval(return_all=True)` exception disappears — the exception
    becomes the rule.
  - **H, method-agnostic score**: one scalar per feature, the single heterogeneity
    quantity consumed by regional splitting (weighted child-H minimization) and the
    future interaction submodule (F2: vector = normalized H; matrix = drop of H under
    conditioning). Aggregation internals + weighting (uniform vs data-density) +
    normalization: deferred.
  - Conscious cost: removing `eval(..., heterogeneity=True)` is a **breaking API
    change** — right time, pre-1.0. (Today's dialects folded into h: ALE returns bin
    *variance* named std, PDP returns `np.var`, ShapDP names a variance-spline
    `spline_std`, docstrings say "std".)
- **R3 — Centering vocabulary** *(agreed 2026-07-02, LOGBOOK #4)*: `{False, "zero_integral" (=True), "zero_start"}` via
  `helpers.prep_centering` (already true). Each class declares its default **once** as a
  class attribute (e.g. `DEFAULT_CENTERING = "zero_integral"`), and `fit/eval/plot`
  signatures use that attribute instead of hardcoding different literals per method.
- **R4 — State schema** *(agreed 2026-07-02, LOGBOOK #4)*: `feature_effect["feature_{i}"]` always contains
  `norm_const: float | None` (use `None`, not the `1e8` `EMPTY_SYMBOL` or `np.nan` —
  today all three conventions coexist) plus a method-specific payload;
  `fit_args["feature_{i}"]` records the kwargs needed to detect refit.
- **R5 — One method registry** *(agreed 2026-07-02, LOGBOOK #5)*: a single `{canonical_name: (cls, needs_jac, uses_data_effect,
  display_name)}` table (plus aliases), used by `FeatureEffect._REGISTRY`,
  `RegionalEffectBase._create_fe_object`, and plot titles. Today the same knowledge lives
  in ≥3 places as if/elif chains and string comparisons.
- **R6 — String-argument registries** *(agreed 2026-07-02, LOGBOOK #5)*: exactly one alias table per concept, resolved by one
  `return_default`-style function, and *asserts always match the resolver*
  (binning: `"fixed" | "greedy" | "dp"`; partitioner: `"best" | "best_level_wise"`).
- **R7 — Plot contract** *(agreed 2026-07-02, LOGBOOK #6)*: every `vis.*` function and every public `.plot` returns
  `(fig, ax)` when `show_plot=False` and `None` otherwise — uniformly.
- **R8 — Constructor contract** *(agreed 2026-07-02, LOGBOOK #3 — concept; exact order decided at constitution-writing)*: canonical parameter order
  `data, model, model_jac=None, *, data_effect, nof_instances, axis_limits,
  feature_types, cat_limit, feature_names, target_name, ...` — everything after
  `model_jac` keyword-only, so the current per-class positional shuffles can't bite.
- **R9 — Errors & messages** *(agreed 2026-07-02, LOGBOOK #6)*: `ValueError`/`TypeError` (not bare `assert`) for user input;
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
  ships dead weight and its ideas belong in Part V.
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

*(Sequencing with Part II — revised 2026-07-02: §1 above is agreed **as text** right
after the functional anchor lands, and is then encoded as the contract layer
(Part II §6 steps 3–4) **before** any step below starts. The steps below turn the
resulting `xfail` markers green; done ≔ functional layer still green + zero xfails.)*

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

Combined with the test-suite work (see Part II §7, ~1 day), the realistic
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
Part I / Part V and should come **before** further feature ideas.

1. **Reproducibility end-to-end** (small, high trust-impact). `nof_instances`
   subsampling uses global `np.random` with no seed; dataset splits are unseeded — two
   runs give two different explanations. Add `random_state`/`seed` to every constructor
   and `datasets.*`, thread it through `prep_nof_instances`, and add a contract test:
   two identical constructions → identical `eval` output. For an XAI package this is a
   core value, not a nice-to-have.
2. **Keep the contract enforced, not conventional.** The §1 rules go into
   `CONTRIBUTING.md` / `docs/design.md`, and the contract layer of Part II
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
closed-form ground-truth test per Part II §8 ("new ground truths over new
smoke tests").


---

# Part IV — Features plan

*(was `FEATURES_PLAN.md` — “Features Plan — new capabilities for v0.3 → v1.0”)*

Builds on Part II (safety net) and Part III (core contract): those two make the codebase *ready* for
features; this one says *which* features and how each one plugs into the homogenized
core. Fills in "Phase 2 — Add new features" of Part I.

The recurring theme *(updated 2026-07-02, LOGBOOK #4)*: after homogenization, every
method exposes the same three-object surface — `eval` (mean effect, one return
type), a method-specific **payload accessor**, and the method-agnostic
**heterogeneity score H** — and the regional machinery consumes only a
`heter_func(mask) -> float` callable (`space_partitioning.Base.compile`), i.e. H on
subsets. Every feature below is written **once against those seams**, never
per-method. That is the test of whether the homogenization was done right.

Five features: **F1** importance, **F2** interaction quantification,
**F3** x-by-design (CALM), **F4** GADGET-style joint splitting, **F5** the
already-specced upgrades bundle (categorical FOI, reproducibility, visualization, …).

---

## F1 — Importance submodule (`effector.importance`)

**Idea:** a score per feature measuring how much its effect *varies* — features whose
effect curve is flat are unimportant; features whose effect swings a lot drive the
prediction. This is the PDP-based importance of Greenwell et al. (2018), generalized
to every method in the package.

**Definition (one, method-agnostic):** for feature $j$ with mean effect curve
$\mu_j(x)$ from any method,

$$ I_j = \mathrm{std}_{x \sim p(x_j)}\,[\mu_j(x)] $$

i.e. the standard deviation of the effect curve **weighted by the data distribution
of $x_j$** (evaluate `eval` on the data column, not on a uniform linspace — a
uniform grid over-weights sparse tails). Offer `weighting="data" | "uniform"` with
`"data"` the default; report both raw and normalized-to-sum-1 scores.

**Why it's nearly free after homogenization:** it needs only base-class `eval`
(Part III §2.2). One implementation:

```python
imp = effector.importance.effect_importance(fe)        # fe: any fitted global method
# or as a base-class method:
ale.importance(features="all") -> np.ndarray (D,)
```

plus a `vis.plot_importance` bar chart (sorted, with method name in the title via the
R5 registry).

**Companion score — heterogeneity-based:** $H_j$, the base-defined heterogeneity
score (LOGBOOK #4). This is *not*
importance — it is an interaction proxy — so it lives in F2, but the two are computed
by the same loop and plotted side-by-side ("effect strength" vs "interaction
strength" per feature) — a genuinely novel two-axis summary no other package ships.

**Regional variant:** same function against `RegionalEffectBase.eval(feature,
node_idx, xs)` → importance per subregion ("temperature matters twice as much on
working days"). Free once the global version exists.

**Ground-truth tests** (per Part II §8 philosophy): linear model
$f = \sum a_j x_j$ with $x_j \sim U(-1,1)$ → $I_j = |a_j| \cdot \mathrm{std}(x_j)$
for PDP/ALE/RHALE exactly; ConditionalInteraction model already has closed-form
curves in the notebooks — derive $I_j$ from them.

**Effort:** ~½ day (core ~2 h, vis ~1 h, tests ~1.5 h). Highest
visibility-per-hour item in this plan; also the best first exercise of the
homogenized `eval`.

---

## F2 — Interaction quantification, global and regional

**Idea:** quantify *how much* each feature interacts — as a $D \times 1$ vector
(feature $j$ vs all others together) and a $D \times D$ matrix (feature $j$ vs
feature $k$ pairwise). For PDP the classical answers are Friedman–Popescu
H-statistics; the ALE-family analogs are heterogeneity-based and are effector's own
contribution.

### F2a — $D \times 1$ vector (cheap, do first)

Feature-vs-rest interaction is exactly what the package already computes:

- **Heterogeneity index (all methods):** $H_j$ — the base-defined heterogeneity
  score (LOGBOOK #4), normalized (e.g. by $\mathrm{Var}[f(X)]$, the same
  normalization H-statistics use, so scores are comparable across models). Zero iff
  the effect of $x_j$ is the same for all instances ⇒ no interaction. This is the
  quantity the regional partitioner minimizes — surfacing it as a number closes the
  loop: *"$H_j$ is high → ask regional effects where it splits."*
- **H-statistic $H^2_j$ (PDP only):** $\mathrm{Var}[f - pd_j - pd_{-j}]$ needs
  $pd_{-j}$ (PDP of all-but-$j$), computable with the existing ICE machinery on the
  complement — one new kernel, no new concepts.

Seam: base `eval` + one small module `effector/interaction.py` (the name is free —
the dead file is deleted in Part III §2.9). API sketch:

```python
ix = effector.interaction.h_index(fe)            # (D,) heterogeneity-based, any method
ix = effector.interaction.h_statistic(pdp)       # (D,) Friedman–Popescu, PDP only
```

### F2b — $D \times D$ matrix

Two routes, both worth shipping:

- **PDP route (H-statistic matrix):** $H^2_{jk}$ needs 2-D PDP $pd_{jk}$ → depends
  on 2-D effect computation (listed in Part III §6.7). ICE machinery
  generalizes directly (evaluate on a 2-D grid); expensive is fine — document cost,
  subsample by default.
- **ALE-family route (conditional-heterogeneity drop — effector-native):** define
  $\mathrm{IX}_{jk}$ = the *reduction* in feature $j$'s heterogeneity achieved by
  the best split on feature $k$. The candidate-split scan in
  `space_partitioning._single_node_split` already evaluates
  `heter_func(mask)` for every candidate split of every conditioning feature — the
  weighted-heterogeneity matrix it builds **is** the $D \times D$ interaction matrix,
  currently computed and thrown away except for the argmin. Expose it:
  `partitioner.heterogeneity_matrix_` after `fit`, wrapped as
  `effector.interaction.matrix(regional_fe)`. Near-zero extra compute; consistent by
  construction with what regional effects will actually split on. This is also
  GADGET's diagnostic view, so F2b and F4 share code.

**Regional variants:** all of the above per tree node (which interactions *remain*
inside a region — the "did splitting resolve the interaction?" check).

**Plot:** heatmap for the matrix, bar for the vector (`vis.plot_interaction_matrix`).

**Ground truths:** `ConditionalInteraction` model — closed-form: only specific pairs
interact; uncorrelated features ⇒ off-pair entries ≈ 0. For $H^2_{jk}$ on
$f = x_1 x_2 + x_3$: exact value derivable analytically.

**Effort:** F2a ~1 day (incl. $pd_{-j}$ kernel + tests). F2b ALE-route ~1 day
(mostly surfacing + API + tests); F2b PDP-route ~1.5 days because it drags in 2-D
PDP (which is then reusable for 2-D plots in F5).

---

## F3 — X-by-design (CALM integration)

**What exists:** branch `origin/calm` holds a complete standalone package —
interpretability-by-design models: fit a blackbox (XGB), detect regions with
`effector.RegionalPDP/RHALE` (`RegionDetector.detect_regions` returns the tree),
then fit a masked/regional GAM (EBM, NAM, PyGAM) restricted to those regions →
an accurate, locally additive, fully interpretable model. Paper: "Interpretability-by-Design
with Accurate Locally Additive Models and Conditional Feature Effects".

**Decision to make: where does it live?**

- **Option A — separate package (recommended).** CALM stays its own repo/package
  depending on effector (the calm branch already states this design). Its
  dependency set (xgboost, interpret/EBM, NAM, pygam, sklearn) is far heavier than
  effector's and pulls in training-time concerns effector deliberately doesn't have.
  Effector's job is to provide **stable seams** CALM consumes:
  1. programmatic tree access (Part V "programmatic access to the partition
     tree" + Part III §6.5) — regions as data: condition lists, masks,
     `tree.to_dict()` / `region.mask(X)`;
  2. a stable `space_partitioner` API (post-homogenization R6 registry);
  3. reproducibility (§6.1) so region detection is deterministic.
- **Option B — `effector.xbd` submodule** with optional extras
  (`pip install effector[xbd]`), vendoring a *minimal* CALM: region detection +
  masked PyGAM only (skip EBM/NAM/XGB). Gets the story into effector's docs at the
  cost of maintaining modeling code.

**Recommendation:** A, plus one docs tutorial in effector ("interpretability by
design with effector + CALM") and the seams (1)–(3) implemented in effector proper.
The seams are independently valuable (they're already in the plans); CALM then works
against public API instead of reading `self.tree` internals. Revisit B only if the
separate package proves too much friction.

**Effort (effector-side only):** the seams are F5 items (tree UX ~1 day,
reproducibility ~½ day, already counted there); tutorial notebook ~½ day.
CALM-side packaging/refresh is its own project, out of scope here.

---

## F4 — GADGET-style joint regional splitting

**Today (REPID-style):** `RegionalEffectBase.fit` finds a partition **per feature
independently** — feature $j$'s tree minimizes feature $j$'s heterogeneity only.
Result: $D$ different partitions of the space, which don't compose into a single
interpretable segmentation.

**GADGET (Herbinger et al., 2023):** find **one** partition minimizing the
*aggregated* heterogeneity over a set $S$ of features of interest — one tree, valid
simultaneously for all features in $S$; within each region, all effects in $S$ are
approximately homogeneous (the model is approximately additive there). This is also
exactly the region detector CALM/x-by-design wants (F3), and its split-scan
diagnostics are the F2b matrix.

**Why the seam is nearly drop-in:** `space_partitioning.Base.compile(...,
heter_func)` takes *one* callable `mask -> float`. Regional fit already builds a
heterogeneity callable per feature (`_create_heterogeneity_function`, all four
regional modules). A joint objective is just:

```python
def joint_heter(mask):
    return sum(w_j * heter_funcs[j](mask) for j in features_of_interest)
```

fed into the **existing** `Best` partitioner, unchanged. Weights $w_j$: default
uniform; option to normalize each $H_j$ by its root value so one high-variance
feature doesn't dominate.

**What actually needs building:**

1. **Precompute-all-then-partition flow:** current fit interleaves (per feature:
   build heter_func → partition). Joint mode needs all heter_funcs *first*, then one
   partition. The template-method `fit` from Part III §2.5 (hooks
   `_precompute_global` / `_create_heterogeneity_function`) makes this a loop
   reordering, not a rewrite. **F4 should not start before §2.5 lands.**
2. **One tree, many features:** today `self.feature_effect["feature_{j}"]` stores a
   tree per feature. Joint mode stores one shared tree; `eval/plot(feature,
   node_idx)` work unchanged (fit the global method on the node's subset — that
   machinery, `_create_fe_object`, is feature-agnostic already).
3. **API:** keep it an argument, not a new class:
   `RegionalPDP.fit(features=[...], space_partitioner="gadget")` (registry per R6;
   canonical name e.g. `"joint"` with alias `"gadget"`), and
   `summary()` prints the single shared tree with per-feature heterogeneity drops
   per node.
4. **Candidate conditioning features:** in joint mode a feature can appear both as
   FOI and as conditioning feature; default `candidate_conditioning_features =
   all \ S` mirroring GADGET, overridable.

**Tests:** `ConditionalInteraction4Regions` (already has notebook ground truth):
joint split on $x_2$ must resolve heterogeneity for *both* dependent features with
one tree; a REPID-vs-GADGET comparison notebook is the flagship demo ("$D$ trees
vs 1 tree, same regions").

**Effort:** ~1.5–2 days after §2.5 (flow reordering ~½ day, shared-tree state + summary
~½ day, API/registry ~2 h, tests + demo notebook ~½ day).

---

## F5 — Already-specced upgrades bundle

These are specified elsewhere; listed here so Phase 2 has one complete inventory.
References, not duplication:

| item | spec lives in | est. |
|---|---|---|
| Categorical features as FOI (all methods, regional-on-cat) | Part III §6.4a (full spec + phasing) | ~2 days |
| Reproducibility end-to-end (`random_state` everywhere) | Part III §6.1 | ~½ day |
| pandas/DataFrame ingestion + classification story | Part III §6.4 | ~1 day |
| Regional UX: regions as data, no magic `node_idx` | Part III §6.5 + Part V (tree access) | ~1 day |
| Facade phase 2+3: ±std bands, global-vs-regional overlay | Part V ⭐ (progress list) | ~1 day |
| Visualization upgrades: `ax=` seam, multi-feature grid `plot(features=[...])`, scale-at-construction, units/denormalization | Part V (bike-sharing items) | ~1.5 days |
| 2-D effect plots (PDP first; shared kernel with F2b H-matrix) | Part III §6.7 | ~1.5 days |
| Performance posture: benchmark script + per-node caching | Part III §6.6 | ~1 day |

Plus the running Part V log, which keeps growing as the notebook walkthrough
continues (02_california_housing, 03_tabpfn, 04_no2 still pending).

---

## Dependency graph & suggested order

```
Part II (done first)
  └── Part III steps 1–7
        ├── §6.1 reproducibility ──────────────┐
        ├── F1 importance  ── needs base eval  │
        ├── F2a interaction vector             │  (F5 foundations
        ├── F5 categorical FOI (§6.4a)         │   interleave here)
        ├── §2.5 template regional fit
        │     └── F4 GADGET joint splitting
        │           └── F2b DxD matrix (ALE route = F4's scan surfaced)
        ├── 2-D PDP (F5) ──► F2b DxD (H-statistic route)
        └── tree-as-data UX (F5) ──► F3 CALM seams + tutorial
```

Suggested sequence (each item = one PR/branch, ships independently per Part I
release policy — minor version bumps as features land):

1. **F5-foundations:** reproducibility, then categorical FOI (per §6 of the
   homogenization plan these precede new feature ideas).
2. **F1 importance** — cheapest, high visibility, first consumer of the unified `eval`.
3. **F2a interaction vector** — pairs with F1 (the two-axis feature summary).
4. **F4 GADGET** — the biggest methodological addition; unlocks F2b and F3.
5. **F2b interaction matrix** — ALE route first (free from F4), PDP H-matrix with 2-D PDP.
6. **F5-UX:** tree-as-data, visualization upgrades, facade phases 2–3.
7. **F3 x-by-design** — effector-side seams done in (6); tutorial + CALM refresh.

## Effort summary

| feature | est. |
|---|---|
| F1 importance | ~½ day |
| F2a vector | ~1 day |
| F2b matrix (both routes) | ~2.5 days |
| F3 (effector-side + tutorial) | ~1 day |
| F4 GADGET | ~2 days |
| F5 bundle | ~9 days (itemized above) |
| **total Phase 2** | **~16 focused days** |

Spread across small PRs with the contract suite as the merge gate, this is the road
from v0.3 to a credible v1.0: importance + interaction + joint splitting turn
effector from "effect plots" into a complete *effect-based model understanding*
package, and x-by-design extends it from understanding models to building them.


---

# Part V — Ideas log (notebook walkthrough, running)

*(was `Part V.md` — “effector v0.3 — feature ideas from the notebook walkthrough”)*

Running log of ideas collected while going through the notebooks one by one.
Each idea is tagged **[apply-now]** (small, localized) or **[needs-a-plan]**
(restructuring / cross-cutting), with a rough difficulty and the code seam it
touches. Feeds Phase 2 of Part I.

Difficulty legend: 🟢 easy · 🟡 medium · 🔴 large.

---

## ⭐ Headline feature — unified `effector.FeatureEffect` facade + method comparison

**Source:** bike-sharing walkthrough. **Status:** needs-a-plan, but core is 🟢.

**Idea:** a single entry point that homogenizes the user-facing API the way the
base classes homogenize the internals. Compare methods in one figure:

```python
fe = effector.FeatureEffect(X_test, predict, model_jac=jac,
                            feature_names=..., target_name=...)
fe.plot(feature=3, methods=["PDP", "ALE", "RHALE", "ShapDP"], centering=True)
```

**Why it's feasible:** all methods already expose the SAME
`eval(feature, xs, heterogeneity, centering)` returning a mean curve on an
arbitrary shared grid, with lazy self-fit. Overlaying mean curves ≈ 30 lines.

**Caveats found in code (must handle):**
- Constructors diverge: `RHALE`/`DerPDP` need `model_jac`; `ShapDP` needs `shap`,
  has `backend`/`shap_values`, default `nof_instances=1000` and is slow. Facade =
  dispatch + per-method `method_kwargs` passthrough + graceful shap/jac handling.
- Heterogeneity is NOT overlayable (ICE vs bin-var bars on a 2nd axis vs shap
  scatter, 3 different `vis.*` funcs). Overlay only the mean curve; offer optional
  ±std band via `eval(heterogeneity=True)`; leave native heterogeneity to per-method
  small-multiples.
- `eval` returns VARIANCE in all three but names it inconsistently (`std` in ALE);
  plots sqrt it. Centralize the sqrt if building a shared std-band.
- `DerPDP` is in derivative units → never overlay with PDP/ALE/RHALE/SHAP.
- Overlaying on one axis must bypass the existing `vis.*` funcs (each calls its own
  `plt.subplots()`), i.e. write a new dedicated comparison-plot function — sidesteps
  the missing `ax=` seam.

**Bonus:** `RegionalEffectBase.eval(feature, node_idx, xs, ...)` is homogeneous too
→ later, compare a global effect vs its regional sub-effects in one figure (the
"wow" demo). Phase after global.

**Also subsumes** the "pass scaling once" idea below — the facade holds
`data/model/names/scaling` once.

**Suggested phasing (3 commits):** (1) facade + `.plot(methods=[...])` mean-curve
overlay [flagship, mostly 🟢]; (2) optional ±std bands [🟡]; (3) regional comparison [🟡].

**Progress:**
- [x] (1) `effector.FeatureEffect` facade + `.plot(methods=[...])` mean-curve overlay.
      Branch `feat/feature-effect-comparison`, commit `e5ba5be`. Pool = PDP/ALE/RHALE/ShapDP
      (DerPDP excluded); shared subsampled data; centering forced on; `model_jac` optional
      with numeric fallback + warning; new `vis.plot_effect_comparison`; tests in
      `tests/test_feature_effect.py` (6 passing).
- [ ] (2) optional ±std bands (revisit heterogeneity, incl. showing ALE heterogeneity in
      output units so it's comparable).
- [ ] (3) regional comparison (global vs regional sub-effects via `RegionalEffectBase.eval`).

---

## Notebook: `real-examples/01_bike_sharing_dataset.ipynb`

### [needs-a-plan] Pass scaling once, not on every call — 🟡 (subsumed by ⭐ facade)
Every `.plot()` repeats `scale_x=scale_x_list[i], scale_y=scale_y`. Let the user
supply `scale_x_list` / `scale_y` **at construction** (stored on the object) and
have `plot`/`eval`/`summary` default to it, still overridable per-call.
- Seam: constructors of all effect classes + thread defaults into `plot`/`eval`;
  `visualization.py` already consumes `scale_x`/`scale_y`.
- Cross-cutting (all 11 classes) → plan, not on-the-fly. Backward compatible.

### [needs-a-plan] Multi-feature plot (grid) — 🟡
Replace the manual `for i in [2,3,8,9,10]: pdp.plot(...)` with
`pdp.plot(features=[...])` rendering a grid / list of subplots.
- Seam: new orchestration layer above the single-feature `vis.*` funcs; depends
  on the missing `ax=` seam (each `vis.*` calls its own `plt.subplots()`).

### [needs-a-plan] Programmatic access to the partition tree — 🟡
`summary()` is print-only; user reads it by eye and hand-maps "workingday=0" →
`node_idx=1`. Expose the tree as data and/or let `plot`/`eval` select a node by
condition instead of raw index.
- Seam: `tree.py` + `RegionalEffectBase.summary`/`_plot`.

### [needs-a-plan] Smoother unit handling / denormalization — 🔴
The notebook hand-patches `scale_x_list[8]["mean"] += 8; ["std"] *= 47` to undo
standardization for temp/hum/windspeed. Signals unit handling is fiddly.
Explore a first-class "feature scaling / units" concept so users don't build
`scale_x_list` dicts by hand.
- Seam: broad — data preprocessing + every `scale_x` consumer. Needs design.

---

_Next notebooks to walk: 02_california_housing, 03_california_housing_tabpfn, 04_no2._

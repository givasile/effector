# Logbook — what we actually do

Companion to `PLAN.md`. That file is the **map** (the extensive roadmap); this file
is the **trail** — the steps we actually take, written by me (Vasilis), with Claude's
help, so that I am fully aware of everything we do.

The loop, per item:

1. Claude reads `PLAN.md` and names the next item.
2. We discuss it — intuition, diagrams, alternatives — until I understand and agree
   (or we change the plan).
3. I write the entry below; then (and only then) the work happens.

Nothing lands in code before it has an entry here.

Entry format:

```
## <nr>. <date> — <title> (tag: theory | code)  [PLAN.md ref]

<diagram — what we do, at a glance>

**What:** the decision / the thing done, in one or two sentences.
**Why:** the reason I'm convinced.
**Changes:** what changed (in PLAN.md, code, tests).
```

Tags: **theory** = ends in a mental model / verdict; **code** = ends in code changes.
Every entry starts with a diagram.

---

## 1. 2026-07-02 — Test suite: hierarchy of tests, run in two tiers (tag: theory)

```
              the hierarchy                       what breaking it looks like
  ┌────────────────────────────────────┐
  │ FUNCTIONAL — pieces work together  │   wrong numbers end-to-end
  │   closed-form ground truths        │   (curve shifts, split lands wrong)
  ├────────────────────────────────────┤
  │ UNIT — each piece in isolation     │   wrong numbers in one kernel
  │   bin effects, ALE math, helpers   │   (NaN handling, off-by-one)
  ├────────────────────────────────────┤
  │ CONTRACT — the signature/promise   │   a method deviates from the API
  │   one suite, all 11 methods        │   (shape, centering, lazy-fit)
  └────────────────────────────────────┘

  tier 1   make test       ≤ 3 min    every PR          all layers, trimmed
  tier 2   make test-all   ≤ 10 min   merge / nightly   all layers, full + notebooks
```

**What:** 

Three test layers seen as a hierarchy: 
- the **contract** layer tests the *signature* (the promise every method makes), 
- the **unit** layer tests the *actual numbers* of each small piece, 
- the **functional** layer tests that the *pieces work well together*. 


Two run tiers: 
- **≤ 3 min sanity** (`make test`, every PR — all three layers, trimmed) and 
- **≤ 10 min exhaustive** (`make test-all`, pre-merge/nightly — full strength, incl. notebooks).

**Why:** 

Today's suite is not adequate; the refactoring and the new features need a good alarm system and a sanity tier that is never blind to a whole layer.

**Changes:** 
PLAN.md Part II rewritten (intro, §5 two-tier budget, §8 budget rule).

---

## 2. 2026-07-02 — Order of test suite and refactor (tag: theory)

```
   P1–P5          FUNCTIONAL           §1 RULES         CONTRACT           REFACTOR
   fix the   ──►  anchor:        ──►   agreed      ──►  rules as      ──►  turn every
   wiring         ground truths        as text          xfail spec         xfail green
                  ─────────────────────────────────────────────────────────────────►
                  stays green through EVERYTHING

                  done ≔ zero xfails left  +  functional layer still green
```

**What:** 
(1) fix the P-items and write the **functional ground-truth tests first**: 
they test *what* is computed, never *how*, so they stay stable through the whole
effort 
(2) agree the Part III §1 constitution **as text**; 
(3) encode it as the **contract layer** — green where a rule already holds, `xfail(strict=True)` where the refactor must make it true; 
(4) refactor (all part III) until **zero xfails remain and the functional layer is still green** — that is the definition of done.

**Why:** the contract tests don't "follow" Part III — they *are* Part III §1 in
executable form. Functional tests are the refactor-invariant anchor; writing the
contract first turns the refactor into "make the xfails green" (measurable, no
scope drift). The concrete functional test list stays open — we agree on it
test-by-test when we write them.

**Changes:** PLAN.md Part II §6 rewritten (interleaved order), Part III §4 preamble
added.

---

## 3. 2026-07-02 — Method lifecycle: fit = compile, eval = execute, plot = pretty-print (tag: theory)  [Part III §1, R1+R8]

```mermaid
flowchart TB
    C["CONSTRUCT<br/>same shape for all 11 classes"]
    F["FIT — compile<br/>the hard work, once"]
    E["EVAL — execute<br/>much cheaper, runs over what fit stored"]
    P["PLOT — pretty-print<br/>a view over eval, zero own computation"]
    X["regional effects · facade · importance (F1) · interaction (F2)<br/>all built by composing eval"]
    C --> F --> E --> P
    E -.-> X
```

**What:** every method follows the same three-step lifecycle: **fit** does the hard
work once; **eval** is the cheap execution step over fit's results; **plot** is
pretty-print over eval. All classes are constructed the same way. This is a
*mental-model* agreement: the exact signatures (of `eval`, of the constructor, and
whether extra entry points like an `eval_all` exist) are deliberately **not**
decided here — they come later, with the constitution text and its contract tests.

**Why:** everything beyond a single plot is programmatic composition of `eval` —
regional, facade, F1, F2. If plot computes anything on its own, what we *see*
drifts from what we *compute*. This is the core of our mental model.

**Changes:** none yet — the exact-signature decisions and contract tests
(C1/C3/C6) reference this entry when we get there.

---

## 4. 2026-07-02 — Heterogeneity: eval returns the mean only; payload accessor + agnostic score H (tag: theory)  [Part III §1, R2–R4]

```mermaid
flowchart TB
    S["stored state (from fit)<br/>ICE matrix · per-bin variances · shap cloud"]
    E["eval(feature, xs) → y(z)<br/>mean effect, ONE return type, always"]
    A["payload accessor (name TBD)<br/>the method's honest object"]
    H["H(feature) → scalar<br/>method-AGNOSTIC heterogeneity score"]
    P["plot<br/>(method-specific)"]
    R["regional splitting:<br/>minimize weighted child-H"]
    IX["interaction (F2):<br/>vector = normalized H · matrix = H drops"]
    S --> E
    S --> A
    S --> H
    A --> P
    H --> R
    H --> IX
```

**What:** heterogeneity does **not** go through `eval`. Three public objects instead
of one overloaded one:
- `eval` → the mean effect only, one return type, always;
- a **payload accessor** (name/shape TBD) → the method's honest per-instance object
  (ICE table for PDP, per-bin variances for ALE/RHALE, shap cloud for ShapDP) —
  method-specific units are fine, nothing agnostic consumes it;
- **H** → one method-agnostic scalar per feature; the only heterogeneity quantity
  consumed by regional splitting and the future interaction submodule.

Convention for the underlying h: variance internally, sqrt only at plot;
centering-invariant. Also locked: **R3** one centering vocabulary
{False, zero_integral (=True), zero_start}, default declared once per class;
**R4** one null (`None`) for "not computed".

Deliberately not decided: payload accessor name/shape; how H aggregates (weighting
uniform vs data-density); normalization.

**Why:** every agnostic consumer needs either the mean curve or the scalar H —
never h(z) on a grid (for bin methods that was interpolation theater anyway). One
return type for eval kills a whole class of contract warts, and PDP's old
`return_all` exception becomes the rule: every method has a payload accessor.

**Conscious costs:** breaking API change (`eval(heterogeneity=True)` is public
today) — right time, pre-1.0. Functional anchor tests get written against today's
API first and re-pointed mechanically when this lands.

**Changes:** PLAN.md — R1/R2 rewritten (Part III §1); notes where the old seam was
referenced (Part II §3.1 contract items, Part IV intro + F1/F2).

---

## 5. 2026-07-02 — One menu: every list of valid names is written once (tag: theory)  [Part III §1, R5+R6]

```mermaid
flowchart TB
    T["ONE table per concept<br/>methods · binning strings · partitioner strings"]
    V["input validation<br/>(the check)"]
    L["lookup / dispatch<br/>(facade, regional)"]
    D["display<br/>(plot titles)"]
    T --> V
    T --> L
    T --> D
```

**What:** every list of valid names is written **once** and everything else reads
it. R6: for each string option (binning `"fixed" | "greedy" | "dp"`, partitioner
`"best" | "best_level_wise"`) there is one list — the check and the lookup use the
same one, so they *cannot* disagree. R5: same idea one level up — "which methods
exist and what does each need" is one table (`"pdp" → class, needs jacobian?,
display name`), read by the facade, the regional dispatch, and the plot titles
instead of three hand-typed if/elif chains. Exact table shapes: decided later, at
constitution-writing.

**Why:** the hand-typed copies already drifted — 3 of our 9 known bugs (B2, B3:
`"dp"` binning is unreachable today; `"cart"` passes the check then crashes;
`"best_level_wise"` is valid but rejected). One table kills the bug *class*, and
future features become new rows: GADGET = a row in the partitioner table,
categorical support = a column in the method table.

**Changes:** none yet — implemented across Part III steps; contract tests
(registries) enforce it.

---

## 6. 2026-07-02 — Polite surfaces: plots always return the figure, errors always speak (tag: theory)  [Part III §1, R7+R9 — constitution closed]

```
  R7   plot(..., show_plot=False)  →  (fig, ax)     every method, every vis function
       plot(..., show_plot=True)   →  None          no exceptions

  R9   bad user input   →  ValueError/TypeError with a message that names the fix
       internal hiccup  →  warnings.warn (never print, never bare assert)
```

**What:** R7 — every plot hands back `(fig, ax)` when asked, uniformly (that's what
enables saving, tweaking, composing grids). R9 — wrong input raises a proper error
that says what's wrong and what's allowed; internal warnings go through
`warnings.warn`. With these two, the **constitution R1–R9 is closed**: #3 (R1+R8
lifecycle), #4 (R2–R4 semantics), #5 (R5+R6 registries), #6 (R7+R9 surfaces).

**Why:** small rules, big feel — they're the difference between a package that
feels solid and one that prints errors into a progress bar. Both are trivially
contract-testable.

**Changes:** PLAN.md Part III §1 — all nine rules now carry their agreement marker.
Next per entry #2: functional anchor (first `tag: code` work), then the other test
layers as xfail spec, then the refactor.

---

## 7. 2026-07-02 — Functional anchor: the final test list (tag: theory)  [Part II §3.3]

```
  sources surveyed (18 notebooks · 4 scripts · 12 test files)

  notebooks 05/06/07 ──port──►  F1 F3 F4     real asserts already, never ran in CI
  05_regional (stub) ──write──► F2           regional GT known, nothing implemented
  notebook 02        ──port──►  F5           richest closed forms, never asserted (NEW)
  linear/gam/regional tests ──repair──► F6 F7 F8   good structure, no-op asserts (P1)
  TestExample2       ──keep──►  F9
  everything else    ──reject── scripts = benchmarks · quickstart/guides/real-examples
                                = no ground truths → tier-2 execution smoke only
```

| # | file | setup | asserts |
|---|---|---|---|
| F1 | `test_functional_conditional_interaction.py` (port 05_global + 05_heter) | `models.ConditionalInteraction`, `IndependentUniform` dim=3 [-1,1], N=1000, `Fixed(31)` | centered PDP/ALE/RHALE vs closed form (atol 1e-1, ALE jump-bin masked); PDP heterogeneity + ALE `bin_variance` vs closed form |
| F2 | same file, regional section (written fresh — 05_regional is a stub) | same model/data; `RegionalPDP/ALE/RHALE.fit(0)` | root split on x2 ≈ 0; per-region effect ≈ ±x1² centered; per-region heterogeneity ≈ 0 |
| F3 | `test_functional_general_interaction.py` (port 06) | `models.GeneralInteraction`, same recipe | PDP/ALE/RHALE vs closed form, atol 1e-1 |
| F4 | `test_functional_4_regions.py` (port 07) | `models.ConditionalInteraction4Regions`, dim=4, N=1000 | PDP/ALE/RHALE vs closed form (atol 1e-1/2e-1/1e-2); Regional finds x2-then-x3 splits → 4 leaves |
| F5 | `test_functional_correlated.py` (NEW — port notebook 02) | Gkolemis 2023 model, x3 strongly correlated with x1, analytic jac, N≈170 | notebook's `*_gt` closed forms asserted: PDP, d-PDP, ALE, RHALE, SHAP each vs its own GT — only test where methods provably differ; only closed-form SHAP check |
| F6 | `test_functional_linear.py` (repair P1) | f = x1+x2, N=100; 5 methods + jac variants + both SHAP backends | eval ≈ x, heterogeneity ≈ 0, real asserts, parametrized per method; non-SHAP fast, one tiny SHAP case in the gate |
| F7 | `test_functional_gam.py` (repair P1) | same grid, GAM model | per-method eval vs closed-form effect + derivative curves |
| F8 | `test_regional_methods.py` (repair P1) | f = 5·x0 if (x1>0 ∧ x2=0) else 0; 5 regional methods + shapiq | subregion effect ≈ 5x, heterogeneity ≈ 0; `node_idx` derived from fitted tree |
| F9 | `test_functional.py::TestExample2` (keep) | piecewise-linear, correlated x1,x2 | RHALE vs closed-form ALE, Fixed + DP binning |

**What:** the functional layer's concrete test list (the selection §3.3 left open),
chosen from a full survey of every notebook, script, and test file. Nine tests:
port the four asserting synthetic notebooks (F1, F3, F4), write the regional ground
truth fresh (F2), port notebook 02's unasserted closed forms (F5 — new vs the plan),
repair the three no-op-assert tests (F6–F8), keep TestExample2 (F9).

**Why:** the survey confirmed real ground-truth asserts exist in exactly five places
in the whole repo — notebooks 05/06/07 and TestExample2 — and none run in CI. F2
fills the biggest hole (regional methods have zero working numeric assertions);
F5 is the only ground truth where PDP/ALE/SHAP provably differ (correlation), which
is exactly what a refactor can silently break. Scripts and the remaining notebooks
carry no ground truths — smoke tier only.

**Changes:** this entry. PLAN.md §3.3 to be updated with the two deltas (F5 added;
F2 written fresh, not ported); code lands next as the first `tag: code` work.

---

## 8. 2026-07-02 — Functional tests, agreed one-by-one: 8 tests, one source of truth (tag: theory)  [Part II §3.3]

```
        effector.benchmarks.<Model><Distribution>          ← the (f, p) PAIR:
        .model  ·  .dataset  ·  .pdp_gt / .ale_gt /          a GT is a theorem about
        .rhale_gt / .heter_gt  (+ regional GT)               model AND distribution,
                    ┌───────────┴───────────┐                never the model alone
                    ▼                       ▼
     tests/test_functional_*.py      notebooks: derivation prose
     tier-1 gate, parametrized       + the SAME asserts (tier-2 execution)
```

**What:** went through LOGBOOK #7's F1–F9 one by one (Claude proposed, I decided).
Final list is **8 tests in 5 + 2 files**, and the keep-mechanism is **C**: closed-form
ground truths live once, on benchmark pair-objects named `<Model><Distribution>`
(the name states the scope of validity); tests and notebooks both consume them.
Files named by scenario; F7–F9 stay test-local (trivial GT, no notebook twin).

| # | file | benchmark object | decision |
|---|---|---|---|
| F1 | `test_functional_conditional_interaction.py` | `ConditionalInteractionUniform` | keep; trim heter N 10k→~2k |
| F2 | same file | same object | keep; write fresh; **finish 05_regional notebook** from same object |
| F3 | `test_functional_general_interaction.py` | `GeneralInteractionUniform` | keep; **add h(x2)=x2⁴/3** closed form |
| F4 | `test_functional_four_regions.py` | `ConditionalInteraction4RegionsUniform` | keep; **investigate first** (PDP assert fails on current code, 91% of points — regression or stale GT); xfail(strict) if code bug; add two-level regional assert |
| F5 | `test_functional_correlated_features.py` | `CorrelatedInteraction` | keep; asserts never ran — tune tolerances knowingly; pin SHAP seed+budget |
| F7 | `test_functional_all_methods_global.py` | — (test-local) | **absorbs F6** — linear test dropped (its DerPDP check compared eval to x, wrong; GAM file does it right); real asserts, parametrize, jac/no-jac parity, tiny fast SHAP in gate |
| F8 | `test_functional_all_methods_regional.py` | — (test-local) | keep + repair: real asserts, leaf derived from fitted tree (no `node_idx=3`), assert tree structure, ShapDP N→50 |
| F9 | `test_functional_binning.py` | — (test-local; linear is load-bearing: exact under any binning → atol 1e-2) | keep + cleanup: fix lying docstring, seed inside test (P4), N→10k, **add Greedy** → all 3 binning methods covered |

Retired: `test_functional_linear.py`, old `test_functional.py` (TestExample2 → F9,
TestBinEstimation → unit layer), old `test_functional_gam.py` / `test_regional_methods.py`
names (→ F7/F8).

**Why:** (mechanism) a `.pdp_gt` on the model class alone would be a lie — the ground
truth is a functional of (prediction function, data distribution); hanging it on the
pair-object makes it true, and it is the only option where notebook and test cannot
disagree (under plain porting, drift between the two copies is caught by nobody).
(list) F1/F2 anchor method semantics + the only heterogeneity numbers; F3 anchors
averaging; F4 anchors D=4 and multi-level splits — and already caught a real failure;
F5 is the only place methods provably differ (correlation) and the only SHAP GT;
F7/F8 are breadth (every method/backend/jac-path, global and regional); F9 is the
only tight-tolerance and only DP/Greedy binning guard.

**Changes:** PLAN.md §3.3 rewritten to this list (files, benchmark objects, fixes).
Next: the first `tag: code` work — P1–P5 + F4 investigation, then the tests.

---

## 9. 2026-07-03 — Functional anchor landed (tag: code)  [Part II §3.3; branch `tests/functional-anchor`]

```
   F4 investigation ──► B10: model bug, FIXED     benchmarks.py (4 pair-objects)
   (07 never passed)    (exp read x3, not x4)              │
                                                           ▼
   8 tests · 7 files · all green        tier 1:  106 tests   ~15 s   (was 18 / 28 s)
   + P2–P5 wiring fixed                 tier 2:  121 tests   ~2:20   (budget 10 min)
```

**What:** the agreed functional layer (#8) is code. `effector.benchmarks` holds
the four (model, distribution) pairs; F1–F9 live in 5 new + 2 repaired test
files; the old no-op/broken files are retired; P2–P5 fixed (notebooks collected
as tier-2 `test_notebooks.py` with `kernel_name=python3`, BikeSharing marked
slow, no import-time seeding anywhere, utils doctests fixed and gated).

**Found along the way (all recorded in PLAN Part III §3):**
- **B10 (fixed):** `ConditionalInteraction4Regions` read `x[:, 2]` for its exp
  term — x4 unused; notebook 07's asserts had never passed anywhere. Fixing the
  model made 07 green end-to-end for the first time.
- **B11 (open):** `ShapDP.eval` defaults `heterogeneity=True`, contradicting its
  own docstring and every sibling class — bare `eval` returns a tuple only there.
- **Regional-SHAP semantics:** `RegionalShapDP` subsets the *global* shap values
  per region (slope 5·7/12 on the F8 gate, not 5 = within-region recompute).
  F8 locks the current semantics; the regional pass decides deliberately.
- **Notebook-02 errata:** `gt_d_pdp` has a spurious `+1` (corrected in
  benchmarks); the SHAP closed form (−5/6 sin) does not match interventional
  Shapley (exact brute force + shap package agree with each other, not with it)
  — needs re-derivation, its assert is skipped until then.
- Tolerances tuned knowingly where GTs are 0 only in expectation (RHALE of
  x1-like features ~1.5e-1 at N=1000; leaf heterogeneity at bin resolution).

**Why green means something now:** every method class and both SHAP backends sit
behind numeric asserts; heterogeneity values, regional splits (incl. two-level
and categorical), and all three binning strategies each have at least one
closed-form guard. Zero xfails; one honest skip.

**Changes:** effector: `benchmarks.py` (new), `models.py` (B10), `utils.py`
(doctests), `__init__.py`; tests: 7 new/repaired files, 4 retired
(`test_functional.py`, `test_functional_linear.py`, `test_functional_gam.py`,
`test_regional_methods.py`, `notebook_execution.py`); notebook 07 tolerances;
PLAN.md (baseline table, §3 bug register + errata). Next per #2: 05_regional
notebook finish (F2's twin), then the constitution text.

---

## 10. 2026-07-03 — Notebooks rewired to benchmarks: mechanism C complete (tag: code)  [LOGBOOK #8; branch `tests/functional-anchor`]

```
   effector.benchmarks ──────────────► tests/test_functional_*.py   (tier 1, #9)
        one source of truth  └───────► notebooks 02 · 05×3 · 06 · 07 (tier 2, this entry)
                                        derivation prose + the SAME asserts
```

**What:** the notebook half of mechanism C. Notebooks 02, 05_global, 05_heter,
06, 07 now import their ground truths from `effector.benchmarks` (local GT
copies deleted); 05_regional is written in full (F2's prose twin — regional
PDP/ALE/RHALE recover split, position, and ±x1² per region); all execute green
under `test_notebooks.py`, twice, with fresh outputs stored (the outputs ARE
the docs).

**Shine fixes along the way:** kernelspec normalized to `python3` in 8 notebooks
(contributor-local `eff-env` pin); 06 got a heterogeneity section (the new
h(x1), h(x2) closed forms) and its RHALE summary corrected (was copy-pasted
from 05 — claimed x1 has zero effect); 07's ALE mask moved to the feature that
actually has the jump (x2 → x3's step), letting atol tighten 2e-1 → 1e-1, and
its stray duplicated ALE derivation cell removed; 05_heter's empty RHALE
derivation placeholders filled (per-bin variance 4c²/0/0) with a new assert —
also added to F1; notebook 02 got its first-ever asserts (mirroring F5) and an
honest note that its SHAP closed form is disputed; RHALE fits pinned to
`Fixed(31)` where bin-variance GTs assume it (default binning is not
deterministic across kernels — caught by running the suite twice).

**Changes:** 6 notebooks rewritten/executed; `benchmarks.py` +
`rhale_bin_variance_gt`; F1 + RHALE bin-variance test. Docs `.md` regeneration
deliberately untouched — that belongs to the docs-pipeline work
(docs/DOCS_PIPELINE_PLAN.md). Next: constitution → contract layer (#2 step 3–4).

---

## 11. 2026-07-03 — Docs finalized: all notebooks refreshed + signed, pipeline made explicit (tag: code)  [PLAN II §6 step 1; docs/DOCS_PIPELINE_PLAN.md; branch `docs/finalize-notebooks-and-pipeline`]

**What:** the docs-refresh step, complete. All 17 runnable notebooks
(quickstart ×3, guides ×2, synthetic ×9, real ×3) re-executed green with fresh
stored outputs, each timed; every notebook and hand-written guide page now
opens with a signature — `Author / Runtime / Description` (pages that aren't
executed drop the Runtime line). The docs build follows the four-bucket model
of DOCS_PIPELINE_PLAN.md: `docs/notebook_map.txt` declares which notebook is a
`page` (committed converted copy) or an `image` source (figures for authored
pages); `make docs-pages` / `make docs-images` replace the old blanket
`docs-update`; conversion renders saved outputs, never executes; nbconvert
moved to the lightweight `docs` dep group.

**Fixes along the way:** every hardcoded `node_idx` plot (synthetic 04,
readme_example, real 01/02/04 + the same snippet in README.md and index.md)
became a loop over the fitted tree's level-1/level-2 nodes — ids depend on the
fitted tree, which is exactly what crashed synthetic 04 (now un-skipped in
`test_notebooks.py`). Real 01's ShapDP zero-size-array crash did not reproduce
across two full re-runs — treated as fixed by the re-run + robust node lookup.
Dead docs trees deleted (Tutorials output dir, `notebooks/quickstart` copies,
stray static `.md`, `_bak`, orphan `02_regional_*` + `efficiency_comparison_*`
pages); `index.md`/README now read readme-example figures from
`static/quickstart/`; captions in `global_and_regional_effects.md` re-synced to
the fresh trees (PDP's workingday branch now splits by year, SHAP's by temp).

**Deferred:** `03_california_housing_tabpfn.ipynb` NOT re-run (TabPFN
license/CPU cost) — signed with Runtime from its stored metadata (~8 min),
still a docs page (old outputs render fine). Deal with it later.

**Next:** contract layer (PLAN II §6 step 2 / §3.1).

---
## 12. 2026-07-03 — Test safety net complete: contract + unit + plot layers, frozen (tag: code)  [PLAN II §6 steps 2–3 / §3.1–3.5; branch `tests/contract-and-unit-layers`]

**Process (agreed with Vasilis):** steps 2–3 batched and executed
autonomously; the test-by-test agreement moved to an end-of-phase summary
gate. Synthetic tests are the iteration oracle; real-data/guide/timing
notebooks run once at the very end of the refactor, not during it.

**What landed:**
- **Contract layer** — `tests/conftest.py` (the method registry: all 11
  classes constructed one way, tiny linear model N=200 for global, the F8
  gated model N=500 for regional, ShapDP via *analytic* shap values so the
  layer is SHAP-free) + `test_contract_global.py` (C1–C6 + new-surface
  xfails), `test_contract_regional.py` (RC1–RC5),
  `test_contract_registries.py` (binning/partitioner/centering menus, R9).
- **Unit layer** — `test_unit_{utils,helpers,axis_partitioning,
  space_partitioning,tree,pdp_kernels}.py`; absorbed and retired
  `test_unit.py`, `test_tree.py`, `test_space_partitioning.py`.
- **Plot-content layer** — `test_contract_plots.py` (mean line == eval,
  affine scaling, y_limits, nof_ice count, legend labels, band semantics);
  `test_plots.py` trimmed (N=200, precomputed shap): plot suite 15 s → <2 s.
- **Facade/data** — overlay-equals-eval check, jacobian-vs-finite-diff per
  model, seeded-data reproducibility.

**Provisional decisions (encoded as xfail(strict), cheap to rename — need
Vasilis's sign-off):** payload accessor = `method.payload(feature) -> dict`
with at least key `"h"` (ndarray); agnostic score = `method.heterogeneity(
feature) -> float >= 0`, centering-invariant; `eval` loses
`heterogeneity`/`return_all` kwargs (one return type, always).

**xfail ledger (36, all strict):** B11 (C1, ShapDP tuple), B1 (RC4 via a
value-level twin comparison), B2 ("dp" string), B5 ×2 (derivative scale_y;
std_err band), B6 (Fixed on single-unique-value data), R7 (regional
show_plot ×5 + eval-signature ×5), R2 new surface (payload/H/h-invariance
×15), R9 (ValueError not assert, ×4), prep_features range spec (×1).

**Empirical corrections to the B-table:** B9 is *resolved-fine* — the
vectorized finite-diff d-ICE agrees with the non-vectorized path and the
analytic jacobian (unit-tested; delete the TODO in refactor step 2.3). B3's
bad assert is dead code (every `fit()` resolves the partitioner string before
`_fit_feature`), so `"best_level_wise"` already works — RC5 is green; the fix
is deleting the assert. B6 bites only the single-unique-value case
(`min_points` violations do return False). B4 is dead compute, no visible
defect — pinned via "band only when asked".

**Freeze:** gate = 314 passed / 36 xfailed / ~14 s, identical over two runs;
slow SHAP tier 6 passed / 42 s; coverage 77% → **90%** (regional modules
18–27% → 88–99%). Tier-2 notebooks not re-run (cost discipline).

**Next:** Vasilis reviews the phase summary + provisional names → then the
Part III refactor (steps 1–7). Definition of done unchanged: functional layer
green, zero xfail markers left.

---
## 13. 2026-07-03 — Constitution amendment: the four-object heterogeneity surface (tag: theory)  [Part III §1 R2; approval gate for PLAN II §6 steps 2–3]

```mermaid
flowchart TB
    S["stored state (from fit)"]
    E["eval(feature, xs, centering) → y(xs)<br/>mean effect, ONE return type"]
    EH["eval_heter(feature, xs) → h(xs)<br/>heterogeneity curve, NO centering kwarg"]
    P["payload(feature) → dict<br/>the raw honest object"]
    HS["heter_score(feature) → float ≥ 0<br/>method-AGNOSTIC scalar"]
    V["plot layer<br/>bands/error-bars == eval_heter (R1)"]
    R["regional splitting · F2"]
    S --> E
    S --> EH
    S --> P
    S --> HS
    P -.raw for method-specific plot modes.-> V
    EH --> V
    HS --> R
```

**What:** Vasilis approved the Phase-A summary (LOGBOOK #12) and amended the
surface: alongside `eval`, `payload`, and the scalar (now named
**`heter_score`**), a fourth public object exists — **`eval_heter(feature,
xs)`**, the heterogeneity curve. Aggregation ladder: `payload` (raw) →
`eval_heter` (curve) → `heter_score` (scalar); each level has a distinct
consumer. `eval_heter` takes **no centering argument** — invariance is enforced
by the signature, not by convention. Method-specific units (PDP: var of
centered ICE; DerPDP: var of d-ICE; ALE/RHALE: per-bin variance as a step
function — the step function is the honest object, no interpolation; ShapDP:
residual spline). Regional twin: `eval_heter(feature, node_idx, xs)`.

**Why:** the plot layer is the concrete consumer that justifies the curve:
every plot draws heterogeneity vs x, and R1 (plot = thin wrapper, zero own
computation) is only achievable if the band values come from a uniform accessor
instead of per-plot method-specific reductions of the payload. `payload` stays
because curves can't replace the raw objects (ICE lines for the "ice" mode,
shap cloud for the scatter).

**Changes:** contract tests re-pointed/extended on `tests/contract-and-unit-layers`
(PR #24): `heterogeneity` → `heter_score`; new xfail C-items for `eval_heter`
(shape, non-negativity, no-centering signature, centering-invariance — the
invariance test moved off `payload["h"]`); payload item loosened to "non-empty
dict" (schema fixed at refactor steps 2–3). PLAN.md R2 rewritten; refactor step
2 now also builds base `eval_heter`/`payload`/`heter_score`, and step 4 ties
plot bands to `eval_heter`.

**Next:** the Part III refactor, step 1 (helpers/utils), one branch+PR per step.

---
## 14. 2026-07-03 — Refactor step 1: helpers/utils foundation (tag: code)  [Part III §2.1, §4 step 1; branch `refactor/step-1-helpers-utils`]

**What:** the foundations pass, exactly §2.1:
- **`helpers.prep_data`** — the axis-limits→subsample block that lived as three
  copies (GlobalEffectBase, RegionalEffectBase, FeatureEffect) is one function;
  all three constructors now call it. `data_effect` stays row-aligned through
  both steps.
- **R4: `None` for "not computed"** — `norm_const` sentinels unified
  (`EMPTY_SYMBOL`/`np.nan`/`1e8` → `None`); `EMPTY_SYMBOL` deleted from
  `helpers`; `requires_refit`'s `norm_const is None` branch is now live (B7's
  dead branch — the full B7 review completes in step 2).
- **R9 in the input normalizers** — `prep_features` (now also validates
  range), `prep_centering`, `prep_confidence_interval`, `prep_nof_instances`
  raise `ValueError`/`TypeError` instead of bare asserts.
- **R6 groundwork** — `axis_partitioning.VALID_METHODS` alias table;
  `return_default` reads it and raises `ValueError` on junk (B2's other half —
  the RHALE assert — falls in step 3).
- Dead code deleted: `prep_dale_fit_params`, `prep_ale_fit_params`,
  `utils_integrate.py` (live `mean_1d_linspace` moved into `utils` with a
  doctest). One numerical-differentiation scheme:
  `compute_jacobian_numerically` is now central-difference, `eps=1e-6`
  (the `ice_*` kernels rewire to it in step 3).

**xfails flipped green (markers removed, 51 → 46):** prep_features
out-of-range spec, R9 ValueError ×3 (centering / binning / nof_instances), R9
junk-centering-through-eval.

**Verified:** gate 323 passed / 46 xfailed / ~13 s; slow SHAP tier 6 passed /
~45 s; ruff clean.

**Next:** step 2 — `global_effect.py` base (`_eval_unnorm` kernel, base
`eval`/`eval_heter`/`payload`/`heter_score`, base fit loop + norm-const).

---
## 15. 2026-07-03 — Refactor step 2: the base class carries the lifecycle + the new surface (tag: code)  [Part III §2.2, R1–R4; branch `refactor/step-2-global-base`]

**What:** `GlobalEffectBase` now owns the shared machinery:
- **`_fit_loop`** (hoisted from ALEBase; PDP and ShapDP had inline copies) —
  one fit skeleton: prep inputs → `_fit_feature` payload → norm-const →
  `fit_args`/`is_fitted`. `fit_args` now records *all* fit kwargs uniformly.
- **`_compute_norm_const`** (generalized from ALEBase) — works on any method
  through the `_eval_unnorm` kernel; PDP overrides it (its normalization is
  per-instance: each ICE centers on its own — that is what makes h honest).
- **`_eval_unnorm(feature, xs, heterogeneity)` is the abstract kernel** —
  ALE already had it; PDP and ShapDP kernels extracted (PDP's computes ICE and
  h = var of per-instance-centered ICE / raw d-ICE; ShapDP's reads the splines).
- **The new surface (LOGBOOK #13), implemented once on the base:**
  `eval_heter` (kernel's h, no centering kwarg), `payload` (the stored fit
  dict), `heter_score` (mean of `eval_heter` on a 30-grid — same grid
  convention as regional's `points_for_mean_heterogeneity`).
- `DEFAULT_CENTERING` class attribute declared (ALE-family/ShapDP:
  `"zero_integral"`, PDP-family: `False`); signatures converge on it in step 3.
  Dead `self.avg_output` state removed. ALEBase's double `method_name` set
  removed.

**Deliberate behavior change (flagged for the notebook oracle):** ShapDP's
`zero_integral` norm-const now uses the same midpoint scheme as ALE
(`utils.mean_1d_linspace`) instead of a 30-gridpoint mean — its centered
curves shift by ~2e-3 on the contract fixture. The C2 zero-integral contract
test was re-stated semantically (dense grid, atol covering the 30-point
discretization) instead of pinning the old per-method grid accident.

**xfails flipped green (markers removed, 46 → 21):** the 25 new-surface items —
`payload` ×5, `heter_score` ×5, `eval_heter` shape/positivity ×5, no-centering
signature ×5, centering-invariance ×5.

**Verified:** gate 348 passed / 21 xfailed / ~14 s; slow SHAP tier 6 passed;
ruff clean.

**Next:** step 3 — slim the three global-method files onto the base (eval loses
`heterogeneity`/`return_all` → B11; keyword-only constructors R8; `spline_var`
rename B8).

---
## 16. 2026-07-03 — Refactor step 3: one eval, one return type — the global methods slim down (tag: code)  [Part III §2.3, R1–R3, R8–R9; branch `refactor/step-3-global-methods`]

**What:** the breaking API change LOGBOOK #4 accepted, landed:
- **`eval(feature, xs, centering=None→class default)` is written once on the
  base** and always returns the `(T,)` mean effect. The `heterogeneity` and
  PDP's `return_all` kwargs are gone (B11 with them); per-class defaults come
  from `DEFAULT_CENTERING`. The three per-method `eval` overrides are deleted.
- **Consumers re-pointed** to the new surface: regional heterogeneity
  functions use `eval_heter` (identical values); regional PDP/DerPDP build
  their ICE tables from the kernel + stored norms; regional `eval` keeps its
  public signature but is built on `eval`+`eval_heter`; facade drops the dead
  kwarg; ALE plot feeds vis through a small shim (vis redraw is step 4).
- **R8 constructors:** all five global classes are keyword-only after
  `(data, model[, model_jac])`; RHALE's params reordered to the canonical
  `data, model, model_jac, *, data_effect, ...`; three positional call sites
  in the regional files fixed (they only survived by memorizing per-class
  orders — exactly the bug class R8 kills).
- **ShapDP:** B4 fixed (`== "std" or True`); `spline_std` → `spline_var`
  (B8) with sqrt at the plot boundary; `plot` normalizes `centering` like
  every sibling; `_compute_shap_values` factored module-level (the 4×
  duplicated "code behind the scene" docstring blocks now reference it);
  junk backend raises `ValueError`.
- **(RH)ALE:** binning validation delegated to the resolver (B2: `"dp"` now
  works, junk gets one `ValueError`); the 3× duplicated "impossible to
  compute bins" assert is one `utils.raise_if_no_binning`. B9's settled TODO
  deleted.
- Functional tests re-pointed mechanically (`eval(heterogeneity=True)` →
  `eval` + `eval_heter`), values unchanged — as planned in LOGBOOK #4. The
  executed notebooks still call the old API: they are updated once, at the
  end-of-story full pass.

**xfails flipped green (markers removed, 21 → 14):** B11 (C1), the 5
eval-signature specs, B2. Remaining 14: B1, B5×2, B6, R7 regional plots ×5,
regional `eval_heter` ×5.

**Verified:** gate 355 passed / 14 xfailed / ~14 s; slow SHAP tier 6 passed;
ruff clean. B-table updated in PLAN.md (B2/B4/B7/B8/B9/B11 fixed; B3
downgraded to dead code).

**Next:** step 4 — `visualization.py` (fresh session: shared frame, wire
`is_derivative` → flips B5×2, bands from `eval_heter`), then steps 5–7.

---
## 17. 2026-07-03 — Refactor step 4: the plot layer draws, it does not compute (tag: code)  [Part III §2.4, R1+R7, B5; branch `refactor/step-4-visualization`]

**What:** `visualization.py` redrawn as a pure plot layer:
- **Shared frame extracted** — `_scale_x`/`_scale_y` (one scaling rule:
  affine for level curves, std-only for derivatives), `_feature_label`
  (fallback now **0-based**, matching API indices and
  `helpers.get_feature_names`; was `x_%d % (feature+1)`), `_add_avg_output`,
  `_decorate_ax` (xlabel/ylabel/legend/y_limits), `_finalize` (the one R7
  exit: show→`None`, else `(fig, ax)`). File shrinks 377→300 lines while
  gaining docstrings.
- **B5 fixed, both halves:** `is_derivative` wired from a new
  `PDPBase.IS_DERIVATIVE` class attribute (False; DerPDP=True) — DerPDP +
  `scale_y` no longer adds the output mean to dy/dx; `std_err` is now
  `std/sqrt(N)`, not a second std. Both xfails flipped.
- **`ale_plot` re-signatured on data, not callables:** takes the `x`/`y`
  mean curve (the caller's `eval` output — R1) plus the bin payload
  (`bin_effect`, `bin_variance`, `limits`, `dx`); the `(feature, x, het,
  centering)` callable + step-3 shim in `ALEBase.plot` are gone.
  `show_only_aggregated=True` now gets an xlabel (it had none).
- **One vocabulary:** the kwarg is `heterogeneity` in every vis function
  (`plot_pdp_ice` had `confidence_interval`, `ale_plot` had `error`); option
  names documented once in the module docstring (`True`≡`"std"`, enforced by
  `prep_confidence_interval`). `plot_pdp_ice` computes the ICE mean *after*
  scaling (one code path), clamps `nof_ice` once, in the only branch that
  uses it.
- **ALE gets the `nof_points` knob** (`ALEBase.plot(..., nof_points=1000)`)
  — same knob as PDP/ShapDP; default 1000 preserves the old hardcoded grid.

**Verified:** gate 357 passed / 12 xfailed / ~13 s; ruff clean. Slow tier:
the 5 notebook failures are the known step-3 API breakage (verified
identical on the step-3 commit via stash — zero new failures; they are
rewired at the end-of-story pass, LOGBOOK #16).

**Next:** step 5 — regional family (template `fit`, kill `locals()` idioms →
B1, registry `_create_fe_object`, uniform plots → flips R7 regional ×5 and
regional `eval_heter` ×5).

---
## 18. 2026-07-03 — Refactor step 5: one regional skeleton, explicit kwargs, the registry is born (tag: code)  [Part III §2.5, R1–R3, R5–R9, B1+B3; branch `refactor/step-5-regional`]

**What:** the regional family redrawn on the base:
- **Template-method `fit`** — `RegionalEffectBase._fit_loop(features, ccf,
  space_partitioner)`: resolve the partitioner string once (R6, top of the
  loop), then per feature `_precompute_global` (hook: ICE table / global ALE
  effects / shap values) → `_create_heterogeneity_function(feature,
  min_points)` (hook) → `_fit_feature` (now: fresh `deepcopy` of the
  partitioner, dead B3 assert deleted, bounds check is a `ValueError`).
  The five per-method `fit`s keep their public signatures + docstrings and
  end with two explicit dicts + one `_fit_loop` call.
- **B1 fixed — `locals()` idioms killed:** `kwargs_fitting` /
  `kwargs_subregion_detection` are written out per method (the `[:3]` slice
  and the `"binnning_method"` typo are gone), so regional (RH)ALE `eval`/
  `plot` now refit node objects with the user's binning method (RC4 flipped).
- **`method_registry.py` (R5, new):** the one
  `{canonical: (cls, needs_jac, uses_data_effect, display_name)}` table +
  aliases + `resolve()`. `_create_fe_object` uses it (five-way if/elif gone);
  the `global_shap_values` smell moved behind a `_extra_fe_kwargs` hook that
  only `RegionalShapDP` overrides. The facade re-points to it in step 7.
- **`eval_heter(feature, node_idx, xs)` (R2)** — the regional twin, built on
  `_fit_node_effect` (shared with `eval`/`_plot`: create node fe → fit with
  stored kwargs). The 5 xfails flipped.
- **R7 plots:** all five regional `plot`s take `show_plot=True` and return
  the underlying global plot's `(fig, ax)` when `False` (5 xfails flipped);
  explicit parameters passed as an explicit dict (no `locals()`);
  `RegionalPDP.plot`'s `heterogeneity: bool = "ice"` annotation fixed.
- **R3:** regional `eval`/`plot` `centering=None` now means the underlying
  class default (ALE-family/ShapDP `"zero_integral"`, (d-)PDP `False`) —
  was a hardcoded `True` for every method (the docstring itself warned it
  was wrong for DerPDP) and an inconsistent `False` on the PDP plots.
- **R8:** the five regional constructors are keyword-only after
  `(data, model[, model_jac])`, canonical parameter order.
- **R9:** heterogeneity-function failures `warnings.warn` instead of
  `print`; stray debug print deleted. Unified `features="all"` default
  (was required-positional in RegionalALE/RegionalShapDP);
  `RegionalShapDP.fit` gains `points_for_mean_heterogeneity` (was a
  hardcoded 30); `RegionalDerPDP.plot`'s `node_idx` is required like every
  sibling.

**xfails flipped green (markers removed, 12 → 1):** B1 (RC4), R7 regional
plots ×5, regional `eval_heter` ×5. Remaining 1: B6 (step 6).

**Verified:** gate 368 passed / 1 xfailed / ~13 s; slow SHAP tier 7 passed;
notebooks: same 5 known step-3-API failures as the pre-step baseline, the
4 regional/real-example ones still pass; ruff clean. B-table: B1, B3 fixed.

**Next:** step 6 — partitioning (`axis_partitioning.py` registries +
`Fixed` control flow → B6; Best/BestLevelWise dedup in
`space_partitioning.py`).

---
## 19. 2026-07-03 — Refactor step 6: partitioning — zero xfails, the net is all green (tag: code)  [Part III §2.6–2.7, R6+R9, B6; branch `refactor/step-6-partitioning`]

**What:**
- **`axis_partitioning.py` (§2.6):** B6 fixed — `Fixed.find_limits`
  restructured into early returns like Greedy/DP, so the
  `_none_valid_binning` verdict is no longer overwritten by the linspace
  (single-unique-value data now returns `False`, xfail flipped). `Base.find`
  (never called, returned `NotImplementedError` instead of raising) replaced
  by an abstract `find_limits` with the real contract; `Base.__init__`
  docstring rewritten (described a signature from another life);
  `min_points_per_bin` float defaults (`2.0`/`0.0`) are ints; the 3×
  commented `_is_categorical` blocks and unused `_cat_limit` locals deleted;
  `Base.plot` labels 0-based; `effector.binning_methods.*` docstring
  leftovers renamed to `axis_partitioning`.
- **`space_partitioning.py` (§2.7):** `Best` and `BestLevelWise` deduped —
  the shared constructor (params + the long docstring, once) and the shared
  exhaustive split search `_evaluate_splits(active_indices_list)` hoisted to
  `Base`; the two classes differ only in recursion strategy (node-wise
  `_recursive_split` vs level-wise `_search_all_splits`). `Best` no longer
  names itself `"cart"` (R6: the registry name is `"best"`). The `"all"`
  normalization of `candidate_conditioning_features` moved to
  `helpers.prep_conditioning_features` (next to `prep_features`); `compile`'s
  default is now `"all"` instead of `None`. `_splits_to_tree`'s
  set-to-`None` "hack to check usage" removed; inline ==/</!=/> chain
  replaced by `_get_comparison_symbol`; dead commented code deleted.

**xfails flipped green (markers removed, 1 → 0):** B6. **The contract/unit
net has zero xfails left — every B-bug (B1–B11) is fixed or resolved, and
all constitution rules R1–R9 hold on the surfaces the tests pin.**

**Verified:** gate 369 passed / 0 xfailed / ~14 s; slow tier: SHAP 6 + the
4 regional/real notebooks pass, same 5 known step-3-API notebook failures
as the baseline (end-of-story pass pending); ruff clean. B-table: B6 fixed.

**Next:** step 7 — facade + cleanup (facade onto `method_registry` +
`eval`; delete `interaction.py`; fold `utils_integrate`; datasets seed),
then the end-of-story full pass (notebooks rewired to the new API).

---
## 20. 2026-07-03 — Refactor step 7: facade on the registry, dead weight overboard — steps 1–7 complete (tag: code)  [Part III §2.8–2.10, §4; branch `refactor/step-7-facade-cleanup`]

**What:** the last application-order step:
- **Facade (§2.8):** `FeatureEffect` re-pointed to `method_registry` (R5) —
  its private `_REGISTRY`/`_ALIASES`/`_DISPLAY` deleted; only a `_POOL`
  list remains (which registry methods are comparable — DerPDP excluded,
  derivative units). The registry grew a `canonical(name)` helper. The
  facade gained **`eval(feature, xs, methods, centering)` →
  `{display_name: y}`** — the promised "grow eval for free"; `plot` is now
  a thin wrapper over it (shared grid → `eval` → one `vis.*` call, R1).
- **Dead modules (§2.9):** `interaction.py` deleted (293 fully-commented
  lines importing functions that no longer exist); `tree.py`'s 50-line
  commented `DataTransformer` deleted. (`utils_integrate.py` was already
  folded into `utils.py` in step 1.)
- **`datasets.py` (§2.10):** `RealDatasetBase.split` takes `seed=21`
  (train/test splits reproducible between runs; was unseeded);
  `standarize` → `standardize`; `np.array` → `np.ndarray` annotations;
  `BikeSharing.postprocess`'s magic constants documented (UCI id=275
  normalizations — temp `(t+8)/47`, hum `/100`, windspeed `/67`; indices
  after dropping dteday/atemp: 8/9/10).
  **Deliberate deviation:** `IndependentUniform.generate_data`'s "pointless"
  extra shuffle is *kept* — removing it changes every seeded data stream,
  which would invalidate the executed notebooks as the end-of-story
  regression oracle. Revisit after the notebook pass if it still bothers.
- **The constitution is now user-facing:** R1–R9 copied (as-built) to
  `docs/design.md`; CONTRIBUTING.md links it as the contract new features
  must follow (PLAN III §4 closing requirement).

**Verified:** gate 369 passed / 0 xfailed / ~13 s; slow tier: SHAP 6 + 4
notebooks pass, same 5 known step-3-API notebook failures; ruff clean.

**Steps 1–7 of PLAN III §4 are done: zero xfails, B1–B11 all closed, R1–R9
hold everywhere the net reaches.**

**Next (separate story):** the end-of-story full pass — rewire the 5
old-API notebooks (`eval(heterogeneity=...)` → `eval`+`eval_heter`,
`spline_std` → `spline_var`), re-execute all of them against the new API as
the regression oracle, re-sign, and merge the step 1–7 branch chain.

---
## 21. 2026-07-04 — End-of-story pass: notebooks rewired + re-executed, docs refreshed, the homogenization is done (tag: code)  [PLAN III §4 close-out; branch `refactor/end-of-story-notebooks`]

**What:** the regression-oracle pass that closes the refactor.
- **Old-API call sites rewired** (the two breaking changes steps 3+5
  introduced):
  - `eval(..., heterogeneity=True)` → `eval` (mean) + `eval_heter` (curve)
    everywhere it returned a tuple — 4 synthetic notebooks, the two
    efficiency guides' `measure_time` benchmarks, and `simple_api.md`'s 5
    global snippets + `api_global.md`.
  - **R8 positional constructors** (the failure the pre-pass baseline missed
    because it crashed before the eval cells): `effector.PDP(x, model,
    axis_limits)` → `axis_limits=...` across 4 synthetic notebooks. Only
    the constructor's 3rd positional arg was affected (everything after
    `model_jac` is keyword-only now).
- **All 17 runnable notebooks re-executed in-place against the new API**,
  fresh outputs, each re-signed with a measured Runtime (quickstart ×3,
  guides ×2, synthetic ×9, real ×3-minus-TabPFN). The synthetic oracle's
  internal closed-form asserts all pass (02's "all closed-form checks
  passed", 05_heter's PDP-heterogeneity `assert_allclose`, 06's, 07's).
  `test_notebooks.py` tier: **9 passed** (was 5 failing).
- **Regression check:** diffed every re-executed notebook's printed outputs
  against the pre-pass version — only expected drift (the ShapDP centering
  shift noted in LOGBOOK #15; tqdm rates; regional trees re-fit on the
  seeded splits). Two guide benchmarks (`efficiency_global/regional`) had a
  cell hang for hours on the old-API path before this pass; both now run in
  ~11 / ~7 min.
- **Docs refreshed:** `make docs-images` + `make docs-pages` re-harvested
  every figure and re-converted every committed page; `make docs-build`
  clean (only pre-existing griffe docstring warnings). Hand-authored pages
  re-synced to the fresh trees — `global_and_regional_effects.md` (3 tree
  summaries), `index.md` + `README.md` (readme-example tree: the second
  workingday split is now `yr`, not `temp`; prose + captions updated),
  `simple_api.md`/`api_*.md` eval snippets. One renamed bike figure
  (`_34_494` → `_34_489`) re-pointed.
- **TabPFN notebook 03** still deferred (license/CPU) — its converted page
  re-converts unchanged saved outputs, as before.

**Deferred, unchanged:** notebook-02 SHAP closed form (math, Vasilis's call);
notebook 03 TabPFN execution.

**Verified:** gate 369 passed / 0 xfailed / ~15 s; ruff clean; docs build
clean; all 17 executed notebooks green.

**Done.** This closes PLAN Part III (homogenization). The step 1–7 branch
chain + this pass are one stacked history off `main`; the single final PR
opens from here for Vasilis's end-to-end verdict.

---
## 22. 2026-07-04 — Reproducibility: `random_state` on every constructor (tag: code)  [PLAN III §6.1 close-out; branch `reproducibility-random-state`]

**Decision (Vasilis):** default is a **fixed seed, `random_state=21`**
(matching `datasets.py`) — explanations reproducible out of the box;
`None` opts into fresh randomness.

**What:**
- `helpers.prep_nof_instances`/`prep_data` take `random_state` and draw via
  `np.random.default_rng` — the unseeded global `np.random.choice` is gone;
  no effect-class code touches global `np.random` state anymore
  (datasets.IndependentUniform keeps its legacy seeded global draw — LOGBOOK #20).
- Keyword-only `random_state=21` on all 10 public constructors + the
  `FeatureEffect` facade, appended last on the two base `__init__`s (their
  positional call shapes untouched), stored as `self.random_state` and
  forwarded to every internally built object (regional `_precompute_global`,
  node fe objects, facade `_get_method`).
- Seed also reaches plot-time sampling (`vis.plot_pdp_ice` ICE selection,
  `ShapDP.plot` `nof_shap_values`) and the shap backends: `seed=` (shap) /
  `random_state=` (shapiq) default to the constructor seed, user
  `shap_explainer_kwargs` still override.
- New contract layer `tests/test_contract_determinism.py` (D1–D7, 28 tests):
  same seed → identical data/eval/eval_heter/regional tree + node evals/ICE
  plot lines; different seeds differ; `None` works; global `np.random` state
  untouched. `regional_shapdp` D5 runs two *real* shap fits with no explicit
  seed kwargs — the end-to-end proof the constructor seed reaches the backend.
- Bookkeeping: R8 in `docs/design.md` now states reproducibility as contract;
  CHANGELOG unreleased entry; PLAN §6.1 ticked (datasets half was already
  done in Part III §2.10).

**Noted, out of scope:** conftest's `make_global("shapdp", ...)` setdefaults
`shap_values` computed on the *full* data — row-misaligned when the
constructor subsamples (latent, pre-existing); the determinism tests
recompute analytic shap values on the kept rows instead.

**Verified:** gate 397 passed / 0 xfailed / ~15 s (369 + 28 new); ruff
check + format clean.

---

## 23. 2026-07-04 — Input layer decided: pandas ingestion, one `schema` argument, three-way feature types (tag: theory)  [PLAN III §6.4 (new), §6.4a revised; docs/design.md R10; docs/method_semantics.md]

Planning session for "categorical features end to end". Agreed the work is two
sequential stages — the input layer first, method behavior on non-continuous
features second — and pinned the decisions:

- **Input types:** numpy 2-D + pandas DataFrame, converted to numpy at the door;
  pandas stays optional (lazy detection via `sys.modules`, numpy path never
  imports it). No narwhals/polars — convert-at-the-door numeric core.
- **One metadata argument (breaking, pre-1.0):** `schema=` (`effector.Schema`
  dataclass or plain dict) holds `feature_names, feature_types, cat_limit,
  target_name, scale_x_list, scale_y`; the separate `feature_names=`/
  `target_name=`/`feature_types=`/`cat_limit=` kwargs are removed everywhere.
  Define-or-infer: explicit schema field > DataFrame dtype inference > numpy
  heuristic; heuristic-decided types (int-with-few-uniques) emit one
  `UserWarning` nudging an explicit declaration — inference is a fallback, not
  trusted.
- **Three-way taxonomy** `continuous / ordinal / nominal` (aliases cont→continuous,
  cat→nominal). Chosen over sklearn's two-way because in effector the distinction
  is algorithmically real: DerPDP → continuous only; RHALE → continuous + ordinal
  (discrete derivative + adaptive level grouping), nominal rejected; PDP/ALE/ShapDP
  → all three, ALE-nominal defaulting to ascending encoded order with a documented
  order-dependence caveat and `order=[...]` / `order="similarity"` overrides.
  Nominal is never inferred from numpy input.
- **Model-call rule:** DataFrame in → model always called with a reconstructed
  DataFrame (original names/dtypes, codes decoded); escape hatch
  `lambda X: f(X.to_numpy())`.
- **Scaling** moves to construction (schema) with plot-time override
  (plot dict > schema > None; `False` disables) — closes the F5 item.
- **Exactness contract:** `docs/method_semantics.md` states the
  eval/eval_heter/heter_score/plot formulas per method × feature type; it is the
  acceptance spec Stage B will be reviewed against.
- **Signature harmonization rides the same break:** one `nof_instances` rule
  (10k, SHAP classes 1k), unified plot defaults, one `heterogeneity` vocabulary,
  ALE plots at bin edges, SHAP config on the constructor, keyword-only `fit`
  with one canonical order. Deliberately untouched: per-method DEFAULT_CENTERING
  divergence, regional `eval`'s bool heterogeneity return shape.

Stage B (ordinal/nominal kernels, regional-on-cat, similarity ordering) gets its
own planning round after the input layer merges; deferred there: rare-level
pooling vs K-cap, PR split, `requires_refit` × `order=`.

---

## 24. 2026-07-04 — Input layer + categorical FOI shipped end to end (tag: code)  [PLAN III §6.4 + §6.4a; branch `feat/input-layer-schema`]

Implementation of LOGBOOK #23, both stages on one branch:

- **Stage A — input layer:** `effector/ingestion.py` (R10: `Schema`, dtype
  table, numpy heuristics + `UserWarning` nudge, DataFrame encoding, the
  model-call rule with `_make_frame_builder`, one validation point);
  `schema=` threaded through all 11 classes + facade (breaking — the old
  metadata kwargs are gone); signature harmonization (one `nof_instances`
  rule, unified plot defaults, one heterogeneity vocabulary, SHAP config on
  the constructor, keyword-only `fit`); scale-at-construction with
  plot-kwarg override (`False` disables). Tree display fixed for sparse
  `scale_x_list`.
- **Stage B — categorical FOI:** kernels per `docs/method_semantics.md`
  (PDP ICE-at-levels + freq-weighted centering; ALE two-sided adjacent-level
  transitions in code space, exact at levels; RHALE discrete derivative +
  Greedy/DP adaptive level grouping via `axis_partitioning.
  adapt_for_categorical`; ShapDP per-level step lookup; DerPDP and
  RHALE-nominal raise with the capability matrix mirrored into the R5
  registry). Frequency-weighted `heter_score`/centering. Categorical plot
  layer (`plot_categorical_effect`, `plot_pdp_ice_categorical`,
  `plot_shap_categorical` — bars, whiskers, seeded jitter, label ticks from
  DataFrame encodings). Regional-on-cat: `search_partitions_when_categorical`
  flipped to True, level-aware heterogeneity closures (freq-weighted within
  the candidate region; ALE re-bins masked contributions through
  `instance_idx`). Nominal `order=`: explicit list or `"similarity"`
  (`effector/ordering.py`, scipy-only KS + classical-MDS seriation);
  `zero_start` honors the fit order.
- **Deviations from the plan, decided solo:** `Fixed.min_points_per_bin`
  stays 0 (aligning to 2 broke default ALE on small N — UX regression);
  the heuristic warning fires only for int→ordinal (int→continuous is the
  expected reading); PDP/ShapDP display-only `order="effect"` skipped
  (small win, more API); rare-level pooling deferred (no K-cap guard added
  either — revisit when a real high-cardinality case appears).
- **Tests:** +36 unit (ingestion), +28 contract (R10 parity/metadata/scale),
  +22 functional categorical closed-form GT (`models.ConditionalCategorical`),
  +10 regional-categorical GT. Notebooks migrated to the schema API and
  re-executed; new `08_categorical_features.ipynb` is the categorical
  walkthrough.

---

## 25. 2026-07-06 — House theme: one palette, the red clouds go quiet (tag: code)  [PLAN IV / VISION A9 "do-first"; branch `feat/house-theme`]

The plots were dressed by ~20 hardcoded literals scattered across 7 functions —
`"b-"` here, `color="red"` there, no palette, no rcParams. A9 is the cheapest big
win: one designed, colorblind-safe look, ported from the validated
`scripts/vision_sketches/style.py` the mockups already run on.

**The one decision — two layers, applied differently:**

```
              import effector            effector.set_theme("light")
                    │                             │
  palette COLORS ───┼─ always on ────────────────┼─► red clouds → quiet gray,
                    │  (per-artist)               │   mean line = palette blue
  chrome RCPARAMS ──┘  (rcParams untouched)       └─► background / font / grid / spines
```

Colors are the package's identity → they ship **by default**, baked into every
draw call through semantic tokens (`MEAN`, `BAND`, `CLOUD`, `BAR_FACE`…), never
touching global state. Chrome (fonts/grid/bg) is a global rcParams push, so it
only fires when the user **opts in** with `set_theme()`. Why not wrap each plot in
one `rc_context`? matplotlib re-reads grid/tick rcParams at *draw* time — after we
have handed back `(fig, ax)` — so a context wrapper reverts at render.
Global-on-opt-in is the only thing that survives (the seaborn model). Tokens
resolve from the active theme, so `set_theme("dark")` flips colors and background
as one.

**What lands:**
- `effector/theme.py` — the ported palette, a frozen `Theme`, three instances
  (`light`/`dark`/`paper`), `set_theme(name)` + a `"default"` reset.
- `visualization.py` — the ~20 literals become tokens; `"b-"`/`"rx"` fmt-strings
  unpacked; ICE/shap clouds go gray with an explicit thin linewidth (the 2.0
  rcParam would otherwise fatten them).
- `set_theme` exported on `effector`; the `axis_partitioning.py` `"bo"` straggler
  token-swapped (its pyplot-state design left as flagged debt).
- `tests/test_theme.py` — default tokens apply, heterogeneity is **gray-not-red**,
  no rcParams leak without opt-in, `set_theme` switches + resets.

**The tail:** default colors change → every committed effect-plot doc figure
shifts. Re-running notebooks + regenerating docs is the last, expensive step,
gated behind a visual color review (the alpha values are the one taste knob).

**Out of scope (later waves):** per-plot `title/figsize/ax` (B4), the A6
categorical dot-interval redesign, A5 stable per-method colors.

---

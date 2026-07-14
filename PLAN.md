# effector — the plan

One living document: how we work (Part I), what is done (Parts II–III, stubs),
what comes next (Part IV, the ordered backlog), and the parking lot (Part V).
The trail of every decision is `LOGBOOK.md`; the destination is
`EFFECTOR_VISION.md`; the rules of the codebase are `docs/design.md` (R1–R14).
History that used to live here in full is in the LOGBOOK and git history.

---

# Part I — Workflow

**Git & releases (GitHub Flow).**
- `main` is protected and always releasable: 1 approving review + green
  required checks; the admin merges his own PRs with `gh pr merge N --admin`.
- Short-lived `feat/fix/chore/docs/refactor` branches off `main` → PR → CI
  green → squash-merge → delete. No integration branch.
- SemVer, ship-early. The stable-API milestone that ends this effort is
  **`1.0.0`**. Release mechanics: bump `pyproject.toml` + `uv.lock` +
  `CHANGELOG.md` (cut `[Unreleased]` → `[X.Y.Z] - date`) in a PR, merge, then
  tag `vX.Y.Z` on main → `publish_to_pypi.yml` (gated behind the test suite)
  publishes to PyPI and auto-creates the GitHub Release.
- ⚠️ Renaming a CI job breaks the exact-match required-status-check names —
  update branch protection via `gh api` whenever `run_tests.yml`/`lint.yml`
  job names change.
- ⚠️ Stacked PRs: verify each merge actually reaches `main` (they once merged
  into their base branches only; recovered via cherry-pick PR #44).

**Test tiers.**
- Fast gate: `make test` (~15 s) — contract + unit + functional layers,
  every PR.
- Slow non-notebook: `pytest tests/ -m slow --ignore tests/test_notebooks.py`.
- Notebooks: `pytest tests/test_notebooks.py -m slow` (~3 min; executes the
  synthetic-examples notebooks end-to-end).
- `tests/conftest.py`'s method registry is the single construction path for
  all effect classes — extend it, don't bypass it. `tests/toy_method.py` is
  the R14 reference implementation.

**The LOGBOOK protocol.** Nothing lands in code before it has a LOGBOOK
entry. Loop per item: name the next item from this plan → discuss until
convinced (simple explanations, concrete examples, diagrams) → the USER
writes the entry (Claude drafts, user restyles; every entry opens with a
diagram and carries `tag: theory|code`) → then code on a short-lived branch →
PR.

**Docs pipeline.** See `docs/README.md`: authored pages + static images
(`make docs-images`), selected notebooks → committed converted pages
(`make docs-pages`, mapping in `docs/notebook_map.txt`), `make docs-serve` /
`docs-build`. Conversion renders saved outputs, never executes.

---

# Part II — Testing plan ✅ DONE

The three-layer safety net (contract / unit / functional, two runtime tiers)
was designed and landed 2026-07-02/03 — **LOGBOOK #1–#12**. Closed-form
ground truths live in `effector.benchmarks` and are shared by the functional
tests and the executed notebooks (regression oracles). The gate has stayed
green through every refactor since; extend it via `tests/conftest.py`.

# Part III — Homogenization & the new API ✅ DONE

The 2026-07 refactor wave, in LOGBOOK order:

- **R1–R9 constitution + 7-step refactor** (one `eval`, heterogeneity ladder,
  registries, plot contract) — **LOGBOOK #13–#21**.
- **Reproducibility** (`random_state` contractual) — **#22**.
- **Input layer** (numpy-only door, `schema=`, `from_dataframe`, three-way
  feature types, categorical FOI end-to-end) — **#23–#24**; v0.3.0 released.
- **House theme** (`set_theme`) — **#25**.
- **Regional ≡ masked global** (R11) — **#26**.
- **Regional\* deleted; `find_regions` → `Partition`** (values-not-state,
  R12), **importance** (R13), **one-click `explain` → `Report`** — **#27–#29**.
- **Rules algebra + rule-primary Partition + finder/proposer seams** — **#30**
  (branch `feat/rules-algebra`).
- **Two-block lifecycle** (R14: frame-gated local effects + epoch-keyed
  summaries memo; kernel cont/cat dispatch; model-call budgets
  contract-tested) — R14 in `docs/design.md`; efficiency guides rewritten.
- **The API shell** (adapters, feature names everywhere, `rule=` sugar,
  plural `find_regions`, `compare`, `plot_triage`, mental-model docs) — **#31**.

---

# Part IV — Next steps (the ordered backlog)

## 1. Land the open PR chain ← FIRST, blocks everything

`#49 → #50 → #52 → #53 → (open a PR for feat/rules-algebra) → #54 → #55 →
chore/cleanup-and-docs`. Review/merge bottom-up, retargeting the next PR on
each merge; after the last merge, confirm the commits are on `main` (Part I
warning). Then delete merged branches and prune the ~20 stale remotes.

## 2. F2a — the interaction vector (the only new math this quarter)

*Why now:* the pipeline answers "which features matter" (importance) and
"where is the model inconsistent" (heterogeneity) but not **why** — the D×1
interaction vector closes that loop and routes
`candidate_conditioning_features` instead of guessing.

- **Heterogeneity index (all methods):** normalized `heter_score` (e.g. by
  `std[f(X)]`, the H-statistic normalization) — zero iff no interaction; it is
  exactly the quantity `find_regions` minimizes, surfaced as a number.
  *(Half done by the units contract: `heter_score` is now a std-type quantity
  in output units, so the index is just `heter_score / std(f(X))`.)*
- **H-statistic `H²_j` (PDP only):** needs `pd_{-j}` (PDP of all-but-j) — one
  new kernel on the existing ICE machinery.
- Module `effector/interaction.py`; ground truths from
  `ConditionalInteraction` (closed-form) and `f = x1·x2 + x3` (analytic).
- ~1 day. Pairs with `plot_triage` for the two-axis feature summary.

## 3. First-user hardening (adoption beats features)

- **Classification story:** `classifier_proba` exists; document and test the
  full convention — class selection, probability vs logit units, one
  explanation per class — in `explain` and the manual.
- **Big-N posture:** sensible subsampling defaults / warnings so a first
  `explain()` on 500k rows doesn't hang; document the cost model (the
  efficiency guides count model calls — surface the summary in the manual).
- **Trust layer, minimum viable:** extrapolation guard (fade/mask effect
  segments outside the data envelope; rug plots by default), low-evidence
  flags for rare categorical levels and wide ALE bins. (Vision B8.)

## 4. Phase D — the interactive layer (the second half of the end goal)

Spec harvested from the executed find_regions plan:

- `ipywidgets` as an optional extra (`pip install effector[interactive]`),
  imported lazily; spike first, fall back to plotly-Dash/marimo only if
  ipywidgets feels too limited (record the decision in the LOGBOOK).
- `effector.interact(effect)` (`effector/interactive.py`): feature dropdown,
  centering + heterogeneity-mode dropdowns, a **mask builder** (range slider
  per continuous / multiselect per categorical conditioning feature →
  conjunction mask), and a "find regions" button whose leaf dropdown sets the
  mask to `partition.mask(i)` — auto and manual masking share one code path.
  On any change: `effect.plot(feature, mask=..., show_plot=False)` into an
  `Output` widget; the summaries memo makes slider-back-to-a-prior-state
  instant.
- **State discipline (the whole point):** widget state lives only in the
  widget layer; the estimator stays a pure query surface. If latency is poor,
  raise the memo cap or lower `nof_instances` — never add per-widget caches
  to the effect.
- Tests headless (skip without ipywidgets; assert values-not-state: no new
  public attribute on the effect). Docs: `quickstart/interactive.md` + GIF.

## 5. Bigger methodology (each its own branch + LOGBOOK entry)

- **F2b — D×D interaction matrix.** Two routes: (i) *ALE-native, near-free*:
  the finder's candidate-split scan already computes `heter_func(mask)` for
  every (conditioning feature, split) — the weighted-heterogeneity matrix it
  builds IS the interaction matrix, currently discarded except the argmin;
  expose it (`effector.interaction.matrix`). Consistent by construction with
  what `find_regions` splits on. (ii) *PDP H-statistic route*: needs 2-D PDP
  (then reusable for 2-D effect plots).
- **F4 — GADGET-style joint splitting.** ONE partition minimizing aggregated
  heterogeneity over a feature set S: `joint_heter(mask) = Σ w_j·heter_j(mask)`
  fed into the existing finder seam. Needs precompute-all-then-partition flow
  + a shared-partition result; API = a finder argument, not a new class;
  default `candidate_conditioning_features = all∖S`. Test on
  `ConditionalInteraction4Regions`; flagship demo: REPID (D trees) vs GADGET
  (1 tree). Shares its diagnostics with F2b route (i).
- **F3 — CALM / x-by-design.** Decided: separate package consuming effector's
  seams — and the seams now all exist (Partition-as-data, finder API,
  reproducibility). Remaining: a docs tutorial ("interpretability by design
  with effector + CALM") + CALM-side refresh in its own repo.

## 6. Release

After (1)–(3): version bump → tag. If the API has settled — and R1–R14 +
the shell is the settled API — this is the **`1.0.0`** that ends "road to
v3". Update README GIF/story, the arxiv/JOSS text ("v0.2 → full reformat"),
and announce.

---

# Part V — Parking lot

Small items and research probes, one line each. Promote to Part IV via a
LOGBOOK entry.

**Viz / UX (vision doc A-items still open):**
- A2 overview grid: ranked small multiples on a SHARED centered y-axis
  (importance visible as amplitude), band on every curve.
- A3 visual partition tree (`plot_tree`): node = mini effect plot, edges =
  split condition + share, heterogeneity chip — the screenshot figure.
- A5 `compare` upgrade: disagreement strip + likely-cause annotation
  (correlated features → PDP extrapolation → prefer ALE).
- A6 categorical plots redesign: dot-interval marks, per-level ICE cloud,
  evidence margin (n per level), low-evidence hollow markers.
- A8 explain-one-row (`locate`): instance position on each curve +
  prediction-anatomy bar; residual = interaction share.
- FeatureEffect ±std bands (scalars are DONE — heter_score/importance are in
  output units since the units contract; what remains is the facade's ±std
  *band* display); global-vs-regional overlay; multi-feature grid
  `plot(features=[...])` (needs an `ax=` orchestration layer); 2-D effect
  plots (PDP first, shares the F2b kernel).
- Units/denormalization as a first-class concept (today: hand-built
  `scale_x_list` dicts) — needs design. 🔴

**API (vision doc B-items still open):**
- B6 effects-as-data: `to_dataframe(feature)` tidy export; `save`/`load` of
  fitted effects (fitting is the expensive part; cannot persist today).
- B7 `method="auto"` + presets `fast|balanced|exact` (encode the README
  decision table; say *why* in the caption).

**Engine / methods:**
- DerPDP explained-variance ledger: integrate the derivative curves to
  output scale (∂f/∂x → cumulative effect, RHALE-style accumulation) so the
  surrogate R² / decision sequence applies to it too — today DerPDP is the
  one method whose report has no ledger/§3 (derivative units, sums of
  curves don't approximate f̂). Agreed 2026-07-13: leave as is for now.
- Binners: **ChangePoint/CUSUM first** (edges where the running mean of the
  effect shifts; O(N); continuous-only), CART-split (top-down), curve-simplify
  (Douglas–Peucker on the cumulative curve); give Agglomerative an
  O(N + K log K) heap (matters only for large K).
- Research probe: PDP is implicitly fixed-grid binning — try adaptive grids
  (quantile/agglomerative over the axis) to sharpen curves / cut T×N calls.
- Research probe: does SHAP-DP need bins at all, or spline the raw (x, φ)
  scatter directly?
- RHALE 2-level-ordinal + DP-binning failure (hit in the regional guide,
  sidestepped with a nominal schema — pre-existing).
- `FeatureEffect` calls the model for `show_avg_output` — could reuse the
  `_y_pred` cache.
- ShapDP `shapiq` backend: `IndexError` in `_fit_feature` (pre-existing) —
  verify & fix.
- nan-poisoning accept-test wart in the partitioner; `space_partitioning`
  finder-constructor arg clump.

**Docs / repo hygiene:**
- Seed synthetic notebooks 03 + 09 (unseeded → doc pages not byte-comparable
  across re-executions).
- Re-execute the tabpfn notebook when a `TABPFN_TOKEN` is available (page
  renders from saved outputs meanwhile).
- Docs guidance: ordinal codes may want scaling for the *model* while the
  schema keeps them categorical for the *explanation* (the worse-model trap
  with raw codes into an MLP).
- `py.typed` marker + drop the stale `[tool.setuptools]` block in
  `pyproject.toml`; prune stale remote branches (with #1).

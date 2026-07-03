# Docs pipeline refactor — plan for another Claude session

> Goal: make the documentation build follow one simple, explicit model with four
> clearly separated buckets, an explicit notebook→page mapping, and automated
> targets — **without breaking any existing links**.
>
> This plan was written after auditing the repo. Every "safe to delete" claim
> below was verified with grep against `docs/` and `.github/`. Re-run the
> verification commands before deleting anything; if a command returns matches
> that contradict this plan, STOP and re-audit.

---

## The four buckets (target model)

| Bucket | What | Where it lives | How it is updated |
|---|---|---|---|
| **(a)** Authored pages + static images | Hand-written mkdocs/Material markdown; images are static files, refreshed from notebooks only on demand | Pages: `docs/docs/index.md`, `docs/docs/quickstart/*.md`, `docs/docs/api_docs/*.md`, nav landing pages (`guides.md`, `examples.md`, `quickstart.md`). Images: `docs/docs/static/<group>/<nb>_files/` | Edit `.md` by hand. Refresh images with `make docs-images` when you choose. |
| **(b)** Notebooks you don't care about | Any notebook not in the mapping file | `notebooks/…` only | Nothing — ignored automatically |
| **(c)** Selected notebooks → doc pages | nbconvert output (saved outputs, no execution), committed | `docs/docs/notebooks/<group>/<nb>.md` + `<nb>_files/` | `make docs-pages` (reads the mapping file) |
| **(d)** Standalone scripts | Benchmark/paper figures, unrelated to docs | `scripts/` | Untouched; not part of docs |

Decisions already made by the maintainer:
- **Convert saved outputs, do NOT execute** notebooks at build time.
- **Commit** the generated converted-notebook markdown.

---

## Current state audit (facts, verified)

**LIVE — must not break:**
- `docs/docs/index.md` reads readme-example images from
  `./notebooks/quickstart/readme_example_files/…` (7 references: lines ~108, 179, 180, 206, 207, 210, 211).
- `docs/docs/quickstart/simple_api.md` and `flexible_api.md` read images from
  `../static/quickstart/{simple_api_files,flexible_api_files}/`.
- `docs/docs/quickstart/global_and_regional_effects.md` reads images from
  `../static/real-examples/01_bike_sharing_dataset_files/`.
- `docs/docs/api_docs/api_global.md` and `api_regional.md` read images from
  `../static/quickstart/simple_api_files/`.
- Nav pages link to converted pages under `docs/docs/notebooks/{guides,synthetic-examples,real-examples}/`
  (see `guides.md`, `examples.md`, and `index.md` lines 21).
- CI (`/.github/workflows/publish_documentation.yml`) only runs
  `mkdocs build -f docs/mkdocs.yml` (+ copies CHANGELOG/CONTRIBUTING). It does
  **not** run nbconvert. So the committed converted `.md` must remain in git.

**DEAD — safe to delete (verified no references):**
- `docs/docs/Tutorials/` — only *written* by the current `make docs-update`; no page reads it.
- `docs/docs/static/**/*.md` — the stray `.md` files inside `static/quickstart/` and
  `static/real-examples/`. Only their sibling `<nb>_files/` dirs are referenced, never the `.md`.
- `docs/docs/static/real-examples/01_bike_sharing_dataset_files_bak/` — backup copy.
- `docs/docs/notebooks/synthetic-examples/02_regional_pdp.md`,
  `02_regional_rhale.md`, `02_regional_shapdp.md` — orphans: no source notebook, not in nav.
- `notebooks/synthetic-examples/.ipynb_checkpoints/` — Jupyter cruft.

**Inconsistencies to fix:**
- `make docs-update` writes to `docs/docs/Tutorials/` but the nav reads
  `docs/docs/notebooks/`. It also globs all notebooks and pulls the heavy
  `tutorials`+`shap` extras.
- `(a)` images come from two different locations: most from `static/`, but
  `index.md`'s readme-example images come straight from `notebooks/quickstart/`.
  Standardize on `static/`.
- The `docs` dependency group has no `nbconvert`/`jupyter`; conversion currently
  only works via the `tutorials` extra. Since we convert saved outputs (no
  execution), add lightweight `nbconvert` to the `docs` group instead.

---

## Selected notebooks (the mapping)

Create `docs/notebook_map.txt`. Format: `kind  source_notebook  dest_subdir`.
`kind` is `page` (bucket c → `docs/docs/notebooks/<dest>`) or `image`
(bucket a → copy `<nb>_files/` into `docs/docs/static/<dest>`).

```
# --- PAGES (bucket c): converted notebooks shown as documentation pages ---
page  notebooks/guides/efficiency_global.ipynb                                   guides
page  notebooks/guides/efficiency_regional.ipynb                                 guides
page  notebooks/synthetic-examples/01_linear_model.ipynb                         synthetic-examples
page  notebooks/synthetic-examples/02_global_effect_methods_comparison.ipynb     synthetic-examples
page  notebooks/synthetic-examples/03_regional_effects_synthetic_f.ipynb         synthetic-examples
page  notebooks/synthetic-examples/04_regional_effects_real_f.ipynb              synthetic-examples
page  notebooks/synthetic-examples/05_conditional_interaction_independent_uniform_global.ipynb  synthetic-examples
page  notebooks/synthetic-examples/06_general_interaction_independent_uniform_global.ipynb      synthetic-examples
page  notebooks/real-examples/01_bike_sharing_dataset.ipynb                      real-examples
page  notebooks/real-examples/02_california_housing.ipynb                        real-examples
page  notebooks/real-examples/03_california_housing_tabpfn.ipynb                 real-examples
page  notebooks/real-examples/04_no2.ipynb                                       real-examples

# --- IMAGES (bucket a): notebooks whose figures feed authored pages ---
image notebooks/quickstart/simple_api.ipynb                                      quickstart
image notebooks/quickstart/flexible_api.ipynb                                    quickstart
image notebooks/quickstart/readme_example.ipynb                                  quickstart
image notebooks/real-examples/01_bike_sharing_dataset.ipynb                      real-examples
```

**Bucket (b), explicitly NOT in docs** (leave in `notebooks/`, do nothing):
`05_conditional_interaction_independent_uniform_heter.ipynb`,
`05_conditional_interaction_independent_uniform_regional.ipynb`,
`07_conditional_interaction_4_regions_independent_uniform_global.ipynb`.

Note `01_bike_sharing_dataset.ipynb` appears as both `page` and `image` (its
figures also feed `global_and_regional_effects.md`). That is intentional.

---

## Steps to apply (in order)

### Step 0 — Re-verify before touching anything
Run and confirm output matches the audit above:
```bash
cd /home/givasile/github/packages/effector
# Tutorials referenced only by the Makefile?
grep -rn "Tutorials" docs/ .github/ Makefile | grep -v "^Makefile"
# static .md files referenced anywhere? (expect: no hits for the .md themselves)
grep -rn "static/quickstart/[a-z_]*\.md\|static/real-examples/[a-z0-9_]*\.md" docs/
# orphan synthetic pages referenced?
grep -rn "02_regional_pdp\|02_regional_rhale\|02_regional_shapdp" docs/
```
If any command shows a real page depending on a "dead" item, stop and re-audit.

### Step 1 — Add lightweight conversion dep
In `pyproject.toml`, add `"nbconvert"` (and `"ipython"` if needed for the
markdown exporter) to the `[dependency-groups] docs = [...]` list. This lets
`--group docs` run nbconvert without the heavy `tutorials` extra. Then
`uv sync`.

### Step 2 — Add the mapping file
Create `docs/notebook_map.txt` with the content in the section above.

### Step 3 — Rewrite the Makefile docs section
Replace the current `docs-update` target. Keep `docs-serve` and `docs-build`
as-is. Add two map-driven targets (bash reading `docs/notebook_map.txt`):

```makefile
# Documentation -------------------------------------------------------------
.PHONY: docs-serve
docs-serve:  ## serve the documentation locally
	uv run --no-default-groups --group docs mkdocs serve -f docs/mkdocs.yml

.PHONY: docs-build
docs-build:  ## build the documentation site
	uv run --no-default-groups --group docs mkdocs build -f docs/mkdocs.yml

.PHONY: docs-pages
docs-pages:  ## (c) convert selected notebooks -> committed doc pages
	@grep '^page' docs/notebook_map.txt | while read _ src dest; do \
		echo "converting $$src -> docs/docs/notebooks/$$dest"; \
		uv run --no-default-groups --group docs jupyter nbconvert --to markdown \
			"$$src" --output-dir "docs/docs/notebooks/$$dest"; \
	done

.PHONY: docs-images
docs-images:  ## (a) refresh static images for authored pages, on demand
	@grep '^image' docs/notebook_map.txt | while read _ src dest; do \
		nb=$$(basename "$$src" .ipynb); tmp=$$(mktemp -d); \
		echo "harvesting figures from $$src -> docs/docs/static/$$dest/$${nb}_files"; \
		uv run --no-default-groups --group docs jupyter nbconvert --to markdown \
			"$$src" --output-dir "$$tmp"; \
		rm -rf "docs/docs/static/$$dest/$${nb}_files"; \
		mkdir -p "docs/docs/static/$$dest"; \
		cp -r "$$tmp/$${nb}_files" "docs/docs/static/$$dest/" 2>/dev/null || true; \
		rm -rf "$$tmp"; \
	done
```
(If `nbconvert` isn't exposed as `jupyter nbconvert` under the docs group, use
`uv run --no-default-groups --group docs python -m nbconvert ...`.)

### Step 4 — Standardize bucket (a) image source to `static/`
`index.md` currently reads readme-example figures from
`notebooks/quickstart/readme_example_files/`. After `make docs-images` populates
`docs/docs/static/quickstart/readme_example_files/`, rewrite the 7 image paths
in `docs/docs/index.md`:
```
./notebooks/quickstart/readme_example_files/  ->  ./static/quickstart/readme_example_files/
```
Verify every rewritten image resolves to a file on disk. After this, `(a)`
images come exclusively from `docs/docs/static/`.

### Step 5 — Delete dead trees (only after Steps 3–4 succeed)
```bash
rm -rf docs/docs/Tutorials
rm -rf docs/docs/static/real-examples/01_bike_sharing_dataset_files_bak
find docs/docs/static -name '*.md' -delete          # stray converted .md, keep *_files/
rm -f docs/docs/notebooks/synthetic-examples/02_regional_pdp.md \
      docs/docs/notebooks/synthetic-examples/02_regional_rhale.md \
      docs/docs/notebooks/synthetic-examples/02_regional_shapdp.md
rm -rf docs/docs/notebooks/quickstart      # quickstart notebooks are image-only now (see note)
rm -rf notebooks/synthetic-examples/.ipynb_checkpoints
```
**Caution on `docs/docs/notebooks/quickstart`:** only delete it AFTER Step 4,
because `index.md` used to read from it. Confirm with
`grep -rn "notebooks/quickstart" docs/docs` returning nothing first.

### Step 6 — Gitignore the cruft
Add to `.gitignore`:
```
.ipynb_checkpoints/
docs/docs/Tutorials/
```

### Step 7 — Update `docs/README.md`
Replace the manual nbconvert + `cp` instructions with the new flow:
- `make docs-pages` — regenerate converted example/guide pages (bucket c).
- `make docs-images` — refresh static images for authored pages (bucket a).
- `make docs-serve` / `make docs-build` — preview / build.
- Note that bucket (b) notebooks are simply absent from `docs/notebook_map.txt`.

### Step 8 — Verify the whole thing builds
```bash
make docs-pages          # regenerates committed pages; git diff should be ~noise-free
make docs-build          # must succeed with no missing-file / broken-link warnings
git diff --stat          # review
```
Optionally run `make docs-images` and confirm `git diff` on `static/` is empty
(proves the committed images already match a fresh notebook conversion).

---

## Final directory shape

```
notebooks/                         # (b)+(c) all notebooks — single source of truth
  quickstart/  guides/  synthetic-examples/  real-examples/

docs/
  notebook_map.txt                 # the explicit selection/mapping
  docs/
    index.md  quickstart.md  guides.md  examples.md   # (a) nav + authored
    quickstart/  *.md                                 # (a) authored pages
    api_docs/    *.md                                 # (a) mkdocstrings pages
    static/
      quickstart/<nb>_files/                          # (a) images (make docs-images)
      real-examples/<nb>_files/                       # (a) images (make docs-images)
      *.png / logos                                   # site assets
    notebooks/                                        # (c) converted pages (committed)
      guides/  synthetic-examples/  real-examples/

scripts/                           # (d) standalone, untouched
```

## Safety summary
- Nothing that a page references is deleted (verified by grep in Step 0).
- CI keeps working: it only runs `mkdocs build`, and all converted pages stay
  committed.
- Notebooks are never executed; only their saved outputs are converted.
- The only link edit is `index.md`'s 7 readme-example image paths (Step 4),
  done before removing their old source dir (Step 5).

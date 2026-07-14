# Documentation

The docs build follows one explicit model with four buckets:

| Bucket | What | Where | How to update |
|---|---|---|---|
| (a) | Authored pages + static images | `docs/docs/**/*.md`, images in `docs/docs/static/<group>/<nb>_files/` | Edit `.md` by hand; refresh images with `make docs-images` |
| (b) | Notebooks not in the mapping | `notebooks/` only | Nothing — ignored by the docs |
| (c) | Selected notebooks → doc pages | `docs/docs/notebooks/<group>/` (committed) | `make docs-pages` |
| (d) | Standalone scripts | `scripts/` | Untouched; not part of docs |

Which notebooks feed the docs — and whether as a page (c) or as images for an
authored page (a) — is declared in `docs/notebook_map.txt`. Notebooks absent
from that file are simply not part of the docs.

Conversion never executes notebooks; it renders their **saved outputs**
(`jupyter nbconvert --to markdown`). Re-run a notebook first if you want its
doc page refreshed. The generated markdown is committed: CI only runs
`mkdocs build` and deploys, it never converts.

## Commands

```bash
make docs-pages    # (c) regenerate converted notebook pages from the mapping
make docs-images   # (a) refresh static images for authored pages
make docs-serve    # preview locally
make docs-build    # build the site into docs/site/
```

## Before publishing

CI (`publish_documentation.yml`) copies `CHANGELOG.md` and `CONTRIBUTING.md`
into `docs/docs/` automatically; for a local preview do the same by hand:

```bash
cp CHANGELOG.md docs/docs/changelog.md
cp CONTRIBUTING.md docs/docs/contributing.md
```

## Connection between files

- `docs/docs/index.md` → images from `docs/docs/static/quickstart/interactive_guide/`
- `docs/docs/quickstart/simple_api.md` / `interactive/customize_fit.md` → images from `docs/docs/static/quickstart/{simple_api,flexible_api}_files/`
- `docs/docs/quickstart/global_and_regional_effects.md` → images from `docs/docs/static/real-examples/01_bike_sharing_dataset_files/`
- `docs/docs/api_docs/api_global.md` → images from `docs/docs/static/quickstart/simple_api_files/`

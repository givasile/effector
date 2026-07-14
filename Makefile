# global variables
SHELL := /bin/bash
PROJECT_NAME = effector

# Environment ---------------------------------------------------------------
.PHONY: install
install:  ## create/update .venv with all dev dependencies (uv)
	uv sync

# Tests ---------------------------------------------------------------------
.PHONY: test
test:  ## run the fast test suite (the merge gate: -m "not slow")
	uv run --no-default-groups --group test pytest tests -m "not slow"

.PHONY: test-all
test-all:  ## run the full test suite, including slow tests
	uv run --no-default-groups --group test pytest tests

# Code style ----------------------------------------------------------------
.PHONY: format
format:  ## format the source code with ruff
	uv run --no-default-groups --group dev ruff format $(PROJECT_NAME) tests

.PHONY: lint
lint:  ## lint the source code with ruff
	uv run --no-default-groups --group dev ruff check $(PROJECT_NAME) tests

# Documentation -------------------------------------------------------------
.PHONY: docs-serve
docs-serve:  ## serve the documentation locally
	uv run --no-default-groups --group docs mkdocs serve -f docs/mkdocs.yml

.PHONY: docs-build
docs-build:  ## build the documentation site
	uv run --no-default-groups --group docs mkdocs build -f docs/mkdocs.yml

.PHONY: docs-pages
docs-pages:  ## convert selected notebooks (docs/notebook_map.txt) -> committed doc pages
	@grep '^page' docs/notebook_map.txt | while read _ src dest; do \
		echo "converting $$src -> docs/docs/notebooks/$$dest"; \
		uv run --no-default-groups --group docs jupyter nbconvert --to markdown \
			"$$src" --output-dir "docs/docs/notebooks/$$dest"; \
	done

.PHONY: docs-images
docs-images:  ## refresh static images for authored pages from notebooks (docs/notebook_map.txt)
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

.PHONY: docs-reports
docs-reports:  ## harvest one-click report pages from notebooks (docs/notebook_map.txt)
	@mkdir -p docs/docs/static/reports
	@grep '^report' docs/notebook_map.txt | while read _ src method; do \
		nb=$$(basename "$$src" .ipynb); dir=$$(dirname "$$src"); \
		out="docs/docs/static/reports/$${nb}_$${method}.html"; \
		echo "harvesting $$dir/reports/$$nb/report_$$method.html -> $$out"; \
		cp "$$dir/reports/$$nb/report_$$method.html" "$$out" || \
			{ echo "  MISSING: run the notebook first (reports/ is gitignored)"; exit 1; }; \
	done

# Housekeeping --------------------------------------------------------------
.PHONY: clean
clean:  ## delete compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

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

.PHONY: docs-update
docs-update:  ## regenerate tutorial markdown from the notebooks
	uv run --extra tutorials --extra shap jupyter nbconvert --to markdown ./notebooks/real-examples/* --output-dir docs/docs/Tutorials/real-examples/
	uv run --extra tutorials --extra shap jupyter nbconvert --to markdown ./notebooks/synthetic-examples/* --output-dir docs/docs/Tutorials/synthetic-examples/

# Housekeeping --------------------------------------------------------------
.PHONY: clean
clean:  ## delete compiled Python files
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

# global variables
SHELL := /bin/bash

PROJECT_NAME = effector
PYTHON ?= 3.10
ENV ?= sandbox
REQUIREMENTS := $(if $(findstring sandbox,$(ENV)),requirements.txt,requirements-$(ENV).txt)

# Conda related commands
.PHONY: conda-remove-env
conda-remove-env:
	@conda env list | grep -q "^$(PROJECT_NAME)-$(ENV) " && conda env remove --name $(PROJECT_NAME)-$(ENV) -y || echo "Environment $(PROJECT_NAME)-$(ENV) does not exist, skipping removal."

.PHONY: conda-create-env
conda-create-env:
	@conda create --name $(PROJECT_NAME)-$(ENV) python=$(PYTHON) -y

.PHONY: conda-install-requirements
conda-install-requirements:
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install --upgrade pip
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install -r $(REQUIREMENTS)
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install -e .

.PHONY: conda-init
conda-init: conda-remove-env conda-create-env conda-install-requirements

.PHONY: conda-update
conda-update: conda-install-requirements

# Pip related commands
.PHONY: venv-remove
venv-remove:
	rm -rf .venv-$(ENV)

.PHONY: venv-create
venv-create:
	python -m venv .venv-$(ENV)

.PHONY: venv-install-requirements
venv-install-requirements:
	source .venv-$(ENV)/bin/activate && python -m pip install --upgrade pip
	source .venv-$(ENV)/bin/activate && python -m pip install -r $(REQUIREMENTS)
	source .venv-$(ENV)/bin/activate && python -m pip install -e .

.PHONY: venv-init
venv-init: venv-remove venv-create venv-install-requirements

.PHONY: venv-update
venv-update: venv-install-requirements

# Documentation related commands
.PHONY: docs-update
docs-update:
	@source .venv-dev/bin/activate && jupyter nbconvert --to markdown ./notebooks/real-examples/* --output-dir docs/docs/Tutorials/real-examples/
	@source .venv-dev/bin/activate && jupyter nbconvert --to markdown ./notebooks/synthetic-examples/* --output-dir docs/docs/Tutorials/synthetic-examples/
	@source .venv-dev/bin/activate && jupyter nbconvert --to markdown ./notebooks/getting-started/* --output-dir docs/docs/Tutorials/getting-started/

docs-serve:
	@source .venv-dev/bin/activate && cd docs/ && mkdocs serve

# Test related commands
.PHONY: test
test:
	@source .venv-test/bin/activate && pytest -v

# Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml $(PROJECT_NAME)

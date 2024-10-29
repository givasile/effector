#################################################################################
# GLOBALS                                                                       #
#################################################################################
SHELL := /bin/bash

PROJECT_NAME = effector
PYTHON ?= 3.10
ENV ?= sandbox

REQUIREMENTS := $(if $(findstring sandbox,$(ENV)),requirements.txt,requirements-$(ENV).txt)

# Conda related commands
.PHONY: remove-conda-env
remove-conda-env:
	@conda env list | grep -q "^$(PROJECT_NAME)-$(ENV) " && conda env remove --name $(PROJECT_NAME)-$(ENV) -y || echo "Environment $(PROJECT_NAME)-$(ENV) does not exist, skipping removal."

.PHONY: create-conda-env
create-conda-env:
	@conda create --name $(PROJECT_NAME)-$(ENV) python=$(PYTHON) -y

.PHONY: install-conda-requirements
install-conda-requirements:
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install --upgrade pip
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install -r $(REQUIREMENTS)

.PHONY: conda-init
conda-init: remove-conda-env create-conda-env install-conda-requirements

.PHONY: conda-update
conda-update: install-conda-requirements

# Pip related commands
.PHONY: remove-venv
remove-venv:
	rm -rf .venv-$(ENV)

.PHONY: create-venv
create-venv:
	python -m venv .venv-$(ENV)

.PHONY: install-venv-requirements
install-venv-requirements:
	source .venv-$(ENV)/bin/activate && python -m pip install --upgrade pip
	source .venv-$(ENV)/bin/activate && python -m pip install -r $(REQUIREMENTS)

.PHONY: venv-init
venv-init: remove-venv create-venv install-venv-requirements

.PHONY: venv-update
venv-update: install-venv-requirements


# Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml $(PROJECT_NAME)

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = effector
PYTHON_VERSION = 3.10

#################################################################################
# COMMANDS                                                                      #
#################################################################################
.PHONY: setup-conda-env-generic
setup-conda-env-generic:
	# if conda is not installed, halt
	@conda --version

	# if environment exists, remove it
	@conda env list | grep -q "^$(PROJECT_NAME)-$(ENV) " && conda env remove --name $(PROJECT_NAME)-$(ENV) -y || echo "Environment $(PROJECT_NAME)-$(ENV) does not exist, no need to remove it."

	# create environment
	@conda create --name $(PROJECT_NAME)-$(ENV) python=$(PYTHON_VERSION) -y

	# populate environment
	@conda run -n $(PROJECT_NAME)-$(ENV) pip install --upgrade pip

	# if env=sandbox, install requirements.txt
	@if [ "$(ENV)" = "sandbox" ]; then \
		conda run -n $(PROJECT_NAME)-$(ENV) pip install -r requirements.txt; \
	else \
		conda run -n $(PROJECT_NAME)-$(ENV) pip install -r requirements-$(ENV).txt; \
	fi


# Targets to set up specific environments
.PHONY: setup-conda-env setup-conda-env-dev setup-conda-env-test
setup-conda-env:
	$(MAKE) setup-conda-env-generic ENV=sandbox
setup-conda-env-dev:
	$(MAKE) setup-conda-env-generic ENV=dev
setup-conda-env-test:
	$(MAKE) setup-conda-env-generic ENV=test


## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Format source code with black
.PHONY: format
format:
	black --config pyproject.toml $(PROJECT_NAME)

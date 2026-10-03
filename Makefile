SHELL := /bin/bash
.DEFAULT_GOAL := help

DEV_IMAGE := psyb0t/torchbase-dev
UID := $(shell id -u)
GID := $(shell id -g)
DEPENDENCY_CUTOFF := 2026-09-20T00:00:00Z
MIN_TEST_COVERAGE := 90
TUPLE ?= all

DEV_RUN := docker run --rm --init --user $(UID):$(GID) -e HOME=/tmp -v $(CURDIR):/work -w /work $(DEV_IMAGE)

.PHONY: help dev-image model-lock format lint lint-fix test test-unit test-coverage build build-test build-test-cuda-gpu ci-targets

help: ## List supported operations
	@awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "%-22s %s\\n", $$1, $$2}' $(MAKEFILE_LIST)

dev-image: ## Build the isolated dependency-lock toolchain
	docker build -f Dockerfile.dev -t $(DEV_IMAGE) .

model-lock: dev-image ## Generate hash-locked dependencies for the selected tuple set
	$(DEV_RUN) python scripts/tuple_matrix.py lock --cutoff $(DEPENDENCY_CUTOFF) $(TUPLE)

format: dev-image ## Format Python source
	$(DEV_RUN) bash -ceu 'python -m ruff check --fix scripts tests && python -m ruff format scripts tests'

lint: dev-image ## Validate the registry and run static Python checks
	$(DEV_RUN) bash -ceu 'python scripts/tuple_matrix.py validate && python -m ruff format --check scripts tests && python -m ruff check scripts tests && python -m pyright scripts tests && python -m mypy --strict scripts tests && python -m bandit -q -r scripts'

lint-fix: format ## Apply safe formatting fixes, then lint
	@$(MAKE) lint

test: test-unit ## Run the normal test suite

test-unit: dev-image ## Run tuple-registry command tests
	$(DEV_RUN) python -m unittest discover --start-directory tests --verbose

test-coverage: dev-image ## Enforce unit coverage for the tuple renderer
	$(DEV_RUN) bash -ceu 'python -m coverage erase && COVERAGE_RUN=1 python -m unittest discover --start-directory tests && python -m coverage combine && python -m coverage report --include="scripts/*" --fail-under $(MIN_TEST_COVERAGE) && python -m coverage erase'

build: ## Build one tuple, or every tuple when TUPLE=all
	python scripts/tuple_matrix.py docker-build $(TUPLE)

build-test: build ## Build and verify the selected tuple or tuple set
	python scripts/tuple_matrix.py docker-verify $(TUPLE)

build-test-cuda-gpu: ## Build and GPU-verify every selected CUDA tuple
	python scripts/tuple_matrix.py docker-build $(if $(filter all,$(TUPLE)),cuda,$(TUPLE))
	python scripts/tuple_matrix.py docker-verify --gpu $(if $(filter all,$(TUPLE)),cuda,$(TUPLE))

ci-targets: dev-image ## Print the reusable Docker workflow matrix JSON
	$(DEV_RUN) python scripts/tuple_matrix.py ci-targets

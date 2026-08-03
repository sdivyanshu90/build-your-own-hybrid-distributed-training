# Convenience targets.  Everything here is a thin wrapper around a command that
# is also documented verbatim in README.md, so the Makefile never becomes the
# only place a command lives.
# `python3`, not `python`: a stock Debian/Ubuntu has only the former unless
# `python-is-python3` is installed.  Override with `make PYTHON=... <target>`.
PYTHON ?= python3
TORCHRUN ?= torchrun
NPROC ?= 2

.DEFAULT_GOAL := help

.PHONY: help
help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-28s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Editable install with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

.PHONY: lint
lint:  ## Ruff lint
	ruff check src tests examples scripts

.PHONY: format
format:  ## Ruff format (in place)
	ruff format src tests examples scripts

.PHONY: format-check
format-check:  ## Ruff format check only
	ruff format --check src tests examples scripts

.PHONY: typecheck
typecheck:  ## mypy over the package
	mypy

.PHONY: test
test:  ## Full test suite
	pytest -q

.PHONY: test-cpu
test-cpu:  ## Everything that does not need a GPU
	pytest -q -m "not cuda and not multigpu"

.PHONY: test-unit
test-unit:  ## Fast, single-process unit tests only
	pytest -q tests/unit

.PHONY: test-distributed
test-distributed:  ## Multi-process Gloo tests
	pytest -q tests/distributed tests/integration

.PHONY: test-e2e
test-e2e:  ## End-to-end training / checkpoint scenarios
	pytest -q tests/end_to_end

.PHONY: coverage
coverage:  ## Coverage over the non-GPU suite
	pytest -q -m "not cuda and not multigpu" --cov=hybrid_training --cov-report=term-missing --cov-report=xml

.PHONY: check
check: lint format-check typecheck test-cpu  ## Everything CI runs on CPU

.PHONY: train-ddp
train-ddp:  ## torchrun custom-DDP example
	$(TORCHRUN) --standalone --nproc-per-node=$(NPROC) examples/train_ddp.py

.PHONY: train-fsdp
train-fsdp:  ## torchrun FSDP-style example
	$(TORCHRUN) --standalone --nproc-per-node=$(NPROC) examples/train_fsdp.py

.PHONY: train-tp
train-tp:  ## torchrun tensor-parallel example
	$(TORCHRUN) --standalone --nproc-per-node=$(NPROC) examples/train_tensor_parallel.py

.PHONY: train-sp
train-sp:  ## torchrun sequence-parallel example
	$(TORCHRUN) --standalone --nproc-per-node=$(NPROC) examples/train_sequence_parallel.py

.PHONY: train-hybrid
train-hybrid:  ## torchrun 4-rank hybrid example
	$(TORCHRUN) --standalone --nproc-per-node=4 examples/train_hybrid.py

.PHONY: benchmark
benchmark:  ## Single-process benchmark sweep
	$(PYTHON) scripts/benchmark.py --world-size 1

.PHONY: clean
clean:  ## Remove caches and build artefacts
	rm -rf build dist *.egg-info src/*.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	rm -f .coverage .coverage.* coverage.xml
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

.PHONY: help install test test-unit test-integration test-e2e lint type-check format clean docs-build docs-serve benchmark profile dist release check

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	uv sync --extra dev

install-prod: ## Install production dependencies only
	uv sync

test: ## Run all tests
	uv run pytest tests/ -v

test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -v -m integration

test-e2e: ## Run end-to-end tests only
	uv run pytest tests/e2e/ -v -m e2e

test-parallel: ## Run tests in parallel
	uv run pytest tests/ -n auto --dist=worksteal -v

lint: ## Run ruff linter
	uv run ruff check pymars tests

type-check: ## Run ty type checker
	uv run ty check pymars/

format: ## Format code with ruff
	uv run ruff check --fix pymars tests
	uv run ruff format pymars tests

check: lint type-check test ## Run all quality checks

clean: ## Clean build artifacts and caches
	rm -rf build dist *.egg-info .mypy_cache .pytest_cache .ruff_cache
	rm -rf coverage.xml htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

docs-build: ## Build documentation
	uv run mkdocs build

docs-serve: ## Serve documentation locally
	uv run mkdocs serve

benchmark: ## Run benchmarks
	uv run pytest tests/test_benchmark.py -v --benchmark-only

profile: ## Run Scalene profiler
	uv run python -m scalene pymars/earth.py -- --profile

dist: ## Build distribution packages
	uv build

release: dist ## Publish to PyPI
	uv run twine upload dist/*

mutmut: ## Run mutation testing
	uv run mutmut run

mutmut-results: ## Show mutation test results
	uv run mutmut results

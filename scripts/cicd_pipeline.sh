#!/bin/bash
# Complete CI/CD pipeline script for pymars

set -e  # Exit on any error

echo "Starting CI/CD pipeline for pymars..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: pyproject.toml not found. Run this script from the project root."
    exit 1
fi

echo "1. Installing dependencies..."
pip install -e .[dev]

echo "2. Running linting with Ruff..."
pip install ruff
ruff check pymars tests
ruff format --check pymars tests

echo "3. Running type checking with MyPy..."
mypy pymars/

echo "4. Running tests with coverage..."
python -m pytest tests/ --cov=pymars --cov-report=term-missing

echo "5. Running pre-commit hooks..."
pre-commit run --all-files

echo "6. Checking coverage percentages..."
bash scripts/check_coverage.sh

echo "7. Running profiling..."
python scripts/profile_pymars.py

echo "8. Running tox tests..."
if command -v tox &> /dev/null; then
    tox
else
    echo "Tox not found, skipping tox tests"
fi

echo "CI/CD pipeline completed successfully!"
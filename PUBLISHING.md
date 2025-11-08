# Publishing pymars

This document describes how to publish the pymars package to PyPI and TestPyPI.

## Prerequisites

1. Install the build and publishing tools:
```bash
pip install build twine
```

2. Update the version in `pyproject.toml` to the desired version

3. Ensure all tests pass:
```bash
python -m pytest tests/
```

4. Build the package:
```bash
python -m build
```

## Publishing Process

### To TestPyPI (for testing)

1. Register and get API token from TestPyPI: https://test.pypi.org/account/register/

2. Upload to TestPyPI using twine:
```bash
twine upload --repository testpypi dist/*
```

3. You will be prompted for username and password (use `__token__` as username and your API token as password)

### To PyPI (for production release)

1. Register and get API token from PyPI: https://pypi.org/manage/account/token/

2. Upload to PyPI using twine:
```bash
twine upload dist/*
```

3. You will be prompted for username and password (use `__token__` as username and your API token as password)

## Testing the Installation

### From TestPyPI (for testing)
```bash
pip install --index-url https://test.pypi.org/simple/ pymars
```

### From PyPI (production)
```bash
pip install pymars
```

## Current Package Information

- Name: pymars
- Version: 1.0.0
- Description: Pure Python Earth (MARS) algorithm
- Dependencies: numpy, scikit-learn, matplotlib
- Optional dependencies: pandas (for CLI functionality)

## Features Included in Release

- Core MARS algorithm with forward and backward passes
- Scikit-learn compatibility with EarthRegressor and EarthClassifier
- Generalized Linear Models with GLMEarth
- Cross-validation helper with EarthCV
- Feature importance calculations
- Missing value and categorical feature support
- Plotting utilities
- Advanced interpretability tools (partial dependence, ICE plots, model explanations)
- Command-line interface for model fitting, prediction, and scoring
- Comprehensive test suite with property-based tests and benchmarks
- State-of-the-art CI/CD pipeline with automated testing, linting, type checking, and security scanning
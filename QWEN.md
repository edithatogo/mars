# pymars Project Context

## Project Overview

**pymars** is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS), inspired by the popular `py-earth` library by Jason Friedman and an R package `earth` by Stephen Milborrow. The goal is to provide an easy-to-install, scikit-learn compatible version of the MARS algorithm without C/Cython dependencies.

### Key Features

- **Pure Python:** Easy to install and use across different platforms
- **Scikit-learn Compatible:** Integrates with the scikit-learn ecosystem (estimators, pipelines, model selection tools)
- **MARS Algorithm:** Implements the core MARS fitting procedure, including:
  - Forward pass to select basis functions (both hinge and linear terms)
  - Pruning pass using Generalized Cross-Validation (GCV) to prevent overfitting
  - Support for interaction terms (including interactions involving linear terms)
  - Refined `minspan` and `endspan` controls for knot placement, aligning more closely with `py-earth` behavior
- **Feature Importance:** Calculation of feature importances using methods like 'nb_subsets', 'gcv', and 'rss'
- **Regression and Classification:** Provides `EarthRegressor` and `EarthClassifier` classes
- **Generalized Linear Models:** The `GLMEarth` subclass fits logistic or Poisson models
- **Cross-Validation Helper:** The `EarthCV` class integrates with scikit-learn's model selection utilities
- **Plotting Utilities:** Simple diagnostics built on `matplotlib`

### Architecture

The project is structured as follows:

- **Core Module:** `pymars/earth.py` contains the main `Earth` class implementing the MARS algorithm
- **Scikit-learn Compatibility:** `pymars/_sklearn_compat.py` contains `EarthRegressor` and `EarthClassifier` wrappers
- **GLM Support:** `pymars/glm.py` contains `GLMEarth` for generalized linear models
- **Cross-validation:** `pymars/cv.py` contains `EarthCV` class
- **Plotting:** `pymars/plot.py` provides visualization utilities
- **Basis Functions:** `pymars/_basis.py` defines the various basis functions
- **Missing Value Handling:** `pymars/_missing.py` handles missing values
- **Categorical Features:** `pymars/_categorical.py` handles categorical features
- **Forward/Pruning Logic:** `pymars/_forward.py` and `pymars/_pruning.py` implement the algorithm steps

## Building and Running

### Installation

Install `pymars` inside a Python virtual environment:

```bash
# Install from PyPI
pip install pymars

# Or work with the latest source
git clone https://github.com/pymars/pymars.git
cd pymars
pip install -e .
```

Optional pandas dependency for full scikit-learn estimator checks:
```bash
pip install "pymars[pandas]"
```

### Running Tests

Install dependencies and run tests:

```bash
# Install dependencies
pip install -r requirements.txt
pip install "pymars[pandas]"  # optional, for estimator checks

# Run tests
pytest
```

Alternatively, use the helper script:
```bash
bash scripts/setup_tests.sh
pytest
```

### Usage Examples

Basic usage:
```python
import numpy as np
import pymars as earth

X = np.random.rand(100, 3)
y = np.sin(X[:, 0]) + X[:, 1]

model = earth.Earth(max_degree=1, penalty=3.0)
model.fit(X, y)
predictions = model.predict(X)
```

Run included demos:
```bash
python -m pymars.demos.basic_regression_demo
python -m pymars.demos.basic_classification_demo
```

## Development Conventions

### Coding Standards

- **Pure Python only:** No C or Cython extensions
- **Scikit-learn compatibility** is mandatory:
  - Inherit from `sklearn.base.BaseEstimator`, `RegressorMixin`, or `ClassifierMixin`
  - Implement `fit`, `predict`, `score`, `get_params`, and `set_params`
  - Validate inputs with `sklearn.utils.validation.check_X_y` and `check_array`
  - Return `self` from `fit`
  - Store learned attributes with trailing underscores (e.g., `coef_`)

### Project Structure

- `pymars/`: Main library code
- `tests/`: Test files using pytest
- `docs/`: Documentation files
- `examples/`: Usage examples
- `scripts/`: Helper scripts

### Documentation Files

- `ROADMAP.md`: Multi-phase development plan
- `TODO.md`: Detailed task checklist
- `AGENTS.md`: Guidelines for AI agents working on the project
- `CONTRIBUTING.md`: Contribution guidelines
- `SESSION_LOGS.md`: Development session logs

### Key Directives

1. **MARS Algorithm Implementation:** Focus on forward and backward passes
2. **Scikit-learn Compatibility:** Ensure full compliance with scikit-learn estimator interface
3. **Pure Python:** Maintain pure Python implementation without external dependencies beyond NumPy and SciPy
4. **Performance:** Optimize Python code while maintaining correctness and clarity
5. **Testing:** Maintain high test coverage with pytest
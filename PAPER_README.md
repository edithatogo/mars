# pymars: Multivariate Adaptive Regression Splines in Python

Welcome to pymars, a pure Python implementation of Multivariate Adaptive Regression Splines (MARS) with scikit-learn compatibility.

## Repository Structure

This repository follows a modular branch structure:

- **`main` branch**: Contains the core software library (pymars), source code, tests, documentation, and examples
- **`paper` branch**: Contains the research paper describing pymars and its applications in health economics

## About pymars

pymars is a pure Python implementation of the Multivariate Adaptive Regression Splines (MARS) algorithm. Key features include:

- Pure Python implementation with no C/Cython dependencies
- Full scikit-learn compatibility
- Support for feature importance calculations
- Missing value handling
- Categorical variable support
- Generalized linear model extensions
- Interpretability tools including partial dependence plots
- Cross-validation helper for scikit-learn integration

## Paper

The research paper describing pymars and its applications in health economics is available in the [`paper` branch](https://github.com/edithatogo/pymars/tree/paper). The paper demonstrates the use of pymars with Australian and New Zealand health datasets, showcasing its effectiveness in modeling complex relationships in health economic data.

To access the paper, switch to the paper branch:

```bash
git checkout paper
```

The paper includes:

- Comprehensive introduction to MARS and pymars
- Technical implementation details
- Health economic applications using real-world examples
- Future directions including planned JAX/XLA backend
- Comparison with alternative methods

## Installation

```bash
pip install pymars
```

## Usage

```python
import numpy as np
from pymars import Earth

# Create sample data
X = np.random.randn(100, 4)
y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.5*X[:, 2]*X[:, 3] + np.random.randn(100)*0.1

# Fit MARS model
model = Earth(max_degree=2, penalty=3.0, max_terms=21)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"R-squared: {model.score(X, y):.3f}")
```

For more examples, check the examples directory.

## Branch Management

- Switch to the paper branch: `git checkout paper`
- Switch back to main: `git checkout main`
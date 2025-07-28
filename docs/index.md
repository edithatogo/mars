# pymars Documentation

`pymars` provides a pure Python implementation of Multivariate Adaptive Regression Splines (MARS) with a scikit‑learn compatible API. The library exposes the classic `Earth` model along with utilities for generalized linear models, cross‑validation, and plotting diagnostics.

## Installation

Install the latest stable release from PyPI:

```bash
pip install pymars
```

To work with the current development version, clone the repository and install in editable mode:

```bash
git clone https://github.com/your-repository-url/pymars.git
cd pymars
pip install -e .
```

## Usage Examples

Fit a basic regression model using `Earth`:

```python
import numpy as np
import pymars as earth

X = np.random.rand(100, 3)
y = np.sin(X[:, 0]) + X[:, 1]

model = earth.Earth(max_degree=2)
model.fit(X, y)
print(model.predict(X[:5]))
```

`GLMEarth` supports logistic and Poisson models:

```python
from pymars import GLMEarth

clf = GLMEarth(family="binomial")
clf.fit(X, (y > 0).astype(int))
```

Perform cross‑validation with `EarthCV`:

```python
from pymars import EarthCV

cv = EarthCV(max_degree=[1, 2], penalty=[2, 3])
cv.fit(X, y)
best_model = cv.best_estimator_
```

## API Reference

### `pymars.earth.Earth`
The core MARS estimator. Implements the forward and pruning algorithms and is compatible with scikit‑learn.

### `pymars.glm.GLMEarth`
Extension of `Earth` that fits generalized linear models such as logistic and Poisson regression.

### `pymars.cv.EarthCV`
Grid‑search helper built on scikit‑learn's `GridSearchCV` for tuning `Earth` hyperparameters.

### Plotting Tools (`pymars.plot`)
Utility functions for visualizing model diagnostics and basis functions.


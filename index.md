# pymars: Multivariate Adaptive Regression Splines in Python

`pymars` is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS) that provides scikit-learn compatibility without C/Cython dependencies.

## Quick Start

```python
import numpy as np
from pymars import Earth

# Sample data
X = np.random.randn(100, 4)
y = X[:, 0]**2 + np.sin(X[:, 1]) + 0.5*X[:, 2]*X[:, 3] + np.random.randn(100)*0.1

# Fit MARS model
model = Earth(max_degree=2, penalty=3.0)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"R² Score: {model.score(X, y):.3f}")
```

## Features

- Pure Python implementation with no compilation required
- Full scikit-learn compatibility
- Support for missing values and categorical variables
- Feature importance calculations
- Generalized linear model extensions
- Cross-validation helpers
- Automatic changepoint detection through knot identification

## Documentation Sections

- [Installation](installation.md) - How to install pymars
- [Basic Tutorial](tutorial_basic.md) - Getting started with pymars
- [Advanced Usage](tutorial_advanced.md) - Advanced features and techniques
- [API Reference](api_reference.md) - Complete API documentation
- [Health Economics Applications](health_applications.md) - Specific examples in health economics
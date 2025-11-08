# Usage Guide

## Basic Usage

The pymars library provides MARS (Multivariate Adaptive Regression Splines) functionality with scikit-learn compatibility.

### Basic Example

```python
import numpy as np
from pymars import Earth

# Generate sample data
X = np.random.rand(100, 4)
y = X[:, 0]**2 + np.sin(X[:, 1]) + X[:, 2] * X[:, 3]

# Create and fit MARS model
model = Earth(max_degree=2, penalty=3.0)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print(f"R-squared: {model.score(X, y):.3f}")
```

### Regression Example

```python
from pymars import EarthRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = EarthRegressor(max_degree=2, penalty=3.0, max_terms=21, allow_bias=True)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Training Score: {train_score:.3f}")
print(f"Test Score: {test_score:.3f}")
```

### Classification Example

```python
from pymars import EarthClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

classifier = EarthClassifier(max_degree=2, penalty=3.0)
classifier.fit(X, y)

accuracy = classifier.score(X, y)
print(f"Accuracy: {accuracy:.3f}")
```

### Cross-Validation

pymars includes a cross-validation helper:

```python
from pymars import EarthCV
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)

# Define parameter grid
param_grid = {
    'max_degree': [1, 2],
    'penalty': [2.0, 3.0, 4.0],
    'max_terms': [10, 15, 20]
}

# Perform cross-validation
cv_model = EarthCV(param_grid=param_grid, cv=5)
cv_model.fit(X, y)

print(f"Best parameters: {cv_model.best_params_}")
print(f"Best score: {cv_model.best_score_:.3f}")
print(f"Best model: {cv_model.best_estimator_}")
```

### Feature Importance

pymars provides multiple methods to assess feature importance:

```python
import numpy as np
from pymars import Earth

# Generate sample data
X = np.random.rand(500, 5)
y = X[:, 0]**2 + 2*X[:, 1] + X[:, 2]*X[:, 3] + 0.1*np.random.randn(500)

model = Earth(max_degree=2, penalty=3.0, feature_importance_type='gcv')
model.fit(X, y)

# Get feature importances
importances = model.feature_importances_
print("Feature importances:", importances)

# Summarize feature importances
print(model.summary_feature_importances())
```
# Tutorial

This tutorial will walk you through the basics of using the mars library for Multivariate Adaptive Regression Splines (MARS) modeling. 

## Introduction to MARS

Multivariate Adaptive Regression Splines (MARS) is a flexible regression technique that models non-linear relationships using piecewise linear basis functions. It's particularly effective for datasets where the relationship between predictors and the response variable is complex and non-linear.

MARS builds models in two stages:
1. **Forward Stage**: Add basis functions (hinge functions) that best improve the model
2. **Backward Stage**: Remove basis functions that do not significantly contribute, using Generalized Cross-Validation (GCV)

## Getting Started

Let's start with a basic example using synthetic data:

```python
import numpy as np
import matplotlib.pyplot as plt
from mars import Earth

# Generate sample data with a non-linear relationship
np.random.seed(42)
n = 200
x1 = np.random.uniform(0, 10, n)
x2 = np.random.uniform(0, 10, n)
y_true = 2 * np.maximum(0, x1 - 5) + 1.5 * np.maximum(0, 3 - x2) + 0.1 * np.random.randn(n)
X = np.column_stack([x1, x2])

# Fit the MARS model
model = Earth(max_degree=1, penalty=3.0)
model.fit(X, y_true)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = np.mean((y_true - y_pred) ** 2)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {model.score(X, y_true):.4f}")
```

## Understanding Model Parameters

### Controlling Model Complexity

The key parameters that control model complexity are:

- `max_degree`: Maximum degree of interaction between variables
- `max_terms`: Maximum number of terms in the model  
- `penalty`: Penalty parameter used in GCV calculation
- `minspan` and `endspan`: Constraints on knot placement

```python
# Example with different complexity settings
from mars import Earth

# Simple model (less flexible)
simple_model = Earth(max_degree=1, max_terms=10, penalty=3.0)

# Complex model (more flexible)
complex_model = Earth(max_degree=3, max_terms=50, penalty=2.0)
```

### Controlling Knot Placement

```python
# Control knot placement with minspan and endspan
model = Earth(
    minspan=3,      # Minimum observations between knots
    endspan=3,      # Minimum observations at variable ends
    max_terms=21
)
model.fit(X, y_true)
```

## Cross-Validation for Hyperparameter Tuning

Use cross-validation to select optimal hyperparameters:

```python
from mars import EarthCV
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.2, random_state=42)

# Use cross-validation to select the best model
cv_model = EarthCV(
    max_degree=2,
    max_terms=(5, 10, 20, 30),
    penalty=(2.0, 3.0, 4.0),
    cv=5
)
cv_model.fit(X_train, y_train)

# Make predictions with the best model
y_pred_cv = cv_model.predict(X_test)

print(f"Best model parameters: {cv_model.best_params_}")
print(f"CV R² Score: {cv_model.score(X_test, y_test):.4f}")
```

## Working with Real Data

Let's work with a more realistic example:

```python
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from mars import EarthRegressor
import numpy as np

# Generate a more complex regression dataset
X, y = make_regression(n_samples=500, n_features=10, noise=0.1, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit a MARS model
mars_model = EarthRegressor(
    max_degree=2,
    penalty=3.0,
    max_terms=21
)
mars_model.fit(X_train, y_train)

# Evaluate the model
train_score = mars_model.score(X_train, y_train)
test_score = mars_model.score(X_test, y_test)

print(f"Training R² Score: {train_score:.4f}")
print(f"Test R² Score: {test_score:.4f}")

# Get feature importance
print(f"Number of selected terms: {len(mars_model.coef_)}")
```

## Model Interpretation

MARS models are highly interpretable. You can examine the model components:

```python
# Get a detailed summary of the model
summary = mars_model.summary()
print(summary)

# Access the model's basis functions
print(f"Number of basis functions: {len(mars_model.basis_) if hasattr(mars_model, 'basis_') else 'Not available'}")

# Feature importance
importances = mars_model.summary_feature_importances() 
print("Feature importances:", importances)
```

## Visualization

Visualize the model behavior:

```python
from mars import plot_partial_dependence, plot_residuals

# Partial dependence plots show the relationship between features and prediction
plot_partial_dependence(mars_model, X_test, features=[0, 1, 2])

# Residual plots to check model assumptions
plot_residuals(mars_model, X_test, y_test)
```

## Classification Example

MARS can also be used for classification tasks:

```python
from mars import EarthClassifier
from sklearn.datasets import make_classification

# Create a classification dataset
X_cls, y_cls = make_classification(n_samples=300, n_features=8, n_classes=2, random_state=42)

# Split the data
X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

# Fit a classification model
classifier = EarthClassifier(max_degree=2, penalty=3.0)
classifier.fit(X_train_cls, y_train_cls)

# Evaluate the classifier
accuracy = classifier.score(X_test_cls, y_test_cls)
print(f"Classification Accuracy: {accuracy:.4f}")

# Get prediction probabilities
probabilities = classifier.predict_proba(X_test_cls)
print(f"First 5 probability predictions:\n{probabilities[:5]}")
```

## Advanced Techniques

### Feature Engineering

MARS automatically performs feature selection, but you can also use it in combination with other preprocessing steps:

```python
from sklearn.pipeline import Pipeline
from mars import EarthRegressor

# Create a pipeline with preprocessing
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Optional: scale features
    ('mars', EarthRegressor(max_degree=2, penalty=3.0))
])

# Fit the pipeline
pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)
print(f"Pipeline R² Score: {score:.4f}")
```

### Regularization Path

Explore how model complexity affects performance:

```python
import matplotlib.pyplot as plt

penalties = np.logspace(0, 2, 20)  # 20 penalty values from 1 to 100
train_scores = []
test_scores = []

for penalty in penalties:
    model = Earth(penalty=penalty, max_terms=30)
    model.fit(X_train, y_train)
    
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

# Plot the regularization path
plt.figure(figsize=(10, 6))
plt.semilogx(penalties, train_scores, label='Training Score', marker='o')
plt.semilogx(penalties, test_scores, label='Test Score', marker='s')
plt.xlabel('Penalty Parameter')
plt.ylabel('R² Score')
plt.title('Regularization Path')
plt.legend()
plt.grid(True)
plt.show()
```

## Best Practices

1. **Start Simple**: Begin with `max_degree=1` and default parameters
2. **Validate Thoroughly**: Always validate models on held-out test data
3. **Check Residuals**: Use residual plots to verify model assumptions
4. **Consider Interactions**: If domain knowledge suggests interactions, try `max_degree=2`
5. **Tune Hyperparameters**: Use cross-validation for optimal settings
6. **Interpret Results**: Examine model summary and feature importances

## Summary

MARS (via the mars library) provides a powerful and interpretable approach to modeling non-linear relationships in data. Its key advantages include:
- Automatic feature selection and interaction detection
- Interpretability of the resulting models
- Flexibility to capture complex non-linear patterns
- Integration with the scikit-learn ecosystem

With proper validation and tuning, MARS models can be very effective for many regression and classification tasks.
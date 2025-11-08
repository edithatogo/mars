# Basic Tutorial

This tutorial introduces the basic concepts and usage of mars for multivariate adaptive regression splines.

## Loading and Preparing Data

First, let's load a sample dataset and prepare it for modeling:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import mars as earth

# Generate sample data with non-linear relationships
np.random.seed(42)
n_samples = 500

# Create features with non-linear relationships
x1 = np.random.uniform(-3, 3, n_samples)
x2 = np.random.uniform(-3, 3, n_samples)
x3 = np.random.uniform(-3, 3, n_samples)
x4 = np.random.uniform(-3, 3, n_samples)

# True function with non-linearities and interactions
y_true = (
    2 * np.maximum(0, x1 - 1)  # Spline for x1 > 1
    - 1.5 * np.maximum(0, -x1 - 0.5)  # Spline for x1 < -0.5
    + 3 * np.maximum(0, x2 - 0) * np.maximum(0, x3 - 1)  # Interaction between x2 and x3
    + 0.5 * x4  # Linear effect
    + np.random.normal(0, 0.1, n_samples)  # Noise
)

# Create feature matrix and target vector
X = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3,
    'x4': x4
})

y = y_true

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

## Fitting a Basic MARS Model

Now let's fit a basic MARS model:

```python
# Create MARS model instance with basic parameters
model = earth.Earth(
    max_degree=2,      # Allow up to 2-way interactions
    penalty=3.0,       # GCV penalty parameter
    max_terms=21,      # Maximum number of terms (including intercept)
    minspan_alpha=0.05,  # Minimum span control as fraction of data
    endspan_alpha=0.05,  # End span control as fraction of data
    allow_linear=True,   # Allow linear basis functions
    feature_importance_type='gcv'  # Calculate feature importance using GCV reduction
)

# Fit the model
model.fit(X_train.values, y_train)

# Evaluate the model
train_score = model.score(X_train.values, y_train)
test_score = model.score(X_test.values, y_test)

print(f"Training R²: {train_score:.3f}")
print(f"Test R²: {test_score:.3f}")
print(f"GCV Score: {model.gcv_:.3f}")
```

## Examining the Model

After fitting, let's examine what the model learned:

```python
# Print number of selected basis functions
n_basis = len(model.basis_) - 1  # Subtract 1 for intercept
print(f"Number of selected basis functions: {n_basis}")

# Print the selected basis functions
print("\nSelected basis functions:")
for i, (bf, coef) in enumerate(zip(model.basis_, model.coef_)):
    if i == 0:  # Intercept
        print(f"  {bf}: {coef:.3f}")
    else:
        print(f"  {bf}: {coef:.3f}")

# Check feature importances
print("\nFeature importances:")
importances = model.feature_importances_
feature_names = X.columns.tolist()
for name, importance in zip(feature_names, importances):
    print(f"  {name}: {importance:.4f}")
```

## Making Predictions

Once trained, the model can make predictions:

```python
# Make predictions
y_pred_train = model.predict(X_train.values)
y_pred_test = model.predict(X_test.values)

# Calculate residuals
residuals_train = y_train - y_pred_train
residuals_test = y_test - y_pred_test

# Plot residuals
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(y_pred_train, residuals_train, alpha=0.5)
ax1.axhline(y=0, color='r', linestyle='--')
ax1.set_xlabel('Predicted Values')
ax1.set_ylabel('Residuals')
ax1.set_title('Training Residuals')

ax2.scatter(y_pred_test, residuals_test, alpha=0.5)
ax2.axhline(y=0, color='r', linestyle='--')
ax2.set_xlabel('Predicted Values')
ax2.set_ylabel('Residuals')
ax2.set_title('Test Residuals')

plt.tight_layout()
plt.show()
```

## Model Interpretation

mars provides tools for interpreting models:

```python
# Generate partial dependence plots to understand feature effects
from mars.explain import plot_partial_dependence

# Plot partial dependence for the most important features
top_features_idx = np.argsort(importances)[-3:][::-1]  # Top 3 features

fig, axes = plot_partial_dependence(
    model, X_train.values, top_features_idx,
    feature_names=feature_names,
    n_cols=3,
    figsize=(15, 5)
)

plt.tight_layout()
plt.show()
```

## Hyperparameter Tuning

Like other scikit-learn compatible estimators, mars works well with scikit-learn's model selection tools:

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_degree': [1, 2],
    'penalty': [2.0, 3.0, 4.0],
    'max_terms': [11, 16, 21]
}

# Create grid search object
grid_search = GridSearchCV(
    earth.Earth(minspan_alpha=0.05, endspan_alpha=0.05),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

# Fit grid search
grid_search.fit(X_train.values, y_train)

# Get best model
best_model = grid_search.best_estimator_

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Evaluate best model
best_train_score = best_model.score(X_train.values, y_train)
best_test_score = best_model.score(X_test.values, y_test)

print(f"Best model - Training R²: {best_train_score:.3f}")
print(f"Best model - Test R²: {best_test_score:.3f}")
```

## Handling Missing Values

mars can handle missing values directly:

```python
# Add some missing values to the dataset
X_with_missing = X_train.copy()
X_with_missing.iloc[::50, 0] = np.nan  # Add missing values to first column
X_with_missing.iloc[::75, 2] = np.nan  # Add missing values to third column

# Fit model with missing values
model_with_missing = earth.Earth(
    max_degree=2,
    penalty=3.0,
    max_terms=21
)

model_with_missing.fit(X_with_missing.values, y_train)

# Evaluate
train_score_missing = model_with_missing.score(X_with_missing.values, y_train)
print(f"Model with missing values - Training R²: {train_score_missing:.3f}")

# Show the missingness basis functions that were created
print("Basis functions involving missingness:")
for bf, coef in zip(model_with_missing.basis_, model_with_missing.coef_):
    if 'missing' in str(bf).lower():
        print(f"  {bf}: {coef:.3f}")
```

## Summary

This basic tutorial covered:
- Loading and preparing data for MARS modeling
- Fitting a basic MARS model
- Examining model components and feature importances
- Making predictions and evaluating the model
- Interpreting the model with partial dependence plots
- Hyperparameter tuning with scikit-learn
- Handling missing values directly with MARS

For more advanced usage, see our [Advanced Features tutorial](advanced_features.md).
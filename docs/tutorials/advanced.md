# Advanced Features Tutorial

This tutorial covers advanced features of mars including model interpretation, changepoint detection, cross-validation, and specialized health economic applications.

## Model Interpretation Tools

mars provides several tools for understanding model behavior:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate more complex data
X, y = make_regression(n_samples=1000, n_features=8, noise=0.1, random_state=42, 
                       effective_rank=8, tail_strength=0.5)

# Convert to DataFrame for better feature names
feature_names = [f"x{i}" for i in range(X.shape[1])]
X_df = pd.DataFrame(X, columns=feature_names)

X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42)

# Fit a MARS model with interpretability features enabled
model = earth.Earth(
    max_degree=2,      # Allow two-way interactions
    penalty=3.0,       # GCV penalty
    max_terms=21,      # Maximum number of terms
    minspan_alpha=0.05,  # Minimum span control
    endspan_alpha=0.05,  # End span control
    allow_linear=True,    # Allow linear terms
    feature_importance_type='gcv'  # Calculate feature importance
)

model.fit(X_train.values, y_train)

# Generate comprehensive model summary
print("Model Summary:")
print(f"Number of basis functions: {len(model.basis_) - 1}")  # -1 for intercept
print(f"Training R²: {model.score(X_train.values, y_train):.3f}")
print(f"Test R²: {model.score(X_test.values, y_test):.3f}")
print(f"GCV Score: {model.gcv_:.3f}")

# Detailed feature importance analysis
print("\nDetailed Feature Importance:")
print(f"Method used: {model.feature_importance_type}")
feature_importance = model.feature_importances_
for i, (name, imp) in enumerate(zip(feature_names, feature_importance)):
    print(f"  {name}: {imp:.4f}")

# Get more detailed breakdown of different importance metrics
print("\nAlternative Importance Metrics:")
if hasattr(model, 'summary_feature_importances'):
    print(model.summary_feature_importances())
```

## Partial Dependence and ICE Plots

Understanding how individual features affect predictions:

```python
from mars.explain import plot_partial_dependence, plot_individual_conditional_expectation

# Plot partial dependence for top 3 most important features
top_features_idx = np.argsort(feature_importance)[-3:][::-1]

fig, axes = plot_partial_dependence(
    model, 
    X_train.values, 
    top_features_idx,
    feature_names=feature_names,
    n_cols=3,
    figsize=(15, 5)
)

plt.suptitle('Partial Dependence Plots for Top 3 Features', fontsize=16)
plt.tight_layout()
plt.show()

# Generate ICE plots for the most important feature
most_important_idx = top_features_idx[0]

fig, ax = plot_individual_conditional_expectation(
    model,
    X_train.values,
    most_important_idx,
    feature_names=feature_names,
    n_grid_points=50,
    alpha=0.1
)

ax.set_title(f'Individual Conditional Expectation Plot for {feature_names[most_important_idx]}')
plt.show()
```

## Cross-Validation and Model Selection

mars provides specialized cross-validation helpers:

```python
from mars.cv import EarthCV
from sklearn.model_selection import KFold

# Define parameter grid for cross-validation
param_grid = {
    'max_degree': [1, 2],
    'penalty': [2.0, 3.0, 4.0],
    'max_terms': [11, 16, 21, 31]
}

# Initialize cross-validation model
cv_model = EarthCV(
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='r2',
    n_jobs=-1,  # Use all available cores
    verbose=1   # Show progress
)

# Fit the cross-validation model
cv_model.fit(X_train.values, y_train)

# Display results
print(f"Best parameters: {cv_model.best_params_}")
print(f"Best cross-validation score: {cv_model.best_score_:.3f}")

# Compare with original model
original_score = model.score(X_test.values, y_test)
cv_score = cv_model.score(X_test.values, y_test)

print(f"Original model test score: {original_score:.3f}")
print(f"Cross-validated model test score: {cv_score:.3f}")
```

## Changepoint Detection Capabilities

One of the powerful features of MARS is its ability to detect changepoints as knots in the data:

```python
# Create data with clear changepoints
np.random.seed(42)
n_points = 1000
x = np.linspace(0, 10, n_points)

# Simulate data with changepoints (structural breaks)
y_changepoints = (
    2 * x[:300] + np.random.normal(0, 0.5, 300)  # Linear growth until x=3
    + np.concatenate([
        np.full(200, 6) + np.random.normal(0, 0.5, 200),  # Flat segment (x=3 to x=5)
        0.5 * (x[500:700] - 5) + 6 + np.random.normal(0, 0.5, 200),  # Gentle slope (x=5 to x=7)
        -1 * (x[700:] - 7) + 7 + np.random.normal(0, 0.5, 300)  # Negative slope (x=7 to x=10)
    ])
)

# Fit MARS model to detect changepoints
changepoint_model = earth.Earth(
    max_degree=1,      # Only linear relationships for changepoint detection
    penalty=2.0,       # Lower penalty to allow more knots
    max_terms=21,      # Allow more terms to detect multiple changepoints
    minspan_alpha=0.01,  # Smaller minspan for more sensitive changepoint detection
    endspan_alpha=0.01
)

changepoint_model.fit(x.reshape(-1, 1), y_changepoints)

print(f"\nChangepoint Detection Results:")
print(f"Number of basis functions (knots): {len(changepoint_model.basis_) - 1}")
print(f"Knot locations: {[bf for bf in changepoint_model.basis_ if 'hinge' in str(bf).lower()]}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(x, y_changepoints, 'b.', alpha=0.5, label='Data', markersize=2)
plt.plot(x, changepoint_model.predict(x.reshape(-1, 1)), 'r-', linewidth=2, label='MARS Fit')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('MARS Changepoint Detection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Show the actual changepoints in the model
print("\nModel Knots (Changepoints):")
for i, bf in enumerate(changepoint_model.basis_):
    if i == 0:  # Intercept
        continue
    print(f"  {bf} (coefficient: {changepoint_model.coef_[i]:.3f})")
```

## Generalized Linear Models

mars supports logistic and Poisson regression through generalized linear models:

```python
from sklearn.datasets import make_classification

# Generate binary classification data
X_binary, y_binary = make_classification(
    n_samples=1000, 
    n_features=8, 
    n_informative=5, 
    n_redundant=1,
    n_clusters_per_class=1, 
    random_state=42
)

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
    X_binary, y_binary, test_size=0.2, random_state=42
)

# Fit logistic regression model with MARS basis functions
from mars.glm import GLMEarth

glm_model = GLMEarth(
    family='binomial',  # Logistic regression
    max_degree=2,
    penalty=3.0,
    max_terms=21
)

glm_model.fit(X_train_bin, y_train_bin)

# Evaluate the model
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred_proba = glm_model.predict_proba(X_test_bin)[:, 1]
y_pred = glm_model.predict(X_test_bin)

accuracy = accuracy_score(y_test_bin, y_pred)
auc = roc_auc_score(y_test_bin, y_pred_proba)

print(f"\nLogistic Regression Results:")
print(f"Accuracy: {accuracy:.3f}")
print(f"AUC: {auc:.3f}")

# Feature importance for logistic model
print(f"Feature Importances:")
for name, imp in zip([f"x{i}" for i in range(X_binary.shape[1])], glm_model.feature_importances_):
    print(f"  {name}: {imp:.4f}")
```

## Cross-Validation Helper for GLMs

Using cross-validation with GLMs:

```python
from mars.cv import GLMEarthCV

# Parameter grid for GLM
glm_param_grid = {
    'family': ['binomial'],
    'max_degree': [1, 2],
    'penalty': [2.0, 3.0, 4.0],
    'max_terms': [11, 16, 21]
}

# Cross-validate GLM model
cv_glm_model = GLMEarthCV(
    param_grid=glm_param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)

cv_glm_model.fit(X_train_bin, y_train_bin)

print(f"\nBest GLM parameters: {cv_glm_model.best_params_}")
print(f"Best GLM CV score: {cv_glm_model.best_score_:.3f}")
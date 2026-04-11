# Usage

## Basic Usage

Here's a basic example of how to use mars:

```python
import numpy as np
import mars as earth

# Generate sample data
X = np.random.rand(100, 3)
y = np.sin(X[:, 0]) + X[:, 1]

# Create and fit the model
model = earth.Earth(max_degree=1, penalty=3.0)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

# Get model summary
print(model.summary())
```

## Regression Examples

### Simple Regression

```python
from mars import EarthRegressor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample regression data
np.random.seed(42)
X = np.random.rand(200, 5)
y = (X[:, 0] > 0.5) * X[:, 1] + np.sin(X[:, 2]) + 0.1 * np.random.randn(200)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
regressor = EarthRegressor(max_degree=2, penalty=3.0)
regressor.fit(X_train, y_train)

# Make predictions
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

### With Cross-Validation

```python
from mars import EarthCV

# Use cross-validation to select hyperparameters
cv_model = EarthCV(
    max_degree=2, 
    penalty=3.0,
    cv=5,
    enable_classification=False  # For regression tasks
)
cv_model.fit(X_train, y_train)

# Make predictions with the best model
y_pred_cv = cv_model.predict(X_test)

# Evaluate the cross-validated model
mse_cv = mean_squared_error(y_test, y_pred_cv)
r2_cv = r2_score(y_test, y_pred_cv)

print(f"Cross-Validated MSE: {mse_cv:.4f}")
print(f"Cross-Validated R²: {r2_cv:.4f}")
```

## Classification Examples

```python
from mars import EarthClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Generate sample classification data
X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the classifier
classifier = EarthClassifier(max_degree=2, penalty=3.0)
classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

## Advanced Features

### Feature Importance

```python
# Calculate and display feature importances
importances = model.summary_feature_importances()
print("Feature Importances:")
for i, imp in enumerate(importances):
    print(f"Feature {i}: {imp:.4f}")
```

### Model Interpretation and Visualization

```python
from mars import plot_partial_dependence, plot_basis_functions, plot_residuals

# Plot partial dependence for important features
plot_partial_dependence(model, X, features=[0, 1])

# Plot the basis functions
plot_basis_functions(model)

# Plot residuals to check model assumptions
plot_residuals(model, X, y)
```

### Accessing Model Details

```python
# Get detailed model information
print("Model Summary:")
print(model.summary())

# Access coefficients and basis functions
print(f"Number of terms: {len(model.coef_)}")
print(f"Selected variables: {model.used_vars_}")

# Get the GCV score
print(f"GCV Score: {model.gcv_:.4f}")
```

## Parameters

The main parameters for the Earth model are:

- `max_degree`: Maximum degree of interaction for the model (default: 1)
  - Controls the maximum interaction between variables
  - Higher values allow more complex interactions but increase computation
  
- `penalty`: GCV penalty per knot (default: 3.0)
  - Used in GCV calculation during model selection
  - Higher values penalize model complexity more heavily
  
- `minspan`: Minimum number of observations between knots (default: 0)
  - Controls how close knots can be to each other
  - Higher values enforce smoother functions
  
- `endspan`: Minimum number of observations at the end of a variable (default: 0)
  - Controls knot placement near boundaries
  - Higher values prevent knots near data boundaries
  
- `max_terms`: Maximum number of terms in the model (default: 21)
  - Limits model complexity
  - Setting too high may lead to overfitting
  
- `max_knots`: Maximum number of knots per variable (default: 20)
  - Limits the number of knots placed on each variable
  - Controls granularity of model fitting

## Model Selection Tips

1. **Start Simple**: Begin with `max_degree=1` and default values
2. **Tune Complexity**: Gradually increase `max_degree` if needed
3. **Regularization**: Adjust `penalty` to balance model complexity and fit
4. **Validation**: Always validate models on held-out data

## Examples

You can find more detailed examples in the `examples` directory of the repository.
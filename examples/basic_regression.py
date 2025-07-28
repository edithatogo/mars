# -*- coding: utf-8 -*-
"""Basic regression demo using EarthRegressor."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pymars._sklearn_compat import EarthRegressor


def run_regression_example() -> None:
    """Run a simple regression example."""
    print("--- pymars Basic Regression Example ---")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 200
    X = np.random.rand(n_samples, 5)
    y = 2 * X[:, 0] + np.sin(3 * X[:, 1]) - 1.5 * X[:, 2] ** 2 + np.random.normal(0, 0.5, n_samples)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    print(f"Generated data: X shape {X.shape}, y shape {y.shape}")

    # Fit model
    model = EarthRegressor(max_degree=1, penalty=3.0, max_terms=20)
    model.fit(X_train, y_train)

    if hasattr(model, "summary"):
        print(model.summary())

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = model.score(X_test, y_test)

    print("\nEvaluation on Test Set:")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  R-squared (R2): {r2:.4f}")


if __name__ == "__main__":
    run_regression_example()

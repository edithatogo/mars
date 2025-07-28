# -*- coding: utf-8 -*-
"""Basic classification demo using EarthClassifier."""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pymars._sklearn_compat import EarthClassifier


def run_classification_example() -> None:
    """Run a simple classification example."""
    print("--- pymars Basic Classification Example ---")

    np.random.seed(123)
    n_samples = 200
    X = np.random.rand(n_samples, 5)
    latent_y = 2 * X[:, 0] + np.sin(3 * X[:, 1]) - 1.5 * X[:, 2] ** 2 + np.random.normal(0, 1.0, n_samples)
    y = (latent_y > np.median(latent_y)).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123, stratify=y
    )
    print(f"Generated data: X shape {X.shape}, y shape {y.shape}")

    model = EarthClassifier(max_degree=1, penalty=3.0, max_terms=15)
    model.fit(X_train, y_train)

    if hasattr(model, "earth_") and hasattr(model.earth_, "summary"):
        print(model.earth_.summary())

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nEvaluation on Test Set:")
    print(f"  Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    run_classification_example()

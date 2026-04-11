"""
Advanced example demonstrating complex Earth model usage with interpretability tools.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import pymars as earth


def advanced_example():
    print("Running advanced pymars example with interpretability...")

    # Generate a more complex synthetic dataset
    np.random.seed(42)
    n_samples = 200
    n_features = 5

    X = np.random.randn(n_samples, n_features)

    # Create a complex target with interactions and non-linearities
    y = (
        2 * X[:, 0]
        + 1.5 * np.sin(X[:, 0])
        + 0.8 * X[:, 1] * X[:, 2]
        + 0.5 * np.maximum(0, X[:, 3] - 1)  # Hinge function
        + 0.3 * np.maximum(0, -X[:, 4] - 0.5)  # Another hinge
        + np.random.normal(0, 0.1, n_samples)
    )  # Add some noise

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Dataset shape: {X_train.shape}")
    print(f"Number of features: {n_features}")

    # Fit Earth model with feature importance
    model = earth.Earth(
        max_degree=3,  # Allow higher-order interactions
        penalty=3.0,
        max_terms=20,
        feature_importance_type="gcv",  # Use GCV-based importance
    )

    model.fit(X_train, y_train)

    print("\nModel summary:")
    print(f"Number of selected basis functions: {len(model.basis_)}")
    print(f"GCV score: {model.gcv_:.4f}")

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("\nPerformance metrics:")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}")

    # Show feature importances
    print("\nFeature importances (GCV-based):")
    for i, importance in enumerate(model.feature_importances_):
        print(f"  Feature {i}: {importance:.4f}")

    # Plot predictions vs actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(y_train, y_train_pred, alpha=0.6)
    axes[0].plot(
        [y_train.min(), y_train.max()], [y_train.min(), y_train.max()], "r--", lw=2
    )
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predicted")
    axes[0].set_title(f"Training Data (R² = {train_r2:.3f})")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y_test, y_test_pred, alpha=0.6)
    axes[1].plot(
        [y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2
    )
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predicted")
    axes[1].set_title(f"Test Data (R² = {test_r2:.3f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("advanced_example_predictions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nPrediction plots saved to 'advanced_example_predictions.png'")

    # Use interpretability tools if available
    try:
        # Plot partial dependence for first 3 features
        fig, axes = earth.plot_partial_dependence(
            model,
            X_train,
            features=[0, 1, 2],
            feature_names=[f"Feature {i}" for i in [0, 1, 2]],
            figsize=(15, 5),
        )
        plt.savefig(
            "advanced_example_partial_dependence.png", dpi=150, bbox_inches="tight"
        )
        plt.close()
        print(
            "Partial dependence plots saved to 'advanced_example_partial_dependence.png'"
        )

        # Generate model explanation
        explanation = earth.get_model_explanation(
            model, X_train, feature_names=[f"Feature {i}" for i in range(n_features)]
        )

        print("\nModel explanation:")
        print(f"  Number of features: {explanation['model_summary']['n_features']}")
        print(
            f"  Number of basis functions: {explanation['model_summary']['n_basis_functions']}"
        )
        print(f"  GCV score: {explanation['model_summary']['gcv_score']:.4f}")

    except AttributeError:
        print("\nInterpretability tools not available in this version")

    return model, test_r2


if __name__ == "__main__":
    model, score = advanced_example()
    print(f"\nAdvanced example completed with test R² = {score:.4f}")

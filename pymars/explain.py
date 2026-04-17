from __future__ import annotations

"""
Advanced interpretability features for Earth models.
"""

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import partial_dependence

from .earth import Earth


def _normalize_axes(axes: Any) -> list[Any]:
    """Return a flat list of matplotlib axes."""
    flat_axes = np.atleast_1d(axes).ravel().tolist()
    return list(flat_axes)


def plot_partial_dependence(
    earth_model: Earth,
    X: Any,
    features: Sequence[int],
    feature_names: list[str] | None = None,
    grid_resolution: int = 100,
    n_cols: int = 2,
    figsize: tuple[int, int] | None = None,
) -> tuple[Any, Any]:
    """
    Create partial dependence plots for Earth model features.
    """
    if not earth_model.fitted_:
        raise ValueError("Model must be fitted before plotting partial dependence")

    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    try:
        fig, axes = plt.subplots(
            nrows=(len(features) + n_cols - 1) // n_cols,
            ncols=n_cols,
            figsize=figsize or (15, 5 * ((len(features) + n_cols - 1) // n_cols)),
        )

        axes = _normalize_axes(axes)

        for i, feature in enumerate(features):
            pd_result = partial_dependence(
                earth_model, X, [feature], grid_resolution=grid_resolution
            )

            axes[i].plot(pd_result["values"][0], pd_result["average"][0])
            axes[i].set_xlabel(
                feature_names[feature]
                if feature_names is not None and feature < len(feature_names)
                else f"Feature {feature}"
            )
            axes[i].set_ylabel("Partial dependence")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"Partial Dependence: Feature {feature}")

        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig, axes
    except ValueError as e:
        print(
            f"Warning: sklearn partial dependence failed ({e}), using custom implementation"
        )

        fig, axes = plt.subplots(
            nrows=(len(features) + n_cols - 1) // n_cols,
            ncols=n_cols,
            figsize=figsize or (15, 5 * ((len(features) + n_cols - 1) // n_cols)),
        )

        axes = _normalize_axes(axes)

        for i, feature in enumerate(features):
            feature_values = np.linspace(
                X[:, feature].min(), X[:, feature].max(), grid_resolution
            )

            predictions = np.zeros(grid_resolution)
            X_temp = X.copy()

            for j, val in enumerate(feature_values):
                X_temp[:, feature] = val
                pred = earth_model.predict(X_temp)
                predictions[j] = np.mean(pred)

            axes[i].plot(feature_values, predictions)
            axes[i].set_xlabel(
                feature_names[feature]
                if feature_names is not None and feature < len(feature_names)
                else f"Feature {feature}"
            )
            axes[i].set_ylabel("Partial dependence")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"Partial Dependence: Feature {feature}")

        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig, axes


def plot_individual_conditional_expectation(
    earth_model: Earth,
    X: Any,
    features: Sequence[int],
    feature_names: list[str] | None = None,
    grid_resolution: int = 50,
    n_cols: int = 2,
    n_samples: int | None = None,
    figsize: tuple[int, int] | None = None,
) -> tuple[Any, Any]:
    """
    Create Individual Conditional Expectation (ICE) plots for Earth model features.
    """
    if not earth_model.fitted_:
        raise ValueError("Model must be fitted before plotting ICE")

    X = np.asarray(X)
    n_samples = min(n_samples or 50, X.shape[0])

    sample_indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
    X_sample = X[sample_indices]

    fig, axes = plt.subplots(
        nrows=(len(features) + n_cols - 1) // n_cols,
        ncols=n_cols,
        figsize=figsize or (15, 5 * ((len(features) + n_cols - 1) // n_cols)),
    )

    axes = _normalize_axes(axes)

    for i, feature in enumerate(features):
        feature_values = np.linspace(
            X[:, feature].min(), X[:, feature].max(), grid_resolution
        )

        ice_values = np.zeros((n_samples, grid_resolution))
        for j, val in enumerate(feature_values):
            X_temp = X_sample.copy()
            X_temp[:, feature] = val
            ice_values[:, j] = earth_model.predict(X_temp)

        for k in range(n_samples):
            axes[i].plot(feature_values, ice_values[k, :], alpha=0.5, linewidth=0.8)

        mean_values = np.mean(ice_values, axis=0)
        axes[i].plot(feature_values, mean_values, color="red", linewidth=2, label="PDP")
        axes[i].legend()

        axes[i].set_xlabel(
            feature_names[feature]
            if feature_names is not None and feature < len(feature_names)
            else f"Feature {feature}"
        )
        axes[i].set_ylabel("Prediction")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"ICE Plot: Feature {feature}")

    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig, axes


def get_model_explanation(
    earth_model: Earth, X: Any, feature_names: list[str] | None = None
) -> dict[str, Any]:
    """
    Generate a comprehensive explanation of the Earth model.
    """
    if not earth_model.fitted_:
        raise ValueError("Model must be fitted before generating explanation")

    X_array = np.asarray(X)
    basis = earth_model.basis_ or []
    coef = earth_model.coef_ if earth_model.coef_ is not None else np.array([])

    explanation: dict[str, Any] = {
        "model_summary": {
            "n_features": X_array.shape[1] if X_array.ndim > 1 else 1,
            "n_basis_functions": len(basis),
            "gcv_score": earth_model.gcv_,
            "r2_score": earth_model.score(X_array, earth_model.predict(X_array))
            if X_array.shape[0] < 10000
            else "Too large to compute",
        },
        "basis_functions": [],
        "feature_importance": {},
    }

    if basis:
        for i, bf in enumerate(basis):
            explanation["basis_functions"].append(
                {
                    "index": i,
                    "type": type(bf).__name__,
                    "description": str(bf),
                    "coefficient": coef[i] if i < len(coef) else 0,
                }
            )

    if (
        hasattr(earth_model, "feature_importances_")
        and earth_model.feature_importances_ is not None
    ):
        explanation["feature_importance"] = {
            "values": earth_model.feature_importances_.tolist(),
            "feature_names": feature_names
            or [f"feature_{i}" for i in range(len(earth_model.feature_importances_))],
        }

    return explanation

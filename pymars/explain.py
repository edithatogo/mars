"""
Advanced interpretability features for Earth models.
"""
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import partial_dependence

from .earth import Earth


def plot_partial_dependence(
    earth_model: Earth,
    X,
    features,
    feature_names: Optional[list] = None,
    grid_resolution: int = 100,
    n_cols: int = 2,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Create partial dependence plots for Earth model features.
    
    Parameters
    ----------
    earth_model : Earth
        Fitted Earth model
    X : array-like
        Training data used to fit the model
    features : list
        List of feature indices or names to plot
    feature_names : list, optional
        Names of features for labeling
    grid_resolution : int, default=100
        Number of points to evaluate each feature
    n_cols : int, default=2
        Number of columns in the subplot grid
    figsize : tuple, optional
        Figure size as (width, height)
    
    Returns
    -------
    fig : matplotlib figure
        The figure object
    axes : matplotlib axes
        The axes objects
    """
    if not earth_model.fitted_:
        raise ValueError("Model must be fitted before plotting partial dependence")

    # Convert to numpy array if needed
    if not isinstance(X, np.ndarray):
        X = np.asarray(X)

    # Try to use sklearn's partial dependence, but if it fails with Earth model,
    # implement a custom approach
    try:
        # Calculate partial dependence for each feature
        fig, axes = plt.subplots(
            nrows=(len(features) + n_cols - 1) // n_cols,
            ncols=n_cols,
            figsize=figsize or (15, 5 * ((len(features) + n_cols - 1) // n_cols))
        )

        if len(features) == 1:
            axes = [axes]
        elif len(features) <= n_cols:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, feature in enumerate(features):
            # Calculate partial dependence
            pd_result = partial_dependence(
                earth_model, X, [feature], grid_resolution=grid_resolution
            )

            # Plot
            axes[i].plot(pd_result["values"][0], pd_result["average"][0])
            axes[i].set_xlabel(feature_names[feature] if feature_names and
                              isinstance(feature, int) and feature < len(feature_names)
                              else f"Feature {feature}")
            axes[i].set_ylabel("Partial dependence")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"Partial Dependence: Feature {feature}")

        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig, axes
    except ValueError as e:
        # If sklearn partial dependence fails, implement a simplified version
        # For Earth models, we can implement our own version
        print(f"Warning: sklearn partial dependence failed ({e}), using custom implementation")

        fig, axes = plt.subplots(
            nrows=(len(features) + n_cols - 1) // n_cols,
            ncols=n_cols,
            figsize=figsize or (15, 5 * ((len(features) + n_cols - 1) // n_cols))
        )

        if len(features) == 1:
            axes = [axes]
        elif len(features) <= n_cols:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

        for i, feature in enumerate(features):
            # Create grid of values for the feature
            feature_values = np.linspace(
                X[:, feature].min(),
                X[:, feature].max(),
                grid_resolution
            )

            # For each feature value, create a copy of X with that feature varied
            predictions = np.zeros(grid_resolution)
            X_temp = X.copy()

            for j, val in enumerate(feature_values):
                X_temp[:, feature] = val
                pred = earth_model.predict(X_temp)
                predictions[j] = np.mean(pred)

            # Plot
            axes[i].plot(feature_values, predictions)
            axes[i].set_xlabel(feature_names[feature] if feature_names and
                              isinstance(feature, int) and feature < len(feature_names)
                              else f"Feature {feature}")
            axes[i].set_ylabel("Partial dependence")
            axes[i].grid(True, alpha=0.3)
            axes[i].set_title(f"Partial Dependence: Feature {feature}")

        # Hide unused subplots
        for i in range(len(features), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        return fig, axes


def plot_individual_conditional_expectation(
    earth_model: Earth,
    X,
    features,
    feature_names: Optional[list] = None,
    grid_resolution: int = 50,
    n_cols: int = 2,
    n_samples: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None
):
    """
    Create Individual Conditional Expectation (ICE) plots for Earth model features.
    
    Parameters
    ----------
    earth_model : Earth
        Fitted Earth model
    X : array-like
        Training data used to fit the model
    features : list
        List of feature indices to plot
    feature_names : list, optional
        Names of features for labeling
    grid_resolution : int, default=50
        Number of points to evaluate each feature
    n_cols : int, default=2
        Number of columns in the subplot grid
    n_samples : int, optional
        Number of samples to show ICE curves for (default: min(50, n_samples))
    figsize : tuple, optional
        Figure size as (width, height)
    
    Returns
    -------
    fig : matplotlib figure
        The figure object
    axes : matplotlib axes
        The axes objects
    """
    if not earth_model.fitted_:
        raise ValueError("Model must be fitted before plotting ICE")

    X = np.asarray(X)
    n_samples = min(n_samples or 50, X.shape[0])

    # Sample some instances to show ICE curves for
    sample_indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
    X_sample = X[sample_indices]

    fig, axes = plt.subplots(
        nrows=(len(features) + n_cols - 1) // n_cols,
        ncols=n_cols,
        figsize=figsize or (15, 5 * ((len(features) + n_cols - 1) // n_cols))
    )

    if len(features) == 1:
        axes = [axes]
    elif len(features) <= n_cols:
        axes = axes if hasattr(axes, '__len__') else [axes]
    else:
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for i, feature in enumerate(features):
        # Create grid of values for the feature
        feature_values = np.linspace(
            X[:, feature].min(),
            X[:, feature].max(),
            grid_resolution
        )

        # For each sample, calculate predictions across feature values
        ice_values = np.zeros((n_samples, grid_resolution))
        for j, val in enumerate(feature_values):
            X_temp = X_sample.copy()
            X_temp[:, feature] = val
            ice_values[:, j] = earth_model.predict(X_temp)

        # Plot ICE curves
        for k in range(n_samples):
            axes[i].plot(feature_values, ice_values[k, :], alpha=0.5, linewidth=0.8)

        # Also plot the average (partial dependence)
        mean_values = np.mean(ice_values, axis=0)
        axes[i].plot(feature_values, mean_values, color='red', linewidth=2, label='PDP')
        axes[i].legend()

        axes[i].set_xlabel(feature_names[feature] if feature_names and
                          isinstance(feature, int) and feature < len(feature_names)
                          else f"Feature {feature}")
        axes[i].set_ylabel("Prediction")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"ICE Plot: Feature {feature}")

    # Hide unused subplots
    for i in range(len(features), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    return fig, axes


def get_model_explanation(earth_model: Earth, X, feature_names: Optional[list] = None):
    """
    Generate a comprehensive explanation of the Earth model.
    
    Parameters
    ----------
    earth_model : Earth
        Fitted Earth model
    X : array-like
        Training data used to fit the model
    feature_names : list, optional
        Names of features
    
    Returns
    -------
    explanation : dict
        Dictionary containing model explanations
    """
    if not earth_model.fitted_:
        raise ValueError("Model must be fitted before generating explanation")

    explanation = {
        'model_summary': {
            'n_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]),
            'n_basis_functions': len(earth_model.basis_) if earth_model.basis_ else 0,
            'gcv_score': earth_model.gcv_,
            'r2_score': earth_model.score(X, earth_model.predict(X)) if X.shape[0] < 10000 else "Too large to compute",
        },
        'basis_functions': [],
        'feature_importance': {}
    }

    # Extract basis function details
    if earth_model.basis_:
        for i, bf in enumerate(earth_model.basis_):
            bf_info = {
                'index': i,
                'type': type(bf).__name__,
                'description': str(bf),
                'coefficient': earth_model.coef_[i] if i < len(earth_model.coef_) else 0
            }
            explanation['basis_functions'].append(bf_info)

    # Include feature importance if available
    if hasattr(earth_model, 'feature_importances_') and earth_model.feature_importances_ is not None:
        explanation['feature_importance'] = {
            'values': earth_model.feature_importances_.tolist(),
            'feature_names': feature_names or [f'feature_{i}' for i in range(len(earth_model.feature_importances_))]
        }

    return explanation

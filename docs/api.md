# API Reference

The pymars library provides a comprehensive API for Multivariate Adaptive Regression Splines (MARS) with full scikit-learn compatibility.

## Core Classes

### Earth

The main MARS implementation class that provides the core algorithm functionality:

::: pymars.Earth
    handler: python
    options:
      members: 
        - fit
        - predict
        - score
        - transform
        - summary
        - gcv_
        - rss_
        - feature_importances_
        - feature_importance_type
        - coef_
        - basis_
        - max_terms
        - max_degree
        - penalty
        - minspan_alpha
        - endspan_alpha
        - allow_linear
      show_root_heading: true
      show_root_full_path: false

### EarthRegressor

Scikit-learn compatible regressor wrapper for the MARS algorithm:

::: pymars.EarthRegressor
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

### EarthClassifier

Scikit-learn compatible classifier wrapper for the MARS algorithm:

::: pymars.EarthClassifier
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

## Generalized Linear Models

### GLMEarth

Extended class for Generalized Linear Models with MARS basis functions:

::: pymars.GLMEarth
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

## Cross-Validation Helpers

### EarthCV

Grid search helper with cross-validation for MARS models:

::: pymars.EarthCV
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

## Advanced Features

### Cached Earth

Earth model with caching for improved performance:

::: pymars.CachedEarth
    handler: python
    options:
      members: 
        - enable_basis_function_caching
        - disable_basis_function_caching
        - get_basis_function_cache_info
      show_root_heading: true
      show_root_full_path: false

### Parallel Earth

Earth models with parallel computation support:

::: pymars.ParallelEarth
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

### Sparse Earth

Earth models with sparse matrix support:

::: pymars.SparseEarth
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

## Utilities and Explanation Tools

### Explain Module

Functions for model interpretation and explanation:

::: pymars.explain
    handler: python
    options:
      members: 
        - get_model_explanation
        - plot_individual_conditional_expectation
        - plot_partial_dependence
      show_root_full_path: false

### Plot Module

Visualization utilities for MARS models:

::: pymars.plot
    handler: python
    options:
      members: 
        - plot_basis_functions
        - plot_residuals
        - plot_partial_dependence
      show_root_full_path: false

## Advanced GLM Extensions

### Advanced GLMs

Advanced generalized linear models with MARS basis functions:

::: pymars.AdvancedGLMEarth
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

### Specialized Regressors

Specialized regressors for different distribution families:

::: pymars.GammaRegressor
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

::: pymars.TweedieRegressor
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

::: pymars.InverseGaussianRegressor
    handler: python
    options:
      members: true
      show_root_heading: true
      show_root_full_path: false

## Feature Importance Methods

pymars implements multiple methods for calculating feature importance:

- **nb_subsets**: Number of subsets method counting basis function appearances
- **gcv**: Generalized Cross-Validation improvement method
- **rss**: Residual Sum of Squares reduction method

These can be accessed through the `feature_importances_` attribute of fitted models.
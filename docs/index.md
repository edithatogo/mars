# mars Documentation

Welcome to the documentation for mars, a pure Python implementation of Multivariate Adaptive Regression Splines (MARS).

## Origins and Relationship to py-earth

`pymars` is a pure Python re-implementation of the Multivariate Adaptive Regression Splines (MARS) algorithm, and it owes a great deal to the original `py-earth` library by Jason Friedman, which was itself an implementation of the MARS algorithm developed by Jerome Friedman.

The `py-earth` library was written in Cython for performance, but has since been archived and is no longer actively maintained. You can find the original `py-earth` repository here: [scikit-learn-contrib/py-earth](https://github.com/scikit-learn-contrib/py-earth).

`pymars` aims to be a spiritual successor to `py-earth`, providing a pure Python implementation that is easy to install and use, while maintaining compatibility with the scikit-learn ecosystem. It is designed to be a drop-in replacement for `py-earth` in many cases, with a similar API and functionality.

## What is MARS?

Multivariate Adaptive Regression Splines (MARS) is a non-parametric regression technique that automatically models non-linearities and interactions between variables. It models the relationship between variables using piecewise linear basis functions (hinge functions), making it particularly effective for complex datasets with non-linear relationships.

## Key Features

- **Pure Python**: Easy to install and use across different platforms
- **Scikit-learn Compatible**: Integrates with the scikit-learn ecosystem (estimators, pipelines, model selection tools)
- **MARS Algorithm**: Implements the core MARS fitting procedure, including:
  - Forward pass to select basis functions (both hinge and linear terms)
  - Pruning pass using Generalized Cross-Validation (GCV) to prevent overfitting
  - Support for interaction terms (including interactions involving linear terms)
  - Refined `minspan` and `endspan` controls for knot placement, aligning more closely with `py-earth` behavior
- **Feature Importance**: Calculation of feature importances using methods like 'nb_subsets', 'gcv', and 'rss'
- **Regression and Classification**: Provides `EarthRegressor` and `EarthClassifier` classes
- **Generalized Linear Models**: The `GLMEarth` subclass fits logistic or Poisson models
- **Cross-Validation Helper**: The `EarthCV` class integrates with scikit-learn's model selection utilities
- **Plotting Utilities**: Simple diagnostics built on `matplotlib`
- **Advanced Interpretability**: Partial dependence plots, Individual Conditional Expectation (ICE) plots, and model explanation tools

## How MARS Works

The MARS algorithm builds models in two stages:

1. **Forward Stage**: The algorithm adds basis functions in a greedy fashion, selecting the ones that most reduce the model's residual error.

2. **Backward Stage**: The algorithm removes basis functions using GCV to prevent overfitting.

This results in a model that can capture complex non-linear relationships while remaining interpretable.

## When to Use MARS

MARS is particularly useful when you have:

- Complex non-linear relationships in your data
- Need for interpretability in your model
- Mixed variable types (continuous and categorical)
- Need for automatic feature selection
- Limited theoretical understanding of the relationships between variables

## Installation

For installation instructions, see the [Installation](installation.md) page.

## Quick Start

For usage examples, see the [Usage](usage.md) page.

## API Reference

For detailed information about the classes and functions, see the [API Reference](api/index.md).

## Contributing

Contributions are welcome! For more information, see the [Contributing](contributing.md) page.
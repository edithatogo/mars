# mars: Pure Python Earth (Multivariate Adaptive Regression Splines)

![CI](https://github.com/edithatogo/mars/actions/workflows/ci.yml/badge.svg)
![Security](https://github.com/edithatogo/mars/actions/workflows/security.yml/badge.svg)
![Code Quality](https://github.com/edithatogo/mars/actions/workflows/code-quality.yml/badge.svg)
![Documentation](https://github.com/edithatogo/mars/actions/workflows/docs.yml/badge.svg)
![Code Coverage](https://codecov.io/gh/edithatogo/mars/branch/main/graph/badge.svg)
![PyPI](https://img.shields.io/pypi/v/mars.svg)
![Python Version](https://img.shields.io/pypi/pyversions/mars.svg)
![License](https://img.shields.io/github/license/edithatogo/mars.svg)

## Overview

**mars** is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS), also known as Earth. The goal is to provide an easy-to-install, scikit-learn compatible version without C/Cython dependencies.

MARS is a non-parametric regression technique that automatically models nonlinearities and interactions between variables. The method works by creating a piecewise linear model with basis functions that can capture non-linear relationships and interactions automatically.

## Key Features

- **Pure Python**: Easy to install and use across different platforms
- **Scikit-learn Compatible**: Integrates with the scikit-learn ecosystem (estimators, pipelines, model selection tools)
- **MARS Algorithm**: Implements the core MARS algorithm with:
  - Forward pass to select basis functions (both hinge and linear terms)
  - Pruning pass using Generalized Cross-Validation (GCV) to prevent overfitting
  - Support for interaction terms (including interactions involving linear terms)
  - Refined `minspan` and `endspan` controls for knot placement
- **Feature Importance**: Multiple methods for calculating feature importance ('nb_subsets', 'gcv', 'rss')
- **Missing Value Handling**: Robust handling of missing data using specialized basis functions
- **Categorical Variable Support**: Direct handling of categorical variables
- **Generalized Linear Models**: Extensions for logistic and Poisson regression
- **Cross-Validation Helper**: Simplified cross-validation with scikit-learn integration
- **Interpretability Tools**: Built-in explanation functions including partial dependence plots
- **Changepoint Detection**: Automatic identification of knots as changepoints in the data

## Motivation

mars was developed to address specific needs in health economic outcomes research, particularly in the analysis of complex health system reforms such as New Zealand's Pae Ora (Healthy Futures) Act 2022. The analysis of such reforms is complicated by the presence of multiple confounding factors, including COVID-19 pandemic effects, systemic changes, and dozens of concurrent policy modifications.

Traditional approaches focusing solely on dates for intervention analysis were insufficient; instead, changepoint detection methods were required to identify significant shifts in health outcomes and utilization patterns. The MARS approach of auto-fitting knots and optimizing to a specific number of knots proved particularly useful for health economic analysis.

The journey toward mars began with the R implementation of MARS ("earth"), but the author's primary workflow was in Python with scikit-learn. The existing Python implementation "py-earth" by Jason Friedman had limitations including Python 2 compatibility issues and difficulty integrating into automated machine learning tools. These practical challenges motivated the development of mars as a pure Python implementation maintaining full scikit-learn compatibility.

## Health Economic Applications

mars is particularly valuable for health economic outcomes research where understanding complex relationships between health outcomes, costs, and utilization patterns is crucial. The automatic identification of non-linearities and interactions makes MARS particularly well-suited for health economic applications where:

- Healthcare costs often increase exponentially with age and morbidity
- The effects of socioeconomic factors may have threshold effects
- Interactions between demographic, clinical, and geographic factors are important
- Health disparities exist across different population subgroups

## Getting Started

For installation instructions, see the [Installation page](installation.md). For usage examples, see the [Usage page](usage.md).

## Contributing

We welcome contributions to mars! Please see our contributing guidelines in the repository.

## License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.
# pymars Documentation

Welcome to the documentation for `pymars`.

`pymars` is a pure Python implementation of Multivariate Adaptive Regression Splines (MARS), designed to be compatible with scikit-learn.

## Contents (Planned)

*   **Introduction**
    *   What is MARS?
    *   Features of `pymars`
*   **Installation**
*   **User Guide**
    *   Basic Usage
    *   Regression with `EarthRegressor`
    *   Classification with `EarthClassifier`
    *   Understanding MARS Model Components (Basis Functions, Coefficients)
    *   Interpreting Model Summaries
    *   Hyperparameter Tuning
    *   Using `GLMEarth` for logistic and Poisson models
    *   Cross-validation with `EarthCV`
    *   Plotting model diagnostics
*   **API Reference**
    *   `pymars.earth.Earth`
    *   `pymars.glm.GLMEarth`
    *   `pymars.cv.EarthCV`
    *   `pymars.sklearn_compat.EarthRegressor`
    *   `pymars.sklearn_compat.EarthClassifier`
    *   Basis Functions (`pymars._basis`)
    *   Utility functions (`pymars._util`)
    *   Plotting utilities (`pymars.plot`)
*   **Examples**
    *   Links to example notebooks or scripts.
*   **Contributing**
*   **Changelog**

## Installation

Install the latest development version from the cloned repository:

```bash
git clone https://github.com/your-repository-url/pymars.git
cd pymars
pip install -e .
```

Once a release is published on PyPI you will be able to install it directly:

```bash
pip install pymars
```

This documentation is currently under construction. Please refer to the main `README.md` and `ROADMAP.md` for project status and plans.

---
*This file serves as a placeholder for more detailed documentation, potentially generated using tools like Sphinx.*
